"""Cross-layer route predictability probe (the prerequisite for SPICE's same-token cross-layer
LoRE forecaster). Question: from a token's layer-l hidden state, how well can layer (l+h)'s routed
experts be predicted? If a cheap linear probe's recall@k >> a popularity baseline, cross-layer
routing is predictable -> a LoRE forecaster is viable. If ~= popularity, the cross-layer forecast
idea is dead (save the training). NOT oracle-prefetch -- this measures predictability on real routes.

Real model (Qwen2MoE or DeepSeek-V2-Lite), real text, real per-layer hidden + routing.
真实跨层路由可预测性探针:训 LoRE 前的决定性前提检查。
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_TEXT_FALLBACK = "The history of mixture of experts models in large language modeling is long and varied. "


def _moe_block(layer):
    """Return the MoE block of a decoder layer across archs (Qwen .mlp, DeepSeek .mlp, Mixtral
    .block_sparse_moe), or None if this layer is dense."""
    for name in ("mlp", "block_sparse_moe"):
        blk = getattr(layer, name, None)
        if blk is not None and hasattr(blk, "experts") and hasattr(blk, "gate") \
                and len(getattr(blk, "experts", [])) > 0:
            return blk
    return None


def capture(model, ids, dev):
    """Run one forward; capture per-MoE-layer (input hidden [S,d], top-k expert idx [S,k]).
    Handles 3 gate conventions: Linear->logits (Qwen/Mixtral), MoEGate->(topk_idx,...) (DeepSeek)."""
    hid, routes = [], []
    handles = []

    def pre_hook(store):
        def h(m, inp):
            store.append(inp[0].detach()[0].float().cpu())   # [S, d] -> CPU (works for sharded models)
        return h

    def gate_hook(store, k):
        # gate input is FLATTENED [B*S, d] -> output [B*S, *] (no batch dim); for batch=1 that's [S, *].
        def h(m, inp, out):
            first = out[0] if isinstance(out, tuple) else out
            if isinstance(out, tuple) and first.dtype in (torch.int32, torch.int64):
                store.append(first.detach()[:, :k].cpu())                        # DeepSeek: top-k idx [S,k]
            else:
                store.append(first.detach().float().topk(k, dim=-1).indices.cpu())  # logits [S,E] -> [S,k]
        return h

    for layer in model.model.layers:
        blk = _moe_block(layer)
        if blk is None:
            continue
        handles.append(blk.register_forward_pre_hook(pre_hook(hid)))
        tk = getattr(blk, "num_experts_per_tok", getattr(blk, "top_k", 6))
        handles.append(blk.gate.register_forward_hook(gate_hook(routes, tk)))
    with torch.inference_mode():
        model(input_ids=ids.to(dev), use_cache=False)
    for h in handles:
        h.remove()
    return hid, routes


def recall_at_k(pred_logits, true_idx, k):
    pred_top = pred_logits.topk(k, dim=-1).indices            # [N, k]
    hit = 0; tot = 0
    for n in range(true_idx.shape[0]):
        t = set(true_idx[n].tolist()); p = set(pred_top[n].tolist())
        hit += len(t & p); tot += len(t)
    return hit / max(1, tot)


def train_probe(X, Y_idx, n_exp, k, dev, steps=300):
    """Linear probe X[N,d] -> n_exp logits, multi-label top-k target. Return held-out recall@k."""
    N = X.shape[0]; ntr = int(N * 0.8)
    Xtr, Xte = X[:ntr], X[ntr:]; Ytr, Yte = Y_idx[:ntr], Y_idx[ntr:]
    target = torch.zeros(ntr, n_exp, device=dev)
    target.scatter_(1, Ytr.to(dev), 1.0)
    W = torch.zeros(X.shape[1], n_exp, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W], lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xtr.to(dev) @ W, target)
        loss.backward(); opt.step()
    with torch.inference_mode():
        return recall_at_k((Xte.to(dev) @ W).cpu(), Yte, k)


def main():
    p = argparse.ArgumentParser(description="cross-layer route predictability probe")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--n_tokens", type=int, default=1024)
    p.add_argument("--leads", default="1,2,3,4,6,8")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device_map", default="", help="set to 'auto' for models too big for one GPU (Mixtral)")
    args = p.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=args.trust_remote_code)
    load_kw = dict(torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
                   trust_remote_code=args.trust_remote_code)
    if args.device_map:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map=args.device_map, **load_kw).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kw).to(dev).eval()
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(t for t in ds["text"] if t and t.strip())
    except Exception:
        text = _TEXT_FALLBACK * 4000
    ids = tok(text, return_tensors="pt").input_ids[:, :args.n_tokens]

    hid, routes = capture(model, ids, dev)
    nL = len(hid)
    n_exp = int(max(r.max().item() for r in routes)) + 1
    k = routes[0].shape[1]
    print(f"[probe] MoE layers={nL} experts={n_exp} top_k={k} tokens={hid[0].shape[0]} d={hid[0].shape[1]}")

    # popularity baseline per target layer
    for lead in [int(x) for x in args.leads.split(",")]:
        probe_recalls, pop_recalls = [], []
        for t in range(lead, nL):
            X = hid[t - lead]                       # source layer hidden
            Y = routes[t]                           # target layer routing
            probe_recalls.append(train_probe(X, Y, n_exp, k, dev))
            # popularity baseline: always predict the top-k most frequent experts of target layer t
            freq = torch.zeros(n_exp);
            for n in range(Y.shape[0]):
                freq[Y[n]] += 1
            pop_top = set(freq.topk(k).indices.tolist())
            ph = sum(len(set(Y[n].tolist()) & pop_top) for n in range(Y.shape[0])) / (Y.shape[0] * k)
            pop_recalls.append(ph)
        pr = sum(probe_recalls) / len(probe_recalls); po = sum(pop_recalls) / len(pop_recalls)
        print(f"[lead {lead}] linear_probe_recall@{k}={pr:.3f}  popularity_recall@{k}={po:.3f}  "
              f"gain={pr - po:+.3f}  (avg over {len(probe_recalls)} layer pairs)")


if __name__ == "__main__":
    main()
