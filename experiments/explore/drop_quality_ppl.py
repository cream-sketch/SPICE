"""Gating #2: quality cost (PPL) of importance-based expert dropping in real Qwen MoE.

度量 miss-handling 的 task-performance 轴: 按 gate 权重 drop 最低 rank 的 routed 专家,
测 WikiText perplexity vs 不 drop. Qwen 有常驻 shared expert; routed 是其上增量.
若 drop rank-4 / rank-3+4 的 PPL 退化很小 -> importance-aware miss-handling 头room 大.
"""
from __future__ import annotations
import argparse, json, types, math
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DROP_N = 0  # 全局: 丢弃 routing_weights 最低的 DROP_N 个 routed 专家

def patched_moe_forward(mlp, hidden_states):
    b, s, d = hidden_states.shape
    h = hidden_states.view(-1, d)
    router_logits = mlp.gate(h)
    rw = F.softmax(router_logits, dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, mlp.top_k, dim=-1)
    if mlp.norm_topk_prob:
        rw = rw / rw.sum(dim=-1, keepdim=True)
    rw = rw.to(h.dtype)
    if DROP_N > 0:
        # 置零最低 DROP_N 个 rank 的权重 (topk 已降序, 最后 DROP_N 列最低)
        rw = rw.clone(); rw[:, mlp.top_k - DROP_N:] = 0.0
    final = torch.zeros((b * s, d), dtype=h.dtype, device=h.device)
    mask = F.one_hot(sel, num_classes=mlp.num_experts).permute(2, 1, 0)
    for ei in range(mlp.num_experts):
        idx, topx = torch.where(mask[ei])
        if topx.numel() == 0:
            continue
        cur = h[None, topx].reshape(-1, d)
        out = mlp.experts[ei](cur) * rw[topx, idx, None]
        final.index_add_(0, topx, out.to(h.dtype))
    shared = mlp.shared_expert(h)
    shared = F.sigmoid(mlp.shared_expert_gate(h)) * shared
    final = final + shared
    return final.view(b, s, d), router_logits

@torch.no_grad()
def ppl(model, tok, texts, device, max_len):
    nll, ntok = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        ids = enc["input_ids"]
        if ids.shape[1] < 2: continue
        out = model(input_ids=ids, labels=ids)
        # HF returns mean loss over tokens; weight by token count
        n = ids.shape[1] - 1
        nll += out.loss.item() * n; ntok += n
    return math.exp(nll / max(1, ntok))

def main():
    global DROP_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--drops", type=str, default="0,1,2,3")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,
                                                 local_files_only=True, low_cpu_mem_usage=True).to(device).eval()
    # patch all MoE blocks
    import transformers.models.qwen2_moe.modeling_qwen2_moe as M
    for mod in model.modules():
        if isinstance(mod, M.Qwen2MoeSparseMoeBlock):
            mod.forward = types.MethodType(patched_moe_forward, mod)
    texts = [l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:args.max_samples]
    res = {}
    for dn in [int(x) for x in args.drops.split(",")]:
        DROP_N = dn
        res[f"drop_{dn}"] = ppl(model, tok, texts, device, args.max_len)
        print(f"DROP_N={dn} (drop lowest {dn} of top-4 routed) PPL={res[f'drop_{dn}']:.4f}")
    base = res.get("drop_0", 1.0)
    res_rel = {k: (v/base - 1.0)*100 for k, v in res.items()}
    out = {"ppl": res, "ppl_rel_pct_vs_full": res_rel}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
