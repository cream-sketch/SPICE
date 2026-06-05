"""Single-GPU offloaded-MoE TTFT/TPOT benchmark across datasets x policies. For each (dataset, policy):
prefill real prompts (TTFT = prefill latency) then greedy-decode (TPOT = mean per-token latency),
averaged over N prompts. Offload runtime + policies come from offload_mixtral.setup (cache auto-sized
by GPU free memory; no hand-picked cache). 真实数据集上的单卡 offload TTFT/TPOT 矩阵。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from offload_mixtral import setup


def _wikitext_prompts(tok, n, plen):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    stream = tok("\n".join(t for t in ds["text"] if t and t.strip()), return_tensors="pt").input_ids[0]
    span = stream.numel() // n
    return [stream[i * span: i * span + plen].unsqueeze(0) for i in range(n)]


def _humaneval_prompts(tok, n, plen):
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    out = []
    for i in range(min(n, len(ds))):
        out.append(tok(ds[i]["prompt"], return_tensors="pt").input_ids[:, :plen])
    return out


def _gsm8k_prompts(tok, n, plen):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in range(min(n, len(ds))):
        out.append(tok(ds[i]["question"], return_tensors="pt").input_ids[:, :plen])
    return out


def _narrativeqa_prompts(tok, n, plen, jsonl, ctx_tokens):
    rows = [json.loads(l) for l in open(jsonl)][:n]
    out = []
    for r in rows:
        text = r["context"] + "\n\nQuestion: " + r["input"] + "\nAnswer:"
        ids = tok(text, return_tensors="pt", truncation=True, max_length=ctx_tokens).input_ids
        out.append(ids)
    return out


def load_prompts(name, tok, n, plen, args):
    if name == "wikitext":
        return _wikitext_prompts(tok, n, plen)
    if name == "humaneval":
        return _humaneval_prompts(tok, n, plen)
    if name == "gsm8k":
        return _gsm8k_prompts(tok, n, plen)
    if name == "narrativeqa":
        return _narrativeqa_prompts(tok, n, plen, args.narrativeqa_jsonl, args.ctx_tokens)
    raise ValueError(name)


@torch.inference_mode()
def measure(model, prompt_ids, decode_tokens, warmup, dev):
    """Return (ttft_ms, tpot_ms). TTFT = prefill latency; TPOT = mean per-token decode latency."""
    ids = prompt_ids.to(dev)
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize(dev)
    ttft = (time.perf_counter() - t0) * 1000.0
    kv = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    t_start = None
    for k in range(decode_tokens):
        if k == warmup:
            torch.cuda.synchronize(dev); t_start = time.perf_counter()
        out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize(dev)
    tpot = (time.perf_counter() - t_start) * 1000.0 / (decode_tokens - warmup)
    return ttft, tpot


def main():
    p = argparse.ArgumentParser(description="offloaded-MoE TTFT/TPOT matrix (datasets x policies)")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--policies", default="cpu_serve,on_demand,fused_compressed,split_cpu_gpu")
    p.add_argument("--datasets", default="wikitext,humaneval,gsm8k,narrativeqa")
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--prompt_len", type=int, default=128)
    p.add_argument("--ctx_tokens", type=int, default=4096, help="narrativeqa context truncation")
    p.add_argument("--decode_tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--reserve_gb", type=float, default=4.0)
    p.add_argument("--split_g", type=float, default=0.5)
    p.add_argument("--narrativeqa_jsonl", default="/data/jiangjunmin/abc/spice_offload/narrativeqa.jsonl")
    p.add_argument("--out", default="")
    args = p.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    results = {}
    for policy in args.policies.split(","):
        model, tok, info = setup(args.model_dir, args.gpu, policy, args.reserve_gb, args.split_g)
        print(f"[setup] policy={policy} {info}", flush=True)
        for dsname in args.datasets.split(","):
            prompts = load_prompts(dsname, tok, args.n_prompts, args.prompt_len, args)
            ttfts, tpots = [], []
            for pr in prompts:
                ttft, tpot = measure(model, pr, args.decode_tokens, args.warmup, dev)
                ttfts.append(ttft); tpots.append(tpot)
            ttft_m = sum(ttfts) / len(ttfts); tpot_m = sum(tpots) / len(tpots)
            results[(policy, dsname)] = (ttft_m, tpot_m)
            print(f"[result] policy={policy} dataset={dsname} TTFT_ms={ttft_m:.1f} TPOT_ms={tpot_m:.2f} "
                  f"(n={len(prompts)}, plen~{prompts[0].shape[1]})", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n=== MATRIX TTFT_ms / TPOT_ms ===")
    pols = args.policies.split(","); dss = args.datasets.split(",")
    print(f"{'dataset':14s} " + " ".join(f"{p:>22s}" for p in pols))
    for ds in dss:
        cells = []
        for pol in pols:
            t = results.get((pol, ds))
            cells.append(f"{t[0]:7.1f}/{t[1]:6.2f}" if t else "   -   ")
        print(f"{ds:14s} " + " ".join(f"{c:>22s}" for c in cells))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()}, f, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
