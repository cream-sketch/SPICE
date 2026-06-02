"""Gating: per-rank gate-weight distribution of real Qwen MoE routing.

度量真实 Qwen top-k 路由里, 各 rank 专家的归一化 gate 权重分布.
若 rank-3/4 权重很小 -> miss 时 drop 低重要专家近乎免费 -> miss-handling 有头room.
读取 qwen routing trace (router_probs), 不需加载模型.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--norm_topk", action="store_true", help="renormalize over top-k (Qwen norm_topk_prob=False)")
    args = ap.parse_args()
    man = json.loads((Path(args.trace_dir)/"manifest.json").read_text())
    rank_weight_sum = torch.zeros(args.top_k)
    rank_weight_sqsum = torch.zeros(args.top_k)
    n = 0
    for f in man["trace_files"]:
        d = torch.load(Path(args.trace_dir)/f, map_location="cpu", weights_only=False)
        for probs in d["router_probs"]:
            p = probs.float()
            if p.ndim == 3: p = p.reshape(-1, p.shape[-1])
            tw, _ = torch.topk(p, k=args.top_k, dim=-1)  # [N, k] already softmax probs
            if args.norm_topk:
                tw = tw / tw.sum(-1, keepdim=True).clamp_min(1e-9)
            rank_weight_sum += tw.sum(0)
            rank_weight_sqsum += (tw*tw).sum(0)
            n += tw.shape[0]
    mean = (rank_weight_sum / max(1,n))
    var = (rank_weight_sqsum / max(1,n)) - mean*mean
    std = var.clamp_min(0).sqrt()
    res = {"tokens_layers": n, "top_k": args.top_k, "norm_topk": args.norm_topk,
           "mean_weight_by_rank": mean.tolist(), "std_weight_by_rank": std.tolist(),
           "cum_weight_by_rank": torch.cumsum(mean,0).tolist()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
