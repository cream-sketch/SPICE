"""Estimate cross-layer expert transition statistics from collected MoE traces.

Input: trace .pt files produced by collect_hf_moe_traces.py (router logits per
layer per token). Output: transition[l][e][e'] = P(expert e' selected at layer
l+1 | expert e selected at layer l), plus per-layer expert popularity priors.

Used by the AdapMoE-style baseline (cross-layer prefetching) in
qwen3_offload_bench.py via --transition_stats.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=48)
    ap.add_argument("--num_experts", type=int, default=128)
    ap.add_argument("--smoothing", type=float, default=1e-3)
    args = ap.parse_args()

    L, E, K = args.num_layers, args.num_experts, args.top_k
    counts = torch.zeros(L - 1, E, E, dtype=torch.float64)
    popularity = torch.zeros(L, E, dtype=torch.float64)
    n_tokens = 0

    files = sorted(glob.glob(str(Path(args.trace_dir) / "*.pt")))
    if not files:
        raise SystemExit(f"no .pt traces in {args.trace_dir}")
    for f in files:
        blob = torch.load(f, map_location="cpu")
        logits = blob.get("router_logits")
        if logits is None:
            continue
        # logits: list of L tensors [tokens, E] (one per layer, in layer order)
        if isinstance(logits, list):
            per_layer = logits
        else:
            per_layer = list(logits)
        if len(per_layer) % L != 0:
            print(f"warn: {f} has {len(per_layer)} router captures, not a multiple of {L}; skipping")
            continue
        # captures may repeat per forward pass: reshape into groups of L
        for g in range(len(per_layer) // L):
            group = per_layer[g * L : (g + 1) * L]
            topk = [t.topk(K, dim=-1).indices for t in group]  # L x [tokens, K]
            tokens = topk[0].shape[0]
            n_tokens += tokens
            for l in range(L):
                popularity[l].index_add_(
                    0, topk[l].reshape(-1),
                    torch.ones(topk[l].numel(), dtype=torch.float64),
                )
            for l in range(L - 1):
                cur, nxt = topk[l], topk[l + 1]  # [tokens, K]
                for k1 in range(K):
                    for k2 in range(K):
                        idx = cur[:, k1] * E + nxt[:, k2]
                        counts[l].view(-1).index_add_(
                            0, idx, torch.ones(tokens, dtype=torch.float64)
                        )
        print(f"[stats] {f}: cumulative tokens={n_tokens}", flush=True)

    transition = (counts + args.smoothing) / (counts + args.smoothing).sum(dim=-1, keepdim=True)
    popularity = popularity / popularity.sum(dim=-1, keepdim=True).clamp(min=1)
    torch.save(
        {
            "transition": transition.to(torch.float32),
            "popularity": popularity.to(torch.float32),
            "num_tokens": n_tokens,
            "top_k": K,
            "source_files": files,
        },
        args.out,
    )
    print(f"[stats] saved {args.out}: transition {tuple(transition.shape)}, tokens={n_tokens}")


if __name__ == "__main__":
    main()
