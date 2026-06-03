"""Speculative-microbatch expert-reuse kill-test (codex decisive cheap test for the ONLY remaining
exact lever). If K consecutive decode tokens (a speculative verify batch) route to OVERLAPPING experts
per layer, then ONE expert weight-sweep serves multiple positions -> amortization changes the reuse
unit (the only physical lever left at request-batch=1, exact same-precision). If reuse C/U ~= 1
(memoryless -> no overlap), speculative amortization is DEAD.

Measured from existing target-decode traces (oracle-draft upper bound = real consecutive target tokens).
Per layer, per window of K consecutive tokens: C = K*top_k expert-slots, U = unique (layer,expert),
reuse = C/U. GPU-fetch break-even (codex): each fetched expert must serve >= 0.782/0.167 ~= 4.7
positions. All printed English. Core params: no defaults.
"""
import argparse, json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser(description="Speculative expert-reuse vs K kill-test")
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--max_traces", type=int, required=True)
    ap.add_argument("--ks", type=str, required=True, help="comma list of speculative verify lengths K")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load(trace_dir, max_traces):
    files = sorted(glob.glob(str(Path(trace_dir) / "dec_*.pt")))[:max_traces]
    man = json.loads((Path(trace_dir) / "manifest.json").read_text())
    seqs = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        seq = [[[int(e) for e in topk] for topk in pl] for (_t, pl) in d["steps"]]
        if seq: seqs.append(seq)
    return seqs, man["num_layers"], man["experts"], man["top_k"]


def main():
    a = parse_args()
    seqs, L, E, top_k = load(a.trace_dir, a.max_traces)
    ks = [int(x) for x in a.ks.split(",")]
    print(f"[data] seqs={len(seqs)} layers={L} experts={E} top_k={top_k}", flush=True)
    rows = []
    for K in ks:
        reuse_vals = []; U_vals = []
        for seq in seqs:
            for start in range(0, len(seq) - K + 1, K):  # non-overlapping windows of K tokens
                window = seq[start:start + K]
                for l in range(L):
                    slots = [e for tok in window for e in tok[l]]   # C = K*top_k expert-slots at layer l
                    C = len(slots); U = len(set(slots))
                    if C > 0:
                        reuse_vals.append(C / U); U_vals.append(U)
        reuse = float(np.mean(reuse_vals)); Umean = float(np.mean(U_vals))
        # per-position served experts under grouped serve = U/K (vs top_k at K=1)
        per_pos = Umean / K
        rows.append({"K": K, "reuse_C_over_U": reuse, "unique_U_per_layer": Umean,
                     "experts_per_position": per_pos, "vs_topk": per_pos / top_k,
                     "gpu_fetch_breakeven_pos_needed": 4.7})
        print(f"K={K:>3} reuse(C/U)={reuse:.3f} unique_U/layer={Umean:.2f} experts/pos={per_pos:.3f} "
              f"(vs top_k={top_k}: {per_pos/top_k:.2f}x) -> amortization {'YES' if reuse>1.3 else 'weak/NO'}", flush=True)
    Path(a.out).write_text(json.dumps({"rows": rows, "top_k": top_k}, indent=2))
    # verdict
    big = rows[-1]
    print(f"\n[verdict] at K={big['K']}: reuse={big['reuse_C_over_U']:.2f}. "
          f"GPU-fetch needs >=4.7 positions/expert; CPU-grouped benefits if reuse>1. "
          f"{'Lever EXISTS' if big['reuse_C_over_U']>1.3 else 'Memoryless -> amortization WEAK, likely incremental'}", flush=True)


if __name__ == "__main__":
    main()
