# Ablation: verified gate-weight vs fixed rank miss-admission (2026-06-02)

online teacher-forced, Qwen 10% cache, bw12, 4 texts x 96 tokens. drop = on-miss low-importance drop.
LESSON: must compare at MATCHED drop rate; unmatched points misled me into a premature "rank competitive" read.

## Matched-drop comparison (lower stall + lower PPL = better)
| drop% | policy | stall/tok | PPL |
|---|---|---|---|
| 0    | both baseline | 104.0 | 10.31 |
| ~21  | rank keep-3   | 77.8  | 10.24 |
| ~27  | gate 0.05     | 69.9  | 10.49 |
| ~42  | rank keep-2   | 51.2  | 10.63 |
| ~44.5| gate 0.07     | 48.3  | 10.13 |  <- dominates rank keep-2 on BOTH axes
| ~63  | gate 0.115    | 25.2  | 12.34 |  <- beats rank keep-1 PPL at same stall
| ~64  | rank keep-1   | 25.0  | 12.63 |

## Conclusion
- At matched drop rate, verified gate-weight thresholding Pareto-dominates fixed rank dropping in the mid-high region (per-token adaptivity: drop by weight not position). Supports the "verified post-router gate" novelty differentiator vs SpecMD drop-by-rank.
- CAVEAT: 4-text sample -> PPL noisy (gate PPL non-monotonic 10.49@27% vs 10.13@44%). MUST rerun with many more texts (e.g. 30-50) to confirm significance before claiming domination.
- Note: gate 0.07 PPL 10.13 < baseline 10.31 (dropping 44% lowest-weight misses slightly IMPROVES PPL -- a DropExpert-like regularization; verify it is not noise).

## Next
1. Bigger-sample gate-vs-rank (30+ texts, 128 tokens) to confirm domination + significance.
2. DeepSeek-V2-Lite same ablation (generalization).
3. Gate A (miss-shadow recovery) + Gate B (bandit vs tuned static) -- the novelty-critical unvalidated pillars.
