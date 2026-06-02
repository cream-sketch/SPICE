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

## CONFIRMED at n=16 texts (cleaner, near-monotonic)
| matched drop% | gate stall/PPL | rank stall/PPL | gate advantage |
|---|---|---|---|
| ~41% | 50.8 / 14.18 | 50.8 / 14.78 | same stall, PPL -0.60 |
| ~62% | 26.8 / 16.92 | 25.2 / 18.35 | PPL -1.44 |
| ~20% | 71 / 14.31 | 77 / 14.21 | ~tied (low-drop region) |
Conclusion: verified gate-weight thresholding Pareto-dominates fixed rank-dropping at matched drop, clearly in the mid-high region. The verified per-token gate-weight novelty vs SpecMD drop-by-rank HOLDS. (16 texts x 96 tokens, Qwen 10% cache/bw12.)

## DeepSeek-V2-Lite generalization (24 texts, full-seq teacher-forced) -- gate>rank does NOT generalize
gate: thr0.02->0.9% drop/15.73 PPL; thr0.05->44.1%/16.81; thr0.10->82%/25.69 (bimodal weights -> coarse drop control)
rank: n1->16.7%/15.90; n2->33.3%/16.40; n3->50%/17.32; n4->66.7%/19.87
Matched: ~33% rank 16.40 vs gate ~16.53; ~50% rank 17.32 vs gate ~18.19 -> RANK slightly better at high drop on DeepSeek.
INFERENCE: DeepSeek routing is near winner-take-all (rank-1~=0.96 gate mass) -> rank order == weight order, and discrete rank gives finer drop control -> gate has no advantage. The "verified gate-weight > rank" novelty is MODEL-DEPENDENT: holds for spread-routing MoE (Qwen top-4) NOT winner-take-all (DeepSeek top-6).
DIRECTION UPDATE: miss-handling DROP lever generalizes (cheap drop both models) BUT the gate>rank differentiator does NOT -> lossy miss-handling alone looks incremental (lever known to SpecMD/AdapMoE; the one differentiator is model-specific). Non-incrementality now hinges on D (miss-shadow oracle) / C (bandit) / a NEW signal-decision coupling.
