# A kill-shot verdict: bandit DEAD; constant small depth optimal — 2026-06-02

oracle-prefetch latency sim, Qwen real traces, regimes bw{5,12,24} x cache{72,144,288}, prefetch-depth policies.
RESULT: oracle-best depth = 2 in ALL 9 regimes. static depth-2 (tuned@bw12/cap144) regret = 0.0% everywhere.
roofline l_min depth (12 @low bw, 6 @high bw) regret 4-15% (mean 10.1%) — OVER-prefetches, pollutes cache (LS evicts far-layer prefetches), and at low bw the transfer is unhideable anyway.
=> The prefetch-depth knob is NOT regime-sensitive: a small CONSTANT depth dominates. Nothing to adapt -> bandit/roofline adaptive scheduling is decoration AND the roofline formula is actively harmful (+10%). Bet A KILLED.
This is itself a paper-C characterization result: debunks adaptive-lookahead (SPICE's own) and learned scheduling for the depth knob; recommends a small constant lookahead.

DECISION: non-incremental search concluded. No surviving non-incremental mechanism in single-GPU offloaded-MoE miss/eviction/prefetch. CONSOLIDATE paper C (notes/paper_plan_C.md).
Next for C: (1) source-only equal-budget baselines (SpecMD/AdapMoE/HOBBIT/MoE-Infinity/FineMoE) via codex moe-baseline-integrity; (2) finalize cross-model importance-drop Pareto + breakdown; (3) replay-vs-live validation; (4) bw x cache shift study (constant-depth finding is part of it).
