# Drop-policy comparison: gate vs rank vs mass(top-p) — 2026-06-02

Full-seq teacher-forced (on-policy quality), 24 texts x 256 len. drop = on-miss low-importance drop.
DATA CORRECTION: earlier importance_ds.json claimed DeepSeek "rank-1=0.957 winner-take-all" — that was an ARTIFACT of double-softmax in trace collection (collect_hf_moe_traces softmaxed the MoEGate topk_weight output again). ds_gate_vs_rank uses the REAL topk_weight; DeepSeek routing is more moderate. Always use real router weights, not trace-derived probs, for weight-magnitude claims.

## QWEN (mass covers to 35.8% drop)
gate: 30.5%/13.88, 57%/14.43, 76%/18.58 ; rank: 25%/13.81, 50%/14.53, 75%/19.21 ; mass: 25.4%/13.88, 35.8%/14.48
Matched ~25%: rank 13.81 ~ mass 13.88 ~ gate(30%) 13.88. ~35%: mass 14.48 vs gate~14.0/rank~14.1 (mass worse).
=> on Qwen the three policies are within noise; NO policy dominates.

## DEEPSEEK (real weights; mass low-p pending)
gate: 44.1%/16.81 ; rank: 16.7%/15.90, 33.3%/16.40, 50%/17.32, 66.7%/19.87 ; mass(<=0.9): 12.8%/15.86
(mass low-p sweep 0.85..0.5 running to get matched drop.)

## DIRECTION UPDATE (per methodology)
top-p mass does NOT Pareto-dominate gate AND rank on both models (Qwen: ~tied/worse). The miss-DROP policy axis (gate/rank/mass) appears SATURATED — no policy is non-incrementally better cross-model. The drop LEVER exists (Pareto real) but is known (SpecMD/AdapMoE). => lossy miss-handling, on the policy axis, is INCREMENTAL. Non-incrementality must come from elsewhere: bandit/adaptive scheduler across hardware/workload shift (Gate C, untested), or a different action space.

## DeepSeek mass low-p completed -> FINAL drop-policy verdict
DeepSeek mass: 21.2%/16.06, 28.1%/16.41, 41.3%/17.17, 52.3%/18.12, 61.1%/19.50
Matched: ~21% mass~=rank(16.0); ~33% rank 16.40 < mass ~16.69; ~50% rank 17.32 < mass ~17.92.
=> DeepSeek: rank >= gate > mass (mass WORST). Qwen: all within noise, rank marginally best.
FINAL: no drop policy dominates cross-model; simple RANK is the most robust. top-p mass (entropy-adaptive) FALSIFIED. The miss-handling DROP-POLICY axis is SATURATED and INCREMENTAL (lever known to SpecMD/AdapMoE). This direction is CLOSED as a non-incremental contribution.
Remaining non-incremental bets: (A) conservative contextual bandit adaptive scheduler (regret across hardware/workload shifts; risk=roofline-derivable), or (C) consolidate a strong systems paper (unified verified resource-constrained scheduler + validated Pareto + clean ablations). Decision pending codex strategic consult (notes/codex_run_strategic.log).
