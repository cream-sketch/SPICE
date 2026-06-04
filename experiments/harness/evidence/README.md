# harness/evidence/ — quality & routing-structure measurement

Standalone tools (nothing imports them). Measure the quality cost / characterization
that justifies the scheduler's decisions.

- `drop_quality_ppl.py` / `deepseek_drop_quality.py` — WikiText PPL cost of substituting (dropping) experts.
  `--select {lowest,random,highest}` gives the gate-vs-random ablation (lowest +1% vs random +40% PPL).
- `qwen_gate_vs_rank.py` / `ds_gate_vs_rank.py` — gate / rank / mass drop-quality ablation.
- `obs_routing_structure.py` / `obs_expert_contribution.py` / `obs_cross_expert_structure.py` — routing
  observation battery (near-memoryless access, shared-expert dominance, no low-rank/shared basis).
- `token_conditional_analysis.py` — token-id predicts ~58-60% of top-k routing.
- `ds_collect_routing.py` — collect real DeepSeek expert ids + 64-dim scores.
