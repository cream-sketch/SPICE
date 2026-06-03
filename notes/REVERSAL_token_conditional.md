# CONCLUSION REVERSAL: routing IS strongly token-determined (audit fix) — 2026-06-02
codex audit found the token_conditional_consistency metric was mis-scaled (divided by n*top_k; max possible = 1/top_k). Corrected leave-one-out token->expert prediction:
- Qwen: recall 0.582 vs random 0.067 (8.7x). early-layer 0.575, late 0.553.
- DeepSeek: recall 0.595 vs random 0.094 (6.3x). early 0.655, late 0.560.
=> token IDENTITY alone (training-free token->modal-expert table, leave-one-out) predicts ~58-60% of top-k experts. Routing is substantially TOKEN-DETERMINED. Earlier "not token-determined (0.23)" was a SCALING BUG. This REOPENS the search.
Reconciles with low autocorr: different consecutive tokens route differently (low autocorr) but the SAME token-id routes consistently (high token-conditional).
Signal=token-id (timely: current id known pre-layer; next id via LM-head/draft), verified/cheap (frequency table), actionable (prefetch token's modal experts), resource-valued (58% recall). Cross-token capable via next-token-id -> the cross-token bridge I wrongly buried.
OTHER audit bugs to fix before any further conclusion: contribution rank agreement is tautological (sorted-index compare); AUC mishandles ties; Qwen router probs ALSO double-softmaxed; cross-token predictability never tested with hidden-state/token-id/position; "memoryless" overstated (lag1 = 3.7-3.9x random).
