AUDIT my observation experiments for measurement errors that could FALSELY produce a "no exploitable structure" conclusion. Be brutal. Read the actual files in experiments/. Output to stdout. I previously had a real data bug (collect_hf_moe_traces double-softmaxed DeepSeek gate output, losing expert IDs) -- so I make mistakes; find them.

The conclusion I'm about to commit to (a characterization paper claiming offloaded-MoE expert access is near-uniform/memoryless/independent -> not cacheable/predictable/compressible) rests on these scripts + numbers. If any measurement is wrong in a way that UNDER-detects real structure, the whole conclusion flips.

Files + the numbers they produced (Qwen / DeepSeek):
1. experiments/obs_routing_structure.py
   popularity Gini 0.17/0.25; static hot-set hit (cache N hottest/layer) N=32->0.65/0.67; autocorr top-k Jaccard lag1 0.13/0.19 -> lag8 0.06/0.09; token_conditional_consistency 0.23/0.16; working_set 37%/43% over 8-token window.
2. experiments/cross_token_info.py
   AUC of full router prob[t] predicting same-layer selection at t+H: h1 0.59 -> window8 ~0.51; frequency ~0.58; selected/recency ~0.57.
3. experiments/obs_expert_contribution.py
   ||routed||/||h_in|| = 0.098; ||shared||/||routed|| = 1.38; contribution gini 0.33; gate-order == contribution-order (top 1.00, bottom 0.96).
4. experiments/obs_cross_expert_structure.py
   per-layer stack experts [N,P], centered SVD: shared-mean energy 2%; rank90/N = 0.88 (both models). Claim: experts independent, no shared basis.
5. experiments/microbatch_multiplex.py
   unique experts/layer for B sampled tokens -> byte ratio; Qwen B=32 2.46x reduction, matches coupon-collector theory.
6. experiments/expert_lowrank_spectrum.py (exp0a)
   per-expert weight effective_rank ~850/1408 (near full-rank).

Check specifically:
A. Correctness bugs in each metric (AUC rank formula; Jaccard random-baseline; Gini; SVD axis/centering; token_conditional alignment of input_ids vs routing positions under truncation/padding; microbatch sampling = independent streams?).
B. Did I use the RIGHT INFORMATION? For CROSS-TOKEN predictability I used the 60-dim router PROB, NOT the 2048-dim hidden state. Is that a signal-choice error that under-states predictability, or is next-token routing fundamentally unpredictable from current state (because it needs the next input token)? Should I test a hidden-state predictor / (token,position) / prev-token context / clustering before concluding "memoryless"?
C. Is obs_cross_expert_structure even the right test for "compressibility"? It tests weight-space shared basis across experts. Should it instead test activation-aware / input-conditioned shared structure, or product-quantization, before concluding "not compressible"?
D. Is the static-hotset / autocorr methodology biased (e.g., mixing many short sequences, cross-sequence concatenation, too few traces)?
E. Any single error that, if fixed, would reveal exploitable structure and flip the conclusion? Rank the risks.
For each issue: the bug, why it matters, the exact fix, and whether it could flip "no structure". If the measurements are sound, say so explicitly and state the conclusion is safe.
