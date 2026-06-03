Brainstorm non-incremental directions, GROUNDED in the measured first-principles structure below (not assumptions). Be creative but rigorous. Output to stdout. Give 3-6 concrete ideas, each as (Signal -> Decision -> Utility -> why non-incremental vs named prior art -> single fast kill experiment), then RANK them and name the single best next experiment. It is OK to propose changing the problem boundary.

## Measured first-principles structure (real Qwen1.5-MoE-A2.7B decode routing, WikiText)
- Popularity Gini = 0.174 -> experts used nearly UNIFORMLY per layer, NO hot-set. Static "cache N hottest/layer": N=8->21% hit, N=16->38%, N=32(53% of experts)->65%. Caching needs >half the experts for decent hit.
- Routing autocorrelation (top-4 set Jaccard, same layer): lag1=0.13, lag2=0.067, lag8=0.056 (random ~0.035). Very short memory; cross-token reuse barely above random.
- Token-conditional consistency = 0.23 (1=token fully determines expert; ~0.07=random). Routing is CONTEXT-driven, NOT token-determined -> token->expert tables won't work.
- Cross-token next-use predictability: full 60-dim router prob predicts next-token same-layer selection at AUC 0.59 (h=1), decaying to ~0.50 over an 8-token window; frequency is best at long window but only AUC 0.58. So NO realizable per-token signal predicts cross-token reuse well.
- Working set = 22/60 experts (37%) per layer over an 8-token window.
- CONTRAST: WITHIN-token cross-layer IS structured: a training-free draft (frozen attn+router, shared-expert-only propagation) predicts the next layers' top-4 with recall 0.70-1.0 across horizon 1-8. SPICE already exploits this for prefetch.
- Eviction oracle (Belady) gets 0.90 hit at 20% cache vs realizable LRU/LS 0.72 -- a LARGE gap that is provably CROSS-TOKEN and (per the above) NOT capturable by realizable signals.
- Fundamental bottleneck: per token a MoE layer needs top-4 experts x 24 layers ~= 96 expert-loads ~= 1.6 GB at 17MB/expert; at 5GB/s that is ~326 ms/token if uncached. Caching is near-useless (near-random access) so the bytes must be either hidden (within-token prefetch; we showed depth-2 constant is optimal, roofline over-prefetches) or REDUCED.

## Already pruned (do not re-propose):
draft->eviction (fails, cross-token); lossless miss-shadow @batch=1 (no win); drop-policy gate/rank/top-p-mass (saturated, no policy dominates; lever known to SpecMD/AdapMoE); bandit/adaptive-depth scheduler (constant depth-2 optimal -> nothing to adapt). Verified importance-drop Pareto is real but incremental.

## Prior art that occupies the obvious couplings:
MoE-Infinity (activation-trace cache), SpecMD (Least-Stale eviction), Skliar (cache-aware routing), HOBBIT (mixed-precision miss), FineMoE (semantic prefetch), SP-MoE/MoE-SpeQ (draft-token speculative prefetch), AdapMoE (sensitivity-adaptive active count).

## Questions
1. Given the access pattern is near-uniform/memoryless cross-token, is the RIGHT non-incremental move to STOP trying to predict/cache and instead attack the 1.6GB/token bytes (e.g., exploit the within-token cross-layer structure or the near-uniform structure for a fundamentally different execution/transfer scheme)? Propose concrete mechanisms.
2. Is there a non-incremental angle in the near-UNIFORM structure itself (e.g., uniform access -> a statistical/streaming/coded-transfer or a fixed-schedule execution that beats demand-driven caching)?
3. Does the within-token cross-layer predictability (draft recall 0.7) enable anything beyond prefetch -- e.g., reordering/fusing expert transfers across layers, or computing in a transfer-optimal order?
4. Is the strongest honest output a CHARACTERIZATION+LIMIT paper ("MoE expert access is near-uniform/memoryless -> a provable ceiling on caching; gains must come from byte-reduction or within-token pipelining"), with a derived bound? If so, what bound/theorem and what experiments make it rigorous and publishable?
5. Would changing the boundary (batch>1: cross-request expert sharing; or NVMe tier; or prefill) reopen a non-incremental coupling that the measured single-stream structure forecloses? Which is most promising and what's the kill experiment?
Be brutally honest; if the answer is "characterization paper", say it, but try hard for a real mechanism first.
