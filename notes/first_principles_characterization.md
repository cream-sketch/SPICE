# First-principles characterization of offloaded-MoE decode (paper C backbone)
2026-06-02. Cross-model: Qwen1.5-MoE-A2.7B (60e,k4) + DeepSeek-V2-Lite (64e,k6), real WikiText routing/weights.

## The complete picture (8 observation experiments, all consistent + cross-model)
| MoE information axis | measurement (Qwen / DeepSeek) | exploitable? |
|---|---|---|
| per-expert weight rank (exp0a) | effective_rank ~850/1408; rank64=18% energy | NO low-rank compression |
| CROSS-expert structure | shared-mean 2% energy; rank90/N=0.88 / 0.88 | NO shared basis; experts independent |
| access popularity | Gini 0.17 / 0.25; cache 50% experts->65% hit | NO hot-set |
| temporal autocorrelation | top-k Jaccard lag1 0.13->lag8 0.06 / 0.19->0.09 | NO memory (near-random by lag2-8) |
| token-conditional | consistency 0.23 / 0.16 | NOT token-determined |
| cross-token next-use predictability | full-prob AUC 0.59->0.50 over window; freq 0.58 | NO realizable signal |
| output contribution vs gate | gate order == contribution order (top 100%/bot 96%) | NOT a new signal |
| within-token cross-layer | draft recall 0.70-1.0 (h1-8) | YES -- the ONLY structure (SPICE prefetch) |

## Consequences (the ceiling)
- Eviction oracle (Belady) 0.90 hit @20% cache vs realizable LRU/LS 0.72: the gap is CROSS-TOKEN, future-only info, provably uncapturable (above measurements).
- Routed experts are a ~10% perturbation on the residual; shared expert contributes 1.38x the entire routed sum. ~1.6GB/token moved to change the layer output ~10%.
- => Offloaded MoE routed experts are a high-entropy, diverse, INDEPENDENT, UNIFORMLY-accessed weight set: structurally resistant to caching, prediction, AND structural compression.

## The only escapes (all known / not new mechanisms)
1. byte reduction per expert: quantization (HOBBIT/int4), drop (AdapMoE/importance-drop, lossy).
2. batch multiplexing: uniformity -> coupon-collector amortization; byte/token Qwen B=32->2.46x, B=64->4.38x; DeepSeek B=32->3.31x (matches theory E*(1-(1-k/E)^B)/(kB)). = standard big-batch weight amortization; needs serving concurrency.
3. within-token pipelining: SPICE prefetch; optimal lookahead is a small CONSTANT (depth 2); roofline l_min over-prefetches +10% (kill-shot).

## Honest conclusion
No non-incremental MECHANISM exists in single-stream GPU offloaded-MoE miss/eviction/prefetch -- the structure required (cacheability/predictability/compressibility) is measurably absent. The non-incremental-IN-KIND contribution is this CHARACTERIZATION + ceiling: it debunks the implicit assumptions of the offloaded-MoE caching literature (MoE-Infinity/SpecMD/FineMoE/SP-MoE/HOBBIT/AdapMoE) and reframes the problem (cache -> byte-reduce / multiplex / pipeline). Plus a verified on-policy replay methodology + cross-model evidence. This is paper C.
