I need a genuinely non-incremental approach for offloaded MoE decode, beyond fetch/drop/precision/cache (those are saturated/known). Pressure-test these two first-principles candidates. Be brutal, cite prior art, output to stdout. Goal: which (if either) is non-incremental, and the single decisive experiment for each.

MEASURED first-principles facts (real Qwen1.5-MoE + DeepSeek-V2-Lite decode):
- routed experts perturb the hidden by only ~10% (||routed||/||h||=0.098); shared expert norm = 1.38x the entire routed sum (shared backbone dominates).
- weight movement 17MB/expert vs activation ~8KB (hidden 2048xfp16) = ~2000x asymmetry.
- CPU DRAM bandwidth ~100GB/s >> PCIe 5-24GB/s (~20x). Experts are fine-grained (intermediate 1408, tiny FLOP).
- cross-token access near-uniform/memoryless -> caching futile; misses unavoidable. within-token cross-layer predictable (draft 0.7). prefetch already solved by SPICE draft; optimal prefetch depth = constant 2.
- dropping low-importance routed experts is cheap (drop lowest = +1% PPL).

CANDIDATE A — compute-placement (CPU-compute on miss): on a miss, do NOT fetch the 17MB weight; send the 8KB activation to CPU, compute the expert exactly using CPU DRAM bandwidth, return the 8KB output. ~0.2ms/expert vs 3.4ms fetch, EXACT, no cache pollution. Reframes the miss from expensive stall to cheap CPU compute.
Q: Is this non-incremental vs Fiddler (CPU-computes MoE experts to avoid weight movement) and HybriMoE (hybrid CPU-GPU scheduling) and ktransformers? What does SPICE's draft add (GPU/CPU placement decision, pipelining CPU expert compute behind GPU attention)? Is there a non-incremental version, or is it just "apply Fiddler to fine-grained MoE + SPICE prefetch for the hot ones"? What is the ONE decisive experiment (measure CPU expert-compute latency for Qwen/DeepSeek experts at batch=1, vs PCIe fetch, + the CPU-GPU pipeline TPOT vs SPICE)?

CANDIDATE B — selective routed execution (uncertainty-gated): routed is a 10% perturbation; shared+attention may determine the OUTPUT TOKEN for most tokens. Compute shared-only (resident, free) logits; only FETCH routed experts for tokens where routed would flip the argmax. Skip most of the 1.6GB/token weight movement; verified-where-it-matters.
Q: Is this non-incremental vs (i) MoE layer-skipping / early-exit, (ii) AdapMoE adaptive active-count, (iii) shared-expert-only approximation, (iv) self-speculative decoding with a shared-only draft? What is the decisive experiment (measure: fraction of greedy tokens whose argmax is UNCHANGED by dropping ALL routed experts = shared+attn only; if high, B works)? Will it hold given drop-to-rank-1 already costs +37% PPL (PPL != argmax flip rate)?

For EACH: verdict (non-incremental? y/n + why), closest prior art, the single decisive cheap experiment + go/no-go, and any fatal flaw. Then pick the more promising one. If both are incremental, say so and propose a better third idea grounded in the measured facts.
