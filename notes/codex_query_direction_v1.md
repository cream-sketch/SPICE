# Codex consultation: is the SPICE non-incremental thesis sound?

You are a senior systems-ML researcher. Critique hard. Do NOT be agreeable. I need to find a genuinely NON-INCREMENTAL contribution, not a 6th expert-predictor paper.

## Project
SPICE = verified speculative expert-prefetching for **offloaded MoE inference** (experts live in CPU DRAM / NVMe, only a subset fits in GPU VRAM; each decode step must fetch the routed experts over PCIe). The task is the MoE *expert routing → prefetch → cache/eviction* path on a single A800-80GB box.

## The 5 recent papers in this space and their COMMON blind spot
1. SP-MoE (2510.10302): draft-model attention -> target gating cross-prediction; async prefetch; static "cutoff layer"; LRU. Admits: no dynamic fallback when prefetch partially fails; misprediction cost unquantified; breaks at batch>1.
2. Pre-Attention Prediction (2511.10676): same-layer pre-attention signal -> 2-layer predictor; over-provision to 98% hit. Admits: on miss just falls back to synchronous load (no system-level miss-rate curve); eviction not integrated.
3. MoE-SpeQ (2511.14102): INT4 quantized draft predicts experts; Expert Lookahead Buffer; amortization roofline. Admits: 9.1% misprediction penalty unquantified, no on-demand fallback cost analysis.
4. EARTH (ACM, FPGA/ASIC): dual-entropy base/delta split, routing-history prefetch, LUT delta reuse; on miss load base-only. Admits: expert-switch pipeline flush; miss penalty unquantified.
5. SpecBranch (2506.01979): speculative-decoding rollback parallelism (not MoE offloading per se).

**Observed common gap: every paper's contribution is a better PREDICTOR. The failure path (prefetch miss / eviction mistake) is universally handled by "just synchronously load and stall", and NONE quantify the miss penalty or design a recovery mechanism. The design axis "better predictor" is saturated.**

## Current SPICE code state (already implemented, Python sim)
- Draft model: frozen target attention/router + trainable LoRE low-rank expert SURROGATE + GRU routing-history context + route-KL + hidden-alignment loss. Produces per-expert confidence.
- Verified prefetch loop: adaptive lookahead depth gated by confidence; on miss -> synchronous fetch (stall). Eviction = LRU.
- IMPORTANT WEAKNESS: target MoE is SYNTHETIC (random weights, randn inputs) -> reported 0.99 hit is a closed synthetic loop. On REAL Qwen1.5-MoE traces, realizable predictors get hit=0.34-0.39 (oracle=1.0). Draft predictor never run on real traces.
- "verified" so far only means: fallback changes latency, not logits (correctness invariant). The latency of the miss is the unsolved cost.

## My proposed NON-INCREMENTAL thesis (tear it apart)
"Treat the prefetch miss not as a failure to minimize but as a RESOURCE to schedule." Two coupled mechanisms:

(A) **Uncertainty-aware admission/eviction**: the draft predictor emits a probability per expert. Instead of binarizing top-k + LRU, drive cache admission AND eviction by expected-future-need (predicted access prob x reuse horizon x transfer cost). The predictor's soft output becomes the cache policy. No paper uses predictor confidence for eviction.

(B) **LoRE approximate-then-correct miss recovery**: SPICE's draft ALREADY has a low-rank surrogate of every expert. On a cache miss, instead of stalling, compute the expert's APPROXIMATE output from the resident low-rank surrogate, let downstream layers proceed speculatively, and CORRECT (recompute the delta) when the true expert finishes loading over PCIe — overlapping the H2D transfer with useful approximate compute. Bounded-error / verified: final token logits either match exactly (if we correct before commit) or carry a certified error bound.

## Questions (answer each, concretely)
1. Is (A)+(B) genuinely non-incremental vs the 5 papers, or is it "X + small mod"? Name the closest prior art for EACH of (A) and (B) and whether it subsumes us.
2. Fatal flaws? Specifically: for (B), if we must produce exact logits we still need the true expert before committing the layer — so where is the real latency win? Is the only win in *downstream* speculative execution (expert-granularity self-speculation)? Does the error compound across layers and kill accuracy?
3. Is "approximate-then-correct at expert granularity" already done under another name (e.g., delta/residual MoE, low-rank MoE, speculative MoE execution)? Search your memory hard.
4. What is the SINGLE highest-value first experiment on REAL traces (Qwen1.5-MoE-A2.7B / DeepSeek-V2-Lite, A800-80GB) that would either kill or validate this thesis fastest? Give exact metric + go/no-go threshold.
5. What baselines MUST we beat for a top-venue (MLSys/ASPLOS/OSDI) non-incremental claim, and what is the minimum credible evaluation?

Be brutal and specific. Cite mechanisms, not vibes.

## OUTPUT REQUIREMENTS (do this)
1. Use web search to find the CLOSEST prior art for mechanisms (A) and (B). Search terms to try: "approximate expert MoE offloading", "low-rank expert surrogate inference", "residual / delta MoE expert", "speculative expert execution MoE", "uncertainty-aware cache eviction LLM expert", "MoE prefetch miss penalty", "cost-aware expert caching MoE inference", recent (2024-2025) MLSys/ASPLOS/OSDI/NSDI/arXiv.
2. For each found work give: title, venue/year, 1-line method, and whether it subsumes (A) or (B).
3. Write your COMPLETE structured assessment as clean Markdown to the file:
   `notes/codex_direction_assessment_v1.md`
   Sections: (1) Verdict: incremental or not, (2) Closest prior art per mechanism, (3) Fatal flaws + fixes, (4) The single highest-value first experiment with exact metric and go/no-go threshold, (5) Required baselines + minimum credible evaluation, (6) A sharper one-sentence thesis if mine is weak.
Keep it rigorous and concrete. This file will be read by another engineer to decide the next experiment.
