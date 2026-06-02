Design review (be terse, concrete, output to stdout). I am building the core evaluation for completing SPICE: a latency-quality Pareto for VERIFIED IMPORTANCE-AWARE MISS-HANDLING on resource-constrained GPU MoE offloading. Confirm the protocol or fix it; flag pitfalls; do NOT be agreeable.

## Established (real Qwen1.5-MoE + DeepSeek-V2-Lite, A800, WikiText)
- SPICE reproduced; training-free draft built (within-token recall@4 0.7+); forecast-driven eviction proven DEAD (cannot beat SpecMD-LS); we ADOPT SpecMD-LS as eviction.
- Miss-handling lever confirmed cross-model: dropping the single lowest-importance routed expert on a miss costs ~+1% PPL (Qwen +1.4%, DeepSeek +0.88%); dropping 2 = 4.5-6.3%. Graceful degradation.
- SPICE's verification gives the TRUE gate weight per token -> per-miss decide fetch (high importance) vs drop (low importance) under a latency SLO. lossless-where-it-matters.

## Proposed Pareto protocol (decoupled latency + quality)
Part A (latency + drop-set, from real routing traces, per text, sequential decode stream): deadline-aware DMA sim (multi-layer in-flight transfers, demand-priority, expert 17MB, bw sweep 5/12/24 GB/s, cache 5/10/20%); eviction = SpecMD-LS; optional draft prefetch. On a DEMAND MISS of expert e with true gate weight w: if w >= threshold -> FETCH (stall += fetch_ms, admit to cache); else -> DROP (no stall, record (pos,layer,e) dropped). Output: exposed_stall_ms/token + the per-(pos,layer) drop set.
Part B (quality): run the real model once per text with the Part-A drop set applied (zero those experts' gate weights at those positions), measure PPL (and optionally MMLU/GSM8K later).
Sweep threshold -> trace (TPOT, PPL) Pareto. Endpoints: threshold=0 = fetch-all = SPICE (max latency, exact PPL); threshold=inf = drop-all-misses (min latency, worst PPL).

## Baselines on the same Pareto axes
- SPICE original (fetch-all on miss).
- SpecMD miss policies: Fetch / Drop-by-rank / Substitution (nearest cached expert).
- HOBBIT-style: low-precision fetch on miss (approx, lossy) -- model as reduced fetch_ms + a quantization PPL penalty.
- Ours: verified importance-threshold fetch/drop (+ optional miss-window draft prefetch).

## Questions
1. Is the decoupled (sim drop-set -> model PPL) design sound, or must latency+quality be measured in ONE token-by-token run with a live KV-cache (which is far slower)? What error does decoupling introduce (e.g., dropping an expert at token t changes hidden states -> changes routing at t+1; the trace was collected WITHOUT drops, so the drop-set's downstream routing drift is not captured). How serious is this and how to bound/handle it?
2. What is the single most important pitfall that would make reviewers reject the Pareto? (e.g., the drift above, or comparing at unequal memory/bandwidth, or PPL not being a real task metric.)
3. Minimal correct version: should the first Pareto be on a SHORT decode (few tokens, token-by-token, exact drift) to validate, then scale with the decoupled approximation? Or is decoupling fine for PPL on these models?
4. For the "verified / lossless-where-it-matters" claim: the high-importance experts are fetched exactly; only low-importance dropped. Is "bounded quality loss with a tunable SLO" a defensible claim, and what guarantee can we actually state (not vacuous)?
5. How to make this clearly NON-incremental vs HOBBIT (low-precision miss) and AdapMoE (sensitivity-adaptive gating) and SpecMD (drop-by-rank)? They all already trade quality for latency on misses. What is the crisp differentiator (verified true-gate + joint latency-SLO controller + the specific Pareto-optimality)? Is it enough?
Cite mechanisms.
