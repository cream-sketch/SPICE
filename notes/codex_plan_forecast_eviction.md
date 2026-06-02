# Codex review request: implementation plan for "forecast-driven eviction on top of SPICE"

Review this PLAN for soundness and the cheapest path. Be brutal, terse. Output to stdout, no file writes. Flag any step that is wasted, wrong, or out of order, and give the minimal viable sequence.

## Established so far (real Qwen1.5-MoE-A2.7B traces, A800)
- Reproduced SPICE: synthetic draft verified hit 0.9965; system sim SPICE hit 0.972 / LRU 57.44GB; real-trace oracle 1.0, anchor_repeat 0.323.
- Eviction kill test (demand-only, deadline-aware exposed stall, bug-fixed per your review): oracle Belady cuts exposed stall vs LRU (SPICE's eviction) by 26% (10% cache), 37% (20%), 58% (50%); at <=5% cache LRU is ~0 hit (no temporal locality, matches SpecMD); at 1-2% everything is cold-miss bound. Cheap static proxies fail (LFU worse than LRU). => capturing the oracle headroom needs a per-token multi-layer FORECAST.
- The repo's real-model path only has simple history predictors (0.32 hit). SPICE's draft predictor exists ONLY for a synthetic random-weight target MoE in the repo; the real-model wrapper the paper claims is NOT in the repo.

## Thesis (your v3 framing, adopted)
SPICE turns a verified routing draft into an online stochastic Belady scheduler under hard cache+PCIe budgets; objective = minimize exposed demand bytes under DMA deadlines, not maximize hit rate. Concretely: reuse SPICE's draft multi-layer forecast (already computed for prefetch) to ALSO drive eviction: value(e)=P(next-use before deadline | draft)/bytes(e); evict min value; hard-protect current-layer demand + in-flight demand fetches.

## Proposed implementation plan (critique + reorder)
Step 1. Build the real-Qwen SPICE draft (the repo's missing piece): wrap HF Qwen2MoE; reuse frozen attention + frozen router (mlp.gate); add per-layer LoRE low-rank state transition E(z)=z+B A z + routing-history logit bias; train route-KL + hidden-align on real traces (SlimPajama/WikiText). ~270M-scale, few GPU-hours.
   - Sub-question: do we even need to TRAIN first? A training-free degenerate draft (LoRE=identity, just frozen attn+router propagated skipping experts) might already give a usable multi-layer routing forecast. Should we measure the training-free draft's layer l+1..l+K exact-expert prediction accuracy FIRST (cheap), before committing to training?
Step 2. Extend the deadline-aware cache simulator with a `forecast_belady` eviction policy driven by the draft's per-token forecast; add a WITH-prefetch mode so SpecMD-LS is compared fairly (its advantage is protecting prefetched experts).
Step 3. Compare LRU / SpecMD-LS(+prefetch) / oracle-Belady / forecast-Belady across cache(1-50%) x bandwidth(5-24GB/s); metric exposed_stall/token + TPOT; assert lossless (forecast only schedules; demand verified).
Step 4. If forecast-Belady captures >=50% of the (oracle - LRU) gap and beats LRU by >=25% exposed stall in 5-50% cache: GO to a real runtime prototype (CUDA streams, pinned mem, demand-priority DMA). Else iterate or kill.

## Questions
1. Is Step 1's training-free-draft-first sub-step the right cheapest gate, or must we train the LoRE draft before any eviction conclusion is trustworthy? What exact accuracy metric on real Qwen would justify proceeding to training (give a number)?
2. Is reusing the SAME forecast for prefetch AND eviction legitimate and novel, or does any prior (FineMoE uses prob maps for both prefetch+eviction) already cover "one forecast drives both"? How do we differentiate concretely?
3. For forecast_belady, how should P(next-use before deadline) be turned into an evictable score from the draft, given the draft gives per-future-layer top-K sets + confidence (not a clean probability per resident)? Give the concrete scoring formula you'd implement.
4. Biggest risk that this collapses to "incremental better-prefetcher": name it and the experiment that pre-empts the reviewer.
5. Minimal viable sequence to a defensible result with least wasted work.
