**1. Novelty**
The sharp framing is not “predictor-driven eviction.” The sharp framing is:
**SPICE turns a verified routing draft into an online stochastic Belady scheduler under hard cache and PCIe budgets.**
That means the contribution is an admission/eviction/DMA scheduler whose inputs are `(expert, layer deadline, confidence, byte cost, demand-priority)` from SPICE’s cross-layer draft. Eviction is only one consequence. The real claim is: SPICE’s existing policy maximizes hit rate/overlap in bandwidth-rich regimes; your system minimizes exposed critical-path bytes under scarce cache and scarce bandwidth.
If you cannot formalize it that way, it is “SPICE + LS + throttle,” and that is not top-venue material.
**2. Difference From Least-Stale**
SpecMD-LS uses deterministic layer order plus stale/current state. It does not know which concrete expert in a future layer is likely for this token. Your policy must use SPICE’s actual multi-layer forecast as a next-use distribution.
A real policy would score residents roughly as:
`value(e) = P(next_use before deadline) * exposed_stall_saved(e) / bytes(e)`
and evict/admit by lowest expected value, with hard protection for current-layer demand and in-flight demand fetches. This is closer to uncertain Belady than to LS.
Using the draft qualifies only if you prove three things:
1. **Identity advantage:** for experts with identical LS staleness/layer position, the SPICE forecast correctly distinguishes soon-needed vs dead experts.
2. **Oracle gap:** offline true-future Belady beats LS meaningfully under 1-5% cache / low bandwidth; otherwise LS is already good enough.
3. **Forecast gap:** SPICE-forecast Belady closes a large fraction of the oracle-vs-LS gap. If it does not, the predictor is not useful for eviction.
Ablations must include `LS`, `SPICE+LS`, `forecast eviction only`, `throttle only`, and `forecast eviction + throttle`.
**3. Miss-Stall Window**
Drop “miss stall -> downstream exact prefetch” as a core mechanism.
For batch=1 decode, during a miss the H2D copy engine and PCIe link are busy fetching the missing expert. A second H2D prefetch is not free; it either waits or steals bandwidth from the demand fetch. Also, downstream exact routing is not available until the current expert computation finishes and produces the next hidden state. So “exact downstream prefetch during the stall” is mostly false.
What is realizable:
`demand-priority DMA + chunked speculative copies + cancellation/admission control`.
If a speculative copy is queued or chunked, demand misses preempt it. If a copy is already a monolithic `cudaMemcpy`, you cannot usefully interrupt it; this is exactly why chunking/preemption matters. For batch>1, you can compute other tokens while one token waits, but that becomes cache-aware batching/scheduling, not SPICE batch=1 miss-window prefetch.
**4. Cheapest Decisive First Experiment**
Do not start with full runtime implementation. Use A800 only to generate real SPICE traces; then replay them under synthetic cache/bandwidth constraints.
Run Qwen1.5-MoE-A2.7B first, then DeepSeek-V2-Lite if Qwen passes. Log for each decode token/layer:
`true top-K experts`, `SPICE predicted experts`, `prediction confidence`, `layer compute time`, `expert byte size`.
Replay with:
`cache = 128 slots and 5% expert cache`, `bandwidth = 5 GB/s`, `batch=1`.
Single decisive metric:
`exposed_H2D_stall_ms_per_token`
computed by a deadline-aware replay: a prefetched expert is useful only if its copy completes before the true layer deadline; demand misses add exposed stall.
Go if both are true:
1. Original aggressive SPICE is worse than or close to LS/LRU in exposed stall and uses at least `2x` more H2D bytes/token.
2. Oracle Belady or SPICE-forecast Belady reduces exposed stall by at least `25%` versus LS at the same cache/bandwidth.
No-go if LS is within `10-15%` of oracle, or SPICE-forecast Belady cannot beat LS by `>10%`. Then the forecast is not a strong cache signal.
**5. Required Baselines**
Minimum serious baselines:
`SPICE original`, `SPICE+LRU`, `SPICE+SpecMD-LS`, `SPICE+LS+simple bandwidth throttle`, `oracle Belady`, `LRU/LFU/static`, `on-demand no-prefetch`.
Strong systems: `MoE-Infinity`, `FineMoE/fMoE`, `HOBBIT`, `ProMoE`, `FATE`, `ExpertFlow`, and if code/claims are available, `MoE-SpAc` and `MoE-SpeQ`. MoE-Infinity, FineMoE, and HOBBIT all explicitly target expert offloading/prefetch/cache tradeoffs.    
Baseline integrity matters: source-only baseline or label it invalid. No custom decode loops reported as upstream baseline numbers.
Minimum bar over SPICE:
1. In tight regimes, `1-5% cache` and `5-12 GB/s`, beat SPICE original and `SPICE+LS+throttle` by at least `20% TPOT` or `30% exposed H2D stall`.
2. Cut H2D traffic materially, ideally `>=2x` less than SPICE’s aggressive prefetch, without losing TPOT.
3. Preserve lossless output: max logit diff `0.0`.
4. Do not regress SPICE’s large-cache/bandwidth-rich regime by more than `5%`.
**6. Best Reframing**
Best shot:
**Resource-constrained verified speculation for MoE offloading.**
Pitch it as: SPICE already predicts routing; your system makes that prediction operationally optimal under scarce cache and bandwidth. The novelty is a verified, deadline-aware, uncertainty-aware memory scheduler, not another prefetch predictor and not another eviction heuristic.
The paper should read as:
`SPICE shows lossless routing lookahead works. We show naive lookahead is the wrong objective under real offloading constraints. The correct objective is minimizing exposed demand bytes under cache and DMA deadlines.`
That is the non-incremental version. The current three-part version is not.
208,832
Bottom line: as phrased, this is still too incremental. “SPICE + smarter eviction + throttle” will get killed because SpecMD already owns the LRU/LFU critique and Least-Stale result, and there is already crowded prior art on proactive/predictor-driven MoE prefetching: ProMoE, FATE, ExpertFlow, MoE-SpAc, and MoE-SpeQ all touch parts of this space. SpecMD explicitly reports poor temporal-locality assumptions and Least-Stale collision-miss reductions; ProMoE predicts future expert usage from intermediate results; ExpertFlow adapts prefetch horizon using bandwidth/cache signals; MoE-SpAc uses speculation as a memory-management sensor; MoE-SpeQ uses a draft model plus an adaptive governor for expert prefetch.     ([arxiv.org](https://arxiv.org/abs/2511.14102))
**1. Novelty**
**1. Novelty**
The sharp framing is not “predictor-driven eviction.” The sharp framing is:
**SPICE turns a verified routing draft into an online stochastic Belady scheduler under hard cache and PCIe budgets.**
That means the contribution is an admission/eviction/DMA scheduler whose inputs are `(expert, layer deadline, confidence, byte cost, demand-priority)` from SPICE’s cross-layer draft. Eviction is only one consequence. The real claim is: SPICE’s existing policy maximizes hit rate/overlap in bandwidth-rich regimes; your system minimizes exposed critical-path bytes under scarce cache and scarce bandwidth.
If you cannot formalize it that way, it is “SPICE + LS + throttle,” and that is not top-venue material.
**2. Difference From Least-Stale**
SpecMD-LS uses deterministic layer order plus stale/current state. It does not know which concrete expert in a future layer is likely for this token. Your policy must use SPICE’s actual multi-layer forecast as a next-use distribution.
A real policy would score residents roughly as:
`value(e) = P(next_use before deadline) * exposed_stall_saved(e) / bytes(e)`
and evict/admit by lowest expected value, with hard protection for current-layer demand and in-flight demand fetches. This is closer to uncertain Belady than to LS.
Using the draft qualifies only if you prove three things:
1. **Identity advantage:** for experts with identical LS staleness/layer position, the SPICE forecast correctly distinguishes soon-needed vs dead experts.
2. **Oracle gap:** offline true-future Belady beats LS meaningfully under 1-5% cache / low bandwidth; otherwise LS is already good enough.
3. **Forecast gap:** SPICE-forecast Belady closes a large fraction of the oracle-vs-LS gap. If it does not, the predictor is not useful for eviction.
Ablations must include `LS`, `SPICE+LS`, `forecast eviction only`, `throttle only`, and `forecast eviction + throttle`.
**3. Miss-Stall Window**
Drop “miss stall -> downstream exact prefetch” as a core mechanism.
For batch=1 decode, during a miss the H2D copy engine and PCIe link are busy fetching the missing expert. A second H2D prefetch is not free; it either waits or steals bandwidth from the demand fetch. Also, downstream exact routing is not available until the current expert computation finishes and produces the next hidden state. So “exact downstream prefetch during the stall” is mostly false.
What is realizable:
`demand-priority DMA + chunked speculative copies + cancellation/admission control`.
If a speculative copy is queued or chunked, demand misses preempt it. If a copy is already a monolithic `cudaMemcpy`, you cannot usefully interrupt it; this is exactly why chunking/preemption matters. For batch>1, you can compute other tokens while one token waits, but that becomes cache-aware batching/scheduling, not SPICE batch=1 miss-window prefetch.
**4. Cheapest Decisive First Experiment**
Do not start with full runtime implementation. Use A800 only to generate real SPICE traces; then replay them under synthetic cache/bandwidth constraints.
Run Qwen1.5-MoE-A2.7B first, then DeepSeek-V2-Lite if Qwen passes. Log for each decode token/layer:
`true top-K experts`, `SPICE predicted experts`, `prediction confidence`, `layer compute time`, `expert byte size`.
Replay with:
`cache = 128 slots and 5% expert cache`, `bandwidth = 5 GB/s`, `batch=1`.
Single decisive metric:
`exposed_H2D_stall_ms_per_token`
computed by a deadline-aware replay: a prefetched expert is useful only if its copy completes before the true layer deadline; demand misses add exposed stall.
Go if both are true:
1. Original aggressive SPICE is worse than or close to LS/LRU in exposed stall and uses at least `2x` more H2D bytes/token.
2. Oracle Belady or SPICE-forecast Belady reduces exposed stall by at least `25%` versus LS at the same cache/bandwidth.
No-go if LS is within `10-15%` of oracle, or SPICE-forecast Belady cannot beat LS by `>10%`. Then the forecast is not a strong cache signal.
**5. Required Baselines**
Minimum serious baselines:
`SPICE original`, `SPICE+LRU`, `SPICE+SpecMD-LS`, `SPICE+LS+simple bandwidth throttle`, `oracle Belady`, `LRU/LFU/static`, `on-demand no-prefetch`.
Strong systems: `MoE-Infinity`, `FineMoE/fMoE`, `HOBBIT`, `ProMoE`, `FATE`, `ExpertFlow`, and if code/claims are available, `MoE-SpAc` and `MoE-SpeQ`. MoE-Infinity, FineMoE, and HOBBIT all explicitly target expert offloading/prefetch/cache tradeoffs.    
Baseline integrity matters: source-only baseline or label it invalid. No custom decode loops reported as upstream baseline numbers.
Minimum bar over SPICE:
1. In tight regimes, `1-5% cache` and `5-12 GB/s`, beat SPICE original and `SPICE+LS+throttle` by at least `20% TPOT` or `30% exposed H2D stall`.
2. Cut H2D traffic materially, ideally `>=2x` less than SPICE’s aggressive prefetch, without losing TPOT.
3. Preserve lossless output: max logit diff `0.0`.
4. Do not regress SPICE’s large-cache/bandwidth-rich regime by more than `5%`.
**6. Best Reframing**
Best shot:
**Resource-constrained verified speculation for MoE offloading.**
Pitch it as: SPICE already predicts routing; your system makes that prediction operationally optimal under scarce cache and bandwidth. The novelty is a verified, deadline-aware, uncertainty-aware memory scheduler, not another prefetch predictor and not another eviction heuristic.
The paper should read as:
`SPICE shows lossless routing lookahead works. We show naive lookahead is the wrong objective under real offloading constraints. The correct objective is minimizing exposed demand bytes under cache and DMA deadlines.`
That is the non-incremental version. The current three-part version is not.
