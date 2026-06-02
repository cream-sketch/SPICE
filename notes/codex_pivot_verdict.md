**Sim Audit**
There is one forecast-specific correctness bug that makes the current negative not clean:

- [eval_forecast_eviction.py:141](/home/abc/Placement/HLS/spice/experiments/eval_forecast_eviction.py:141): forecast eviction says “tie-break LRU” but does `min(..., (keep_value, -last))`, which evicts **MRU** on equal keep value. Fix to:
  ```python
  victim = min(candidates, key=lambda k: (keep_value(k), cache[k]["last"]))
  ```
  This can unfairly hurt forecast, especially when many residents have equal `layer_freq`.

Other bugs/limits do **not** rescue forecast:

- [eval_forecast_eviction.py:198](/home/abc/Placement/HLS/spice/experiments/eval_forecast_eviction.py:198): demand preemption advances `dma_free` but does not delay existing pending entries’ `ready` times. This over-credits speculative prefetch. Fix by modeling a DMA queue or shifting all unfinished pending `ready > now` by `fetch_ms` when demand preempts.
- [eval_forecast_eviction.py:180](/home/abc/Placement/HLS/spice/experiments/eval_forecast_eviction.py:180): late in-flight prefetch is counted as `hits += 1` even though it stalls. Exposed stall is still meaningful; hit rate is inflated. Split `on_time_hits`, `late_prefetch_waits`, and true misses.
- [eval_forecast_eviction.py:151](/home/abc/Placement/HLS/spice/experiments/eval_forecast_eviction.py:151): collision only counts pending evicted while still in flight. A delivered-but-unused prefetched expert evicted before use is also wasted; count it separately.
- [eval_forecast_eviction.py:162](/home/abc/Placement/HLS/spice/experiments/eval_forecast_eviction.py:162): demand-time eviction uses `fcast[l+1]`, which is only available after layer `l` compute. That leaks one-layer future state and favors forecast, not LS.

Belady next-use and LS look basically correct. `eval_real_trace_eviction.py` now flattens per-expert accesses and resets per sequence; LS cyclic distance correctly treats same-layer reuse as a full cycle. Oracle is high because it is a real upper bound, not because of an obvious Belady bug.

**Verdict**
Forecast-driven eviction is dead as the contribution, but rerun once after the MRU tie-break fix before you write the final no-go table.

Even if the fix improves forecast, the mechanism is weak: your within-token draft is excellent for **prefetch admission** and protecting already-prefetched future-layer experts. It is structurally weak for **evicting resident experts**, because a resident key is `(layer, expert)`. Once layer `l` has executed for token `p`, that same `(l, e)` can only be reused by future tokens, not later layers of the same token. LS already captures the layer-deadline part. The missing signal is concrete cross-token next use.

So yes: the oracle gap explanation is correct. The oracle knows which same-layer experts recur in future tokens. The current SPICE-style draft in [qwen_spice_draft.py](/home/abc/Placement/HLS/spice/experiments/qwen_spice_draft.py:74) rolls within the current token/layer stack; it does not predict the next sampled token’s routing. This generalizes to DeepSeek-V2-Lite and other decoder MoEs with layer-expert caches. Magnitude depends on routing entropy, top-k, cache size, batch, and bandwidth, but the structural split is model-independent.

**Best Direction**
Pick **lossless miss-window speculation with exact replay**, not cross-token prefetch as the main claim.

Mechanism: when an expert miss is unavoidable, do not idle. Run a tiny resident proxy for the missing expert, propagate a speculative hidden state several layers forward, use that to schedule downstream exact expert fetches, then replay the exact path once the missed expert arrives before committing logits. The final output remains exact; the proxy is only a scheduler.

Why this is sharper:

- It attacks the physics problem: at 5 GB/s, 17 MB costs about 3.3 ms, or 8+ layer times. Better eviction cannot hide a transfer that starts too late.
- It is not HOBBIT: HOBBIT dynamically uses lower-precision cache-miss experts; your claim would be uncommitted proxy execution plus exact correction.
- It is not SP-MoE/MoE-SpeQ: those already use future-token speculative/draft predictions for expert prefetch. MoE-SpeQ explicitly predicts required experts for future tokens; SP-MoE does SD-aware expert prefetch. Cross-token prefetch alone is crowded.
- It is not FineMoE: FineMoE already combines predicted expert probability and cache frequency for eviction. A better `p * freq / deadline` policy is incremental.

Gate it brutally: same exact logits, 5-10% cache, 5 GB/s, beat SPICE+SpecMD-LS and a FineMoE-style probability/frequency eviction by at least 1.25x TPOT. If replay overhead or proxy drift eats the stall window, kill it.

Sources checked: SpecMD/Least-Stale arXiv, SP-MoE arXiv, MoE-SpeQ arXiv, HOBBIT arXiv, and FineMoE PDF.
tokens used
91,810
**Sim Audit**
There is one forecast-specific correctness bug that makes the current negative not clean:

- [eval_forecast_eviction.py:141](/home/abc/Placement/HLS/spice/experiments/eval_forecast_eviction.py:141): forecast eviction says “tie-break LRU” but does `min(..., (keep_value, -last))`, which evicts **MRU** on equal keep value. Fix to:
  ```python
  victim = min(candidates, key=lambda k: (keep_value(k), cache[k]["last"]))
  ```
  This can unfairly hurt forecast, especially when many residents have equal `layer_freq`.

Other bugs/limits do **not** rescue forecast:

- [eval_forecast_eviction.py:198](/home/abc/Placement/HLS/spice/experiments/eval_forecast_eviction.py:198): demand preemption advances `dma_free` but does not delay existing pending entries’ `ready` times. This over-credits speculative prefetch. Fix by modeling a DMA queue or shifting all unfinished pending `ready > now` by `fetch_ms` when demand preempts.
- [eval_forecast_eviction.py:180](/home/abc/Placement/HLS/spice/experiments/eval_forecast_eviction.py:180): late in-flight prefetch is counted as `hits += 1` even though it stalls. Exposed stall is still meaningful; hit rate is inflated. Split `on_time_hits`, `late_prefetch_waits`, and true misses.
- [eval_forecast_eviction.py:151](/home/abc/Placement/HLS/spice/experiments/eval_forecast_eviction.py:151): collision only counts pending evicted while still in flight. A delivered-but-unused prefetched expert evicted before use is also wasted; count it separately.
- [eval_forecast_eviction.py:162](/home/abc/Placement/HLS/spice/experiments/eval_forecast_eviction.py:162): demand-time eviction uses `fcast[l+1]`, which is only available after layer `l` compute. That leaks one-layer future state and favors forecast, not LS.

Belady next-use and LS look basically correct. `eval_real_trace_eviction.py` now flattens per-expert accesses and resets per sequence; LS cyclic distance correctly treats same-layer reuse as a full cycle. Oracle is high because it is a real upper bound, not because of an obvious Belady bug.

**Verdict**
Forecast-driven eviction is dead as the contribution, but rerun once after the MRU tie-break fix before you write the final no-go table.

Even if the fix improves forecast, the mechanism is weak: your within-token draft is excellent for **prefetch admission** and protecting already-prefetched future-layer experts. It is structurally weak for **evicting resident experts**, because a resident key is `(layer, expert)`. Once layer `l` has executed for token `p`, that same `(l, e)` can only be reused by future tokens, not later layers of the same token. LS already captures the layer-deadline part. The missing signal is concrete cross-token next use.

So yes: the oracle gap explanation is correct. The oracle knows which same-layer experts recur in future tokens. The current SPICE-style draft in [qwen_spice_draft.py](/home/abc/Placement/HLS/spice/experiments/qwen_spice_draft.py:74) rolls within the current token/layer stack; it does not predict the next sampled token’s routing. This generalizes to DeepSeek-V2-Lite and other decoder MoEs with layer-expert caches. Magnitude depends on routing entropy, top-k, cache size, batch, and bandwidth, but the structural split is model-independent.

**Best Direction**
Pick **lossless miss-window speculation with exact replay**, not cross-token prefetch as the main claim.

