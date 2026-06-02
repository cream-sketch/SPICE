Review experiments/eval_forecast_eviction.py for CORRECTNESS BUGS ONLY. Terse, concrete, line-level fixes, no rewrite. Output to stdout, no file writes.

Context: it consumes per-text dumps (true_top[L][S][k] real demand; fcast[L][H][S][k] draft forecast, fcast[a][h][p]=predicted top-K for layer a+h at position p). It replays a decode stream (per text independent, cold cache; positions p=0..S-1 each go through layers 0..L-1, cache persists across positions within the text). At each (p,l): resolve demand true_top[l][p] (miss=sync fetch, stall+=fetch_ms); then if prefetch=='draft', prefetch forecast experts for upcoming layers l+1..l+H from fcast[l+1][*][p] within an overlap budget = bandwidth*t_layer_ms (these are 'pending'); a pending expert evicted before use = collision. Eviction policies: lru, ls(cyclic), forecast(protect near-forecast), oracle(Belady next-use over this text's demand stream). Goal: 2x2 (prefetch off/draft x eviction lru/ls/forecast/oracle), key contrast draft-prefetch lru vs forecast to isolate eviction-only benefit.

Check brutally:
1. Belady next_use: occ built only over DEMAND stream (true_top), gidx=p*L+l. Is next_use monotonic-correct across the whole text replay and across many residents per eviction? Does Belady evict the farthest-next-use among candidates? Any pointer corruption?
2. forecast eviction scoring: min-key tuple (in_forecast 0/1, forecast_layer-cur_layer, -last). Does this correctly evict non-forecast first, then farthest-forecast, then LRU? Does it ever evict a current-demand or about-to-admit key (should be protected via `protect` set)?
3. collision accounting: counted in caller when evicted ent['pending'] True. Is a pending->used transition (prefetch_useful) correctly flipping pending=False on demand hit? Double-count risk?
4. overlap budget: one layer's budget = bandwidth*t_layer_ms; prefetch for up to H future layers drawn from this single budget. Is that a fair model or does it over/under-credit overlap? Demand misses charge full fetch_ms (no overlap). Is exposed stall well-defined?
5. capacity boundary `while len(cache) >= capacity` then insert: final size == capacity? off-by-one?
6. fcast indexing: at current layer l, upcoming layer tl=l+1+h uses fcast[l+1][h][p]. Off-by-one vs the dumper (fcast[anchor][h]=pred for layer anchor+h)? Confirm anchor=l+1 maps to predicting layers l+1..l+H correctly.
7. Any policy given an unfair (artificially good/bad) deal that would invalidate the 2x2 conclusion?

Read the file. Be specific.
