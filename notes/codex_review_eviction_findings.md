Findings:

1. [experiments/eval_real_trace_eviction.py:115](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:115), [131](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:131), [158](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:158): Belady’s `pos` is per layer-step, not per expert access. All top-K experts in one layer share the same `cur_pos`, so `next_use(..., cur_pos)` skips `<= cur_pos` and can treat an unprocessed current top-K resident as already past. Belady can evict an expert needed later in the same `experts` list, making the oracle non-oracle and usually artificially worse.
   Fix: flatten to expert-slot positions for Belady/simulation:
   ```python
   flat_stream = [(l, e) for l, experts in stream for e in experts]
   for i, key in enumerate(flat_stream): occ.setdefault(key, []).append(i)
   for pos, (l, e) in enumerate(flat_stream): ...
   evict_one(l, pos)
   ```
   If top-K is meant as one batch demand, instead protect the whole current `(l, experts)` set during evictions.

2. [experiments/eval_real_trace_eviction.py:147](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:147): `specmd_ls` distance is wrong for same-layer residents. `(key_layer - cur_layer) % num_layers == 0` treats same-layer experts as nearest reuse, but after the current layer they are farthest: next token’s same layer. This makes LS retain stale same-layer entries and can make Belady headroom look artificially larger.
   Fix:
   ```python
   def ls_dist(k):
       d = (key_layer[k] - cur_layer) % num_layers
       return num_layers if d == 0 else d
   victim = max(cache, key=lambda k: (ls_dist(k), -last_used[k]))
   ```

3. [experiments/eval_real_trace_eviction.py:147](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:147): LS tie-break is MRU, because `max(..., last_used[k])` evicts the most recently used among equal cyclic distance. If LS is supposed to be “cyclic distance, LRU fallback”, this is wrong and can perturb LS vs LRU. Use `-last_used[k]` as above, optionally add `k` for deterministic ties.

4. [experiments/eval_real_trace_eviction.py:170](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:170): capacity boundary is correct only for `capacity > 0`. For `capacity == 0`, the loop empties cache then still adds the missed key, producing cache size 1 and possible hits under zero cache.
   Fix: reject nonpositive caps in `main`, or special-case `capacity <= 0` as “all misses, never add”.

5. [experiments/eval_real_trace_eviction.py:86](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:86): sequence boundaries are discarded and cache persists across independent traces. Belady also sees future uses in the next sequence. If sequences are meant independent, this biases results, especially giving oracle cross-sequence knowledge. Fix by simulating per sequence and aggregating, or emitting boundaries and clearing `cache/last_used/freq/key_layer` at each boundary. The layer cycle `0..23` is preserved within each sequence.

Confirmed OK:

- The rolling `occ_ptr` idea itself is correct with a true monotonically increasing access index; querying many residents in one eviction does not corrupt other keys.
- Belady uses `max(next_use)` and `INF`, so it does pick farthest/never-again first once `next_use` is correct.
- Miss count and hit rate are independent of bandwidth in this demand-only code. Bandwidth only sets `fetch_ms`; stall scales linearly, so one bandwidth is enough for stall-headroom ratio.
tokens used
31,007
Findings:

1. [experiments/eval_real_trace_eviction.py:115](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:115), [131](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:131), [158](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:158): Belady’s `pos` is per layer-step, not per expert access. All top-K experts in one layer share the same `cur_pos`, so `next_use(..., cur_pos)` skips `<= cur_pos` and can treat an unprocessed current top-K resident as already past. Belady can evict an expert needed later in the same `experts` list, making the oracle non-oracle and usually artificially worse.
   Fix: flatten to expert-slot positions for Belady/simulation:
   ```python
   flat_stream = [(l, e) for l, experts in stream for e in experts]
   for i, key in enumerate(flat_stream): occ.setdefault(key, []).append(i)
   for pos, (l, e) in enumerate(flat_stream): ...
   evict_one(l, pos)
   ```
   If top-K is meant as one batch demand, instead protect the whole current `(l, experts)` set during evictions.

2. [experiments/eval_real_trace_eviction.py:147](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:147): `specmd_ls` distance is wrong for same-layer residents. `(key_layer - cur_layer) % num_layers == 0` treats same-layer experts as nearest reuse, but after the current layer they are farthest: next token’s same layer. This makes LS retain stale same-layer entries and can make Belady headroom look artificially larger.
   Fix:
   ```python
   def ls_dist(k):
       d = (key_layer[k] - cur_layer) % num_layers
       return num_layers if d == 0 else d
   victim = max(cache, key=lambda k: (ls_dist(k), -last_used[k]))
   ```

3. [experiments/eval_real_trace_eviction.py:147](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:147): LS tie-break is MRU, because `max(..., last_used[k])` evicts the most recently used among equal cyclic distance. If LS is supposed to be “cyclic distance, LRU fallback”, this is wrong and can perturb LS vs LRU. Use `-last_used[k]` as above, optionally add `k` for deterministic ties.

4. [experiments/eval_real_trace_eviction.py:170](/home/abc/Placement/HLS/spice/experiments/eval_real_trace_eviction.py:170): capacity boundary is correct only for `capacity > 0`. For `capacity == 0`, the loop empties cache then still adds the missed key, producing cache size 1 and possible hits under zero cache.
   Fix: reject nonpositive caps in `main`, or special-case `capacity <= 0` as “all misses, never add”.
