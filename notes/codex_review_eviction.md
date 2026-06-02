Review this simulator for CORRECTNESS BUGS only (be terse, list concrete bugs + line-level fixes; do not rewrite wholesale). Output to stdout, do not write files.

File: experiments/eval_real_trace_eviction.py

Context: it replays real MoE per-token per-layer top-K routing as a sequential decode stream, demand-only (no speculative prefetch), and compares eviction policies (lru, lfu, specmd_ls, oracle_belady) on exposed H2D stall and hit rate, to test whether oracle Belady eviction has >=25% headroom over SpecMD Least-Stale under tight cache. Demand miss = synchronous fetch on a single PCIe channel (stall += expert_bytes/bandwidth).

Check specifically and brutally:
1. oracle_belady: the rolling occ_ptr / next_use logic. Is next_use(key,cur_pos) correct when called for MANY residents during one eviction, and across the whole replay with monotonically increasing cur_pos? Any case where the pointer is advanced wrongly or a resident's true next-use is mis-computed? Does Belady correctly pick the resident with FARTHEST next use (evict it)?
2. specmd_ls: is "evict resident with max (key_layer - cur_layer) mod num_layers" a faithful Least-Stale (evict the expert whose next cyclic reuse is farthest)? Does this actually differ from LRU on this cyclic stream, and is the tie-break sane?
3. Is miss-count (hence headroom) truly independent of bandwidth here (we claim it is, sweep only one bw)? Confirm or refute.
4. Eviction loop: `while len(cache) >= capacity and cache: evict_one(...)` then add — correct boundary (final size == capacity)? off-by-one?
5. Any bug that would make Belady look artificially better/worse than LS (which would invalidate the go/no-go)?
6. Does treating cross-sequence concatenation as one stream bias the tight-cache conclusion? Is the within-pass cyclic structure (layer 0..23 per token) preserved correctly?

Read the file yourself. Be concrete.
