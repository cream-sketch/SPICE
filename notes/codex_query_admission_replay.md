Quick design check (be concise & blunt) for the decisive SPICE-X admission TPOT experiment you specified (ALL/SELECT/NONE/ORACLE, exposed-stall economics). I will build a single-stream deadline replay (not full CUDA yet) driven by real decode traces + measured components. Confirm the model is faithful enough to trust a TPOT verdict, or name the one thing that makes it invalid.

Measured (real A800, batch=1 decode): per-layer t_attn=0.48ms, t_gate=0.067, t_shared=0.14, t_expert_gpu=0.079 (resident routed), C_fetch=0.78ms/expert (17MB, one copy engine). CPU-serve burst cost table (wall ms for N concurrent misses): {1:0.18, 2:0.56, 3:1.62, 4:2.19} (intra-op contention). 24 layers, top_k=4, 60 experts.

Per-layer critical path model I plan:
  cp_layer = t_attn + t_gate + expert_region
  expert_region = max( gpu_side, cpu_side, demand_fetch_side )
    gpu_side  = t_shared + n_resident_routed * t_expert_gpu
    cpu_side  = cpu_burst_cost(n_cpu_served)   [from table]
    demand_fetch_side = (n_demand_fetch * 0.78) serialized on the one copy engine, MINUS what was prefetched in earlier layers (deadline-aware queue)
  ALL: every miss demand-fetched to GPU + admitted (LS evict). Fetches can be prefetched if predicted earlier (SPICE within-token draft) -> hidden.
  SELECT: every miss CPU-served THIS layer; admit to cache (so future tokens hit) only if predicted reuse passes ms-threshold; admitted ones fetched async in background (off critical path if bandwidth free).
  NONE: static resident, all misses CPU-served, never admit.
  ORACLE: admit iff true-future-reuse-before-eviction * (C_cpu_marginal - C_gpu) > 0.78 + V_victim.

QUESTIONS:
1. Is this critical-path max() model faithful for batch=1 (GPU stream || CPU threadpool || one copy engine, all feeding the per-layer merge barrier)? What's wrong?
2. How should SELECT's background admission-fetch contend with ALL's prefetch on the single copy engine fairly (equal bandwidth budget)?
3. The CPU burst table is the marginal exposed cost only if CPU work is NOT hidden under gpu_side. My max() captures that. But for ALL, CPU isn't used; for SELECT, CPU does the misses. Is comparing max(gpu,cpu,fetch) across policies apples-to-apples?
4. What's the minimal correct way to credit SPICE within-token draft prefetch (recall@k known) so ALL/SELECT both benefit from it and the only difference is admission? 
5. One thing that would make this replay invalid as a TPOT proxy vs real wall-clock?
