Harsh review of a NEW direction "SPICE-REC: bandwidth-decoupled miss recovery" (design below). Context: single-GPU batch-1 offloaded fine-grained MoE (Qwen 60-expert top4 / DeepSeek 64 top6), artificially capped GPU cache + PCIe so misses are frequent. Prior: I killed lossless miss-shadow earlier because during a miss PCIe was busy with demand weight-fetch so downstream prefetch had no free bandwidth. The new claim: Fiddler-style CPU miss service frees PCIe, reviving miss-shadow as a bandwidth-free downstream-prefetch window.

DECIDE:
1. Is the bandwidth-decoupling REAL and does it genuinely revive miss-shadow? My data: cpu_always(Fiddler) uses 0 MB/tok PCIe = 100% idle; CPU window ~1.7ms/layer at 5% residency -> ~37MB ~2 experts prefetchable/layer at 22GB/s. Is harvesting idle-PCIe-during-CPU-serve for draft downstream prefetch a sound, real mechanism, or is there a hidden serialization (CPU<->GPU activation transfer also uses PCIe? merge barrier? host mem bandwidth contention) that eats it?
2. Is this just "overlap CPU compute with prefetch" which HybriMoE already does? What EXACTLY is novel vs HybriMoE (which has CPU expert exec + impact-driven prefetch + caching)? Is "CPU miss service as a bandwidth-decoupling primitive that converts miss latency into verified downstream prefetch progress" a defensible distinct contribution, or appendix? Blunt: defensible / appendix / dead.
3. The within-token draft (recall@1-8 = 1.0..0.67) was USELESS for eviction/residency (I showed oracle-within hurts cache hit-rate) but I now claim it's the RIGHT predictor for in-window downstream prefetch. Is that consistent/correct?
4. The kill test (spice_fetch_fallback / cpu_only_fiddler / spice_rec / oracle_shadow; GO if spice_rec beats cpu_only_fiddler by >=15-20% exposed-stall): right test? right threshold? what's missing? Note CPU-serve activation D2H/H2D is KB-level but still uses PCIe -- does that contend with the downstream weight prefetch on the same PCIe link and partly defeat the decoupling?
5. The single most likely reason SPICE-REC shows NO gain over cpu_only_fiddler, and the one measurement to check first.
Design attached.
# SPICE-REC: Bandwidth-decoupled miss recovery (user-proposed, supersedes rent-or-buy)

## Core thesis
A GPU-fetch miss does TWO harms: (1) stalls current layer, (2) saturates PCIe with 17MB demand weight-fetch, BLOCKING useful downstream prefetch. Fiddler CPU-compute serves the miss on CPU (small activation, exact) -> PCIe FREED. During the CPU-compute window, SPICE draft prefetches downstream-layer experts on the now-free PCIe -> miss latency becomes a bandwidth-free RECOVERY WINDOW for verified prefetch.
Contribution = DAG-level coupling: CPU exact miss service frees PCIe bandwidth for verified SPICE downstream prefetch. Not Fiddler alone (CPU compute), not SPICE alone (draft prefetch).

## Why it revives the KILLED miss-shadow
Earlier lossless miss-shadow died because: during a miss, PCIe busy fetching demanded weight -> downstream prefetch queued behind demand -> no free bandwidth. With CPU serving the miss, demand path uses NO PCIe -> downstream prefetch overlaps the CPU window. The primitive reopens it.

## Measured support (real A800)
- cpu_always (Fiddler) H2D = 0 MB/tok -> PCIe 100% IDLE = wasted bandwidth SPICE-REC harvests.
- T_cpu_exact=0.18ms/expert, fetch=0.78ms (PCIe 22GB/s). 5% residency ~3.4 misses/layer -> CPU window cpu_burst(3.4)~1.7ms -> 1.7ms*22GB/s ~ 37MB ~ 2 experts prefetchable/layer.
- draft within-token recall@1..8 = 1.0/0.87/0.79/0.73/0.67 (STRONG) -> right predictor for in-window downstream prefetch (was useless for eviction; finds its home here).

## Cost DAG
GPU-fetch miss: H2D(weight_e) -> GPU_compute(e)  [fetch predecessor, serial, blocks PCIe]
CPU recovery:   CPU_exact(e,h) -> resume   ||  SPICE_shadow_rollout -> H2D(prefetch downstream experts)
Benefit = (T_fetch - T_cpu_exact) + future_misses_absorbed_by_prefetch, where
useful_prefetch_during_cpu_window ~ min(B_pcie * T_cpu_window, correctly_predicted_future_expert_bytes).

## Kill test (4 policies, tight-cache/bw-constrained/miss-heavy regime, NOT the 0.24% ideal)
- spice_fetch_fallback: miss -> GPU fetch (PCIe blocked), no recovery prefetch
- cpu_only_fiddler: miss -> CPU serve, NO downstream prefetch (PCIe idle, wasted)
- spice_rec: miss -> CPU serve + draft downstream prefetch on freed PCIe
- oracle_shadow: CPU serve + ORACLE downstream prefetch (true future experts) -- upper bound
GO: spice_rec beats cpu_only_fiddler by >=15-20% MORE exposed-stall reduction. Then SPICE genuinely fills miss-handling (not just Fiddler).

## Risks
1. T_cpu_exact must be << fetch (measured 4x faster, OK) AND CPU window long enough to prefetch useful bytes.
2. draft downstream prediction must be accurate (recall 0.7-1.0 within-token, OK) else H2D pollution.
3. Must be tight-cache/bw-constrained regime (artificially capped bandwidth + cache), not SPICE Table I ideal.
4. prior-art: novelty is the DAG coupling (CPU frees PCIe for verified recovery) vs Fiddler(CPU compute)+SPICE(draft)+HybriMoE(overlaps CPU compute & prefetch). MUST differentiate from HybriMoE.
