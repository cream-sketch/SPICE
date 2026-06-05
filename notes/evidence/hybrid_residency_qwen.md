# Hybrid resource-DAG (static-popularity residency + CPU-serve tail) -- real Qwen, A800 (172)

Real offloaded Qwen1.5-MoE-A2.7B decode, clean node 172, batch=1, 64 decode tokens (warmup 8),
3 repeats. `experiments/harness/realrt/offload_qwen.py`, policy `hybrid_resident_cpu`.
Hybrid = pin the C hottest experts GPU-resident (0 PCIe) + CPU-serve every miss (no weight H2D).

## Honesty / fairness controls

- Popularity is estimated on a SEPARATE calibration text (`--calib_prompt`, an unrelated economics
  sentence), NOT the eval sequence -> deployable static popularity, no future leakage. The oracle
  variant (`--oracle_resident`, popularity from the eval sequence itself) is reported ONLY as a
  ceiling/gap reference, never as a result. (codex caught the original same-sequence oracle leak.)
- Resident set is frozen during the timed window (no fetch/evict); the one-time warm fetch is
  excluded (synchronized before timing), analogous to a startup cache warm.
- All policies are argmax-exact vs the full-resident reference; CPU and GPU experts both use the
  same SwiGLU. Single process, isolated node.

## Result (mean of 3 repeats, TPOT ms)

Bars: cpu_serve = 76.8 ms; on_demand_fetch @144 = 76.0 ms.

| cache C | resident % | honest hit % | honest TPOT | vs cpu_serve | oracle ceiling |
|--------:|-----------:|-------------:|------------:|-------------:|---------------:|
| 72      | 5%         | 8.1%         | 79.5        | +3.5% (loses)| --             |
| 144     | 10%        | 14.7%        | 76.2        | ~tie (noise) | 71.1           |
| 288     | 20%        | 26.2%        | 71.6        | -6.8%        | 65.9           |
| 576     | 40%        | 46.4%        | 61.6        | -19.8%       | 53.9           |

Variance is small (e.g. C=576: 61.0 / 61.2 / 62.6).

## Findings

1. Static-popularity residency + CPU-serve tail is a real, deployable, exact win -- but only with
   enough resident budget. Below ~10% cache it loses (too few hits to offset GPU serialization).
2. The win scales with cache: 20% -> -7%, 40% -> -20% vs cpu_serve.
3. Honest (calibration-text) popularity captures most but not all of the oracle ceiling: at C=576,
   honest 61.6 vs oracle 53.9 (popularity-generalization gap ~14% of traffic). The hot experts are
   globally stable enough that an unrelated calibration text transfers well at large C.
4. This is NOT SPICE: it is plain static popularity caching. SPICE's per-token forecast adds nothing
   here -- consistent with the established result that prefetch loses to the H2D volume wall at
   batch=1 (compute headroom ~0.8 ms cannot hide ~75 ms transfer).

## Bottom line

At batch=1, the exact offload win is the resource split (hottest resident + CPU-serve the rest),
sized by GPU memory budget, NOT forecast-driven dynamic prefetch. Composes with lossless
compressed-fetch for the cold tail (see lossless_pcie_qwen.md).
