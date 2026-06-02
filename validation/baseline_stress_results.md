# Baseline Stress Results

Run directory: `F:\ICCD\SPICE\experiments\results\baseline_stress_20260528_113254`

Protocol: 768 decoding steps, 16 layers, 64 experts, 8 MB expert transfers, and
the same tight cache budget for all policies. The run split Top-K values across
four RTX 5090 GPUs.

| Top-K | Naive TPOT | LRU TPOT | MoE-Offloading TPOT | Pre-gated TPOT | SPICE TPOT | SPICE fallback | SPICE PCIe active |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 50.42 | 42.92 | 42.92 | 41.66 | 41.01 | 0.46% | 23.19% |
| 4 | 60.83 | 45.91 | 45.91 | 43.30 | 42.51 | 7.43% | 25.49% |
| 6 | 71.25 | 49.10 | 49.10 | 45.09 | 43.14 | 6.97% | 38.53% |
| 8 | 81.67 | 52.65 | 52.65 | 54.37 | 51.11 | 24.36% | 58.73% |
| 10 | 92.08 | 56.25 | 56.25 | 68.90 | 58.36 | 33.41% | 73.48% |
| 12 | 102.50 | 58.39 | 58.39 | 76.54 | 65.35 | 39.03% | 82.74% |

Notes:

- Naive reaches 20.66-60.98% PCIe active; Pre-gated reaches 34.91-100% for
  `K>=4`.
- SPICE is best through `K=8` because verified prefetching reduces synchronous
  fallback compared with LRU/MoE-Offloading and Pre-gated.
- At `K=10` and `K=12`, the bandwidth limit dominates and LRU/MoE-Offloading
  have lower TPOT than SPICE.
- MoE-Offloading matches LRU in this controlled trace because the implemented
  history-based async prefetch has no extra future oracle beyond cache reuse.
