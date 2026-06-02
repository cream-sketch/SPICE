# SPICE ICCD Supplemental Results For Paper

Result directory: `F:\ICCD\SPICE\experiments\results\20260528_103744`

Remote run directory: `/home/ial-lvyx/workspace/spice_iccd_runs/20260528_103744`

## What Was Run

- Verified lossless correctness harness on GPU0.
- Real-model PPL smoke test with cached GPT-2 on GPU0.
- Main prefetch/offloading comparison on GPU1.
- Offline/online/no-LoRE overhead ablation on GPU2.
- Top-K/PCIe saturation stress on GPU3.
- Controlled proxy comparison for SP-MoE, MoE-SpeQ, ExpertFlow, and AdapMoE on GPU3.
- CUDA copy/compute timeline workload with Nsight Systems on GPU1.
- GPU power traces for idle and copy/compute workload settings.

No datasets were transferred through the local machine. The only remote reads
were from the workstation's own cache or synthetic traces generated on the
workstation.

## Paper-Usable Results

### Correctness / Lossless Claim

The verified execution harness shows that SPICE-style prefetching changes data
movement but not target execution:

- Max logit difference: `0.0`
- Exact argmax match rate: `1.0`
- Baseline pseudo-PPL: `1035.2114`
- SPICE verified pseudo-PPL: `1035.2114`
- Prefetch-slot hit rate: `74.24%`
- Fallback-slot rate: `25.76%`

Safe paper wording:

> In a verified MoE execution harness, SPICE and on-demand offloading produce
> identical logits while differing only in prefetch hit/fallback behavior,
> supporting the claim that verified prefetching preserves target-model
> semantics.

### Main Prefetch / Baseline Comparison

Controlled same-harness simulation with 16 layers, 64 experts, Top-K=6, 512
expert cache entries, and 8 MB expert transfers:

| Policy | TPOT (ms) | Cache hit | Fallback | H2D GB | PCIe active fraction |
|---|---:|---:|---:|---:|---:|
| Naive | 71.25 | 0.00% | 100.00% | 384.00 | 43.86% |
| LRU | 44.67 | 85.04% | 14.96% | 57.44 | 10.46% |
| MoE-Offloading | 44.67 | 85.04% | 14.96% | 57.44 | 10.46% |
| Pre-gated | 41.83 | 94.15% | 5.85% | 147.27 | 28.65% |
| SPICE | 41.03 | 99.76% | 0.24% | 198.80 | 39.43% |

Safe paper wording:

> Under the same cache and PCIe model, SPICE reduces critical-path fallback
> traffic from 5.85% for Pre-gated to 0.24%, improving TPOT from 41.83 ms to
> 41.03 ms while using more overlapped prefetch traffic.

Important caveat: this is a controlled harness result, not a direct official
reproduction of every external system.

### Online Self-Correction Overhead

| Variant | TPOT (ms) | Fallback | Draft overhead (ms/run) | Online overhead (ms/run) |
|---|---:|---:|---:|---:|
| SPICE offline | 41.04 | 0.24% | 491.52 | 0.00 |
| SPICE online | 42.96 | 0.24% | 491.52 | 983.04 |
| SPICE no-LoRE | 42.56 | 5.10% | 491.52 | 0.00 |

Safe paper wording:

> Online self-correction is not free: in our overhead model, enabling online
> updates increases TPOT from 41.04 ms to 42.96 ms. We therefore report the
> offline draft path as the default latency configuration and treat online
> adaptation as an optional mode.

### Top-K / PCIe Saturation

The original stress run was SPICE-only. A follow-up reviewer-facing run now
compares all baselines under the same tight cache budget:

Result directory: `F:\ICCD\SPICE\experiments\results\baseline_stress_20260528_113254`

| Top-K | Naive TPOT | LRU TPOT | MoE-Offloading TPOT | Pre-gated TPOT | SPICE TPOT | SPICE fallback | SPICE PCIe active |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 50.42 | 42.92 | 42.92 | 41.66 | **41.01** | **0.46%** | 23.19% |
| 4 | 60.83 | 45.91 | 45.91 | 43.30 | **42.51** | **7.43%** | 25.49% |
| 6 | 71.25 | 49.10 | 49.10 | 45.09 | **43.14** | **6.97%** | 38.53% |
| 8 | 81.67 | 52.65 | 52.65 | 54.37 | **51.11** | **24.36%** | 58.73% |
| 10 | 92.08 | **56.25** | **56.25** | 68.90 | 58.36 | 33.41% | 73.48% |
| 12 | 102.50 | **58.39** | **58.39** | 76.54 | 65.35 | 39.03% | 82.74% |

Safe paper wording:

> In a high-utilization Top-K stress study, Naive already drives modeled PCIe
> activity from 20.66% to 60.98%, while Pre-gated reaches 34.91--100% for
> K>=4. SPICE remains fastest through K=8 by reducing synchronous fallback, but
> at K=10 and K=12 the conservative LRU/MoE-Offloading cache has lower TPOT.
> We therefore state SPICE as a bounded systems optimization rather than as a
> way to remove the physical PCIe limit.

### Proxy Comparison To Recent Methods

Controlled same-harness proxy variants:

| Variant | TPOT (ms) | Cache hit | Fallback |
|---|---:|---:|---:|
| ExpertFlow proxy | 41.70 | 98.66% | 1.34% |
| AdapMoE proxy | 42.08 | 94.38% | 5.62% |
| SP-MoE proxy | 43.04 | 95.39% | 4.61% |
| MoE-SpeQ proxy | 41.98 | 95.72% | 4.28% |
| SPICE verified | 41.03 | 99.78% | 0.22% |

Safe paper wording:

> We additionally include controlled proxy variants for recent systems to
> isolate the effect of routing-only verified prefetching. These proxy results
> should be presented as same-harness approximations, not official reproductions.

### Power / Timeline Telemetry

- Idle/low-load GPU2 sample: average `160.60 W`, estimated `3185.53 J` over
  19.84 s.
- Copy+compute workload GPU1 sample: average `184.32 W`, peak `452.98 W`,
  estimated `8275.25 J` over 44.90 s.
- Nsight Systems trace generated:
  `results/20260528_103744/gpu1_nsys/copy_timeline.nsys-rep`
- CUDA GPU trace CSV generated:
  `results/20260528_103744/gpu1_nsys/cuda_gpu_trace_cuda_gpu_trace.csv`
- Copy/compute overlap workload: 16 GB transferred in 0.569 s, effective
  28.13 GB/s.

Safe paper wording:

> We report power telemetry to avoid inferring energy from latency alone. The
> current measurements are telemetry evidence, not a full system energy study.

## Results Not Safe To Overclaim

- Do not claim official reproduction of SP-MoE, MoE-SpeQ, ExpertFlow, or
  AdapMoE from the proxy table.
- Do not claim real DeepSeek-V2-Lite perplexity preservation from the GPT-2
  smoke result. The verified MoE harness supports the semantic argument; a
  full DeepSeek PPL table would still be stronger.
- Do not claim system energy savings unless a paired baseline-vs-SPICE energy
  protocol is run with equal duration and token count.

## ICCD System-Focused Addendum

Result directory: `F:\ICCD\SPICE\experiments\results\iccd_system_20260528_121027`

Additional GPU experiments were run for ICCD positioning:

- Paired energy replay on the same GPU:
  `results\iccd_system_20260528_121027\gpu0_energy_paired\energy_per_token.json`
- Cache-budget sweep:
  `results\iccd_system_20260528_121027\cache_sweep_summary.csv`
- Nsight timeline replay:
  `results\iccd_system_20260528_121027\gpu3_timeline\*\*.nsys-rep`

Paste-ready LaTeX:

- `F:\ICCD\SPICE\experiments\overleaf_iccd_system_results.tex`

Safe wording:

> The paired hardware replay shows that SPICE reduces replay J/token relative
> to Naive offloading, but it does not dominate every cache-based baseline.
> We therefore avoid claiming universal energy reduction. The cache-budget
> sweep is stronger for ICCD: SPICE is most useful in the memory-constrained
> region, especially 256--512 cache slots, where verified prefetching reduces
> fallback traffic without requiring a large resident expert set. When the
> cache budget is large enough to hold most hot experts, all methods converge.
