# Batch=1 exact MoE offload: exhaustive lever map (real Qwen1.5-MoE-A2.7B, A800, clean node 172)

All numbers: real offloaded decode, 64 tokens (warmup 8), argmax-exact vs full-resident reference,
single isolated process. `experiments/harness/realrt/offload_qwen.py`. Cache = GPU expert slots out
of 1440 total routed experts (60/layer x 24).

## Fair matrix (TPOT ms; same cache budget per column block; all EXACT)

| cache (resident %) | cpu_serve | on_demand (LRU) | hybrid (honest pop.) | split_cpu_gpu g=0.75 | compressed_fetch (nvCOMP ANS) |
|--------------------|----------:|----------------:|---------------------:|---------------------:|------------------------------:|
| 72  (5%)           | 77        | 88              | 79                   | ~ (loses)            | 96                            |
| 144 (10%)          | 77        | 75              | 76                   | **69.5**             | 85                            |
| 288 (20%)          | 77        | 67              | 72                   | **63.9**             | 79                            |
| 576 (40%)          | 77        | **55.7**        | 62                   | 57.0                 | --                            |

Bars: full-bf16 H2D per token = 75.5 ms; cpu_serve is cache-independent.

## Settled findings (do not re-explore at batch=1)

1. **PCIe weight VOLUME is the fundamental wall.** on_demand TPOT scales monotonically with miss
   rate (88->75->67->56 as cache 72->144->288->576), i.e. it is H2D-volume-bound. Best exact ~55-57 ms
   (on_demand / split at 40% cache), set by the PCIe transfer of the ~50% misses.
2. **All exact levers give only modest gains (5-30%); none is dramatic.** No batch=1 lever escapes
   the volume wall.
3. **on_demand LRU is the strong simple baseline**; at equal cache it beats hybrid residency
   (static-popularity + CPU-serve) for cache >= 144 (LRU hit ~= popularity hit, and H2D-fetch beats
   CPU-serve per miss). Earlier "hybrid wins" was only vs cpu_serve, NOT the fair equal-cache compare.
4. **split_cpu_gpu (CPU||GPU concurrent split) wins modestly at small/medium cache** (-8% @144,
   -5% @288), shrinking to ~0 at 576 (few misses left to offload). Real overlap requires a THREADED
   CPU worker (torch CPU matmul releases GIL); naive single-thread is worse than either endpoint
   (sum, no overlap). The cost-model ~36 ms was too optimistic: per-layer thread dispatch +
   activation D2H/H2D overhead caps the gain (g=0 all-CPU-threaded 86.9 > inline cpu_serve 77).
5. **Lossless compression of dense expert weights does NOT net-win with a separate decode pass.**
   nvCOMP ANS+byteplane ratio 1.39x (bit-exact) but compressed_fetch 79-85 ms LOSES to on_demand.
   Corroborated by SOTA: EuroSys'26 IBP (dense lossless ~1.1x over PCIe), ASPLOS'26 ZipServ
   (~1.43x, 30%). A ZipServ-style FUSED decode-GEMV (decode in-register, no separate pass) is
   projected ~55-62 ms (codex) -- exact, beats baselines, but reproduces published work and needs a
   custom CUDA kernel. Compression is prior work, not a SPICE contribution.
6. **SPICE forecast / prefetch helps nowhere dramatically at batch=1.** Prefetch hides latency, but
   the wall is volume; gos (forecast prefetch) measured 142 ms (worst). The only batch=1 setting where
   forecast could matter (deep compressed pipeline) is gated by the marginal compression ratio.

## Implication for direction

Batch=1 exact MoE offload is fundamentally PCIe-volume-bound; the achievable exact floor is ~55 ms
(cache + modest CPU||GPU overlap), and SPICE's route forecast adds little. For forecast to be a real
contribution it needs a REUSE regime -- batch>1 or a speculative/multi-token window -- where one
expert fetch is amortized across many tokens and forecasting WHICH experts the window needs enables
prefetch+reuse. That is the regime to target next.
