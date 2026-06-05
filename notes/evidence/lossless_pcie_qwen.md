# Lossless expert-weight compression for offloaded-MoE PCIe reduction (real Qwen, A800)

Measured on `moe-server-... clean node 172` GPU, real Qwen1.5-MoE-A2.7B expert weights
(bf16), nvCOMP 5.2.0, via `experiments/harness/realrt/measure_lossless_pcie.py`.
All methods bit-exact verified (`exact=True`): decompress(compress(W)) == W byte-for-byte.

## Premise

Batch=1 offload: GPU expert compute ~0.8 ms/token vs H2D PCIe ~75 ms/token (1.66 GB/token @ 22 GB/s).
Lossless constraint (user): output must stay exact; lossy int4/drop is out. So the only lossless
PCIe lever is: compress experts offline (CPU, free), transfer fewer bytes, decompress on GPU.

## Result (32 expert tensors, 184.5 MB sample, reps=20)

| method               | lossless ratio | exact | decomp GB/s | PCIe ms | decomp ms | serial ms | pipelined ms |
|----------------------|---------------:|:-----:|------------:|--------:|----------:|----------:|-------------:|
| ANS + byteplane      | 1.394          | yes   | 34.7        | 54.1    | 47.9      | 102.0     | **54.1**     |
| Bitcomp + byteplane  | 1.324          | yes   | 36.7        | 57.0    | 45.2      | 102.2     | 57.0         |
| GDeflate + byteplane | 1.411          | yes   | 20.1        | 53.5    | 82.5      | 136.0     | 82.5         |
| Zstd + byteplane     | 1.438          | yes   | 13.2        | 52.5    | 125.7     | 178.2     | 125.7        |
| Deflate + byteplane  | 1.417          | yes   | 7.5         | 53.3    | 220.6     | 273.8     | 220.6        |

Bars: full-bf16 H2D = 75.5 ms; cpu_serve (no weight H2D) = 77 ms.
`byteplane` = bf16 high/low byte-plane separation (exposes low-entropy sign+exponent plane).
`pipelined` = double-buffered steady state = max(PCIe_compressed, decompress); `serial` = sum (no overlap).

## Findings (honest)

1. Lossless ratio ceiling ~1.4x (Zstd+byteplane 1.438 best ratio). Entropy bound: bf16 mantissa
   LSBs are near-random. Cannot exceed without going lossy.
2. GPU decompression is NOT free and NOT "hundreds of GB/s": ~35 GB/s for ANS/Bitcomp on Ampere
   A800 (no hardware decompression engine; SM-bound). Decompressing 1.66 GB ~ 48 ms. The 0.8 ms
   compute headroom does NOT hide it -- decompression is itself ~48 ms of GPU work.
3. SERIAL (transfer then decompress) LOSES: 102 ms >> 75 ms.
4. PIPELINED (transfer compressed chunk N+1 while decompressing chunk N; copy engine vs SM run
   concurrently) is the only win: steady state max(54, 48) ~ 54 ms < cpu_serve 77 ms, EXACT.
   Best = ANS + byteplane: ~54 ms, ~30% faster than cpu_serve, lossless.
5. Decompressor choice matters hugely via throughput, not ratio: Zstd has the best ratio (1.438)
   but decompresses at 13 GB/s -> pipelined 126 ms (loses). ANS (ratio 1.394, 35 GB/s) wins.

## Caveats / fairness

- `pipelined` is a double-buffered steady-state PROJECTION (max of the two stages), not a measured
  end-to-end TPOT. It assumes near-perfect transfer/decompress overlap + the expert GEMV (~0.8 ms)
  also overlaps. Realizing it needs nvCOMP decompress + double buffering integrated into the offload
  runtime (offload_qwen.py). Fill/drain and SM contention would erode some of the margin.
- Microbenchmark itself is real: real Qwen weights, real GPU compress/decompress, byte-exact verify.
- Orthogonal to residency: compression shrinks the fetched tail; residency (hybrid) removes hot
  experts from the fetch set entirely. They compose (resident hot + compressed-fetch cold tail).

## Bottom line

Lossless compression is the strongest EXACT batch=1 PCIe lever: ~1.4x bytes -> ~54 ms pipelined vs
75-77 ms, ~30% TPOT. But it is NOT a 2-4x win (entropy + Ampere decompress speed cap it), and it
requires a pipelined decompress implementation to beat cpu_serve at all.
