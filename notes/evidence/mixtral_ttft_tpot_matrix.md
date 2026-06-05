# Mixtral-8x7B single-GPU offload TTFT/TPOT matrix (hpclab04, A800 80GB, 4 datasets x 4 policies)

Real Mixtral-8x7B-Instruct (90GB experts, offloaded to CPU pinned), single A800. n=4 prompts/cell,
decode 16 (warmup 4), prefill=TTFT. Cache auto-sized from free GPU mem (NOTE: free varied 57-75GB
across policies due to other-user contention -> cache% 63-85%, so cross-policy compare has noise).
Code: offload_mixtral.py + bench_ttft_tpot.py.

MATRIX  TTFT_ms / TPOT_ms
dataset        cpu_serve        on_demand    fused_compressed   split_cpu_gpu
wikitext     12937/306        2575/185        2532/245         6604/226
humaneval    12033/301        2010/179        1966/244         6249/227
gsm8k         6132/308        2040/182        2056/226         3219/225
narrativeqa 140952/316        2542/170        2384/199        70287/231

## Findings
- on_demand (reactive LRU) is BEST on both TTFT and TPOT across all 4 datasets.
- cpu_serve: catastrophic TTFT at long context (narrativeqa 141s = CPU prefill over 4k tokens x all experts);
  worst TPOT (~306). Only viable for short-context pure-TPOT.
- fused_compressed: competitive TTFT but TPOT WORSE than on_demand (199-245 vs 170-185) -- the
  decode-on-every-use cost hurts when cache is high (a resident expert is decoded on every use instead
  of stored bf16). Compression only wins at LOW cache where transfer savings dominate.
- split_cpu_gpu: bad TTFT (CPU prefill + threading; narrativeqa 70s), TPOT 225-231 (> on_demand).

## Conclusion
For a model that nearly fits (Mixtral 90GB on 80GB), plain reactive on_demand wins; offload
"optimizations" (cpu-serve / compression / split) do not help and often hurt. Consistent with the
Qwen finding: offload optimizations matter only at SMALL cache (model >> GPU). forecast/prefetch
policies are NOT in this baseline matrix -- they are the next experiment (DeepSeek-Lite long-context).
