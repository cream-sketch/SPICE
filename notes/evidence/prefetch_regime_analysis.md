# When does expert prefetch help in offloaded MoE? (real measurements + codex/literature review)

## Empirical (real Qwen1.5-MoE-A2.7B, A800, PCIe ~22GB/s, argmax-exact)

batch=1:
- cpu_serve 77ms, on_demand (reactive async H2D) 75ms, gos forecast-prefetch 142ms (WORSE).
- compressed shallow 85ms; ORACLE deep cross-layer prefetch pipeline 78.9ms -- still loses to on_demand 75.
- fused-decode compressed (NO prefetch) 61ms (1.29x, wins via volume reduction).

batch>1 (wikitext test, on_demand reactive, no forecast), per-token TPOT / fetches-per-token:
- B=1 101.0 / 136; B=2 101.9 / 138; B=4 89.1 / 118; B=8 74.7 / 97; B=16 57.4 / 73; B=32 38.2 / 48.
- 2.6x per-token drop B1->B32 is pure batching AMORTIZATION (expert shared across batch tokens), captured by
  a REACTIVE policy; forecast adds nothing.

## Core finding (codex-reviewed, literature-checked)

Offloaded-MoE DECODE at batch=1 short-context is PCIe-BANDWIDTH-saturated, not latency-bound. Prefetch
only moves transfers earlier; it cannot reduce the irreducible expert-byte volume, and reactive async H2D
already keeps PCIe ~100% busy -> no idle link time to fill. Prefetch needs a COMPUTE SHADOW (concurrent
work whose time >= transfer time) + spare bandwidth; at batch=1 each expert weight serves 1 token (compute
~us, weight-HBM-read-bound) << transfer (~0.77ms/17MB expert) -> no shadow.

Published prefetch-MoE-offload systems do NOT contradict this -- they live in different regimes (verified):
- Fiddler: SUPPORTS the premise -- at single-batch decode it CPU-serves missed experts instead of PCIe-copying;
  notes each expert gets <=1 token.
- MoE-Infinity: gains are cache/residency/reuse + avoiding bad prefetch, NOT forecast-beats-saturated-reactive;
  reports inaccurate prediction can be worse than on-demand vLLM.
- Pre-gated MoE: changes the architecture (trained pre-gate), not exact original routing; hides migration under
  attention/non-MoE compute; baseline is serial on-demand / prefetch-all.
- SwapMoE: lossy (virtual experts), not exact decode.
- BrainStorm: batch 8 / dynamic vision, not batch=1 MoE decode. Lina: distributed all-to-all, not PCIe offload.

So "prior prefetch works" because of shadow-rich regimes (long context, batch>1, prefill, heavy dense/attention)
or weak (serial) baselines or bundled residency -- not because forecast beats a good reactive stream at batch=1
short-context exact decode.

## Where prefetch/forecast COULD earn its place (the niche)

Long-context attention provides a real shadow, but the crossover is LATE: for a Qwen-like small model the
per-step attention compute reaches the ~50ms cold-expert-transfer time only around ~32-64k context (4k is far
too short). For Mixtral-scale cold streaming (~22.5GB/token-step ~ 1s) even 32k attention cannot hide it.

=> The only defensible foothold for forecast-prefetch is the COMBINATION: long context (>=32-64k, attention
shadow) + volume reduction (compression / partial caching so the residual misses are small enough to hide under
the shadow). Neither alone suffices: long context alone can't hide Mixtral's 1s transfer; compression alone gives
no prefetch shadow (and fused decode removes it). This combination is not occupied by the systems above.

## What actually wins (all attack VOLUME, not latency)

cpu_serve (avoid transfer) | fused-compressed (1.33x fewer bytes + more residency) | residency (don't fetch hot)
| batch amortization (share experts). Prefetch/forecast is superfluous wherever a good reactive policy runs,
except the long-context + volume-reduced niche above.
