# Validation Notes

Validation was run on `ial-gpu_workstation_1` under
`/home/ial-lvyx/workspace/spice`.

## System Simulator

Result files:

- `device3_system_SUMMARY.md`
- `device3_prefetch_system_summary.csv`

The device3 run reproduced the verified correctness harness, prefetch-system
simulation, cache sweep, top-k stress, timeline replay, and energy replay.

Key checks:

- Lossless verified execution: `max_abs_logit_diff = 0.0`
- Device3 memory probe: two `14500 MiB` buffers allocated, totaling `29000 MiB`
- Correctness invariant preserved: prefetch misses affect latency, not target
  logits

## Draft-Model Path

Result files:

- `draft_train_800.json`
- `draft_prefetch_eval_fixed.json`

This run validates the non-proxy SPICE draft-model code path:

- Frozen target attention/router
- Shared-down LoRE expert surrogates
- GRU routing-history context
- Route-KL loss plus hidden alignment loss
- Checkpoint save/load
- Adaptive prefetching from draft predictions
- Verified fallback
- Optional online self-correction

Key metrics from the fixed evaluation:

- Offline draft checkpoint: 800 training steps
- Draft trainable parameters: `573824`
- Parameter overhead vs synthetic target: `0.016822`
- Before-online slot hit rate: `0.502930`
- After-online slot hit rate: `0.503418`
- Verified prefetch hit rate: `0.857056`
- Verified fallback slot rate: `0.142944`
- Corrected wrong-prefetch rate: `0.519211`
- Average lookahead depth: `1.944336`

Checkpoints and logs are intentionally not committed.
