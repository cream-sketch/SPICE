# Validation Notes

Validation was run on `ial-gpu_workstation_1` under
`/home/ial-lvyx/workspace/spice`.

## System Simulator

Result files:

- `device3_system_SUMMARY.md`
- `device3_prefetch_system_summary.csv`
- `baseline_stress_results.md`
- `iccd_system_results.md`

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
- `anchored_eval_20260602_fix.json`
- `multiseed_2000/`

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

The anchored evaluation validates the paper-style re-initialization path: after
each verified target layer, the draft rollout restarts from the target hidden
state and uses observed target routing history as context.

Key metrics from `anchored_eval_20260602_fix.json`:

- Checkpoint: seed13, 2000 training steps
- Anchor re-initialization: `true`
- Observed routing history context: `true`
- Verified prefetch hit rate: `0.992432`
- Verified fallback slot rate: `0.007568`
- Wrong-prefetch rate: `0.253079`
- Average lookahead depth: `2.378906`

The four-GPU multiseed run in `multiseed_2000/` validates reproducibility across
seeds 7, 13, 23, and 31 at 2000 draft-training steps.

## Qwen1.5-MoE-A2.7B Trace Path

Result files:

- `qwen_moe_trace/`

This run validates the real-checkpoint trace path on Qwen1.5-MoE-A2.7B:

- Downloaded the model to `/home/ial-lvyx/workspace/models/Qwen1.5-MoE-A2.7B`
  through the HuggingFace mirror endpoint.
- Loaded Qwen with `bfloat16` and `device_map=auto`.
- Captured all 24 `mlp.gate` router tensors.
- Confirmed Qwen config: 60 routed experts and top-4 experts per token.
- Evaluated verified prefetching on 256 real routed tokens.

Key trace-prefetch metrics:

- Oracle upper bound: hit `1.000000`, fallback `0.000000`
- Last-observed-layer repeat: hit `0.342122`, fallback `0.657878`
- Layer-prior predictor: hit `0.385824`, fallback `0.614176`

The simple non-draft predictors leave high fallback rates on real Qwen routing
traces, which motivates the model-specific SPICE draft predictor wrapper as the
next experiment.

Checkpoints and logs are intentionally not committed.
