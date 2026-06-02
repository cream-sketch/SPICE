# Qwen1.5-MoE-A2.7B Trace Validation

Validation ran on `ial-gpu_workstation_1`.

## Trace Collection

- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Local path: `/home/ial-lvyx/workspace/models/Qwen1.5-MoE-A2.7B`
- Loader: `bfloat16`, `device_map=auto`
- Max memory: `0=8GiB,1=6GiB,3=8GiB,cpu=40GiB`
- Captured router modules: `model.layers.{0..23}.mlp.gate`
- Layers: `24`
- Experts: `60`
- Experts per token: `4`
- Prompts: `16`
- Valid routed tokens: `256`

Trace tensors are not committed. This directory keeps only compact metadata and
prefetch summaries.

## Verified Prefetch on Real Router Traces

All runs use `top_k=4`, `cache_capacity=256`, `l_min=2`, and `l_max=6`.

| Predictor | Prefetch Hit | Fallback | Wrong Prefetch | Avg Lookahead |
| --- | ---: | ---: | ---: | ---: |
| oracle | 1.000000 | 0.000000 | 0.000000 | 5.375000 |
| anchor_repeat | 0.342122 | 0.657878 | 0.925101 | 5.375000 |
| layer_prior | 0.385824 | 0.614176 | 0.758659 | 5.361654 |

Interpretation: the Qwen router traces are extractable and compatible with the
verified prefetch evaluator. Simple non-draft predictors leave a high fallback
rate, while oracle lookahead shows the available headroom. The next model-
specific step is to train/evaluate a Qwen wrapper for the SPICE draft predictor
using these hidden/router traces.
