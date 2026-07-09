# Qwen2 Memory-Limited Experiments

This note describes the portable single-GPU path used for Qwen2-MoE
memory-limited experiments on A800, L4, RTX 4060, or similar machines.

The entrypoint is:

```bash
MODEL_DIR=/path/to/Qwen2-57B-A14B-Instruct \
RUN_DIR=/path/to/spice_runs/qwen2_57b_l4 \
GPU=0 \
HW_TAG=l4 \
GPU_MEM=22GiB \
CPU_MEM=256GiB \
bash experiments/harness/run_qwen2_memory_limited_compare.sh
```

For RTX 4060 16GB, leave more headroom:

```bash
MODEL_DIR=/path/to/Qwen2-57B-A14B-Instruct \
RUN_DIR=/path/to/spice_runs/qwen2_57b_4060 \
GPU=0 \
HW_TAG=rtx4060 \
GPU_MEM=13GiB \
CPU_MEM=192GiB \
N_PROMPTS=4 \
GEN=32 \
bash experiments/harness/run_qwen2_memory_limited_compare.sh
```

The pipeline performs these stages:

1. Measure hardware-specific resource edges and miss-assignment costs.
2. Collect Qwen2-MoE decode traces with HuggingFace CPU offload.
3. Convert decode traces into real-LoRE training traces.
4. Train the lightweight real-LoRE draft predictor.
5. Build SPICE forecast dumps.
6. Run SPICE exact CPU residual scheduling.
7. Run SPICE low-confidence substitution.
8. Run an AdapMoE-style active-expert replay baseline.
9. Run a Pre-gated-style predictive-prefetch fetch-only baseline.
10. Write a Markdown summary to `$RUN_DIR/summary.md`.

Main result files:

- `spice_exact.json`: SPICE speculative prefetch plus exact CPU residual scheduling.
- `spice_sub_rank*.json`: SPICE with low-confidence substitution enabled.
- `adapmoe.json`: method-level AdapMoE-style replay on the same Qwen2 trace.
- `pregated_fetch_baseline.json`: method-level Pre-gated-style fetch-only baseline on the same Qwen2 trace.
- `summary.md`: compact TPOT comparison table.

Important baseline note: the released Pre-gated MoE artifact targets Switch/T5
FasterTransformer models and is not a drop-in Qwen2-MoE runtime. For Qwen2-57B
comparisons, use the method-level predictive-prefetch/fetch-only baseline unless
a full Qwen2 port is implemented.
