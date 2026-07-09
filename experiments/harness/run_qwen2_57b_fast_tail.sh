#!/usr/bin/env bash
set -euo pipefail

cd /data/ziheng/spice

RUN=/data2/ziheng/spice_runs/qwen2_57b_bf16
PY=/data/ziheng/miniconda3/envs/caproute_vllm/bin/python
MODEL=/data2/ziheng/models/Qwen2-57B-A14B-Instruct
export CUDA_VISIBLE_DEVICES=${GPU:-2}

"$PY" experiments/harness/datagen/make_single_decode_manifest.py \
  --decode_dir "$RUN/decode_smoke" \
  --model_dir "$MODEL" \
  --files dec_00000.pt \
  --prompt_len 128 \
  --gen 32 \
  2>&1 | tee "$RUN/manifest_fast.log"

"$PY" experiments/harness/datagen/decode_to_real_lore_traces.py \
  --decode_dir "$RUN/decode_smoke" \
  --out_dir "$RUN/lore_trace_fast" \
  --max_steps 32 \
  2>&1 | tee "$RUN/convert_lore_trace_fast.log"

"$PY" experiments/core/draft/train_real_lore.py \
  --model_id "$MODEL" \
  --trace_dir "$RUN/lore_trace_fast" \
  --out_dir "$RUN/lore_train_fast" \
  --gpu 0 \
  --rank 64 \
  --route_context 64 \
  --history gru \
  --steps 120 \
  --batch_traces 1 \
  --lr 5e-4 \
  --warmup 20 \
  --log_every 20 \
  --checkpoint_name real_lore.pt \
  2>&1 | tee "$RUN/lore_train_fast.log"

"$PY" experiments/harness/datagen/make_forecast_from_real_lore.py \
  --checkpoint "$RUN/lore_train_fast/real_lore.pt" \
  --decode_dir "$RUN/decode_smoke" \
  --out_dir "$RUN/forecast_fast" \
  --max_horizon 6 \
  --gpu 0 \
  2>&1 | tee "$RUN/forecast_fast.log"

"$PY" experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
  --forecast_dir "$RUN/forecast_fast" \
  --cost_json "$RUN/miss_assign_qwen2_57b_bf16_t16.json" \
  --out "$RUN/runtime_exact_fast.json" \
  --gpu 0 \
  --train_frac 0.0 \
  --residency 0.1 \
  --max_test_tokens 32 \
  --policies deep_fetch_all,deep_cpu,shallow_cpu,shallow_scheduler,gos_cpu \
  --d_model 3584 \
  --d_inter 2560 \
  --top_k 8 \
  --cpu_threads 16 \
  --cpu_dtype bf16 \
  --shallow_depth 2 \
  --filler_compute_dim 4096 \
  --filler_repeats 1 \
  --timed_repeats 3 \
  --allow_train_eval_fallback \
  2>&1 | tee "$RUN/runtime_exact_fast.log"

"$PY" experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
  --forecast_dir "$RUN/forecast_fast" \
  --cost_json "$RUN/miss_assign_qwen2_57b_bf16_t16.json" \
  --out "$RUN/runtime_sub_rank7_fast.json" \
  --gpu 0 \
  --train_frac 0.0 \
  --residency 0.1 \
  --max_test_tokens 32 \
  --policies shallow_scheduler,gos_cpu \
  --d_model 3584 \
  --d_inter 2560 \
  --top_k 8 \
  --cpu_threads 16 \
  --cpu_dtype bf16 \
  --shallow_depth 2 \
  --filler_compute_dim 4096 \
  --filler_repeats 1 \
  --timed_repeats 3 \
  --substitute_ranks 7 \
  --allow_train_eval_fallback \
  2>&1 | tee "$RUN/runtime_sub_rank7_fast.log"

echo DONE | tee "$RUN/fast_tail.done"
