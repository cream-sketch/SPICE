#!/usr/bin/env bash
set -euo pipefail

cd /data/ziheng/spice

BASE=/data2/ziheng/spice_runs/qwen2_57b_bf16
RUN=/data2/ziheng/spice_runs/qwen2_57b_full8
PY=/data/ziheng/miniconda3/envs/caproute_vllm/bin/python
MODEL=/data2/ziheng/models/Qwen2-57B-A14B-Instruct
export CUDA_VISIBLE_DEVICES=${GPU:-2}

mkdir -p "$RUN/decode" "$RUN/offload"

if [[ -f "$BASE/decode_smoke/dec_00000.pt" && ! -f "$RUN/decode/dec_00000.pt" ]]; then
  cp "$BASE/decode_smoke/dec_00000.pt" "$RUN/decode/dec_00000.pt"
fi

echo "[stage] collect/resume Qwen2-57B decode traces" | tee "$RUN/full_pipeline.log"
"$PY" experiments/harness/datagen/gen_decode_traces_offload.py \
  --model_dir "$MODEL" \
  --text_file experiments/harness/sample_prompts.txt \
  --out_dir "$RUN/decode" \
  --gpu 0 \
  --n_prompts 8 \
  --gen 32 \
  --prompt_len 128 \
  --save_hidden_states \
  --gpu_mem 70GiB \
  --cpu_mem 600GiB \
  --offload_folder "$RUN/offload" \
  --resume \
  2>&1 | tee "$RUN/decode_full8.log"

echo "[stage] convert decode traces for LoRE training" | tee -a "$RUN/full_pipeline.log"
"$PY" experiments/harness/datagen/decode_to_real_lore_traces.py \
  --decode_dir "$RUN/decode" \
  --out_dir "$RUN/lore_trace" \
  --max_steps 256 \
  2>&1 | tee "$RUN/convert_lore_trace.log"

echo "[stage] train real LoRE" | tee -a "$RUN/full_pipeline.log"
"$PY" experiments/core/draft/train_real_lore.py \
  --model_id "$MODEL" \
  --trace_dir "$RUN/lore_trace" \
  --out_dir "$RUN/lore_train" \
  --gpu 0 \
  --rank 64 \
  --route_context 64 \
  --history gru \
  --steps 400 \
  --batch_traces 1 \
  --lr 5e-4 \
  --warmup 40 \
  --log_every 50 \
  --checkpoint_name real_lore.pt \
  2>&1 | tee "$RUN/lore_train.log"

echo "[stage] build real-LoRE forecast" | tee -a "$RUN/full_pipeline.log"
"$PY" experiments/harness/datagen/make_forecast_from_real_lore.py \
  --checkpoint "$RUN/lore_train/real_lore.pt" \
  --decode_dir "$RUN/decode" \
  --out_dir "$RUN/forecast" \
  --max_horizon 6 \
  --gpu 0 \
  2>&1 | tee "$RUN/forecast.log"

echo "[stage] SPICE exact residual runtime" | tee -a "$RUN/full_pipeline.log"
"$PY" experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
  --forecast_dir "$RUN/forecast" \
  --cost_json "$BASE/miss_assign_qwen2_57b_bf16_t16.json" \
  --out "$RUN/spice_exact.json" \
  --gpu 0 \
  --train_frac 0.5 \
  --residency 0.1 \
  --max_test_tokens 128 \
  --policies gos_cpu \
  --d_model 3584 \
  --d_inter 2560 \
  --top_k 8 \
  --cpu_threads 16 \
  --cpu_dtype bf16 \
  --shallow_depth 2 \
  --filler_compute_dim 4096 \
  --filler_repeats 1 \
  --timed_repeats 3 \
  2>&1 | tee "$RUN/spice_exact.log"

echo "[stage] SPICE substitution runtime" | tee -a "$RUN/full_pipeline.log"
"$PY" experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
  --forecast_dir "$RUN/forecast" \
  --cost_json "$BASE/miss_assign_qwen2_57b_bf16_t16.json" \
  --out "$RUN/spice_sub_rank7.json" \
  --gpu 0 \
  --train_frac 0.5 \
  --residency 0.1 \
  --max_test_tokens 128 \
  --policies gos_cpu \
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
  2>&1 | tee "$RUN/spice_sub_rank7.log"

echo "[stage] AdapMoE method-level replay baseline" | tee -a "$RUN/full_pipeline.log"
"$PY" experiments/harness/scheduler/adapmoe_trace_replay.py \
  --forecast_dir "$RUN/forecast" \
  --cost_json "$BASE/miss_assign_qwen2_57b_bf16_t16.json" \
  --resource_json "$BASE/resource_edges_a800_qwen2_57b.json" \
  --out "$RUN/adapmoe.json" \
  --residency 0.1 \
  --max_test_tokens 128 \
  --active_m 8,6,4 \
  2>&1 | tee "$RUN/adapmoe.log"

echo "[stage] HybriMoE method-level replay baseline" | tee -a "$RUN/full_pipeline.log"
"$PY" experiments/harness/scheduler/hybrimoe_trace_replay.py \
  --forecast_dir "$RUN/forecast" \
  --cost_json "$BASE/miss_assign_qwen2_57b_bf16_t16.json" \
  --resource_json "$BASE/resource_edges_a800_qwen2_57b.json" \
  --out "$RUN/hybrimoe.json" \
  --residency 0.1 \
  --max_test_tokens 128 \
  --prefetch_size 4 \
  2>&1 | tee "$RUN/hybrimoe.log"

echo "[stage] summarize comparison" | tee -a "$RUN/full_pipeline.log"
"$PY" experiments/harness/summarize_qwen2_compare.py \
  --run_dir "$RUN" \
  --out "$RUN/summary.md" \
  2>&1 | tee "$RUN/summary.log"

echo DONE | tee "$RUN/full_pipeline.done"
