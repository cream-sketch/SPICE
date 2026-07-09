#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/ziheng/spice
MODEL_DIR=/data2/ziheng/models/Qwen2-57B-A14B-Instruct
RUN_DIR=/data2/ziheng/spice_runs/qwen2_57b_bf16
PY=/data/ziheng/miniconda3/envs/caproute_vllm/bin/python
GPU=${GPU:-2}

mkdir -p "$RUN_DIR"
cd "$CODE_DIR"

echo "[wait] Qwen2-57B download" | tee -a "$RUN_DIR/pipeline.log"
while pgrep -af "hf download Qwen/Qwen2-57B-A14B-Instruct" >/dev/null; do
  date | tee -a "$RUN_DIR/pipeline.log"
  du -sh "$MODEL_DIR" 2>/dev/null | tee -a "$RUN_DIR/pipeline.log" || true
  sleep 120
done

if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "[error] model index missing after download: $MODEL_DIR/model.safetensors.index.json" | tee -a "$RUN_DIR/pipeline.log"
  exit 2
fi

if find "$MODEL_DIR/.cache/huggingface/download" -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
  echo "[error] incomplete files remain under $MODEL_DIR/.cache/huggingface/download" | tee -a "$RUN_DIR/pipeline.log"
  exit 3
fi

echo "[stage] offload decode traces" | tee -a "$RUN_DIR/pipeline.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" experiments/harness/datagen/gen_decode_traces_offload.py \
  --model_dir "$MODEL_DIR" \
  --text_file experiments/harness/sample_prompts.txt \
  --out_dir "$RUN_DIR/decode_smoke" \
  --gpu 0 \
  --n_prompts 8 \
  --gen 32 \
  --prompt_len 128 \
  --save_hidden_states \
  --gpu_mem 70GiB \
  --cpu_mem 600GiB \
  --offload_folder "$RUN_DIR/offload" \
  2>&1 | tee "$RUN_DIR/decode_smoke.log"

echo "[stage] convert decode traces for LoRE training" | tee -a "$RUN_DIR/pipeline.log"
"$PY" experiments/harness/datagen/decode_to_real_lore_traces.py \
  --decode_dir "$RUN_DIR/decode_smoke" \
  --out_dir "$RUN_DIR/lore_trace_smoke" \
  --max_steps 192 \
  2>&1 | tee "$RUN_DIR/convert_lore_trace.log"

echo "[stage] train real LoRE smoke" | tee -a "$RUN_DIR/pipeline.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" experiments/core/draft/train_real_lore.py \
  --model_id "$MODEL_DIR" \
  --trace_dir "$RUN_DIR/lore_trace_smoke" \
  --out_dir "$RUN_DIR/lore_train_smoke" \
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
  2>&1 | tee "$RUN_DIR/lore_train_smoke.log"

echo "[stage] build real-LoRE forecast" | tee -a "$RUN_DIR/pipeline.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" experiments/harness/datagen/make_forecast_from_real_lore.py \
  --checkpoint "$RUN_DIR/lore_train_smoke/real_lore.pt" \
  --decode_dir "$RUN_DIR/decode_smoke" \
  --out_dir "$RUN_DIR/forecast_smoke" \
  --max_horizon 6 \
  --gpu 0 \
  2>&1 | tee "$RUN_DIR/forecast_smoke.log"

echo "[stage] runtime policy comparison: exact residual handling" | tee -a "$RUN_DIR/pipeline.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
  --forecast_dir "$RUN_DIR/forecast_smoke" \
  --cost_json "$RUN_DIR/miss_assign_qwen2_57b_bf16_t16.json" \
  --out "$RUN_DIR/runtime_exact_smoke.json" \
  --gpu 0 \
  --train_frac 0.5 \
  --residency 0.1 \
  --max_test_tokens 128 \
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
  2>&1 | tee "$RUN_DIR/runtime_exact_smoke.log"

echo "[stage] runtime policy comparison: SPICE low-confidence substitution proxy" | tee -a "$RUN_DIR/pipeline.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
  --forecast_dir "$RUN_DIR/forecast_smoke" \
  --cost_json "$RUN_DIR/miss_assign_qwen2_57b_bf16_t16.json" \
  --out "$RUN_DIR/runtime_sub_rank7_smoke.json" \
  --gpu 0 \
  --train_frac 0.5 \
  --residency 0.1 \
  --max_test_tokens 128 \
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
  2>&1 | tee "$RUN_DIR/runtime_sub_rank7_smoke.log"

echo "[done] qwen2_57b_bf16 smoke pipeline" | tee -a "$RUN_DIR/pipeline.log"
