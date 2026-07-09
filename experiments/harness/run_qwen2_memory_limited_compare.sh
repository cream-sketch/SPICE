#!/usr/bin/env bash
set -euo pipefail

# End-to-end Qwen2-MoE memory-limited comparison pipeline.
#
# Required:
#   MODEL_DIR=/path/to/Qwen2-57B-A14B-Instruct
#   RUN_DIR=/path/to/output/run
#
# Example:
#   MODEL_DIR=/data2/ziheng/models/Qwen2-57B-A14B-Instruct \
#   RUN_DIR=/data2/ziheng/spice_runs/qwen2_57b_l4 \
#   GPU=0 HW_TAG=l4 GPU_MEM=22GiB CPU_MEM=256GiB \
#   bash experiments/harness/run_qwen2_memory_limited_compare.sh

ROOT=${SPICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"

PY=${PYTHON:-python}
MODEL=${MODEL_DIR:?set MODEL_DIR to a local Qwen2-MoE checkpoint directory}
RUN=${RUN_DIR:?set RUN_DIR to an output directory}
PROMPTS=${PROMPTS:-experiments/harness/sample_prompts.txt}

# If CUDA_VISIBLE_DEVICES is unset, expose one physical GPU. Python tools then
# address it as logical cuda:0 by default.
PHYSICAL_GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$PHYSICAL_GPU}
GPU_ARG=${GPU_ARG:-0}

HW_TAG=${HW_TAG:-gpu${PHYSICAL_GPU}}
RESOURCE_DIR=${RESOURCE_DIR:-$RUN/resource}
OFFLOAD_DIR=${OFFLOAD_DIR:-$RUN/offload}
DECODE_DIR=${DECODE_DIR:-$RUN/decode}
LORE_TRACE_DIR=${LORE_TRACE_DIR:-$RUN/lore_trace}
LORE_DIR=${LORE_DIR:-$RUN/lore_train}
FORECAST_DIR=${FORECAST_DIR:-$RUN/forecast}

N_PROMPTS=${N_PROMPTS:-8}
GEN=${GEN:-32}
PROMPT_LEN=${PROMPT_LEN:-128}
MAX_STEPS=${MAX_STEPS:-256}
MAX_HORIZON=${MAX_HORIZON:-6}
MAX_TEST_TOKENS=${MAX_TEST_TOKENS:-128}

GPU_MEM=${GPU_MEM:-70GiB}
CPU_MEM=${CPU_MEM:-600GiB}
CPU_THREADS=${CPU_THREADS:-16}
CPU_DTYPE=${CPU_DTYPE:-bf16}

# Qwen2-57B-A14B-Instruct defaults. Override for another MoE checkpoint.
D_MODEL=${D_MODEL:-3584}
D_INTER=${D_INTER:-2560}
TOP_K=${TOP_K:-8}
RESIDENCY=${RESIDENCY:-0.1}

RESOURCE_ITERS=${RESOURCE_ITERS:-50}
MISS_ITERS=${MISS_ITERS:-30}
MISS_WARMUP=${MISS_WARMUP:-10}
MISS_BANK=${MISS_BANK:-256}
TIMED_REPEATS=${TIMED_REPEATS:-3}
FILLER_COMPUTE_DIM=${FILLER_COMPUTE_DIM:-4096}
FILLER_REPEATS=${FILLER_REPEATS:-1}

LORE_RANK=${LORE_RANK:-64}
LORE_CONTEXT=${LORE_CONTEXT:-64}
LORE_STEPS=${LORE_STEPS:-400}
LORE_LR=${LORE_LR:-5e-4}
LORE_WARMUP=${LORE_WARMUP:-40}
SUBSTITUTE_RANKS=${SUBSTITUTE_RANKS:-7}
ADAPMOE_ACTIVE_M=${ADAPMOE_ACTIVE_M:-8,6,4}

RUN_RESOURCE=${RUN_RESOURCE:-1}
FORCE_RESOURCE=${FORCE_RESOURCE:-0}
FORCE_DECODE=${FORCE_DECODE:-0}
FORCE_TAIL=${FORCE_TAIL:-0}

RESOURCE_JSON=${RESOURCE_JSON:-$RESOURCE_DIR/resource_edges_${HW_TAG}_qwen2.json}
COST_JSON=${COST_JSON:-$RESOURCE_DIR/miss_assign_${HW_TAG}_qwen2_t${CPU_THREADS}.json}

mkdir -p "$RUN" "$RESOURCE_DIR" "$OFFLOAD_DIR" "$DECODE_DIR"

log() {
  printf '\n[stage] %s\n' "$*" | tee -a "$RUN/pipeline.log"
}

run_if_missing() {
  local marker=$1
  shift
  if [[ "$FORCE_TAIL" == "1" || ! -s "$marker" ]]; then
    "$@"
  else
    echo "[skip] $marker exists" | tee -a "$RUN/pipeline.log"
  fi
}

log "configuration"
cat <<EOF | tee "$RUN/config.env" | tee -a "$RUN/pipeline.log"
ROOT=$ROOT
PYTHON=$PY
MODEL_DIR=$MODEL
RUN_DIR=$RUN
PROMPTS=$PROMPTS
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
GPU_ARG=$GPU_ARG
HW_TAG=$HW_TAG
GPU_MEM=$GPU_MEM
CPU_MEM=$CPU_MEM
D_MODEL=$D_MODEL
D_INTER=$D_INTER
TOP_K=$TOP_K
RESIDENCY=$RESIDENCY
N_PROMPTS=$N_PROMPTS
GEN=$GEN
PROMPT_LEN=$PROMPT_LEN
MAX_TEST_TOKENS=$MAX_TEST_TOKENS
RESOURCE_JSON=$RESOURCE_JSON
COST_JSON=$COST_JSON
EOF

if [[ "$RUN_RESOURCE" == "1" ]]; then
  if [[ "$FORCE_RESOURCE" == "1" || ! -s "$RESOURCE_JSON" ]]; then
    log "measure hardware resource edges"
    "$PY" experiments/harness/microbench/spice_resource_microbench.py \
      --gpu "$GPU_ARG" \
      --d_model "$D_MODEL" \
      --d_inter "$D_INTER" \
      --iters "$RESOURCE_ITERS" \
      --out "$RESOURCE_JSON" \
      2>&1 | tee "$RUN/resource_edges.log"
  else
    echo "[skip] resource edges exist: $RESOURCE_JSON" | tee -a "$RUN/pipeline.log"
  fi

  if [[ "$FORCE_RESOURCE" == "1" || ! -s "$COST_JSON" ]]; then
    log "measure CPU/GPU miss-assignment costs"
    "$PY" experiments/harness/microbench/miss_assignment_microbench.py \
      --gpu "$GPU_ARG" \
      --d_model "$D_MODEL" \
      --d_inter "$D_INTER" \
      --top_k "$TOP_K" \
      --iters "$MISS_ITERS" \
      --warmup "$MISS_WARMUP" \
      --cpu_threads "$CPU_THREADS" \
      --cpu_dtype "$CPU_DTYPE" \
      --bank "$MISS_BANK" \
      --out "$COST_JSON" \
      2>&1 | tee "$RUN/miss_assignment.log"
  else
    echo "[skip] miss-assignment costs exist: $COST_JSON" | tee -a "$RUN/pipeline.log"
  fi
fi

if [[ "$FORCE_DECODE" == "1" ]]; then
  rm -f "$DECODE_DIR"/dec_*.pt "$DECODE_DIR/manifest.json"
fi

log "collect or resume Qwen2 decode traces with CPU offload"
"$PY" experiments/harness/datagen/gen_decode_traces_offload.py \
  --model_dir "$MODEL" \
  --text_file "$PROMPTS" \
  --out_dir "$DECODE_DIR" \
  --gpu "$GPU_ARG" \
  --n_prompts "$N_PROMPTS" \
  --gen "$GEN" \
  --prompt_len "$PROMPT_LEN" \
  --save_hidden_states \
  --gpu_mem "$GPU_MEM" \
  --cpu_mem "$CPU_MEM" \
  --offload_folder "$OFFLOAD_DIR" \
  --resume \
  2>&1 | tee "$RUN/decode.log"

log "convert decode traces for LoRE training"
run_if_missing "$LORE_TRACE_DIR/manifest.json" \
  "$PY" experiments/harness/datagen/decode_to_real_lore_traces.py \
    --decode_dir "$DECODE_DIR" \
    --out_dir "$LORE_TRACE_DIR" \
    --max_steps "$MAX_STEPS"

log "train real LoRE draft"
run_if_missing "$LORE_DIR/real_lore.pt" \
  "$PY" experiments/core/draft/train_real_lore.py \
    --model_id "$MODEL" \
    --trace_dir "$LORE_TRACE_DIR" \
    --out_dir "$LORE_DIR" \
    --gpu "$GPU_ARG" \
    --rank "$LORE_RANK" \
    --route_context "$LORE_CONTEXT" \
    --history gru \
    --steps "$LORE_STEPS" \
    --batch_traces 1 \
    --lr "$LORE_LR" \
    --warmup "$LORE_WARMUP" \
    --log_every 50 \
    --checkpoint_name real_lore.pt

log "build real-LoRE forecast"
run_if_missing "$FORECAST_DIR/manifest.json" \
  "$PY" experiments/harness/datagen/make_forecast_from_real_lore.py \
    --checkpoint "$LORE_DIR/real_lore.pt" \
    --decode_dir "$DECODE_DIR" \
    --out_dir "$FORECAST_DIR" \
    --max_horizon "$MAX_HORIZON" \
    --gpu "$GPU_ARG"

runtime_common=(
  --forecast_dir "$FORECAST_DIR"
  --cost_json "$COST_JSON"
  --gpu "$GPU_ARG"
  --train_frac 0.5
  --residency "$RESIDENCY"
  --max_test_tokens "$MAX_TEST_TOKENS"
  --d_model "$D_MODEL"
  --d_inter "$D_INTER"
  --top_k "$TOP_K"
  --cpu_threads "$CPU_THREADS"
  --cpu_dtype "$CPU_DTYPE"
  --shallow_depth 2
  --filler_compute_dim "$FILLER_COMPUTE_DIM"
  --filler_repeats "$FILLER_REPEATS"
  --timed_repeats "$TIMED_REPEATS"
)

log "SPICE exact CPU residual runtime"
run_if_missing "$RUN/spice_exact.json" \
  "$PY" experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
    "${runtime_common[@]}" \
    --out "$RUN/spice_exact.json" \
    --policies gos_cpu

if [[ -n "$SUBSTITUTE_RANKS" ]]; then
  log "SPICE low-confidence substitution runtime"
  run_if_missing "$RUN/spice_sub_rank${SUBSTITUTE_RANKS//,/}.json" \
    "$PY" experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
      "${runtime_common[@]}" \
      --out "$RUN/spice_sub_rank${SUBSTITUTE_RANKS//,/}.json" \
      --policies gos_cpu \
      --substitute_ranks "$SUBSTITUTE_RANKS"
fi

log "AdapMoE-style active-expert replay baseline"
run_if_missing "$RUN/adapmoe.json" \
  "$PY" experiments/harness/scheduler/adapmoe_trace_replay.py \
    --forecast_dir "$FORECAST_DIR" \
    --cost_json "$COST_JSON" \
    --resource_json "$RESOURCE_JSON" \
    --out "$RUN/adapmoe.json" \
    --residency "$RESIDENCY" \
    --max_test_tokens "$MAX_TEST_TOKENS" \
    --active_m "$ADAPMOE_ACTIVE_M"

log "HybriMoE-style hybrid CPU-GPU replay baseline"
run_if_missing "$RUN/hybrimoe.json" \
  "$PY" experiments/harness/scheduler/hybrimoe_trace_replay.py \
    --forecast_dir "$FORECAST_DIR" \
    --cost_json "$COST_JSON" \
    --resource_json "$RESOURCE_JSON" \
    --out "$RUN/hybrimoe.json" \
    --residency "$RESIDENCY" \
    --max_test_tokens "$MAX_TEST_TOKENS" \
    --prefetch_size "${HYBRIMOE_PREFETCH_SIZE:-4}"

log "summarize comparison"
"$PY" experiments/harness/summarize_qwen2_compare.py \
  --run_dir "$RUN" \
  --out "$RUN/summary.md" \
  2>&1 | tee "$RUN/summary.log"

echo DONE | tee "$RUN/full_pipeline.done"
