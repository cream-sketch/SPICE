#!/usr/bin/env bash
# Noise-controlled admission verdict run.
#
# This runner is deliberately stricter than the ad-hoc decisive_v4 command:
# - refuses to start if another spice_shallow_issuer_runtime.py is already alive
# - records GPU/process snapshots before every policy
# - binds CPU threads and host allocations to the NUMA node local to the selected GPU
#
# Default target is GPU0/NUMA0 on the A800 server:
#   GPU0/GPU1 -> NUMA0 CPUs 0-25,52-77
#   GPU2/GPU3 -> NUMA1 CPUs 26-51,78-103
#
# Usage:
#   experiments/harness/scheduler/run_admit_isolated_v5.sh [gpu] [numa_node] [out_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/ziheng/miniforge3/envs/sparmoe/bin/python}"
FC="${FC:-notes/evidence/qwen_forecast_wiki64_v1}"
COST="${COST:-notes/evidence/miss_assign_qwen_bf16_v2.json}"
GPU="${1:-0}"
NUMA="${2:-0}"
OUT="${3:-notes/evidence/admit_decisive_v5_isolated}"
TOK="${TOK:-1024}"
REP="${REP:-7}"
CPU_THREADS="${CPU_THREADS:-16}"
if [[ "$NUMA" == "0" ]]; then
  CPUSET="${CPUSET:-0-25,52-77}"
else
  CPUSET="${CPUSET:-26-51,78-103}"
fi

mkdir -p "$OUT"
LOG="$OUT/run.log"
: > "$LOG"

snapshot() {
  local label="$1"
  {
    echo "===== snapshot: $label ====="
    date
    pgrep -af 'spice_shallow_issuer_runtime.py' || true
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    echo
  } | tee -a "$LOG"
}

other_runtime_count() {
  pgrep -af 'spice_shallow_issuer_runtime.py' | grep -v "pgrep -af" | wc -l
}

if [[ "$(other_runtime_count)" -ne 0 ]]; then
  snapshot "refuse-start-existing-runtime"
  echo "Refusing to start: another spice_shallow_issuer_runtime.py is running." | tee -a "$LOG"
  exit 2
fi

if command -v numactl >/dev/null 2>&1; then
  BIND=(numactl --cpunodebind="$NUMA" --membind="$NUMA")
  echo "Using numactl NUMA node $NUMA" | tee -a "$LOG"
elif command -v taskset >/dev/null 2>&1; then
  BIND=(taskset -c "$CPUSET")
  echo "numactl not found; using taskset CPU set $CPUSET (first-touch memory locality only)" | tee -a "$LOG"
else
  BIND=()
  echo "numactl/taskset not found; running without explicit CPU/NUMA binding" | tee -a "$LOG"
fi

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export TORCH_NUM_THREADS="$CPU_THREADS"

common() {
  local label="$1"
  shift
  snapshot "before-$label"
  "${BIND[@]}" "$PY" experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
    --forecast_dir "$FC" --cost_json "$COST" --gpu "$GPU" \
    --train_frac 0.5 --residency 0.1 --max_test_tokens "$TOK" \
    --d_model 2048 --d_inter 1408 --top_k 4 --cpu_threads "$CPU_THREADS" --cpu_dtype bf16 \
    --shallow_depth 2 --low_slots 128 --high_slots 8 --bank 256 \
    --max_lead_layers 5 --min_prefetch_lead 1 --timed_repeats "$REP" \
    --filler_compute_dim 4096 --filler_repeats 1 --seed 11 \
    --gos_layer_slack_ms 0.8 --gos_cpu_overlap_ms 0.3 "$@" | tee -a "$LOG"
  snapshot "after-$label"
}

snapshot "start"
common "greedy_transient" --policies gos_cpu --gos_scheduler greedy \
  --prefetch_hit_admission never --out "$OUT/greedy_transient.json"
common "dp_transient" --policies gos_cpu --gos_scheduler dp \
  --prefetch_hit_admission never --out "$OUT/dp_transient.json"
common "always" --policies gos_cpu --gos_scheduler greedy \
  --prefetch_hit_admission always --out "$OUT/always.json"
common "oracle_cost0" --policies gos_cpu --gos_scheduler greedy \
  --prefetch_hit_admission oracle_value --allow_oracle_admission \
  --resident_value_margin_ms 0.0 --resident_admit_cost_ms 0.0 --out "$OUT/oracle_cost0.json"
snapshot "done"
