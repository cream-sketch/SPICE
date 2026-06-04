#!/usr/bin/env bash
# Robust confirmation of the resident-admission verdict: larger trace + more
# repeats to shrink timing noise below the 2% decision threshold.
# Usage: run_admit_decisive_v2.sh <best_admit_cost_ms_from_v1>
# 稳健确认:1024 token + 7 repeat 压低计时噪声,使 2% 判决可信。
set -u
cd /data/ziheng/spice
PY=/home/ziheng/miniforge3/envs/sparmoe/bin/python
FC=notes/evidence/qwen_forecast_wiki64_v1
COST=notes/evidence/miss_assign_qwen_bf16_v2.json
OUT=notes/evidence/admit_decisive_v2
mkdir -p "$OUT"
GPU=1
TOK=1024
REP=7
ADMIT_COST="${1:-0.8}"

common() {
  $PY experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
    --forecast_dir "$FC" --cost_json "$COST" --gpu "$GPU" \
    --train_frac 0.5 --residency 0.1 --max_test_tokens "$TOK" \
    --d_model 2048 --d_inter 1408 --top_k 4 --cpu_threads 16 --cpu_dtype bf16 \
    --shallow_depth 2 --low_slots 128 --high_slots 8 --bank 256 \
    --max_lead_layers 5 --min_prefetch_lead 1 --timed_repeats "$REP" \
    --filler_compute_dim 4096 --filler_repeats 1 --gos_scheduler greedy --seed 0 "$@"
}

echo "===== never (transient-only) ====="
common --policies gos_cpu --prefetch_hit_admission never --out "$OUT/never.json"
echo "===== always ====="
common --policies gos_cpu --prefetch_hit_admission always --out "$OUT/always.json"
echo "===== recent_reuse ====="
common --policies gos_cpu --prefetch_hit_admission recent_reuse --out "$OUT/recent.json"
echo "===== oracle_value admit_cost=$ADMIT_COST ====="
common --policies gos_cpu --prefetch_hit_admission oracle_value --allow_oracle_admission \
  --resident_value_margin_ms 0.0 --resident_admit_cost_ms "$ADMIT_COST" --out "$OUT/oracle_best.json"
echo "===== ALL DONE ====="
