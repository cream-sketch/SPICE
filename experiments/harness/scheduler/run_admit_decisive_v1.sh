#!/usr/bin/env bash
# Decisive resident-admission sweep (same input, held-out test) after the
# oracle_value value-model fix. Verdict rule (user spec): oracle_value must beat
# best(always, never) by >=2% TPOT or resident admission mainline is dropped
# (keep transient-staging only). never-admit == transient-only.
# 判决性常驻准入 sweep:同输入、held-out;oracle_value 须比 best(always,never) 快 >=2% 否则砍掉常驻准入主线。
set -u
cd /data/ziheng/spice
PY=/home/ziheng/miniforge3/envs/sparmoe/bin/python
FC=notes/evidence/qwen_forecast_wiki64_v1
COST=notes/evidence/miss_assign_qwen_bf16_v2.json
OUT=notes/evidence/admit_decisive_v1
mkdir -p "$OUT"
GPU=1
TOK=256

common() {
  $PY experiments/harness/scheduler/spice_shallow_issuer_runtime.py \
    --forecast_dir "$FC" --cost_json "$COST" --gpu "$GPU" \
    --train_frac 0.5 --residency 0.1 --max_test_tokens "$TOK" \
    --d_model 2048 --d_inter 1408 --top_k 4 --cpu_threads 16 --cpu_dtype bf16 \
    --shallow_depth 2 --low_slots 128 --high_slots 8 --bank 256 \
    --max_lead_layers 5 --min_prefetch_lead 1 --timed_repeats 3 \
    --filler_compute_dim 4096 --filler_repeats 1 --gos_scheduler greedy --seed 0 "$@"
}

echo "===== depth0 CPU baseline ====="
common --policies shallow_cpu --out "$OUT/depth0.json"

echo "===== GOS transient-only (never) ====="
common --policies gos_cpu --prefetch_hit_admission never --out "$OUT/never.json"

echo "===== GOS always-admit ====="
common --policies gos_cpu --prefetch_hit_admission always --out "$OUT/always.json"

echo "===== GOS recent_reuse-admit ====="
common --policies gos_cpu --prefetch_hit_admission recent_reuse --out "$OUT/recent.json"

for C in 0.0 0.8 1.6; do
  echo "===== GOS oracle_value admit_cost=$C ====="
  common --policies gos_cpu --prefetch_hit_admission oracle_value \
    --allow_oracle_admission --resident_value_margin_ms 0.0 \
    --resident_admit_cost_ms "$C" --out "$OUT/oracle_c${C}.json"
done

echo "===== ALL DONE ====="
