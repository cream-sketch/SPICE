#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/workspace/spice_iccd_runs/baseline_stress_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"/{gpu0_topk_2_4,gpu1_topk_6_8,gpu2_topk_10,gpu3_topk_12}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES=0 python prefetch_system_sim.py --gpu 0 --out_dir "$ROOT/gpu0_topk_2_4" --mode topk_baselines --steps 768 --expert_mb 8 --topk_values 2,4 > "$ROOT/gpu0_topk_2_4/run.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python prefetch_system_sim.py --gpu 0 --out_dir "$ROOT/gpu1_topk_6_8" --mode topk_baselines --steps 768 --expert_mb 8 --topk_values 6,8 > "$ROOT/gpu1_topk_6_8/run.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 python prefetch_system_sim.py --gpu 0 --out_dir "$ROOT/gpu2_topk_10" --mode topk_baselines --steps 768 --expert_mb 8 --topk_values 10 > "$ROOT/gpu2_topk_10/run.log" 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=3 python prefetch_system_sim.py --gpu 0 --out_dir "$ROOT/gpu3_topk_12" --mode topk_baselines --steps 768 --expert_mb 8 --topk_values 12 > "$ROOT/gpu3_topk_12/run.log" 2>&1 &
P3=$!

echo "$P0 $P1 $P2 $P3" > "$ROOT/pids.txt"
wait "$P0"
wait "$P1"
wait "$P2"
wait "$P3"

python summarize_results.py --root "$ROOT"
echo "$ROOT"
