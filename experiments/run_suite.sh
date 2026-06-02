#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/workspace/spice_iccd_runs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"/{gpu0_correctness,gpu1_main,gpu2_overhead,gpu3_stress}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

(
  CUDA_VISIBLE_DEVICES=0 python lossless_correctness.py --gpu 0 --out_dir "$ROOT/gpu0_correctness" --steps 256
  CUDA_VISIBLE_DEVICES=0 python real_ppl_smoke.py --gpu 0 --out_dir "$ROOT/gpu0_correctness" --model gpt2 --local_only --max_samples 64 --seq_len 128 || true
) > "$ROOT/gpu0_correctness/run.log" 2>&1 &
P0=$!

CUDA_VISIBLE_DEVICES=1 python prefetch_system_sim.py --gpu 0 --out_dir "$ROOT/gpu1_main" --mode main --steps 512 --expert_mb 8 > "$ROOT/gpu1_main/run.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 python prefetch_system_sim.py --gpu 0 --out_dir "$ROOT/gpu2_overhead" --mode overhead --steps 512 --expert_mb 8 > "$ROOT/gpu2_overhead/run.log" 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=3 python prefetch_system_sim.py --gpu 0 --out_dir "$ROOT/gpu3_stress" --mode topk --steps 512 --expert_mb 8 > "$ROOT/gpu3_stress/run.log" 2>&1 &
P3=$!

echo "$P0 $P1 $P2 $P3" > "$ROOT/pids.txt"
wait "$P0"
wait "$P1"
wait "$P2"
wait "$P3"

python summarize_results.py --root "$ROOT"
echo "$ROOT"
