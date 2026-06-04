#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/workspace/spice_iccd_runs/iccd_system_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"/{gpu0_energy,gpu1_energy,gpu2_cache,gpu3_timeline}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES=0 python analysis/energy_per_token.py --gpu 0 --power_gpu 0 --out_dir "$ROOT/gpu0_energy" --policies naive,lru --steps 256 --replay_scale 1.0 --expert_mb 8 --mat_dim 4096 --compute_iters 8 --power_interval 0.1 > "$ROOT/gpu0_energy/run.log" 2>&1 &
P0=$!

CUDA_VISIBLE_DEVICES=1 python analysis/energy_per_token.py --gpu 0 --power_gpu 1 --out_dir "$ROOT/gpu1_energy" --policies moe_offloading,pregated,spice --steps 256 --replay_scale 1.0 --expert_mb 8 --mat_dim 4096 --compute_iters 8 --power_interval 0.1 > "$ROOT/gpu1_energy/run.log" 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=2 python sim/prefetch_system_sim.py --gpu 0 --out_dir "$ROOT/gpu2_cache" --mode cache_sweep --steps 512 --expert_mb 8 --cache_values 128,256,384,512,768,1024,1536,2048 > "$ROOT/gpu2_cache/run.log" 2>&1 &
P2=$!

(
  if command -v nsys >/dev/null 2>&1; then
    for policy in naive pregated spice; do
      mkdir -p "$ROOT/gpu3_timeline/$policy"
      CUDA_VISIBLE_DEVICES=3 nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --output "$ROOT/gpu3_timeline/$policy/timeline_$policy" \
        python analysis/timeline_replay.py --gpu 0 --out_dir "$ROOT/gpu3_timeline/$policy" --policy "$policy" --steps 96 --expert_mb 8
    done
  else
    for policy in naive pregated spice; do
      mkdir -p "$ROOT/gpu3_timeline/$policy"
      CUDA_VISIBLE_DEVICES=3 python analysis/timeline_replay.py --gpu 0 --out_dir "$ROOT/gpu3_timeline/$policy" --policy "$policy" --steps 96 --expert_mb 8
    done
  fi
) > "$ROOT/gpu3_timeline/run.log" 2>&1 &
P3=$!

echo "$P0 $P1 $P2 $P3" > "$ROOT/pids.txt"
wait "$P0"
wait "$P1"
wait "$P2"
wait "$P3"

python analysis/summarize_results.py --root "$ROOT"
python analysis/summarize_iccd_system_results.py --root "$ROOT"
echo "$ROOT"
