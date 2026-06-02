#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/workspace/spice/runs/draft_parallel_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-/home/ial-chency/workspace/envs/fastwam/bin/python}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-3}"
STEPS="${STEPS:-2000}"

mkdir -p "$ROOT"
cd "$(dirname "$0")"
unset CUDA_VISIBLE_DEVICES

run_one() {
  local gpu="$1"
  local seed="$2"
  local name="$3"
  local out="$ROOT/$name"
  mkdir -p "$out"/{train,eval}
  echo "START $name gpu=$gpu seed=$seed steps=$STEPS"
  "$PY" train_draft_model.py \
    --gpu "$gpu" \
    --seed "$seed" \
    --out_dir "$out/train" \
    --layers 8 \
    --experts 16 \
    --top_k 2 \
    --hidden 256 \
    --expert_hidden 512 \
    --rank 16 \
    --route_context 64 \
    --history gru \
    --steps "$STEPS" \
    --batch 8 \
    --eval_batches 32 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup 200 \
    --align_lambda 0.1 \
    --log_every 250
  "$PY" eval_draft_prefetch.py \
    --gpu "$gpu" \
    --seed "$seed" \
    --out_dir "$out/eval" \
    --checkpoint "$out/train/spice_draft.pt" \
    --steps 256 \
    --batch 8 \
    --eval_batches 32 \
    --cache_capacity 64 \
    --expert_mb 64 \
    --pcie_gbps 48 \
    --compute_ms 2.5 \
    --l_max 6 \
    --confidence_threshold 0.7 \
    --online_steps 100 \
    --online_lr 5e-5 \
    --align_lambda 0.1
  echo "DONE $name"
}

echo "ROOT=$ROOT"
echo "PY=$PY"
echo "GPU_A=$GPU_A"
echo "GPU_B=$GPU_B"
echo "STEPS=$STEPS"
date

run_one "$GPU_A" 7 "gpu${GPU_A}_seed7" > "$ROOT/gpu${GPU_A}_seed7.log" 2>&1 &
pid_a=$!
run_one "$GPU_B" 13 "gpu${GPU_B}_seed13" > "$ROOT/gpu${GPU_B}_seed13.log" 2>&1 &
pid_b=$!

echo "$pid_a" > "$ROOT/pid_gpu${GPU_A}.txt"
echo "$pid_b" > "$ROOT/pid_gpu${GPU_B}.txt"
wait "$pid_a"
wait "$pid_b"

date
echo "PARALLEL_DONE $ROOT"
