#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/workspace/spice/runs/draft_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-3}"
PY="${PY:-/home/ial-chency/workspace/envs/fastwam/bin/python}"

mkdir -p "$ROOT"/{train,eval}
cd "$(dirname "$0")"
unset CUDA_VISIBLE_DEVICES

echo "ROOT=$ROOT"
echo "GPU=$GPU"
echo "PY=$PY"
date

"$PY" draft/train_draft_model.py \
  --gpu "$GPU" \
  --out_dir "$ROOT/train" \
  --layers 8 \
  --experts 16 \
  --top_k 2 \
  --hidden 256 \
  --expert_hidden 512 \
  --rank 16 \
  --route_context 64 \
  --history gru \
  --steps 800 \
  --batch 8 \
  --eval_batches 16 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --warmup 100 \
  --align_lambda 0.1 \
  --log_every 100

"$PY" draft/eval_draft_prefetch.py \
  --gpu "$GPU" \
  --out_dir "$ROOT/eval" \
  --checkpoint "$ROOT/train/spice_draft.pt" \
  --steps 128 \
  --batch 8 \
  --eval_batches 16 \
  --cache_capacity 64 \
  --expert_mb 64 \
  --pcie_gbps 48 \
  --compute_ms 2.5 \
  --l_max 6 \
  --confidence_threshold 0.7 \
  --online_steps 100 \
  --online_lr 5e-5 \
  --align_lambda 0.1

date
echo "DONE $ROOT"
