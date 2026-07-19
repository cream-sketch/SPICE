#!/usr/bin/env bash
# Multiseed draft training + trace-level prefetch predictor comparison (GPU 3).
source /home/ial-lvyx/miniforge3/etc/profile.d/conda.sh
conda activate vla
set -u
cd /home/ial-lvyx/workspace/spice_islped/experiments
R=/home/ial-lvyx/workspace/spice_islped/runs

for s in 13 23 31; do
  CUDA_VISIBLE_DEVICES=3 python -u qwen3_train_draft.py \
    --model /home/ial-lvyx/workspace/models/Qwen3-30B-A3B \
    --trace_dir "$R/qwen3_trace_full" \
    --out_dir "$R/draft_qwen3_2000_seed$s" \
    --gpu 0 --steps 2000 --eval_every 500 --depth 6 --seed "$s" \
    > "$R/draft_seed$s.log" 2>&1
done

for p in oracle anchor_repeat layer_prior; do
  python -u eval_hf_trace_prefetch.py \
    --trace_dir "$R/qwen3_trace_full" \
    --out_dir "$R/qwen3_trace_prefetch_$p" \
    --predictor "$p" --top_k 8 --cache_capacity 1024 \
    > "$R/trace_prefetch_$p.log" 2>&1
done

echo done > "$R/SIDE_JOBS_DONE2"
