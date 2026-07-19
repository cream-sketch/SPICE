#!/usr/bin/env bash
# Formal suite: compute on GPU 3 (ours), hot expert tier on the physical GPU
# given as $1 (default 1). Container maps --gpus device=3,$STORE to cuda:0
# (compute) and cuda:1 (store).
set -u
STORE=${1:-1}

PY=/home/ial-lvyx/miniforge3/envs/vla/bin/python
M=/home/ial-lvyx/workspace/models/Qwen3-30B-A3B
E=/home/ial-lvyx/workspace/spice_islped/experiments
R=/home/ial-lvyx/workspace/spice_islped/runs
O=$R/formal_$(date +%Y%m%d_%H%M)
mkdir -p "$O"

COMMON="--model $M --out_dir $O --gpu 0 --store_gpu 1 --store_gpu_gib 17.5 \
  --ram_tier_gib 14 --cold_file $R/qwen3_cold_experts.bin \
  --transition_stats $R/qwen3_trace_full/transition_stats.pt \
  --prompt_file $R/qwen3_prompts.txt --max_prompts 4 --new_tokens 48 \
  --measure_power --power_gpu 0"

run() {
  echo "=== $(date +%H:%M:%S) run: $*" >> "$O/suite.log"
  docker run --rm --name spice_formal --gpus "\"device=3,$STORE\"" --memory 24g \
    --user "$(id -u):$(id -g)" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HOME=/home/ial-lvyx -e USER=ial-lvyx -e LOGNAME=ial-lvyx \
    -v /home/ial-lvyx:/home/ial-lvyx -w "$E" --entrypoint /bin/bash \
    nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04 \
    -c "$PY -u qwen3_offload_bench.py $COMMON $*" >> "$O/suite.log" 2>&1
  echo "=== exit=$?" >> "$O/suite.log"
}

run --policy naive --cache_experts 0
for C in 512 1024 1536; do
  run --policy lru --cache_experts "$C"
  run --policy collab --cache_experts "$C" --cpu_threads 24
  run --policy adapmoe --cache_experts "$C" --prefetch_k 8
  run --policy spice --cache_experts "$C" \
      --draft_checkpoint $R/draft_qwen3_2000/qwen3_spice_draft.pt \
      --prefetch_k 8 --l_max 3 --confidence_threshold 0.0 \
      --draft_history_window 16 --draft_anchor_stride 3
done

echo done > "$O/FORMAL_DONE"
