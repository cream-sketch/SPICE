# SPICE

SPICE is a verified speculative expert-prefetching prototype for offloaded
Mixture-of-Experts inference. The code in this repository reproduces two paths:

1. Proxy system simulation for Naive, LRU, MoE-Offloading, Pre-gated MoE, and
   SPICE-style speculative prefetching.
2. Draft-model-driven SPICE with frozen target attention/router modules, LoRE
   low-rank expert surrogates, routing-history context, route-KL training,
   hidden-state alignment, anchor re-initialized lookahead, adaptive
   prefetching, verified fallback, and optional online self-correction.

The default draft-model implementation uses a synthetic target MoE so that the
full method can be validated without distributing model weights or datasets.
For real checkpoints, `collect_hf_moe_traces.py` records HuggingFace hidden
states and router logits/probabilities from local model paths or cache. A
model-specific wrapper can then reuse the same draft-training and verified
prefetch interfaces with those traces.

## Repository Layout

- `experiments/`: runnable code for draft training, prefetch simulation,
  Qwen/HuggingFace trace collection, and hardware replay.
- `validation/`: compact JSON/CSV/Markdown experiment summaries. Large traces,
  checkpoints, model weights, logs, and profiler reports are intentionally
  excluded from git.
- `experiments/draft_model.py`: target MoE, SPICE LoRE draft model, losses, and
  routing metrics.
- `experiments/train_draft_model.py`: offline draft-model training.
- `experiments/eval_draft_prefetch.py`: draft-driven adaptive prefetching,
  anchor re-initialized lookahead, verified fallback, and online
  self-correction.
- `experiments/collect_hf_moe_traces.py`: optional real MoE trace collector for
  local HuggingFace checkpoints.
- `experiments/eval_hf_trace_prefetch.py`: verified prefetch simulation on
  saved real-router traces.
- `experiments/download_hf_snapshot.py`: utility for downloading HF snapshots
  into a reusable local model directory.
- `experiments/prefetch_system_sim.py`: controlled prefetch system simulator.
- `experiments/lossless_correctness.py`: verified-execution correctness check.
- `experiments/timeline_replay.py`: CUDA H2D overlap replay.
- `experiments/energy_per_token.py`: GPU energy replay.
- `experiments/run_draft_suite.sh`: single-GPU draft training and evaluation.
- `experiments/run_draft_parallel.sh`: two-GPU parallel draft training.

## Quick Start

```bash
cd experiments
python -m py_compile *.py
python train_draft_model.py --gpu 0 --out_dir runs/draft_train --steps 800
python eval_draft_prefetch.py \
  --gpu 0 \
  --out_dir runs/draft_eval \
  --checkpoint runs/draft_train/spice_draft.pt \
  --online_steps 100
python collect_hf_moe_traces.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --out_dir runs/hf_traces \
  --text_file prompts.txt \
  --gpu 0 \
  --device_map auto \
  --allow_download
python eval_hf_trace_prefetch.py \
  --trace_dir runs/hf_traces \
  --out_dir runs/hf_trace_prefetch \
  --top_k 4 \
  --predictor anchor_repeat
```

For GPU workstation runs:

```bash
GPU=3 bash run_draft_suite.sh ~/workspace/spice/runs/manual_draft
GPU_A=0 GPU_B=3 STEPS=2000 bash run_draft_parallel.sh ~/workspace/spice/runs/manual_parallel
bash run_suite.sh ~/workspace/spice/runs/manual_system
```

## Correctness Invariant

SPICE uses draft predictions only to schedule expert weight movement. The target
router remains authoritative. If a target-selected expert is not resident, the
runtime synchronously fetches it before expert execution. Therefore, prefetch
misses affect latency and traffic, not target-model logits.

## Notes

Generated checkpoints, run logs, Nsight reports, and large result artifacts are
ignored by git. Keep reproducible source code and compact summaries in the
repository; store heavyweight artifacts outside the repo.
