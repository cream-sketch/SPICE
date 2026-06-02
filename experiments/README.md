# SPICE ICCD Supplemental Experiments

This code supplements reviewer-requested evidence for the SPICE resubmission.
It intentionally transfers no datasets from the local machine to the GPU
workstation. Scripts either use local Hugging Face cache on the workstation or
generate synthetic routing traces.

## Scripts

- `lossless_correctness.py`: executable verified-MoE harness proving that
  fallback affects latency, not logits.
- `prefetch_system_sim.py`: same-harness system simulator for Naive, LRU,
  MoE-Offloading, Pre-gated, and SPICE; includes PCIe copy microbench.
- `draft_model.py`: implementation of the SPICE draft path: frozen
  target attention/router, LoRE expert surrogates, routing-history context,
  route-KL loss, hidden-state alignment loss, and routing metrics.
- `train_draft_model.py`: trains the LoRE draft predictor on target-generated
  routing distributions and hidden states.
- `eval_draft_prefetch.py`: evaluates draft-model-driven adaptive prefetching
  with anchor re-initialized lookahead, verified fallback, and optional online
  self-correction.
- `collect_hf_moe_traces.py`: captures hidden states and router logits/probs
  from a local HuggingFace MoE checkpoint for model-specific draft wrappers.
- `eval_hf_trace_prefetch.py`: evaluates verified prefetch hit/fallback rates
  directly on saved HuggingFace MoE router traces.
- `download_hf_snapshot.py`: reproducible HuggingFace snapshot downloader for
  remote experiment machines.
- `real_ppl_smoke.py`: target-model perplexity smoke test from local HF cache.
- `summarize_results.py`: writes `SUMMARY.md` and CSV tables.
- `summarize_iccd_system_results.py`: writes the ICCD-specific energy,
  cache-budget, and timeline summary plus paste-ready LaTeX.
- `run_suite.sh`: single-machine run for the four GPU workstation.
- `run_baseline_stress.sh`: four-GPU Top-K stress matrix comparing Naive,
  LRU, MoE-Offloading, Pre-gated, and SPICE under the same tight cache budget.
- `run_iccd_system_suite.sh`: four-GPU ICCD system suite: energy replay,
  cache-budget sweep, and Nsight timeline replay.
- `run_draft_suite.sh`: single-GPU end-to-end draft-model reproduction:
  train draft model, evaluate routing prediction, adaptive prefetching, and
  online self-correction.

## Remote Usage

```bash
source ~/workspace/venv-dsv2/bin/activate
bash run_suite.sh ~/workspace/spice_iccd_runs/manual_run
bash run_baseline_stress.sh ~/workspace/spice_iccd_runs/manual_baseline_stress
bash run_iccd_system_suite.sh ~/workspace/spice_iccd_runs/manual_iccd_system
GPU=3 bash run_draft_suite.sh ~/workspace/spice_iccd_runs/manual_draft
python collect_hf_moe_traces.py --model Qwen/Qwen1.5-MoE-A2.7B --out_dir ~/workspace/spice_iccd_runs/qwen_traces --gpu 3 --device_map auto --allow_download
python eval_hf_trace_prefetch.py --trace_dir ~/workspace/spice_iccd_runs/qwen_traces --out_dir ~/workspace/spice_iccd_runs/qwen_prefetch --top_k 4 --predictor anchor_repeat
```
