# CLAUDE.md — SPICE (offloaded-MoE verified expert prefetch + resource scheduling)

This repo studies offloaded Mixture-of-Experts decoding (experts in CPU DRAM, streamed over
PCIe to GPU). SPICE = a frozen-attention/router draft that FORECASTS future expert routing,
feeding a verified resource scheduler that decides per miss: CPU-serve vs PCIe transient-stage
vs drop. Language/format: Chinese for discussion; English for code/logs; bilingual code comments;
no emojis in code/files.

## Entry points

### Training-free SPICE draft (route forecaster) — UNIFIED, adapter-based
Run from `experiments/harness/datagen/`:
```bash
python -m spice_draft.cli --model_type {qwen|deepseek} --model_dir <DIR> \
    --text_file <wikitext.txt> --out <metrics.json> \
    [--dump_forecast <DIR>] [--oracle_fcast] [--top_k K] [--max_horizon H]
```
- Package `spice_draft/`: `cli.py` / `rollout.py` (model-agnostic anchored shared-expert-only
  rollout) / `forecast_io.py` (dump schema). Per-model differences live in
  `adapters/{base,qwen2_moe,deepseek_v2}.py` (gate logits, true_forward via hooks vs
  output_router_logits, RoPE/position_embeddings, DeepSeek MoE-layer-only dump for the dense
  layer 0). `--oracle_fcast` fills fcast with true future routes (perfect-prediction upper bound).
- Forecast dump (consumed by the GOS runtime): per-text `fc_*.pt` with `true_top[L,S,K]` +
  `fcast[L,H,S,K]` + dir-level `manifest.json`. **d1 recall ~= 1.0 is the adapter-correctness canary.**
- This replaced the old `qwen_spice_draft.py` + `ds_spice_draft.py` (deleted) and the drifted
  `train_real_lore.py` (deleted; it dropped frozen attention). The original paper-faithful
  TRAINED-LoRE-on-SYNTHETIC reference is kept at `experiments/core/draft_model.py` (SPICEDraftModel).

### GOS runtime (the validated scheduler) — LEAN
`experiments/harness/scheduler/spice_shallow_issuer_runtime.py`. Validated policies:
`gos_cpu` (greedy transient overflow staging — the positive result), `shallow_cpu`/`deep_cpu`
(CPU baselines), `gos_dummy_cpu` (perturbation control), `--substitute_ranks` (verified lossy
negative-admission / drop). Consumes a forecast dump dir via `--forecast_dir`.

## Directory map

- `experiments/core/` — original paper-faithful reference reproduction (draft_model.py synthetic
  SPICEDraftModel, draft/train_draft_model.py, sim/, analysis/, data/). Treat as read-only baseline.
- `experiments/harness/` — real-model evidence + runtime:
  - `datagen/` — spice_draft (above), decode-trace + forecast-from-decode generators.
  - `scheduler/` — lean GOS runtime + `miss_assignment_replay.py` (cost table / popularity / LS).
  - `microbench/`, `evidence/`.
  - `ablations/scheduler/` — **QUARANTINED proven-dead levers** (NOT in the lean runtime):
    `resident_admission_runtime.py` (full runtime with resident-cache admission policies
    always/recent/hotter/oracle_value + the DP scheduler), `run_resident_admission_verdict.sh`,
    `test_gos_scheduler.py`. Use these ONLY to reproduce the NEGATIVE results.
- `notes/evidence/*.json` — committed experiment outputs (reproduction baselines). `notes/paper_plan_C.md`
  is the current paper framing.

## Settled findings (do not re-explore from scratch)

- GOS transient-staging is the validated positive: Qwen +15%, DeepSeek +13% TPOT vs CPU baselines
  on REAL drafts; real DeepSeek draft captures ~90% of the perfect-prediction (oracle) gain.
- Resident-cache admission and the global DP scheduler are PROVEN DEAD at batch=1 (clean isolated
  runs; even an optimistic oracle beats never-admit by <0.5% << 2%). Their code is in `ablations/`.
- At batch=1 exact, pure scheduling has little slack; value is the CPU-serve-vs-PCIe split +
  verified-gate drop (resource-DAG framing), not residency management.

## Run environment

- GPU server `moe-server-248` (ssh alias; 4x A800): canonical code `/data/ziheng/spice` (git,
  branch `exp/nonincremental-miss-recovery`), data `/data/ziheng/spice_runs`, conda env `sparmoe`.
- Clean idle timing node `ziheng@10.16.52.172:10548` (2x A800, env `/data/ziheng/conda_envs/mxmoe_clean`):
  use for noise-free TPOT (CPU-contention on a shared node ruins batch=1 timing; isolate single-process).
- Git: edit locally -> push origin -> server `git pull`/`reset --hard` (single-source; avoid both
  machines committing). Major results get tags; non-trivial code is cross-checked with codex.
