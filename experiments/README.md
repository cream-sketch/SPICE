# SPICE Experiments

This tree holds two layers of runnable code:

1. **`core/`** — the original SPICE reproduction (verified speculative expert *prefetch*).
   Reproduces the paper result (up to 2.86x TPOT on PCIe-constrained / weaker GPUs).
2. **`harness/`** — the **resource-DAG / negative-admission** system developed on top of SPICE
   for the **batch=1, fast-GPU (A800), exact same-precision** regime. Real CUDA wall-clock timing.

`miss/` holds one decision-gate kill-test. `../validation/` holds the paper-reproduction artifacts.

---

## Architecture & data flow (upper → lower layers)

```
  datagen/        microbench/                 scheduler/                evidence/
  (routing &      (real A800 cost tables)     (CONSUMES the above)      (quality / char.)
   forecast data)
  qwen_spice_draft  spice_resource_microbench   spice_shallow_            drop_quality_ppl
  gen_decode_traces cpu_expert_bench             issuer_runtime          deepseek_drop_quality
  gen_decode_traces_ds miss_assignment_         (MAIN runtime)           qwen/ds_gate_vs_rank
  make_forecast_from_dec  microbench --cost_json> spice_event_           obs_routing_structure
                    pcie_topology_microbench      scheduler_replay        obs_expert_contribution
                    shallow_h2d_issuer_           spice_forecast_         obs_cross_expert_structure
                     microbench (queue probe)      pressure_replay        token_conditional_analysis
                                                  prefetch_pressure_      ds_collect_routing
                                                   scheduler_replay
                                                  miss_assignment_replay (lib)
```

- **datagen/** produces routing traces (`dec_*.pt`) and SPICE forecast dumps (`true_top`, `fcast`).
- **microbench/** measures real A800 edges (H2D fetch, CPU expert serve, PCIe topology, the
  copy-engine queue probe) and emits the CPU‖PCIe miss-assignment **cost table** (`--cost_json`).
- **scheduler/** is the only import-coupled group: every file does
  `sys.path.insert(0, parent)` then `from miss_assignment_replay import ...`, so these 5 files
  MUST stay co-located. They CONSUME datagen forecasts (`--forecast_dir`) and microbench cost
  tables (`--cost_json`) — coupling is by **file arguments**, not cross-subdir imports.
- **evidence/** independently measures quality (WikiText PPL of substitution, gate-vs-random
  ablation) and routing structure. Standalone tools; nothing imports them.

`miss/vs_cpu_killtest.py` imports `qwen_spice_draft` from `harness/datagen/` (path set in its bootstrap).

---

## Key result: verified-gate NEGATIVE admission (the system's headline)

In the batch=1 / A800 / exact regime, materializing predictions as 17MB H2D prefetch is *harmful*
(PCIe per-expert 0.78ms > CPU 0.167ms; a deep prefetch queue clogs the single copy engine). SPICE's
verified router gate instead drives **negative admission**: low-gate-mass MISSED experts are
shared-expert-substituted (skipped — no CPU, no fetch). Real A800 (only `--substitute_ranks` varies):

| model | substitute | TPOT vs exact all-CPU | PPL (conservative upper bound) |
|---|---|---|---|
| Qwen top-4 | `{3}` / `{2,3}` | -16% / -29% | <=+1.4% / +6.3% |
| DeepSeek top-6 | `{5}` / `{4,5}` | -19% / -31% | <=+0.88% / +4.5% |

Gate-selection is necessary: random substitution costs +40% PPL (40x); see `evidence/qwen_gate_vs_rank.py`.

### Reproduce (from `harness/scheduler/`)
```bash
python spice_shallow_issuer_runtime.py \
  --forecast_dir <runs>/qwen_forecast_dump \
  --cost_json ../../../notes/evidence/miss_assign_qwen_bf16_t16.json \
  --out /tmp/out.json --gpu 0 --train_frac 0.5 --residency 0.1 \
  --max_test_tokens 32 --d_model 2048 --d_inter 1408 --top_k 4 --cpu_threads 16 \
  --shallow_depth 0 --filler_compute_dim 4096 --timed_repeats 3 \
  --policies shallow_cpu --substitute_ranks 3     # vs --substitute_ranks "" for the exact baseline
```
Expected: identical miss counts (79.16 vs 58.50 /tok at residency 0.1) and ~-16% TPOT for `{3}`.
DeepSeek: build a forecast dump first via `datagen/make_forecast_from_dec.py` on `ds_decode_big`.

---

## core/ — original SPICE paper reproduction
`lossless_correctness.py` (fallback affects latency not logits), `prefetch_system_sim.py`
(Naive/LRU/MoE-Offloading/Pre-gated/SPICE policies + PCIe microbench), `draft_model.py` /
`train_draft_model.py` / `eval_draft_prefetch.py` (the LoRE draft path), trace collection and
summary utilities, and the `run_*.sh` suites. See git tags for milestone snapshots.

## Canonical paths (server moe-server-248)
Code: `/data/ziheng/spice` (git). Data: `/data/ziheng/spice_runs/` (traces, forecast dumps).
Models: `/data/Models/Qwen1.5-MoE-A2.7`, DeepSeek-V2-Lite in `/data/ziheng/hf_cache`.
