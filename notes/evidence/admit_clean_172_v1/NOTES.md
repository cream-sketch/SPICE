# admit_clean_172_v1 — clean-machine admission verdict (corroboration)

**Purpose**: re-run the resident-admission comparison on an UNCONTENDED machine,
after discovering the 248 v1/v2 runs were CPU-contention-noisy (a concurrent sweep
oversubscribed CPU -> `never` TPOT spanned 46.9-57.0ms, ~20%). This set is
independent corroboration of the 248 `admit_decisive_v5_isolated` conclusion, on a
SEPARATE machine (different absolute TPOT; do NOT mix the two machines' numbers).

## Machine / isolation
- Node: 172 = `hpclab03`, 2x A800 80GB, system idle (`load avg 0.00`) during the run.
- Single `spice_shallow_issuer_runtime.py` process at a time (sequential driver
  `run_admit_clean_172.sh`); GPU0; `cpu_threads=16`.
- Env: `/data/ziheng/conda_envs/mxmoe_clean` (torch 2.6.0+cu124; the `sparmoe`
  env's torch is cu130 and incompatible with 172's 12.8 driver).
- Forecast: `qwen_forecast_wiki64_v1` (real training-free SPICE draft, recall@1 0.999),
  held-out test split (train_frac 0.5), 1024 test tokens, 7 timed repeats.
- Code: pre-`7e76a72` (this run did NOT include the later GOS-lifecycle/DP fix).

## Results (median TPOT ms, min-max spread ~0.3%)
| file | policy | TPOT | vs never |
|---|---|---:|---:|
| shallow_cpu.json | shallow_cpu (naive shallow prefetch + always-admit, residual CPU) | 62.23 | — |
| never.json       | gos_cpu, transient-staging only (no resident admit) | 52.72 | baseline |
| always.json      | gos_cpu, always-admit staged hits | 52.83 | +0.2% worse |
| recent.json      | gos_cpu, recent_reuse admit (deployable) | 53.55 | +1.6% worse |
| oracle_c0.8.json | gos_cpu, oracle_value admit, resident_admit_cost_ms=0.8 (~10% admit) | 52.44 | -0.5% |

NOTE (correction): `shallow_cpu.json` was originally mis-named `depth0.json`. It is
NOT a pure no-prefetch CPU-only baseline -- `--policies shallow_cpu` still runs the
shallow prefetch issuer (staging_hits/cache_evictions ~38/tok). It is a naive-prefetch
reference, not a CPU-only floor. A true no-prefetch CPU-only baseline was not run here.

## Verdict
Even the optimistic true-future oracle beats `best(always, never)` by only 0.5% (<< 2%);
every deployable admission policy is worse than transient-only. Resident-HBM admission
is dropped as a batch=1 lever. Matches the 248 v5_isolated conclusion.
