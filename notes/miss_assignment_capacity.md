# Capacity-Aware Top-k Miss Assignment

Date: 2026-06-03.

Question: when a MoE layer has `n_miss` routed experts not resident in HBM, should SPICE fallback fetch
all missed experts, CPU-compute all missed experts, or split the top-k miss set across CPU and PCIe/GPU?

This note uses real A800 measurements, not a scalar cost guess. The measured service path includes:

- CPU path: activation D2H -> CPU full expert compute -> output H2D.
- Fetch/GPU path: bf16 expert weight H2D -> GPU expert compute.
- Mixed path: CPU path and fetch/GPU path run concurrently; wall time is the measured critical path.

Model-specific top-k is respected:

- Qwen1.5-MoE-A2.7B: `top_k=4`, `hidden_size=2048`, `moe_intermediate_size=1408`.
- DeepSeek-V2-Lite: `top_k=6`, `hidden_size=2048`, `moe_intermediate_size=1408`.

Evidence:

- `notes/evidence/miss_assign_qwen_bf16_v2.json`
- `notes/evidence/miss_assign_deepseek_bf16_t16.json`
- `notes/evidence/miss_replay_qwen_overlap_rank.json`
- `notes/evidence/miss_replay_deepseek_overlap_rank.json`

## Per-layer miss assignment

Strict bf16, stable median. DeepSeek uses 16 CPU threads because 104-thread bf16 has large tail spikes.

| model | `n_miss` | best `n_fetch/n_cpu` | best ms | all CPU ms | all fetch ms |
|---|---:|---:|---:|---:|---:|
| Qwen top-4 | 1 | 0/1 | 0.588 | 0.588 | 0.848 |
| Qwen top-4 | 2 | 1/1 | 0.884 | 1.057 | 1.671 |
| Qwen top-4 | 3 | 1/2 | 1.293 | 1.579 | 2.498 |
| Qwen top-4 | 4 | 2/2 | 1.706 | 1.999 | 3.322 |
| DeepSeek top-6 | 1 | 0/1 | 0.454 | 0.454 | 0.848 |
| DeepSeek top-6 | 2 | 0/2 | 0.758 | 0.758 | 1.673 |
| DeepSeek top-6 | 3 | 1/2 | 0.920 | 1.141 | 2.498 |
| DeepSeek top-6 | 4 | 1/3 | 1.389 | 1.523 | 3.325 |
| DeepSeek top-6 | 5 | 2/3 | 1.705 | 1.909 | 4.144 |
| DeepSeek top-6 | 6 | 2/4 | 1.868 | 2.015 | 4.964 |

Interpretation:

- Single miss: CPU service beats demand fetch.
- Multiple misses: all-CPU becomes CPU-burst limited; splitting some experts to PCIe/GPU reduces the critical path.
- All-fetch is consistently bad because it serializes 17.3MB per missed expert over PCIe.

## Trace-level replay

Replay uses real decode route traces and measured top-k miss assignment tables. It keeps exact semantics: every
routed expert is computed. It separates the two decisions:

1. Serve current miss: CPU or fetch/GPU.
2. Grant HBM residency: only fetched experts may be admitted; CPU-served experts are not inserted.

Overlap-mode replay is shown below. It is the optimistic boundary where resident-hit GPU work can overlap with
miss service. The additive boundary has the same ordering and similar gains.

### Qwen top-4

| HBM residency | fetch-all TPOT | all-CPU TPOT | hybrid-admit TPOT | gain vs fetch-all | gain vs all-CPU |
|---:|---:|---:|---:|---:|---:|
| 5% | 85.75 | 61.41 | 52.57 | 38.7% | 14.4% |
| 10% | 78.99 | 58.78 | 49.77 | 37.0% | 15.3% |
| 20% | 70.00 | 53.90 | 45.30 | 35.3% | 16.0% |

### DeepSeek top-6

| HBM residency | fetch-all TPOT | all-CPU TPOT | hybrid-admit TPOT | gain vs fetch-all | gain vs all-CPU |
|---:|---:|---:|---:|---:|---:|
| 5% | 125.87 | 68.66 | 60.83 | 51.7% | 11.4% |
| 10% | 108.68 | 66.62 | 57.13 | 47.4% | 14.2% |
| 20% | 90.99 | 61.45 | 51.21 | 43.7% | 16.7% |

## Admission score sensitivity

After the split decides `n_fetch`, the replay chooses which missed experts to fetch/admit. Two cheap scores were tested:

- Static popularity from train traces.
- Current router rank, i.e. higher-rank/gate-priority missed experts are fetched/admitted first.

Router-rank admission is slightly better:

| model | residency | popularity TPOT | rank TPOT | rank gain |
|---|---:|---:|---:|---:|
| Qwen | 5% | 53.14 | 52.57 | 1.1% |
| Qwen | 10% | 50.40 | 49.77 | 1.3% |
| Qwen | 20% | 45.88 | 45.30 | 1.3% |
| DeepSeek | 5% | 61.31 | 60.83 | 0.8% |
| DeepSeek | 10% | 57.63 | 57.13 | 0.9% |
| DeepSeek | 20% | 51.53 | 51.21 | 0.6% |

This means the main gain is the resource split, not a sophisticated admission score. Still, the admission hook is real:
SPICE's verified future-demand signal can replace simple router-rank admission in the next experiment.

## Paper-facing statement

The clean claim is not "CPU is always faster" and not "fetch one expert and CPU one expert".

The claim is:

> SPICE fallback should treat a top-k miss set as a capacity-constrained assignment problem. For each layer, it chooses
> how many missed experts to CPU-serve and how many to fetch/GPU-compute, then admits only the fetched experts with
> sufficient future value. This decouples current-token service from HBM residency.

This fixes a concrete weakness in SPICE's original fallback: fetch-all couples "this expert is needed now" with
"this expert deserves HBM residency". On real Qwen and DeepSeek traces, capacity-aware exact miss assignment reduces
deadline-replay TPOT by 35-52% versus fetch-all and by 11-17% versus CPU-all/Fiddler-style miss service.

## Caveats

- This is deadline replay using real hardware component measurements, not yet a full integrated SPICE runtime.
- CPU bf16 has thread-count-sensitive tail latency; production policy needs a calibrated CPU-load term.
- Prior-art risk: CPU miss service itself is Fiddler-like. The defensible SPICE contribution is the unified controller:
  verified route/future-demand information drives both top-k resource assignment and HBM admission.
