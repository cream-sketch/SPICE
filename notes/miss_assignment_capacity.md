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
- `notes/evidence/prefetch_pressure_qwen_sat58.json`
- `notes/evidence/prefetch_pressure_deepseek_sat58.json`
- `notes/evidence/prefetch_pressure_qwen_sat58_p90.json`
- `notes/evidence/prefetch_pressure_qwen_sat58_cpu15.json`
- `notes/evidence/prefetch_pressure_qwen_floor{0,20,40,58}.json`
- `notes/evidence/forecast_pressure_qwen_draft_v3.json`
- `notes/evidence/pcie_topology_a800_v3.json`
- `notes/evidence/event_scheduler_qwen_main512_v4.json`
- `notes/evidence/event_scheduler_qwen_p90_512_v4.json`
- `notes/evidence/event_scheduler_qwen_cpu15_512_v4.json`
- `notes/evidence/event_scheduler_qwen_spicefetch128_v4.json`
- `notes/evidence/shallow_h2d_issuer_a800_prio_v3.json`
- `notes/evidence/shallow_h2d_issuer_a800_sameprio_v3.json`

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

The clean claim is not "CPU is always faster" and not "fetch one expert and CPU one expert". It is also
not the broad claim "heterogeneous CPU/GPU scheduling", which overlaps with Fiddler/HybriMoE/KTransformers,
and not speculative-window batching/amortized fetch as in SpecMoEOff.
The defensible SPICE-specific claim is narrower:

The claim is:

> Once SPICE's verified future-demand prefetch occupies the H2D copy path, residual fallback misses should be scheduled
> by a deadline-aware resource DAG, not by SPICE's original fetch-all fallback. The controller chooses how many residual
> missed experts may still pay H2D and sends the rest to CPU/DRAM service. Future-demand/admission scores choose which
> fetched experts get residency, but the fetch count itself is a timing decision.

This fixes a concrete weakness in SPICE's original fallback: fetch-all couples "this expert is needed now" with
"this expert deserves HBM residency". When PCIe has slack, exact capacity split beats both fetch-all and CPU-all.
When SPICE prefetch already saturates PCIe, the same scheduler keeps only residual fetches whose CPU burst is more
expensive than the remaining H2D service, and sends the rest to CPU fallback.

## PCIe pressure correction

The unconstrained split table above assumes fallback fetches have usable PCIe slack. That is false in the real SPICE
regime where speculative prefetch already fills the PCIe copy engine. The default scheduler is the `layer_serial`
resource DAG:

```text
base_done_l = clock_l + dense_l + hit_count_l * t_gpu
copy_ready_l = SPICE_prefetch_H2D_floor + previous_residual_fetches * t_fetch_h2d
fetch_service_l = max(0, copy_ready_l - base_done_l)
                  + n_fetch_l * (t_fetch_h2d + t_gpu)
cpu_service_l = measured_cpu_cost(n_miss_l - n_fetch_l)
clock_{l+1} = base_done_l + max(measured_mixed_cost_l, fetch_service_l, cpu_service_l)
token_time = max(clock_L, SPICE_prefetch_H2D_floor + total_residual_fetches * t_fetch_h2d)
```

A small dynamic program over per-layer fetch counts is enough because `top_k` is 4 or 6. The older aggregate
`max(compute_time, SPICE_prefetch_H2D_floor + residual_fetch_H2D)` equation is only a lower-bound/legacy mode
(`--dag_mode aggregate_lower_bound`), not the claim used in the tables below.

Implementation: `experiments/harness/prefetch_pressure_scheduler_replay.py`.

The replay compares four exact same-precision residual-miss policies:

- `spice_fetch_all`: SPICE-style residual fallback fetches every miss over PCIe and admits it.
- `cpu_fallback`: residual misses are served by CPU DRAM compute; no HBM admission.
- `naive_capacity_split`: uses the unconstrained per-layer miss-assignment table, ignoring SPICE prefetch pressure.
- `pressure_aware_dp`: token-level dynamic program that minimizes a sequential layer resource DAG. The older
  aggregate `max(compute_time, SPICE_prefetch_H2D_floor + residual_fetch_H2D)` model remains available only as
  `--dag_mode aggregate_lower_bound`.

The harness now reports not only TPOT but also resource attribution:

- `token_dag_ms_per_tok`
- `layer_clock_ms_per_tok`
- `pcie_h2d_ms_per_tok`
- `fallback_h2d_ms_per_tok`
- `pcie_floor_bound_frac`
- `pcie_wait_bound_frac`
- `layer_clock_bound_frac`
- `compute_bound_frac` (alias for `layer_clock_bound_frac`)
- `cpu_act_roundtrip_mb_per_tok`
- `admitted_never_used_frac`

Two stress knobs are included for robustness:

- `--cost_metric {ms,mean_ms,p90_ms}` switches the measured miss-service table from median to tail costs.
- `--cpu_scale` and `--fetch_scale` model CPU-load and PCIe-bandwidth sensitivity without changing semantics.

### Saturated prefetch regime (`58 ms/token` PCIe floor)

| model | HBM residency | SPICE fetch-all | CPU fallback | naive split | pressure-aware DP |
|---|---:|---:|---:|---:|---:|
| Qwen | 5% | 146.22 | 61.98 | 110.71 | 61.29 / 2.20 fetches/tok |
| Qwen | 10% | 140.05 | 60.15 | 108.49 | 59.68 / 1.08 fetches/tok |
| Qwen | 20% | 131.65 | 58.25 | 104.67 | 58.20 / 0.12 fetches/tok |
| DeepSeek | 5% | 187.75 | 69.46 | 119.43 | 68.47 / 7.38 fetches/tok |
| DeepSeek | 10% | 171.55 | 68.16 | 116.31 | 67.02 / 6.07 fetches/tok |
| DeepSeek | 20% | 154.69 | 64.41 | 111.75 | 63.47 / 3.21 fetches/tok |

Interpretation:

- Naive capacity split is wrong under SPICE prefetch pressure: it still injects many residual fetches into a saturated PCIe path.
- Pressure-aware DP throttles residual fetches to a small set, not to a fixed all-CPU rule.
- The win over SPICE fetch-all is 57-64%; the win over naive split is 42-47%.

### PCIe floor sweep

Qwen, 10% residency:

| prefetch H2D floor | CPU fallback | naive split | pressure-aware DP |
|---:|---:|---:|---:|
| 0 ms | 59.80 | 51.00 / 33.2 fetches/tok | 51.00 / 33.2 fetches/tok |
| 20 ms | 59.80 | 70.07 / 33.2 fetches/tok | 53.80 / 21.5 fetches/tok |
| 40 ms | 59.80 | 90.07 / 33.2 fetches/tok | 56.79 / 10.6 fetches/tok |
| 58 ms | 60.45 | 108.07 / 33.2 fetches/tok | 59.96 / 1.25 fetches/tok |

This is the central scheduler result: the policy is not "always CPU" and not "always split". It continuously reduces
fallback fetches as PCIe pressure rises, and at saturation keeps only residual fetches whose CPU burst is more expensive
than waiting for the remaining PCIe service.

### CPU-tail / load sensitivity

Qwen, saturated `58 ms/token` floor:

| stress | HBM residency | CPU fallback | pressure-aware DP |
|---|---:|---:|---:|
| `p90_ms` cost table | 5% | 92.49 ms | 78.51 ms / 0.55 fetches/tok |
| `p90_ms` cost table | 10% | 140.57 ms | 87.23 ms / 1.86 fetches/tok |
| `p90_ms` cost table | 20% | 254.32 ms | 95.29 ms / 4.16 fetches/tok |
| `cpu_scale=1.5` | 5% | 84.44 ms | 75.38 ms / 11.64 fetches/tok |
| `cpu_scale=1.5` | 10% | 81.02 ms | 73.23 ms / 9.34 fetches/tok |
| `cpu_scale=1.5` | 20% | 73.90 ms | 68.57 ms / 6.01 fetches/tok |

This is the best evidence that the controller is not a disguised all-CPU rule. When CPU service has tail spikes or
higher load, the DP reopens a small number of residual fetches even under SPICE prefetch pressure.

### SPICE forecast-derived pressure

The scalar floor sweep above is a controlled mechanism scan. To connect the result back to SPICE's actual draft signal,
`experiments/harness/spice_forecast_pressure_replay.py` consumes `qwen_spice_draft.py --dump_forecast` output:

- `true_top[L,T,K]`: target routed experts.
- `fcast[anchor,horizon,T,K]`: draft-predicted future experts.

The harness is deliberately conservative: a draft prefetch enters a serial H2D queue and counts as a hit only if it
finishes before a dense-layer lower-bound deadline. Late prefetches may enter HBM only after the token and can help
future tokens; they do not magically fix the current residual miss.

This is still a conservative upper-pressure diagnostic. Prefetch hit/miss uses per-layer ready times, but residual
scheduling collapses the token's issued SPICE prefetch bytes into an H2D floor. That stresses residual fallback fetches
under saturated SPICE traffic; a full runtime needs one copy-engine event queue with cancellation, staging buffers, and
eviction timing.

Qwen training-free SPICE draft, `max_lead_layers=5`, 512 prompt tokens:

| HBM residency | SPICE fetch-all fallback | CPU fallback | naive split | pressure-aware DP |
|---:|---:|---:|---:|---:|
| 5% | 182.49 ms | 90.06 ms | 144.66 ms | 90.06 ms / 0.00 fetches/tok |
| 10% | 171.81 ms | 84.17 ms | 136.63 ms | 84.17 ms / 0.00 fetches/tok |
| 20% | 149.74 ms | 71.32 ms | 119.39 ms | 71.32 ms / 0.01 fetches/tok |

This SPICE-native scalar replay says the same thing more sharply than the hand-set floor: draft information exists, but
17MB H2D deadlines convert many forecasted experts into late/wasted prefetches. In this collapsed floor model, residual
fallback fetches almost disappear. The full event-queue replay below refines that conclusion: residual fetches should
not be zeroed out globally; they should be reopened only when the shared H2D queue has enough timing slack or CPU service
has become the critical path.

## Unified event-queue replay

The scalar forecast replay above is intentionally pessimistic because it collapses all SPICE prefetch traffic into a
single H2D floor before residual miss assignment. The next harness, `experiments/harness/spice_event_scheduler_replay.py`,
uses the same Qwen SPICE forecast dump but replays a single H2D event queue:

- draft prefetches, residual fetches, and CPU-result H2D share one serialized H2D engine;
- activation D2H is modeled separately;
- HBM has a main cache plus an optional staging buffer;
- all routed experts are still computed exactly, in bf16, with no drop/quantization/compression;
- the replay asserts no H2D interval overlap.

This is still diagnostic, not a source-only baseline reproduction. In particular, `fiddler_cpu` below means
"SPICE forecast prefetch stream + all residual misses served by CPU", not upstream Fiddler. It is the right diagnostic
row for isolating residual miss handling under SPICE traffic, but it must not be cited as a prior-art Fiddler result.

The v4 replay fixes three earlier replay bugs: residual fetches are no longer popped before same-layer prefetch issue;
staging expires by logical `(token, layer)` rather than a dense-only wall-clock deadline; and token/layer zero is handled
with explicit `None` checks. It also adds two hardware-assumption controls:

- `shallow_scheduler`: software issuer keeps only a shallow low-priority DMA queue, so "cancellation" means not dispatching
  future prefetch intents rather than aborting a running `cudaMemcpyAsync`.
- `fifo_deadline`: conservative FIFO mode; fallback traffic cannot bypass already-issued low-priority prefetch DMA.

### Normal CPU cost

Qwen, 512 tokens, median CPU miss-service table. All numbers are tail TPOT in ms/token.

| HBM residency | Fiddler-like CPU residual | deadline scheduler | shallow scheduler | conservative FIFO | no cancel | oracle deadline |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 55.58 | 32.67 | 32.41 | 37.01 | 34.16 | 31.11 |
| 10% | 52.49 | 30.94 | 30.29 | 36.19 | 32.45 | 29.58 |
| 20% | 46.77 | 28.60 | 27.62 | 34.54 | 29.73 | 27.69 |

Interpretation:

- The full event queue overturns the scalar-floor conclusion that saturated SPICE should collapse to all-CPU residuals.
  The scheduler reopens a selected subset of residual fetches while throttling low-value draft prefetch traffic.
- `shallow_scheduler` is within 0-1 ms/token of the optimistic deadline scheduler, which is the first evidence that a
  realistic software issuer could preserve most of the gain.
- Even the conservative FIFO mode improves 26-33% over the Fiddler-like all-CPU-residual diagnostic row, so the mechanism
  is not solely an artifact of arbitrary cancellation/reordering.
- Staging is now functional: deadline-scheduler prefetch useful fractions are 0.09, 0.20, and 0.32 across 5/10/20%
  residency. The earlier zero-useful result was a replay expiration bug.

### CPU-tail and CPU-load sensitivity

Tail-sensitive miss costs make the resource split more important.

| stress | HBM residency | Fiddler-like CPU residual | deadline scheduler | shallow scheduler | conservative FIFO | oracle deadline |
|---|---:|---:|---:|---:|---:|---:|
| `p90_ms` | 5% | 123.00 | 36.02 | 36.69 | 42.77 | 35.03 |
| `p90_ms` | 10% | 133.54 | 34.38 | 34.69 | 41.27 | 33.23 |
| `p90_ms` | 20% | 132.91 | 31.75 | 31.78 | 38.35 | 30.60 |
| `cpu_scale=1.5` | 5% | 70.38 | 39.10 | 39.06 | 41.53 | 37.28 |
| `cpu_scale=1.5` | 10% | 66.26 | 36.83 | 36.68 | 39.77 | 34.88 |
| `cpu_scale=1.5` | 20% | 58.35 | 33.62 | 33.13 | 37.02 | 31.77 |

This is the strongest current evidence that the scheduler is not a disguised "always CPU" rule. When CPU service has
tail spikes or load, the scheduler spends H2D on a small set of residual experts and keeps the rest on CPU.

### SPICE fetch-all pressure

The same fixed event queue with `spice_fetch_all` was run for 128 tokens because it is much slower. Tail TPOT remains
dominated by H2D backlog:

| HBM residency | SPICE fetch-all tail TPOT | H2D backlog |
|---:|---:|---:|
| 5% | 132.89 | 47.09 |
| 10% | 119.51 | 40.38 |
| 20% | 101.78 | 31.52 |

This is the concrete weakness in original SPICE fallback: residual demand fetches and draft prefetches both spend
17MB H2D chunks on the same critical path. The scheduler's contribution is not "more prefetch"; it is a resource-DAG
controller that decides which forecasted prefetch intents are worth dispatching, which residual misses deserve H2D/GPU,
and which residual misses should stay on CPU/DRAM.

## Caveats

- This is deadline/event replay using real hardware component measurements. It is not yet a full exact-logit runtime
  and not a source-only reproduction of Fiddler/HybriMoE/KTransformers.
- The shallow software-issuer assumption still needs a CUDA/Nsight implementation check: queued prefetch intents must
  be held in software or in small chunks; already-running 17MB CUDA copies cannot be magically aborted.
- DeepSeek does not yet have a SPICE forecast dump in this workspace, so the unified event-queue replay is Qwen-only.
- CPU bf16 has thread-count-sensitive tail latency; production policy needs a calibrated CPU-load term.
- Prior-art risk: CPU miss service itself is Fiddler-like. The defensible SPICE contribution is the unified controller:
  verified route/future-demand information drives top-k resource assignment, PCIe-pressure-aware fallback, and HBM admission.

## PCIe topology measurement

A separate A800 topology microbench (`experiments/harness/pcie_topology_microbench.py`) tested two hardware hypotheses:

1. H2D and D2H can use PCIe full-duplex bandwidth.
2. Multi-stream tiled H2D can push one expert prefetch above the single-copy bandwidth.

Evidence: `notes/evidence/pcie_topology_a800_v3.json`.

Results for one bf16 Qwen/DeepSeek expert (17.30MB):

| case | median time | effective bandwidth / note |
|---|---:|---:|
| H2D whole expert | 0.792 ms | 21.34 GB/s |
| D2H whole expert | 0.787 ms | 21.46 GB/s |
| H2D + D2H big concurrent | 1.255 ms | overlap exists, but only ~0.41 efficiency; not ideal full-duplex |
| small D2H alone | 0.014 ms | activation-scale copy is tiny |
| small D2H during big H2D | 0.808 ms total | device-wide total still dominated by big H2D |
| small H2D alone | 0.015 ms | CPU-result-scale copy is tiny |
| small H2D completion after whole H2D enqueue | 0.072 ms | small result H2D completion is not delayed by a full 17MB copy on this setup |
| H2D tiled, 1-8 streams, 1-8MB tiles | 0.821-0.876 ms | slower than whole-copy; no bandwidth gain |

Conclusion: PCIe full-duplex should be used for activation D2H/H2D overlap, but it does not create extra H2D capacity
for residual expert fetches. Multi-stream tiled expert prefetch is not a throughput win on this A800 setup; its value
is latency isolation/priority control, not higher steady-state PCIe bandwidth. The event-timed small-H2D result shows
that, in this measured two-stream enqueue pattern, a small H2D can complete quickly; it does not prove general
preemption behind an already in-flight 17MB H2D. The final runtime still needs Nsight/CUDA-event confirmation with its
actual stream priorities.

## Shallow H2D software issuer

The unified event replay's strongest implementability assumption is not "CUDA can cancel a submitted copy". It is
stricter and more realizable: the runtime should keep draft prefetches as software intents and submit only a shallow
number of 17MB H2D DMA copies. Residual miss fetches then wait behind at most this shallow queue.

Implementation: `experiments/harness/shallow_h2d_issuer_microbench.py`.

The v3 evidence records the device and isolation metadata: A800 80GB PCIe, P0, stream-priority range `[0, -3]`, default
compute mode, MIG disabled, and only the benchmark Python process in `nvidia-smi --query-compute-apps` during each run.

The microbench enqueues `N` low-priority expert-size H2D prefetch copies, then times completion of a high transfer.
Latency starts after the low-copy Python enqueue loop returns, so it measures a host-submitted CUDA queue. Some low
copies may already be running while the loop submits the rest; therefore the deep-backlog number is diagnostic, not an
exact worst-case FIFO bound. One expert is 17.30MB; one activation/output copy is 4.10KB.

### Big residual H2D

| low H2D queue before residual fetch | high priority | same priority |
|---:|---:|---:|
| 0 | 0.800 ms | 0.800 ms |
| 1 | 1.551 ms | 1.551 ms |
| 2 | 2.297 ms | 2.298 ms |
| 4 | 3.792 ms | 3.792 ms |
| 8 | 6.784 ms | 6.785 ms |
| 32 deep backlog | 24.724 ms | 24.730 ms |

This directly supports the shallow-issuer design. If SPICE dispatches an entire token worth of prefetches, a later
residual 17MB demand fetch can wait tens of milliseconds. If the software issuer holds the low-priority queue at depth
1-2, the same residual fetch waits only about 1.6-2.3ms. CUDA stream priority is not the main mechanism here: high-priority
and same-priority streams show nearly identical numbers. The robust mechanism is not submitting deep H2D queues in the
first place.

### Small CPU-result H2D and activation D2H

| queued low H2D copies | high small H2D | high small D2H |
|---:|---:|---:|
| 0 | 0.035 ms | 0.034 ms |
| 32 | 0.040 ms | 0.039 ms |

This supports the earlier resource-DAG assumption that isolated pinned 4KB CPU-result H2D and activation D2H copies are
not the dominant bottleneck in this measured stream pattern. It does not prove that arbitrary production small-copy
bursts are always free. The critical scheduler decision remains how many 17MB expert-weight H2D transfers to dispatch
and when.

### Chunking

Submitting only one low-priority prefetch tile before a residual big H2D bounds the head-of-line delay:

| low tile | residual big H2D completion |
|---:|---:|
| 0.5MB | 0.802 ms |
| 1MB | 0.824 ms |
| 2MB | 0.868 ms |
| 4MB | 0.958 ms |

This does not contradict the earlier topology result that tiled full-expert prefetch is not a throughput win. Its value
is latency isolation: chunking or shallow dispatch limits how much useless draft traffic can block an exact residual
fetch.

## Real CUDA runtime bridge: admission beats deeper prefetch

Implementation: `experiments/harness/spice_shallow_issuer_runtime.py`.

This harness consumes a real SPICE forecast dump (`true_top`, `fcast`) and runs a trace-driven CUDA timing loop with
real pinned host expert banks, real device resident/staging/fetch slots, real H2D copies, real GPU expert GEMMs from the
owning slot, and real CPU expert GEMMs. It is still a diagnostic timing harness, not a full source-only model runtime:
dense/attention is represented by a calibrated filler GEMM and logits are not checked.

Second-pass review fixed the earlier fatal modeling bugs:

- CPU fallback uses scoped activation D2H and output H2D events, not device-wide synchronization.
- Staged low-prefetch experts retain their HBM slot until use or deadline expiry.
- Resident, prefetched, and residual-fetched experts compute from owned device slots.
- Demand misses cancel/promote pending low prefetch state rather than silently duplicating it.
- Low-hit admission is delayed until same-layer hit GEMMs have been enqueued, preventing same-layer resident-slot overwrite.
- Missing cost-table entries conservatively choose all-CPU rather than throwing `KeyError`.

Evidence:

- `notes/evidence/filler_gemm_calibration_a800_v1.json`: A800 bf16 4096x4096 GEMM median 0.579 ms.
- `notes/evidence/qwen_forecast_small_v1_metrics.json`: training-free SPICE draft slot recall is 1.00/0.90/0.85/0.82/0.79/0.77 for horizons 1..6.
- `notes/evidence/shallow_runtime_qwen_t16_32_f4096_deep_dagfix_v2.json`
- `notes/evidence/shallow_runtime_qwen_t16_32_f4096_depth{0,1,2,4,8}_dagfix_v2.json`
- `notes/evidence/shallow_runtime_qwen_t16_32_f4096_depth2_minlead{1,2,3,4,5}_dagfix_v1.json`
- `notes/evidence/shallow_runtime_qwen_t16_32_f4096_dummy_dagfix_v1.json`

Qwen top-4, 10% HBM residency, 16 CPU threads, bf16 CPU/GPU, 32 tokens, one 4096 bf16 filler GEMM per layer:

| policy | prefetch rule | TPOT ms | hits/tok | CPU misses/tok | residual fetch/tok | active-prefetch wait/tok |
|---|---|---:|---:|---:|---:|---:|
| deep fetch-all | all forecast H2D + residual fetch | 83.21 | 87.25 | 0.00 | 8.75 | 2.59 |
| deep CPU | all forecast H2D + residual CPU | 72.45 | 87.00 | 9.00 | 0.00 | 11.59 |
| shallow CPU | depth 0, no H2D prefetch | 46.67 | 23.81 | 72.19 | 0.00 | 0.00 |
| shallow CPU | depth 1 | 50.52 | 43.78 | 52.22 | 0.00 | 20.84 |
| shallow CPU | depth 2 | 58.38 | 60.59 | 35.41 | 0.00 | 31.41 |
| shallow CPU | depth 4 | 64.95 | 77.34 | 18.66 | 0.00 | 25.97 |
| shallow CPU | depth 8 | 66.12 | 81.75 | 14.25 | 0.00 | 21.94 |

Deadline-gating near predictions reduces active-H2D waits but still does not beat no-prefetch in this measured regime:

| depth | min prefetch lead | TPOT ms | hits/tok | CPU misses/tok | active-prefetch wait/tok |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 58.37 | 60.56 | 35.44 | 31.41 |
| 2 | 2 | 58.09 | 59.78 | 36.22 | 28.09 |
| 2 | 3 | 56.33 | 59.38 | 36.62 | 25.78 |
| 2 | 4 | 55.85 | 59.19 | 36.81 | 22.91 |
| 2 | 5 | 55.47 | 58.59 | 37.41 | 17.41 |

The apples-to-apples dummy control disables prefetch hits while keeping forecast H2D traffic. Miss counts are identical,
so the remaining difference isolates copy-engine interference:

| policy | TPOT ms | hits/tok | misses/tok | CPU misses/tok | prefetch submitted/tok |
|---|---:|---:|---:|---:|---:|
| deep dummy CPU | 64.72 | 23.81 | 72.19 | 72.19 | 81.88 |
| shallow dummy CPU | 55.15 | 23.81 | 72.19 | 72.19 | 46.91 |

Interpretation:

- The SPICE draft information is strong; the runtime failure mode is materializing too much correct information as
  17MB H2D traffic.
- A forecasted expert that is still in the H2D queue at the demand point is a stall, not a useful hit.
- In this A800/Qwen/16-thread batch-1 regime, the best exact same-precision action is negative admission: do not submit
  forecast H2D, and serve residual misses from CPU/DRAM.
- This does not invalidate the event replay. It sharpens the implementable story: SPICE should be a verified
  expert-demand oracle feeding a resource-DAG admission controller, not a fetch-all prefetcher. The controller's first
  job is deciding when *not* to turn a correct forecast into PCIe traffic.

## SPICE-GOS: global overflow staging, not residency

The next runtime revision separates two actions that SPICE-style prefetch usually couples:

1. **Transient staging service:** fetch a forecasted future expert into a low-priority staging slot, use it exactly once
   when that future miss arrives, then release the staging slot.
2. **Resident cache admission:** copy the staged expert into the long-lived HBM expert cache after use.

This distinction matters on the real resource DAG. A staged expert can reduce future CPU fallback burst, but promoting
that same 17MB expert into the resident cache adds a main-stream D2D copy and extra cache churn. In the earlier runtime,
GOS served the miss and then admitted every low-prefetch hit into the cache, which often erased the scheduling gain.

Implementation: `experiments/harness/spice_shallow_issuer_runtime.py`, policy `gos_cpu` with
`--no_admit_prefetch_hits`.

GOS admission is a rolling-horizon controller over SPICE forecast targets. For each future `(token, layer)`, it admits a
prefix of forecasted experts only if:

- the low-priority H2D backlog can complete before the future-layer deadline,
- the CPU miss-burst cost is covered by measured CPU rows rather than extrapolation, and
- the action reduces the target-layer critical-path increment:
  `max(0, CPU_burst - overlap_window) -> max(prefetched_gpu_compute, remaining_CPU_burst - overlap_window)`.

Evidence:

- `notes/evidence/gos_qwen_t16_64_f4096_depth0_final_v8.json`
- `notes/evidence/gos_qwen_t16_64_f4096_normal_admit_final_v8.json`
- `notes/evidence/gos_qwen_t16_64_f4096_servedonly_final_v8.json`
- `notes/evidence/gos_qwen_t16_64_f4096_dummy_final_v8.json`

Qwen top-4, 10% HBM residency, 16 CPU threads, bf16 CPU/GPU, 64 tokens, one 4096 bf16 filler GEMM/layer:

| policy | TPOT ms | speedup vs CPU-only | resident hit/tok | staging hit/tok | CPU misses/tok | staged H2D/tok | cache evict/tok | active wait/tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| depth0 CPU fallback | 49.00 | 1.00x | 24.83 | 0.00 | 71.17 | 0.00 | 0.00 | 0.00 |
| GOS + resident admission | 47.76 | 1.03x | 20.02 | 35.83 | 40.16 | 39.88 | 35.83 | 0.00 |
| GOS transient staging only | 44.23 | 1.11x | 24.83 | 33.73 | 37.44 | 37.61 | 0.00 | 0.00 |
| GOS perturbation control | 60.74 | 0.81x | 24.83 | 0.00 | 71.17 | 38.12 | 0.00 | 0.00 |

The perturbation control is state-divergent, not an identical replay: disabling staged hits changes later cache and CPU
state. It still rules out the easiest false explanation. Injecting GOS-admitted H2D traffic without consuming the staged
experts is much slower (`60.74ms` in the filler regime), so the gain is not from copy-engine noise; it comes from useful
on-time staging that reduces CPU miss burst.

Interpretation:

- Correct forecasts should not automatically become resident-cache admissions. The useful primitive is **transient
  staging**: a one-shot H2D service path for predicted overflow misses.
- GOS is a positive, SPICE-native miss-handling mechanism in the measured pressure regime: it uses SPICE future-demand
  information to choose which misses should consume PCIe and which should remain CPU-served.
- The gain is regime-dependent. A no-filler stress run showed small positive results in one batch but substantial timing
  noise in a later rerun, so it is not used as main evidence yet. This supports a resource-DAG story rather than a
  universal prefetch story.
- `--gos_cpu_overlap_ms` is a calibrated policy constant in this diagnostic harness. The final system needs to estimate
  it online or replace it with measured per-layer slack.
- Remaining evidence needed before paper-level claims: DeepSeek replication, larger forecast traces, Nsight confirmation
  that the main gain is removal of low-hit D2D cache admission, and a calibrated production overlap model instead of a
  fixed `--gos_cpu_overlap_ms`.
