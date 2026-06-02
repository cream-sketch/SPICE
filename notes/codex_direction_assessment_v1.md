# SPICE Direction Assessment v1

Search date: 2026-06-02. Scope: closest mechanism-level prior art for uncertainty/cost-aware expert caching and approximate or speculative expert miss recovery in offloaded MoE inference.

## 1. Verdict: incremental or not

Blunt verdict: **A is not a non-incremental contribution anymore. B is the only plausible non-incremental axis, and only if it becomes an exact or bounded miss-recovery scheduler with demonstrated critical-path latency reduction on real traces.**

Mechanism A, "uncertainty-aware admission/eviction", is directly challenged by FineMoE/fMoE. FineMoE records per-iteration gate probability distributions ("expert maps"), uses probabilities and confidence-like similarity scores for prefetch selection, and combines expert probability with cache frequency for eviction priority. That is close enough that a reviewer will not accept "no paper uses predictor confidence for eviction" as true. At best, A can be a better implementation or a calibrated utility model over FineMoE/HOBBIT/HybriMoE, not the thesis.

Mechanism B, "low-rank approximate-then-correct miss recovery", is more interesting, but the current story has a fatal ambiguity: if exact logits are required, approximate downstream compute is not automatically reusable. In a transformer tail with RMSNorm, attention, router top-k, and SwiGLU experts, a delta at one missed expert generally changes downstream hidden states and possibly routing. Exactness usually means replaying the downstream tail after the true expert arrives. If replay is full-cost, the approximate pass does not reduce latency; it only gives a speculative window to prefetch downstream exact experts. That can still be a real system idea, but the thesis must say that explicitly.

The sharper non-incremental claim is not "(A)+(B)". It is: **verified expert-level miss recovery that uses a resident low-rank proxy to turn an unavoidable miss stall into a downstream prefetch window, then exact-replays before commit.** That distinguishes SPICE from predictor-only work, but it must beat FineMoE/HOBBIT-like systems under the same memory and transfer budget.

## 2. Closest prior art per mechanism

| Work | Venue/year | 1-line method | Subsumes A? | Subsumes B? |
|---|---:|---|---|---|
| [Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading / FineMoE/fMoE](https://arxiv.org/abs/2502.05370) | arXiv 2025 | Stores iteration-level expert probability distributions, searches by semantic/trajectory similarity, and uses the matched probability map to guide prefetch, caching, and eviction. | **Mostly yes.** It explicitly uses probability distributions for prefetching and eviction priority. This is the closest A prior. | No. Misses are still on-demand loads; no approximate execution. |
| [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433) | arXiv 2024 | On cache miss, dynamically loads lower-precision versions for less-critical experts; also uses adaptive prefetch and multidimensional caching. | Partly. Its cache is cost/miss-penalty aware, not just LRU. | **Partly, and dangerously close.** It is approximate miss recovery via low-precision replacement, but it is lossy and does not exact-correct. |
| [HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference](https://arxiv.org/abs/2504.05897) | DAC 2025 / arXiv 2025 | Hybrid CPU/GPU execution with dynamic intra-layer scheduling, impact-driven inter-layer prefetching, and score-based caching. | Partly. It is utility/score-based cache management, though in a hybrid execution setting. | No low-rank correction; miss mitigation comes from CPU/GPU scheduling. |
| [ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling](https://arxiv.org/abs/2410.17954) | DAC 2026 / arXiv 2024 | Transformer predictor forecasts routing paths; predictive cache plus real-time cache correction and token scheduling. | Partly. It uses predictions to drive cache scheduling, but not the same soft expected-utility formulation. | No. "Correction" means correcting cache contents, not approximate expert outputs. |
| [ProMoE: Fast MoE-based LLM Serving using Proactive Caching](https://arxiv.org/abs/2410.22134) | arXiv 2024/2025 | Predicts future expert usage from intermediate states and proactively fetches experts to remove reactive misses from the critical path. | No, mostly predictor/prefetch. | No approximate miss execution. |
| [MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache](https://arxiv.org/abs/2401.14361) | arXiv 2024/2025 | Request-level activation tracing guides expert replacement and prefetching. | Partly as activation-aware cache baseline, but coarser than A. | No. On-demand load on miss. |
| [Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference](https://arxiv.org/abs/2412.00099) | TMLR 2025 | Adds a cache prior to router logits so cached experts are more likely to be selected, trading routing fidelity for cache locality. | Adjacent. It uses router uncertainty/cache state for locality, but changes routing rather than eviction. | Partly as a lossy alternative to misses: choose cached experts instead of loading uncached ones. No exact correction. |
| [MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators](https://proceedings.mlsys.org/paper_files/paper/2025/hash/9032e5c9ec394ce768a2fa9bdc56af6c-Abstract-Conference.html) | MLSys 2025 | Adds low-rank compensators to highly quantized MoE weights to recover quantization accuracy. | No. | Partly. It owns the "low-rank compensator for expert error" motif, but not runtime cache-miss recovery or exact replay. |
| [Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy / MC-SMoE](https://openreview.net/pdf?id=eFWG9Cy3WK) | ICLR 2024 | Merges experts and decomposes merged weights into low-rank/sparse alternatives to reduce memory/FLOPs. | No. | Only low-rank expert approximation prior; no miss recovery. |
| [Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling](https://arxiv.org/abs/2503.04398) | arXiv 2025 | Predicts routing paths to pre-schedule tokens and experts across devices, reducing distributed expert-parallel communication. | No, different bottleneck. | Adjacent. It is lossless speculative expert pre-scheduling, but not single-GPU CPU/NVMe offload and not low-rank approximate execution. |

Closest prior for A: **FineMoE/fMoE**, then HybriMoE/HOBBIT. A is not enough.

Closest prior for B: **HOBBIT** for cache-miss approximate expert execution, **MiLo** for low-rank expert compensation, and **Speculative MoE/speculative decoding** for verified speculative execution. None cleanly subsumes "low-rank proxy runs during a miss, schedules downstream exact expert movement, then exact-replays before token commit", but each attacks part of the claim.

## 3. Fatal flaws + fixes

### Fatal flaw 1: exact logits erase most naive latency win

Baseline on a miss at layer `l`: wait for H2D load, compute true expert, then compute layers `l+1...L`.

Naive B: compute surrogate expert, run approximate downstream layers while H2D load runs, then exact-correct. If exact correction requires replaying layers `l...L`, the approximate downstream compute is discarded. The critical path is still roughly `H2D load + exact tail compute`, plus surrogate overhead. That is not a speedup.

Fix: define the useful work of the approximate pass as **downstream prefetch scheduling**, not reuse of approximate activations. During the current miss, the proxy predicts future layer expert needs and launches H2D loads. When the true expert arrives, the exact replay sees fewer downstream misses. The paper must measure "stall recovered by speculative downstream prefetch", not just hidden-state approximation error.

### Fatal flaw 2: exact delta correction is not cheap in transformers

The downstream tail is nonlinear: RMSNorm, attention, router top-k, expert MLPs, residuals, and sampling logits. A local delta from one expert cannot be pushed exactly through the tail without replay unless you maintain expensive Jacobians or all intermediate branch states. Router top-k is discontinuous: small hidden error can flip selected experts if margins are small.

Fix: either accept exact replay, or restrict "correction" to a certified approximate mode. For certification, measure router margin distributions and logit margins. A practical guarantee might be "same greedy token if logit error bound is below the top-1/top-2 logit margin", not "exact logits". Full exact logits require exact replay.

### Fatal flaw 3: error bounds are likely vacuous

Layerwise Lipschitz bounds through many transformer blocks will probably explode. Router top-k makes the bound brittle. A mathematically certified logit bound may be too loose to certify anything useful on real prompts.

Fix: do not make certification the first contribution. Start with exact replay and report zero logit mismatch. Treat bounded-error commit as an optional second mode only after empirical local-error and margin data show it might certify nontrivial tokens.

### Fatal flaw 4: A is already covered by stronger papers

FineMoE's expert maps record gate probability distributions and use them in prefetching and eviction. HOBBIT and HybriMoE also make cache decisions depend on miss cost or impact. A new "probability x horizon x transfer cost" heuristic is unlikely to survive as a top-venue novelty claim.

Fix: demote A to a necessary scheduler component. The contribution should be the coupling between cache utility and miss recovery: the cache policy decides which misses are worth turning into speculative windows and which misses should simply block.

### Fatal flaw 5: real-trace predictor quality may kill B

Your synthetic 0.99 hit rate is irrelevant. If real Qwen/DeepSeek predictors sit at 0.34-0.39 hit rate, the proxy may issue many wrong downstream prefetches. Wrong prefetches consume PCIe bandwidth and evict useful experts, making misses worse.

Fix: first measure **post-miss downstream route stability** under the low-rank proxy. If the proxy cannot predict future exact experts after a miss, B dies even if the surrogate output has low MSE.

## 4. The single highest-value first experiment

Run a **real-trace miss-shadow replay** on Qwen1.5-MoE-A2.7B and DeepSeek-V2-Lite on the A800-80GB box.

Experiment:

1. Collect real hidden states, router logits, selected experts, and expert outputs on real prompts. Use at least LMSYS/ShareGPT-style chat prompts plus WikiText for long decode continuity.
2. Fit or load the LoRE surrogate on real target data. Do not use synthetic random weights for this experiment.
3. Replay decode with a constrained expert cache, pinned CPU expert storage, measured A800 PCIe H2D times, and exact target execution.
4. When the baseline policy has a cache miss at layer `l`, SPICE-B does **not** commit approximate logits. It runs the LoRE approximate path only to predict downstream exact expert needs and launch H2D prefetches for layers `l+1...L` while the missed true expert is loading.
5. When the true expert arrives, replay exact layers and measure whether downstream H2D waits were removed.

Primary metric:

`RecoveredStall = (Stall_A_only - Stall_A_plus_B_exact_replay) / Stall_A_only`

where `Stall_*` is critical-path H2D wait time, not total issued transfer time. Compare against the strongest A-only policy you can implement: FineMoE-style probability-guided cache/prefetch/eviction if possible; otherwise LRU, LFU, MoE-Infinity-style activation counts, and an oracle-reuse upper bound.

Go/no-go threshold:

**Go** only if exact-replay SPICE-B recovers at least **30% of critical-path H2D stall** and improves TPOT by at least **1.25x** over the strongest A-only baseline on both Qwen1.5-MoE-A2.7B and DeepSeek-V2-Lite at the same cache budget, while increasing total H2D traffic by at most **25%** and producing exactly matching target logits.

**No-go** if recovered stall is below **15%**, TPOT speedup is below **1.10x**, or total H2D traffic grows above **1.5x**. In that regime, the low-rank proxy is just a noisy prefetcher and the thesis collapses back into predictor work.

Why this experiment is first: it tests the actual non-incremental claim. Hidden MSE, route KL, and synthetic hit rate are secondary. The question is whether a miss can be converted into a useful scheduling window for downstream exact execution.

## 5. Required baselines + minimum credible evaluation

### Baselines that must be beaten

Lossless/offload baselines:

- On-demand expert loading: DeepSpeed/HF Accelerate/kTransformers-style demand paging with no prefetch.
- LRU/LFU expert cache under identical memory budget.
- Mixtral-Offloading-style layerwise speculative prefetch plus LRU.
- MoE-Infinity activation-aware expert cache/prefetch.
- ProMoE proactive caching.
- ExpertFlow predictive cache plus token scheduling if evaluating batch > 1.
- FineMoE/fMoE probability-map prefetch/cache/eviction. This is mandatory because it directly attacks A.
- HybriMoE if CPU execution is allowed or if comparing hybrid CPU/GPU scheduling.

Approximate/lossy baselines:

- HOBBIT mixed-precision miss recovery.
- Mixture of Cache-Conditional Experts / cache-prior rerouting.
- Quantized expert variants with and without MiLo-style low-rank compensators, if SPICE stores low-rank proxy weights.

Speculative baselines:

- User-listed SP-MoE, Pre-Attention Prediction, MoE-SpeQ, EARTH, and SpecBranch/Speculative MoE where code or faithful reimplementation is available.
- Oracle downstream prefetch: uses true future experts to show the maximum possible benefit from turning a miss into a prefetch window.
- Oracle correction-cost lower bound: assumes zero-cost exact correction after approximate pass. If even this is weak, B is dead.

### Minimum credible evaluation

Models:

- At minimum: Qwen1.5-MoE-A2.7B and DeepSeek-V2-Lite.
- Stronger: add Mixtral-8x7B or Phi-3.5-MoE to cover different top-k/expert-count regimes.

Workloads:

- LMSYS/ShareGPT-style chat prompts for serving realism.
- WikiText or similar long sequential text for cache locality.
- MMLU and GSM8K for quality sensitivity if any bounded-error or lossy mode is claimed.
- Azure or comparable serving trace lengths for online request distribution.

Hardware/system:

- Single A800-80GB with artificial expert-cache budgets despite the small models fitting in VRAM.
- Pinned CPU DRAM and, if claimed, NVMe tier.
- CUDA streams with measured H2D bandwidth/latency, not only Python simulation.
- Cache-budget sweep: at least 10%, 25%, 50%, 75% of selectable experts or equivalent GB budgets.
- Batch sizes 1, 2, and 4. If batch > 1 breaks, say so; top-venue reviewers will ask.

Metrics:

- TTFT, mean TPOT, p50/p95/p99 TPOT.
- Critical-path H2D stall, total H2D bytes, copy-engine utilization.
- Expert hit/miss rate, wrong-prefetch rate, evictions of soon-needed experts.
- Miss penalty distribution per layer and per distance from miss.
- Surrogate overhead, exact replay overhead, and memory overhead for proxy weights.
- Exact mode: max logit diff must be zero except floating-point tolerance from identical target execution.
- Bounded-error mode: logit error bound, top-token stability rate, perplexity, MMLU/GSM8K, and failure cases where routing flips.

Ablations:

- A only, B only, A+B.
- LRU vs probability utility eviction.
- Proxy downstream prefetch vs no proxy.
- Exact replay vs approximate commit.
- Oracle future experts vs LoRE proxy future experts.
- Cache budget and prefetch lookahead sensitivity.

Minimum bar for a systems venue:

- At least **1.25x TPOT improvement** over FineMoE-like A-only and HOBBIT-like approximate miss handling at the same memory budget.
- At least **30% critical-path stall recovery** with exact target logits.
- No hidden synthetic-only claims. Real model traces must be the headline.
- A full latency breakdown that shows where the speedup comes from. If the win is only higher hit rate, reviewers will classify it as another predictor/cache paper.

## 6. A sharper one-sentence thesis

**SPICE converts an expert-cache miss into a verified downstream prefetch window: a resident low-rank proxy speculates only to schedule future expert movement, while the original MoE is exact-replayed before token commit so latency improves without changing logits.**

This thesis is narrower and stronger than the current one. It stops pretending A is novel, avoids overclaiming cheap delta correction, and makes the hard falsifiable claim: missed expert latency can be recovered by useful exact downstream scheduling rather than merely reduced by a better predictor.
