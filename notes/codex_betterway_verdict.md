**Bottom Line**
A is the stronger engineering path, but neither A nor B is genuinely non-incremental as stated.

A is basically Fiddler on your fine-grained Qwen/DeepSeek regime, with SPICE used as a placement/prefetch hint. B is a more interesting hypothesis, but it collapses into expert skipping / early exit / self-speculative decoding unless you have a cheap, calibrated certificate for “routed experts cannot change the next token.”

**Candidate A: CPU-Compute On Miss**
Verdict: **not non-incremental.**

Closest prior art: [Fiddler](https://arxiv.org/abs/2402.07033) already states the core move: copy activations to CPU, compute experts there, return activations, because moving weights over PCIe is worse at batch=1. [HybriMoE](https://arxiv.org/abs/2504.05897) adds dynamic CPU/GPU scheduling, prefetch, and cache management. [KTransformers](https://ktransformers.net/en/docs/optimization-techniques/expert-placement) explicitly supports CPU MoE expert compute, GPU expert placement, dynamic updates, and deferred experts; its SOSP paper also overlaps CPU/GPU MoE work via Expert Deferral.

What SPICE adds: a better within-token lookahead signal for “which experts should be hot on GPU” and maybe a cleaner deadline-aware CPU/GPU scheduler. That is useful, but it is not a new principle. The pitch “Fiddler becomes much stronger on tiny fine-grained experts because CPU DRAM wins harder than PCIe” is a regime result, not a mechanism.

Fatal flaw: exact decode has a layer dependency. CPU routed output must be merged before the next layer can run. You can overlap CPU routed experts with same-layer GPU shared/resident work, and across tokens/batches, but “hide it behind downstream GPU attention” is not exact unless you change semantics, like deferral/staleness.

Decisive experiment: **exact microbenchmark plus end-to-end pipeline.**
Measure Qwen1.5-MoE and DeepSeek-V2-Lite routed expert forward at batch=1 on CPU, including D2H/H2D activation transfer, threadpool dispatch, NUMA placement, dtype conversion, and merge. Compare against pinned H2D weight fetch for one 17MB expert. Then run TPOT for:

`SPICE fetch-all`, `A: CPU-compute misses`, `Fiddler-style source-only`, `KTransformers/HybriMoE-style source-only`.

Go if CPU miss handling is at least **2x faster than fetch** exposed latency and end-to-end TPOT is **>1.5x over SPICE fetch-all** and **>1.2x over best hybrid CPU baseline** on both models, exact logits. No-go if CPU kernels/threading push expert latency near fetch latency, or if overlap is mostly imaginary.

**Candidate B: Shared-Only / Uncertainty-Gated Routed Execution**
Verdict: **not non-incremental as stated; higher upside, higher chance of dying.**

Closest prior art: [DeepSeekMoE](https://arxiv.org/abs/2401.06066) already formalizes shared experts as common-knowledge carriers. [AdapMoE](https://arxiv.org/abs/2408.10284) dynamically reduces active experts. [LExI](https://arxiv.org/abs/2509.02753) varies active expert count per layer. [LayerSkip](https://arxiv.org/abs/2404.16710) and [Draft & Verify](https://huggingface.co/papers/2309.08168) cover early-exit/self-speculative execution. [SP-MoE](https://arxiv.org/abs/2510.10302) and [MoE-SpeQ](https://arxiv.org/abs/2511.14102) already combine speculation with MoE offloading/prefetch.

The only possible novelty is a **cheap routed-necessity certificate**: from shared-only logits/hidden state, predict with very low false-skip rate whether routed experts can change the greedy token. Without that, B is just lossy expert skipping or self-speculative decoding.

Fatal flaw: “fetch routed only when routed would flip argmax” is oracle language. You do not know that without computing routed experts. Logit margins are often small; a 10% hidden perturbation can still flip hard tokens. The +37% PPL for drop-to-rank-1 is not a direct argmax-flip measurement, but it is a warning: routed experts may mostly affect probability mass and hard decisions, exactly where quality matters.

Decisive experiment: **drop-all-routed argmax agreement plus gate separability.**
Run full teacher-forced model and shared+attention-only model on the same contexts. Measure:

`argmax_agree = P[argmax(full logits) == argmax(shared-only logits)]`

Also measure whether shared-only top1/top2 margin predicts flips.

Go only if agreement is **>=90-95%** on both Qwen and DeepSeek, and a margin/uncertainty gate can skip **>=70-80%** of routed work with false-skip flips below **0.1-0.5%**. No-go if agreement is below **80%**, or if flips are not cleanly separable by a cheap signal. The existing `experiments/shared_only_argmax.py` is exactly the cheap first cut.

**Pick**
Pick **A** if you need something that can become a working system. It is exact, measurable, and likely to improve TPOT if CPU expert latency is really ~0.2ms. But do not sell it as non-incremental.

B is the only one with real conceptual upside, but it needs a miracle-grade gate. If the argmax agreement experiment fails, kill it immediately.

**Better Third Idea**
If both must be non-incremental, pivot away from single-stream miss handling. The measured facts say single-token expert access is uniform, memoryless, and uncacheable. That is bad for caches but good for **cross-request expert multiplexing**.

Third idea: layer-synchronous decode batching where many active streams at the same layer share one expert transfer/CPU compute per unique expert. Uniform access becomes coupon-collector amortization instead of cache failure. Exact logits, no dropping, no precision change.

Decisive experiment: replay 8/16/32/64 concurrent decode streams, group by layer and expert, fetch or CPU-compute each unique expert once, then scatter outputs. Go if batch 16-32 gives **>2x bytes/token reduction and >1.5x TPOT** under realistic latency SLO. No-go if queueing delay erases the gain.

For batch=1 local decode, the brutal answer is: A is the practical path, B is a cheap hypothesis test, and the genuinely non-incremental story may be the negative characterization itself: caching/prediction are structurally exhausted for these models.
**Bottom Line**
A is the stronger engineering path, but neither A nor B is genuinely non-incremental as stated.

A is basically Fiddler on your fine-grained Qwen/DeepSeek regime, with SPICE used as a placement/prefetch hint. B is a more interesting hypothesis, but it collapses into expert skipping / early exit / self-speculative decoding unless you have a cheap, calibrated certificate for “routed experts cannot change the next token.”

**Candidate A: CPU-Compute On Miss**
Verdict: **not non-incremental.**

Closest prior art: [Fiddler](https://arxiv.org/abs/2402.07033) already states the core move: copy activations to CPU, compute experts there, return activations, because moving weights over PCIe is worse at batch=1. [HybriMoE](https://arxiv.org/abs/2504.05897) adds dynamic CPU/GPU scheduling, prefetch, and cache management. [KTransformers](https://ktransformers.net/en/docs/optimization-techniques/expert-placement) explicitly supports CPU MoE expert compute, GPU expert placement, dynamic updates, and deferred experts; its SOSP paper also overlaps CPU/GPU MoE work via Expert Deferral.

What SPICE adds: a better within-token lookahead signal for “which experts should be hot on GPU” and maybe a cleaner deadline-aware CPU/GPU scheduler. That is useful, but it is not a new principle. The pitch “Fiddler becomes much stronger on tiny fine-grained experts because CPU DRAM wins harder than PCIe” is a regime result, not a mechanism.

Fatal flaw: exact decode has a layer dependency. CPU routed output must be merged before the next layer can run. You can overlap CPU routed experts with same-layer GPU shared/resident work, and across tokens/batches, but “hide it behind downstream GPU attention” is not exact unless you change semantics, like deferral/staleness.

Decisive experiment: **exact microbenchmark plus end-to-end pipeline.**
Measure Qwen1.5-MoE and DeepSeek-V2-Lite routed expert forward at batch=1 on CPU, including D2H/H2D activation transfer, threadpool dispatch, NUMA placement, dtype conversion, and merge. Compare against pinned H2D weight fetch for one 17MB expert. Then run TPOT for:

`SPICE fetch-all`, `A: CPU-compute misses`, `Fiddler-style source-only`, `KTransformers/HybriMoE-style source-only`.
