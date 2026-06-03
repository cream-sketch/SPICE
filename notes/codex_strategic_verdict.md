**Verdict**

Consolidate bet C. Inside single-GPU offloaded-MoE miss/eviction/prefetch, I do **not** see a remaining clean non-incremental `(Signal, Decision)` coupling after your pruning. The space has been mostly reduced to resource allocation under uncertainty, not a new mechanism.

B is incremental. A is worth one fast kill-shot experiment, but my prior is that it collapses to a roofline/threshold scheduler.

**1. Any non-incremental coupling left?**

No, not within the boundary you defined.

The remaining meaningful couplings are already occupied:

- `future expert signal -> prefetch/evict`: MoE-Infinity traces activation sparsity to guide replacement/prefetching, and SpecMD’s Least-Stale explicitly exploits predictable expert-access structure rather than LRU/LFU locality. ([arxiv.org](https://arxiv.org/abs/2401.14361)) ([arxiv.org](https://arxiv.org/abs/2602.03921))
- `cache state -> routing decision`: Skliar et al. already do cache-aware routing to improve expert reuse/locality on memory-constrained devices. ([arxiv.org](https://arxiv.org/abs/2412.00099))
- `miss importance -> approximate service`: HOBBIT replaces less-critical cache-miss experts with low-precision versions and adds token/layer/sequence-level policies. ([arxiv.org](https://arxiv.org/abs/2411.01433))
- `semantic/trajectory signal -> cache/prefetch/offload`: FineMoE uses fine-grained expert-selection patterns plus prompt semantic hints. ([arxiv.org](https://arxiv.org/abs/2502.05370))
- `draft/speculation signal -> expert prefetch`: SP-MoE and MoE-SpeQ already cover SD-aware speculative prefetch, cutoff/governor policies, and roofline-guided speculation. ([arxiv.org](https://arxiv.org/abs/2510.10302)) ([arxiv.org](https://arxiv.org/abs/2511.14102))
- `sensitivity/routing score -> fewer activations`: AdapMoE adjusts activated expert count by sensitivity and combines that with prefetch/cache management. ([arxiv.org](https://arxiv.org/abs/2408.10284))

The only “new” thing left would be a better integration of these knobs with verified replay and equal-budget accounting. That is a systems contribution, not a new primitive.

**2. Is A non-incremental?**

Likely not. In batch=1, single saturated PCIe, the optimal policy is close to:

`expected stall saved - expected wrong-prefetch bytes - cache pollution - drop quality cost`

That is a roofline/threshold controller. If the state already includes bandwidth, DMA queue, occupancy, layer, confidence distribution, and cache state, a good analytic policy can choose:

- prefetch if `p_hit_future * stall_saved > wrong_bytes / BW + pollution_cost`
- increase lookahead only until overlap slack or bandwidth budget is exhausted
- drop experts only when `stall_saved / quality_loss` clears the SLO threshold
- tune chunk size to saturate DMA without queueing the critical path

A bandit becomes non-incremental only if the above model is materially wrong under shifts.

Fastest deciding experiment:

Run trace/replay with abrupt regime shifts and compare four policies:

1. train-regime tuned static scheduler
2. per-regime oracle static grid
3. analytic roofline scheduler
4. contextual bandit

Shifts: PCIe bandwidth throttling, cache budget, prompt entropy/domain, decode length, batch 1 vs 2/4, artificial DMA contention.

Decision rule: if bandit beats analytic roofline and per-regime static by `>10-15% TPOT` or `>30% regret reduction` at the same PPL/SLO across unseen shifts, A has a real story. If it is within `3-5%`, kill A as decoration.

**3. Is C the honest paper?**

Yes. Strongest defensible framing:

“Offloaded MoE decoding is not primarily an eviction-policy problem; it is a verified resource-constrained scheduling problem under bandwidth, cache, and quality budgets.”

Contributions should be:

- A rigorous characterization of which intuitive levers fail: within-token draft routing, lossless miss-shadow at batch=1 PCIe saturation, and specific miss-drop policy choice.
- A positive invariant: importance-aware miss dropping gives a real latency/PPL Pareto across Qwen and DeepSeek, but policy variants do not dominate.
- A verified replay/runtime framework that separates routing, cache, DMA, prefetch, miss handling, and quality effects.
