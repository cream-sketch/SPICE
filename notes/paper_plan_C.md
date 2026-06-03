# Paper plan C (codex-recommended, honest systems paper) — 2026-06-02

## Thesis
"Offloaded MoE decoding is NOT primarily an eviction-policy problem; it is a VERIFIED RESOURCE-CONSTRAINED SCHEDULING problem under bandwidth, cache, and quality budgets."

## Contributions
1. CHARACTERIZATION of which intuitive levers FAIL (this is real value = our prune log):
   - within-token draft routing -> eviction: fails (cannot beat SpecMD-LS; reuse is cross-token).
   - lossless miss-shadow recovery at batch=1 saturated PCIe: no first-order win.
   - miss-drop POLICY choice (gate-weight / rank / top-p mass): no policy dominates cross-model; lever known.
2. POSITIVE INVARIANT: verified importance-aware miss DROP gives a real latency/PPL Pareto across Qwen + DeepSeek (drop low-importance on miss -> big stall cut for small PPL), but policy variants do not dominate -> the value is the Pareto+SLO, not a magic policy.
3. VERIFIED REPLAY/RUNTIME FRAMEWORK separating routing / cache / DMA / prefetch / miss-handling / quality effects (the eval methodology itself: online teacher-forced on-policy; deadline-aware DMA; equal-budget accounting).
4. EQUAL-BUDGET comparisons vs SOURCE-ONLY baselines (SpecMD, AdapMoE, HOBBIT, MoE-Infinity, FineMoE) -- not diagnostic wrappers (moe-baseline-integrity rule).
5. UNIFIED scheduler exposing the Pareto surface honestly (no overclaimed policy).

## Minimum publishable experiments (codex)
- Models: Qwen1.5-MoE-A2.7B + DeepSeek-V2-Lite (+1 more only if cheap).
- Source-only baseline runs for SpecMD/AdapMoE/HOBBIT/MoE-Infinity/FineMoE where possible; label env failures + diagnostic wrappers.
- Equal-budget ablations: cache size, prefetch budget, lookahead, drop SLO, chunk size, confidence threshold.
- Breakdown metrics: TPOT/TTFT, stall, H2D bytes, wrong-prefetch bytes, hit rate, eviction collisions, PPL + 1-2 downstream quality (MMLU/GSM8K).
- Replay-vs-live validation (timing/decision agreement).
- Hardware/workload shifts: bandwidth throttle + cache-size variation (batch>1 appendix).

## Gate A (bandit) status
A = conservative contextual bandit adaptive scheduler. codex prior: roofline-derivable -> decoration. Running kill-shot (scheduler_killshot.py): if analytic-roofline depth ~= per-regime-oracle across bw x cache -> bandit has nothing to learn -> A demoted to scheduler ablation. If roofline regret large -> A may have a story (then build bandit).
