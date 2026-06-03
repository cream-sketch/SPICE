# ARCHIVE_INDEX — explore/ (archived experiments)

One line per file moved into `explore/` during the Phase 0 refactor. Verdict tags:
DEAD = hypothesis killed; CONTESTED = result depends on assumptions / superseded / not
clearly non-incremental; UTILITY = trace-collection / observation / analysis helper (no verdict).

Cross-references: `notes/prior_art_gap.md`, `notes/first_principles_characterization.md`,
and the specific `notes/*verdict*.md` / `notes/codex_*.md` named below.

## Observation / characterization (UTILITY)
- `obs_routing_structure.py` — first-principles routing observation battery (autocorr, modal, lag) → `notes/first_principles_characterization.md`.
- `obs_expert_contribution.py` — expert OUTPUT contribution ||gate*E(x)|| vs gate weight; routed experts perturb hidden ~10%, shared dominates.
- `obs_cross_expert_structure.py` — cross-expert weight basis test; experts independent, no shared basis (no compression). UTILITY.
- `expert_lowrank_spectrum.py` — exp0a: per-expert weight spectrum; experts near full-rank, rank64 ~18% energy → no low-rank compression. CONTESTED/UTILITY, `notes/first_principles_characterization.md`.
- `expert_importance_dist.py` — per-rank gate-weight distribution of real Qwen routing. UTILITY.
- `cross_token_info.py` — full router distribution as cross-token information source. UTILITY.
- `token_conditional_analysis.py` — CORRECTED token→expert recall (audit fix): token-id predicts ~58-60% of top-k → routing is token-determined. UTILITY, `notes/REVERSAL_token_conditional.md`.

## Routing data collection (UTILITY)
- `ds_collect_routing.py` — collect real DeepSeek expert IDs + 64-dim scores for structure analysis. UTILITY.

## Drop / importance-aware quality ablations (CONTESTED — validated Pareto but not non-incremental)
- `drop_quality_ppl.py` — Qwen: PPL cost of importance-based expert dropping (the validated lossy lever). CONTESTED, `notes/drop_policy_comparison.md`.
- `qwen_gate_vs_rank.py` — Qwen gate/rank/mass drop quality ablation (full-seq teacher-forced). CONTESTED, `notes/gate_vs_rank_ablation.md`, `notes/drop_policy_comparison.md`.
- `deepseek_drop_quality.py` — DeepSeek: PPL cost of importance-based dropping (generalization). CONTESTED, `notes/drop_policy_comparison.md`.
- `ds_gate_vs_rank.py` — DeepSeek gate-vs-rank quality ablation; uses REAL topk_weight (fixes double-softmax artifact). CONTESTED, `notes/drop_policy_comparison.md`.

## Miss-admission online (CONTESTED — superseded by miss/)
- `miss_admission_online.py` — Qwen online teacher-forced verified miss-admission (core methodology prototype). CONTESTED, `notes/gate_vs_rank_ablation.md`.
- `deepseek_miss_admission.py` — DeepSeek generalization of the online miss-admission harness. CONTESTED.

## Eviction / cache headroom (DEAD / CONTESTED)
- `eval_real_trace_eviction.py` — exp1: eviction-policy headroom on real traces; small headroom over LRU. CONTESTED, `notes/exp1_eviction_verdict.md`.
- `eval_real_trace_cache_sweep.py` — exp1: real-trace cache-budget x bandwidth sweep. UTILITY/CONTESTED, `notes/exp1_eviction_verdict.md`.
- `eval_forecast_eviction.py` — exp2: forecast-driven eviction vs LRU/LS/oracle; honest negative, does not beat SpecMD-LS (had MRU tie-break bug). DEAD, `notes/codex_pivot_verdict.md`, `notes/codex_plan_forecast_eviction.md`.

## Scheduler / adaptivity (DEAD)
- `scheduler_killshot.py` — is prefetch-depth scheduler roofline-derivable / bandit worthwhile? Small constant depth dominates; adaptive/bandit DEAD (roofline +10% harmful). DEAD, `notes/killshot_verdict.md`.
- `microbatch_multiplex.py` — kill experiment for uniformity→statistical-multiplexing idea. DEAD, `notes/codex_strategic_verdict.md`.

## Candidate B — selective / shared-only routed execution (DEAD)
- `shared_only_argmax.py` — fraction of next-token argmax unchanged when ALL routed experts dropped: only 4.7% agree → cannot skip routed. DEAD, `notes/candidate_B_dead.md`.

## SPICE-X / SPICE-REC — target-conditioned cache value & admission (CONTESTED)
- `spice_x_eviction_value.py` — does target-conditioned cache VALUE beat Least-Stale on hit-rate? (shared by the SPICE-X sims). CONTESTED, `notes/codex_audit_spicex_result.md`, `notes/design_spice_x_value.md`.
- `spice_x_admission.py` — SPICE-X 4-policy decisive admission (demand-fetch, isolate admission lever). CONTESTED, `notes/codex_audit_admission.md`.
- `spice_rec_sim.py` — SPICE-REC faithful discrete-event timeline simulator (codex-mandated over scalar max()). CONTESTED, `notes/codex_audit_recsim.md`, `notes/design_spice_rec.md`.

## Matrix-granular residency (CONTESTED)
- `spice_matrix_microbench.py` — matrix-granular exact residency microbench on A800 (batch=1, exact precision). CONTESTED.
- `spice_matrix_sim.py` — matrix-granular vs whole-expert HBM caching simulator at matched budget. CONTESTED.

## Partial CPU/GPU split & speculative reuse (CONTESTED — A=Fiddler-incremental)
- `spice_partial_split_microbench.py` — partial-expert CPU/GPU split execution microbench (exact, batch=1). CONTESTED, `notes/codex_betterway_verdict.md`.
- `spice_partial_split_fair.py` — fair partial-split microbench addressing codex fairness critiques. CONTESTED.
- `spice_split_window_microbench.py` — windowed partial-split (non-resident-segment CPU compute) microbench. CONTESTED.
- `spice_spec_reuse.py` — speculative-microbatch expert-reuse cheap kill-test. CONTESTED, `notes/codex_strategic_verdict.md`.
- `spice_spec_runtime.py` — real speculative-microbatch + CPU-grouped-Fiddler runtime (codex decisive). CONTESTED.
- `spice_spec_runtime_v2.py` — same runtime, codex 4-fix revision. CONTESTED.
- `spice_grouped_cpu_kernel.py` — grouped CPU MoE-serve kernel microbench (strongest positive exact batch=1 lever per gemini). CONTESTED.
- `spice_zerocopy_microbench.py` — GPU zero-copy remote-expert GEMV microbench (hardware path probe). CONTESTED.
- `spice_hx_components.py` — SPICE-HX kill-shot component calibration (codex decisive-experiment step D). CONTESTED.

## Token-table prefetch (UTILITY)
- `token_table_replay.py` — does a token→expert table beat LFU/LS for offloaded-MoE prefetch? UTILITY/CONTESTED, `notes/REVERSAL_token_conditional.md`.
