Strategic verdict needed. Be brutally honest, cite mechanisms, output to stdout. We have rigorously DERIVED direction from experiments (signal->decision->utility->novelty). Here is the full pruning log on real Qwen1.5-MoE-A2.7B + DeepSeek-V2-Lite (A800, on-policy teacher-forced):

PRUNED (experimentally):
1. draft within-token routing -> EVICTION: fails, cannot beat SpecMD Least-Stale; resident reuse is cross-token, draft predicts within-token. DEAD.
2. miss-shadow rollout -> downstream recovery (lossless): no first-order win at batch=1 single saturated PCIe channel (you confirmed). DEAD as main; appendix/batch>1 only.
3. miss-handling DROP policy axis: gate-weight-threshold vs fixed-rank vs top-p-cumulative-mass (entropy-adaptive). On Qwen the three are within noise (no dominator); gate>rank held only on Qwen, not DeepSeek (and even that is ~noise once mass is added). The DROP LEVER is real (Pareto: drop low-importance experts on miss -> big stall cut for small PPL, e.g. Qwen drop ~25% -> +1.5% PPL) but the POLICY is saturated and the lever itself is known (SpecMD drop-by-rank, AdapMoE sensitivity, Skliar). => lossy miss-handling on the policy axis = INCREMENTAL.

VALIDATED but INCREMENTAL: verified importance-aware miss-drop Pareto (real, on-policy, cross-model) — but no policy beats the others, so "our drop policy is better" is not a claim.

UNTESTED remaining bets:
A. Conservative contextual bandit as the ADAPTIVE SCHEDULER: signal=(bw, cache occupancy, draft confidence dist, DMA queue, layer), decision=knobs(lookahead depth, confidence threshold, prefetch budget, drop SLO, chunk size), utility=regret across hardware/workload SHIFTS without retuning. Correctness via verified replay. Risk: optimal policy may be roofline-derivable -> a tuned static scheduler matches it -> bandit is decoration.
B. Calibrated draft-confidence -> speculative-prefetch ADMISSION (only prefetch high-confidence predictions to bound wrong-prefetch H2D waste under a bandwidth budget). Near SP-MoE cutoff / MoE-SpeQ governor / FineMoE probability.
C. Reframe as characterization + unified verified resource-constrained scheduler (systems paper, not a single novel mechanism), with the validated Pareto + clean equal-budget ablations vs SpecMD/AdapMoE/HOBBIT/MoE-Infinity.

QUESTIONS:
1. Given the pruning, is there ANY non-incremental (Signal,Decision) coupling left in the single-GPU offloaded-MoE miss/eviction/prefetch space that we have NOT tested? Name it concretely with the mechanism and why it would be non-incremental vs the named prior art. If none, say so plainly.
2. Is bet A (bandit adaptive scheduler) likely non-incremental, or is the optimal scheduler roofline-derivable (making it decoration)? What single experiment decides this fastest?
3. Is the honest outcome that this whole line is a solid SYSTEMS paper (bet C) rather than a non-incremental breakthrough? If so, what is the strongest defensible framing and the minimum experiments to make bet C publishable at a systems venue?
4. If you had to bet on ONE next action to maximize the chance of a genuinely non-incremental result, what is it? Or do you recommend consolidating bet C?
Do not be agreeable. If the answer is "consolidate, it's incremental", say it.
