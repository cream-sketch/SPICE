# Baseline Completeness Audit After Re-reading Comments

Source: `F:\ICCD\SPICE\comments.md`

## Reviewer Requirements

| Reviewer | Required comparison / experiment | Current status | Gap |
|---|---|---|---|
| R1 | Scalability when baseline PCIe utilization is greater than 20% | Completed for controlled harness | Added baseline-vs-SPICE Top-K stress run in `results\baseline_stress_20260528_113254`. Naive reaches 20.66--60.98% PCIe active; Pre-gated reaches 34.91--100% for K>=4. |
| R1/R2 | Energy or power gains | Partial | GPU power telemetry exists, but not paired baseline-vs-SPICE energy per token. Safe as telemetry only, not energy-gain claim. |
| R2 | Direct comparison to SP-MoE and MoE-SpeQ | Incomplete | Current result is proxy only. It must not be described as direct comparison. |
| R2 | Baselines Naive, LRU, MoE-Offloading, Pre-gated precisely explained | Mostly complete | We added same-harness baseline definitions and ran controlled comparison. Need final paper text to state cache size, expert size, Top-K, and transfer model. |
| R3 | Direct comparison to AdapMoE and ExpertFlow | Incomplete | Current result is proxy only. AdapMoE has public code; ExpertFlow paper describes components but no public code was found from the current search. |
| R3 | End-to-end accuracy / perplexity | Partial | Verified harness proves semantic equivalence; GPT-2 PPL smoke tests the evaluation path. We still lack a full DeepSeek/Qwen end-to-end PPL or task-accuracy table. |
| R3 | Online training overhead | Complete enough for text | Offline vs online overhead ablation shows online updates increase TPOT from 41.04 ms to 42.96 ms. |
| R3 | Top-K saturation | Completed for controlled harness | Added all-baseline Top-K stress table. SPICE is best through K=8; LRU/MoE-Offloading win at K=10 and K=12, which should be presented as a bandwidth-limit boundary. |

## What Must Not Be Overclaimed

- Do not say "we directly compare with SP-MoE, MoE-SpeQ, AdapMoE, and ExpertFlow" unless official code or faithful reimplementations are actually run.
- Do not call proxy rows "state-of-the-art comparison"; call them "controlled proxy variants" or move them to appendix.
- Do not claim energy reduction from the current telemetry. It is only power/timeline evidence.
- Do not claim full model perplexity preservation for DeepSeek/Qwen from the synthetic correctness harness.

## Highest-Value Missing Runs

1. **High-utilization stress with baselines**:
   Completed in `results\baseline_stress_20260528_113254` and summarized in
   `BASELINE_STRESS_RESULTS.md`.

2. **AdapMoE official/proxy split**:
   Because AdapMoE has public code, inspect whether it can run on the workstation. If its official setup requires Mixtral-specific assets or a different model, report it as "official code available but not directly compatible with our DeepSeek/Qwen offloading harness" and keep proxy rows clearly labeled.

3. **SP-MoE / MoE-SpeQ direct feasibility**:
   Search found arXiv papers but no confirmed public code in the current pass. If no code is available, use a paper-derived approximation and explicitly label it.

4. **Full PPL table**:
   Run on a real target MoE model if feasible. Otherwise, keep the correctness invariant and do not overstate empirical quality.
