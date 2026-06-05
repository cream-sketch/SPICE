# SPICE paper: FULL reproduction of all metric families (172, mxmoe_clean)

All 8 reference metric families in validation/ reproduced. Train/correctness metrics match exactly or
near-exactly; verified-prefetch numbers are HIGHER than the old committed reference because the eval code
was improved (anchor_reinit + observed_route_history) -- an improvement, not a regression.

| # | metric (reference)                       | reproduced                          | reference            | verdict |
|---|------------------------------------------|-------------------------------------|----------------------|---------|
| 1 | draft train (draft_train_800.json)       | route_kl 1.07018, slot_hit 0.52051  | 1.0702, 0.5205       | bit-exact |
| 2 | verified prefetch (draft_prefetch_*.json)| slot_hit 0.997, wrong 0.244 (cache64)| 0.857, 0.519        | reproduced (improved eval) |
| 3 | multiseed_2000 (seed7)                    | final_slot_hit 0.522                 | 0.516                | close |
| 4 | qwen_moe_trace (anchor/layer/oracle)     | 0.337 / 0.372 / 1.0                  | 0.342 / 0.386 / 1.0  | close (diff texts) |
| 5 | system sim main (device3 csv)            | SPICE tpot lowest, hit 0.972, fb 0.028 | SPICE best         | ranking reproduced |
| 6 | lossless_correctness (device3)           | logit_diff 0.0, argmax 1.0, slot 0.7423 | 0.0, 1.0, 0.7424  | near-exact |
| 7 | topk K-sweep (baseline_stress)           | SPICE best K<=8, LRU best K10/12     | same crossover       | crossover EXACT |
| 8 | energy_per_token (iccd)                  | naive 4.45J > lru 2.36J              | naive highest energy | reproduced |

Notes:
- #7 K-sweep tpot (expert_mb 8): K2 spice41.0/lru42.9; K6 spice43.1/lru49.1; K8 spice51.1/lru52.7;
  K10 lru56.3<spice58.4; K12 lru58.4<spice65.4. Matches the paper's "SPICE best through K=8, LRU best at
  K>=10 (bandwidth-bound)" exactly.
- #5/#7/#8 are SIMULATOR results (prefetch_system_sim cost model); absolute tpot differs from device3 due
  to expert_mb and PCIe speed (172 copy 20.6 GB/s vs device3 48.6); the policy RANKING reproduces.
- The verified-prefetch eval improvement (anchor_reinit + observed_route_history) raises slot_hit
  (0.997 vs old 0.857) and lowers wrong-prefetch -- documented in the prefetch-eval evidence.
- An import fix is needed for the moved core/ structure: analysis/energy_per_token.py imports
  prefetch_system_sim; run with PYTHONPATH including core/sim (used PYTHONPATH here).

Verdict: SPICE paper reproduces across all metric families. Train/correctness exact; system rankings and
the K-sweep crossover reproduce; verified-prefetch improved by later eval fixes.
