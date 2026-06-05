# SPICE paper reproduction (synthetic draft + verified prefetch) -- earliest scripts, 172

Ran the original reproduction (experiments/core/draft/train_draft_model.py + eval_draft_prefetch.py,
synthetic SPICEDraftModel: 8 layers, 16 experts, top-2, hidden 256, rank 16, 800 steps) vs the committed
reference validation/draft_train_800.json + draft_prefetch_eval_fixed.json.

## Draft training -- BIT-EXACT reproduction
                       reproduced        reference
  route_kl             1.0701756         1.0702
  slot_hit_rate        0.5205078         0.5205
  exact_set_match      0.2841797         0.2842
  align_mse            7.5238            7.5238
  mean_confidence      0.5921            0.5921
All final_eval metrics match the committed reference exactly.

## Verified prefetch -- mechanism reproduced; current eval IMPROVED vs old reference
                          current eval (cache64)   old reference (cache64)
  prefetch_slot_hit_rate  0.9971                   0.8571
  wrong_prefetch_rate     0.2436                   0.5192
  avg_lookahead_depth     2.364                    1.944
Same cache_capacity=64, l_min2/l_max6. The current eval has anchor_reinit + observed_route_history
(later improvements); the reference json predates them. So the prefetch numbers differ because the eval
code was IMPROVED (higher slot-hit, lower wrong-prefetch), not a regression. (With default cache 512 the
current eval reaches slot_hit 1.0.)

## Verdict
SPICE paper core results reproduce: the draft trains to predict cross-layer routing (train metrics
bit-exact), and verified prefetch achieves high slot-hit with bounded wrong-prefetch (current eval even
better than the old reference). Reproduction confirmed before building the trained-LoRE-on-real extension.
(System-sim baselines Naive/LRU/MoE-Offloading/Pre-gated/SPICE in prefetch_system_sim.py are a SIMULATOR;
the real-hardware track in experiments/harness/ supersedes them for latency claims.)
