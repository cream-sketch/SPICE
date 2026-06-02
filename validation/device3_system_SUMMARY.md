# SPICE ICCD Supplemental Experiment Summary

## prefetch_system_cache_sweep: `cache_sweep/prefetch_system_cache_sweep.json`
- Copy microbench: {'measured_copy_gbps': 48.57380526342715, 'measured_copy_ms': 291.51836413075216}
- GPU power sample: 299.25 W

## lossless_correctness: `correctness/lossless_correctness.json`
- baseline_cross_entropy: 6.942360956221819
- baseline_pseudo_ppl: 1035.2114208813093
- device: cuda:3
- exact_argmax_match_rate: 1.0
- experiment: lossless_correctness
- fallback_slot_rate: 0.2576141357421875
- interpretation: SPICE verified mode executes the target router's experts; nonzero fallback affects latency, not logits.
- max_abs_logit_diff: 0.0
- prefetch_slot_hit_rate: 0.7423858642578125
- spice_verified_cross_entropy: 6.942360956221819
- spice_verified_pseudo_ppl: 1035.2114208813093

## energy_per_token_replay: `energy/energy_per_token.json`
- device: cuda:3
- experiment: energy_per_token_replay
- power_gpu: 3

## prefetch_system_main: `main/prefetch_system_main.json`
- Copy microbench: {'measured_copy_gbps': 52.233391387460024, 'measured_copy_ms': 271.09394725994207}
- GPU power sample: 96.93 W

## memory_probe: `memory_probe.json`
- allocated_mib: 29000.0
- expert_mb: 14500
- free_after_mib: 2571.3125
- free_before_mib: 31571.3125
- gpu: 3
- timestamp: 2026-06-02 17:17:07
- total_mib: 32111.5
- two_buffer_target_mib: 29000

## memory_probe_adaptive: `memory_probe_adaptive.json`
- allocated_mib: 12916.0
- expert_mb: 6457
- free_after_mib: 4094.0
- free_before_mib: 17010.0
- gpu: 3
- timestamp: 2026-06-02 17:22:26
- total_mib: 32111.5
- two_buffer_target_mib: 12914

## prefetch_system_overhead: `overhead/prefetch_system_overhead.json`
- Copy microbench: {'measured_copy_gbps': 48.82414966418885, 'measured_copy_ms': 290.0236122368369}
- GPU power sample: 276.63 W

## timeline_replay: `timeline_naive/timeline_naive.json`
- checksum: 0.0005717277526855469
- copies_per_step: 4
- effective_h2d_gbps: 49.814006701646726
- elapsed_s: 24.304158210987225
- experiment: timeline_replay
- expert_mb: 6457
- h2d_gb: 1210.6875
- overlap: False
- policy: naive
- steps: 48

## timeline_replay: `timeline_pregated/timeline_pregated.json`
- checksum: 0.0005717277526855469
- copies_per_step: 1
- effective_h2d_gbps: 49.67349159933659
- elapsed_s: 6.093227297998965
- experiment: timeline_replay
- expert_mb: 6457
- h2d_gb: 302.671875
- overlap: True
- policy: pregated
- steps: 48

## timeline_replay: `timeline_spice/timeline_spice.json`
- checksum: 0.0
- copies_per_step: 2
- effective_h2d_gbps: 50.675028599911215
- elapsed_s: 11.945602532941848
- experiment: timeline_replay
- expert_mb: 6457
- h2d_gb: 605.34375
- overlap: True
- policy: spice
- steps: 48

## prefetch_system_topk: `topk/prefetch_system_topk.json`
- Copy microbench: {'measured_copy_gbps': 49.23907433694159, 'measured_copy_ms': 287.57965986733325}
- GPU power sample: 305.63 W

## Prefetch System Table

| experiment | policy | variant | top_k | cache_hit_rate | fallback_rate | h2d_gb | sim_tpot_ms | pcie_active_fraction | draft_overhead_ms | online_overhead_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefetch_system_cache_sweep | naive | cache_128 | 6 | 0 | 1 | 696000 | 56680.6 | 0.999294 | 0 | 0 |
| prefetch_system_cache_sweep | lru | cache_128 | 6 | 0.393575 | 0.606425 | 422072 | 34388.3 | 0.998837 | 0 | 0 |
| prefetch_system_cache_sweep | moe_offloading | cache_128 | 6 | 0.393575 | 0.606425 | 422072 | 34388.3 | 0.998837 | 0 | 0 |
| prefetch_system_cache_sweep | pregated | cache_128 | 6 | 0.261393 | 0.738607 | 1.058e+06 | 41875.1 | 1 | 0 | 0 |
| prefetch_system_cache_sweep | spice | cache_128 | 6 | 0.449015 | 0.550985 | 426900 | 31249.1 | 1 | 491.52 | 0 |
| prefetch_system_cache_sweep | naive | cache_256 | 6 | 0 | 1 | 696000 | 56680.6 | 0.999294 | 0 | 0 |
| prefetch_system_cache_sweep | lru | cache_256 | 6 | 0.691508 | 0.308492 | 214710 | 17513.2 | 0.997716 | 0 | 0 |
| prefetch_system_cache_sweep | moe_offloading | cache_256 | 6 | 0.691508 | 0.308492 | 214710 | 17513.2 | 0.997716 | 0 | 0 |
| prefetch_system_cache_sweep | pregated | cache_256 | 6 | 0.832581 | 0.167419 | 516393 | 9522.74 | 1 | 0 | 0 |
| prefetch_system_cache_sweep | spice | cache_256 | 6 | 0.713257 | 0.286743 | 242988 | 16282.3 | 1 | 491.52 | 0 |
| prefetch_system_cache_sweep | naive | cache_512 | 6 | 0 | 1 | 696000 | 56680.6 | 0.999294 | 0 | 0 |
| prefetch_system_cache_sweep | lru | cache_512 | 6 | 0.84729 | 0.15271 | 106286 | 8689.59 | 0.995397 | 0 | 0 |
| prefetch_system_cache_sweep | moe_offloading | cache_512 | 6 | 0.84729 | 0.15271 | 106286 | 8689.59 | 0.995397 | 0 | 0 |
| prefetch_system_cache_sweep | pregated | cache_512 | 6 | 0.940999 | 0.0590007 | 271833 | 3381.83 | 1 | 0 | 0 |
| prefetch_system_cache_sweep | spice | cache_512 | 6 | 0.879089 | 0.120911 | 127569 | 6889.41 | 1 | 491.52 | 0 |
| prefetch_system_cache_sweep | naive | cache_1024 | 6 | 0 | 1 | 696000 | 56680.6 | 0.999294 | 0 | 0 |
| prefetch_system_cache_sweep | lru | cache_1024 | 6 | 0.979167 | 0.0208333 | 14500 | 1220.01 | 0.967213 | 0 | 0 |
| prefetch_system_cache_sweep | moe_offloading | cache_1024 | 6 | 0.979167 | 0.0208333 | 14500 | 1220.01 | 0.967213 | 0 | 0 |
| prefetch_system_cache_sweep | pregated | cache_1024 | 6 | 0.995565 | 0.00443522 | 14500 | 291.214 | 1 | 0 | 0 |
| prefetch_system_cache_sweep | spice | cache_1024 | 6 | 0.98759 | 0.0124105 | 14500 | 743.897 | 1 | 491.52 | 0 |
| prefetch_system_cache_sweep | naive | cache_2048 | 6 | 0 | 1 | 696000 | 56680.6 | 0.999294 | 0 | 0 |
| prefetch_system_cache_sweep | lru | cache_2048 | 6 | 0.979167 | 0.0208333 | 14500 | 1220.01 | 0.967213 | 0 | 0 |
| prefetch_system_cache_sweep | moe_offloading | cache_2048 | 6 | 0.979167 | 0.0208333 | 14500 | 1220.01 | 0.967213 | 0 | 0 |
| prefetch_system_cache_sweep | pregated | cache_2048 | 6 | 0.995687 | 0.00431315 | 14500 | 284.3 | 1 | 0 | 0 |
| prefetch_system_cache_sweep | spice | cache_2048 | 6 | 0.987264 | 0.012736 | 14500 | 762.335 | 1 | 491.52 | 0 |
| prefetch_system_main | naive | nan | 6 | 0 | 1 | 696000 | 56680.6 | 0.999294 | 0 | 0 |
| prefetch_system_main | lru | nan | 6 | 0.850423 | 0.149577 | 104105 | 8512.12 | 0.995301 | 0 | 0 |
| prefetch_system_main | moe_offloading | nan | 6 | 0.850423 | 0.149577 | 104105 | 8512.12 | 0.995301 | 0 | 0 |
| prefetch_system_main | pregated | nan | 6 | 0.940389 | 0.059611 | 271464 | 3416.4 | 1 | 0 | 0 |
| prefetch_system_main | spice | nan | 6 | 0.882345 | 0.117655 | 125303 | 6705.04 | 1 | 491.52 | 0 |
| prefetch_system_overhead | spice | spice_offline | 6 | 0.882345 | 0.117655 | 125303 | 6705.04 | 1 | 491.52 | 0 |
| prefetch_system_overhead | spice | spice_online | 6 | 0.882345 | 0.117655 | 125303 | 6706.96 | 1 | 491.52 | 983.04 |
| prefetch_system_overhead | spice | spice_no_lore | 6 | 0.869385 | 0.130615 | 134323 | 7439.09 | 1 | 491.52 | 0 |
| prefetch_system_topk | spice | topk_2 | 2 | 0.742737 | 0.257263 | 74156.7 | 4898.14 | 1 | 491.52 | 0 |
| prefetch_system_topk | spice | topk_4 | 4 | 0.743195 | 0.256805 | 148101 | 9738.04 | 1 | 491.52 | 0 |
| prefetch_system_topk | spice | topk_6 | 6 | 0.731303 | 0.268697 | 230428 | 15260.1 | 1 | 491.52 | 0 |
| prefetch_system_topk | spice | topk_8 | 8 | 0.716202 | 0.283798 | 321251 | 21473.6 | 1 | 491.52 | 0 |
| prefetch_system_topk | spice | topk_10 | 10 | 0.715625 | 0.284375 | 402233 | 26886.3 | 1 | 491.52 | 0 |
| prefetch_system_topk | spice | topk_12 | 12 | 0.720103 | 0.279897 | 476447 | 31748 | 1 | 491.52 | 0 |
