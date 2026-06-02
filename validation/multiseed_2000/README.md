# SPICE Draft Multiseed Validation (2000 steps)

Four seeds were trained in parallel on workstation1 GPUs 0-3. Checkpoints are not committed.

| GPU | Seed | Final Slot Hit | After Online Slot Hit | Prefetch Hit | Fallback | Wrong Prefetch | Lookahead | Sim TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 7 | 0.516113 | 0.534424 | 0.856873 | 0.143127 | 0.516917 | 1.942871 | 44.334574 |
| 1 | 23 | 0.510986 | 0.515137 | 0.860840 | 0.139160 | 0.515347 | 1.922852 | 43.673359 |
| 2 | 31 | 0.500977 | 0.503418 | 0.857544 | 0.142456 | 0.522249 | 1.932617 | 44.222676 |
| 3 | 13 | 0.512451 | 0.517578 | 0.859924 | 0.140076 | 0.521988 | 1.937500 | 43.825947 |

## Mean +- Std

- final_slot_hit: 0.510132 +- 0.005606
- after_online_slot_hit: 0.517639 +- 0.011071
- verified_prefetch_hit: 0.858795 +- 0.001637
- verified_fallback_rate: 0.141205 +- 0.001637
- wrong_prefetch_rate: 0.519125 +- 0.003046
- avg_lookahead_depth: 1.933960 +- 0.007368
- sim_tpot_ms: 44.014139 +- 0.272815
