# Cross-layer route predictability (prereq for SPICE cross-layer LoRE) -- real models, linear probe

crosslayer_probe.py: from a token's layer-l TRUE hidden, train a cheap LINEAR probe to predict layer
(l+h)'s routed top-k; recall@k vs a popularity baseline. 1024 tokens (wikitext), avg over layer pairs.

| model           | experts | top_k | lead1 probe / pop | lead8 probe / pop | gain (lead1..8) |
|-----------------|--------:|------:|------------------:|------------------:|----------------:|
| Qwen1.5-MoE     | 60      | 4     | 0.935 / 0.236     | 0.922 / 0.241     | +0.70 .. +0.68  |
| DeepSeek-V2-Lite| 64      | 6     | 0.923 / 0.309     | 0.907 / 0.311     | +0.61 .. +0.60  |
| Mixtral-8x7B    | 8       | 2     | 0.744 / 0.338     | 0.667 / 0.340     | +0.41 .. +0.33  |

## Findings
- Cross-layer routing is HIGHLY predictable from the current layer's hidden, even 8 layers ahead, by a
  CHEAP LINEAR map (no attention). Fine-grained models (Qwen/DeepSeek) reach ~0.92-0.94 recall and barely
  decay with lead; coarse Mixtral lower (~0.67-0.74) but still 2x popularity.
- This is the prerequisite for a cheap cross-layer LoRE forecaster: a trainable low-rank transition can
  learn future-layer routing. Fine-grained models give BOTH high predictability AND a large selection
  space -> the regime where forecast-driven scheduling has room.

## Note (what this is and is NOT)
- This is a PROBE (predictability check), not the SPICE draft.
- The current real Qwen/DeepSeek forecast EVIDENCE is the TRAINING-FREE draft (shared-only rollout, no
  trained params). A TRAINED LoRE on a real model does NOT exist yet (train_real_lore.py was deleted;
  trained LoRE only ran on synthetic data). See lore-vs-trainingfree memory.
- The probe (true-hidden-anchored, per-layer) is EASIER than a rolling forecaster (which accumulates error).
  The real LoRE test = cheap rolling LoRE (no attention) vs expensive rolling shared-only, both anchored.

## Next
Rebuild trained-LoRE-on-real (recovered train_real_lore.py = frozen router + low-rank transition +
route-history, no attention = the cheap design) and compare recall-by-lead + forecast cost vs training-free
shared-only. Foothold thesis: cheap LoRE matches shared-only recall WITHOUT attention -> does not eat the
long-context attention shadow that hides the prefetch.
