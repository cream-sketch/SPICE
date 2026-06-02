# Codex: forecast-eviction is an honest negative — review the sim, then give the sharpest surviving non-incremental direction

Be brutal and concrete. Output to stdout. You MUST first sanity-check the simulator for any bug that would INVALIDATE the negative result (we don't want to abandon a direction due to a bug). Then give the verdict and the single best surviving direction. The repo is the current working dir; read experiments/eval_forecast_eviction.py and experiments/qwen_spice_draft.py.

## What we established (real Qwen1.5-MoE-A2.7B, A800, real WikiText traces)
1. Reproduced SPICE (synthetic draft verified hit 0.9965; real-trace simple predictors 0.32).
2. Built the repo's missing real-model draft, TRAINING-FREE (reuse frozen attn + frozen router, propagate hidden with shared-expert-only surrogate, anchored per true layer, roll K layers). It predicts WITHIN-TOKEN future-layer experts very well: recall@4 = 1.00/0.87/0.79/0.73/0.67 at horizon 1/2/4/6/8 vs baselines ~0.07-0.20. Strong within-token forecast, zero training.
3. Deadline-aware DMA cache sim (transfers span multiple layer windows; a 17MB expert at 5GB/s needs ~8.5 layers to hide; demand-priority; collision tracking). 2x2 (prefetch off/draft x eviction lru/ls/forecast/oracle), real Qwen, bw=12GB/s, expert 17MB, t_layer 0.4ms:

cache  prefetch  evict      hit
5%(72)  draft    lru        0.337
5%(72)  draft    ls         0.522   <- SpecMD Least-Stale best realizable
5%(72)  draft    forecast   0.482   <- our draft-forecast eviction
5%(72)  draft    oracle     0.729
10%(144) draft   lru        0.599
10%(144) draft   ls         0.637
10%(144) draft   forecast   0.579
10%(144) draft   oracle     0.827
20%(288) draft   lru        0.724
20%(288) draft   ls         0.745
20%(288) draft   forecast   0.719
20%(288) draft   oracle     0.902

Q4 eviction-only (draft prefetch fixed) forecast-vs-LRU exposed-stall gain: +9.3% (5%), -2.7% (10%), -2.4% (20%).

## The key analysis (confirm or refute)
- draft PREFETCH gives the big gains (off+lru hit 0.32 -> draft+lru 0.72 at 20%). That is SPICE's EXISTING contribution ("just a better prefetcher").
- forecast-driven EVICTION only beats LRU at the tightest cache and NEVER beats SpecMD-LS. So forecast-eviction is not a defensible contribution (your own no-go: forecast-Belady must beat LS by >10%).
- oracle eviction is far above all realizable policies. We believe its headroom is dominated by CROSS-TOKEN reuse (knowing which of the 60 experts at a layer will recur in FUTURE tokens). The SPICE draft only predicts WITHIN the current token's remaining layers; it does NOT predict the next token's routing (that needs predicting the next sampled input token). Hence the within-token draft structurally cannot capture the oracle (cross-token) headroom. Is this analysis correct?

## Questions
1. Sim correctness: any bug in eval_real_trace_eviction.py / eval_forecast_eviction.py (Belady next_use, LS cyclic distance, forecast score = near-forecast(1000+1/d) else layer_freq, demand-priority DMA, collision counting, multi-layer in-flight ready times) that could make forecast look unfairly bad or LS unfairly good or oracle unfairly high? If yes, exactly what and the fix.
2. Verdict: is forecast-driven eviction dead as the contribution? Yes/no, why.
3. Is the cross-token-vs-within-token explanation of the oracle gap correct, and does it generalize (would it hold on DeepSeek-V2-Lite, other bandwidths)?
4. SHARPEST surviving non-incremental direction, pick ONE and justify:
   (a) Cross-token speculative expert prefetching: extend the draft to predict the NEXT token(s)'s routing (predict next input token's experts) to capture the cross-token oracle headroom nobody captures. Feasible? How (the draft would need to roll past the token boundary / predict the sampled token)? Is this genuinely novel vs SP-MoE/MoE-SpeQ (which use a draft over future TOKENS already)? If SP-MoE/MoE-SpeQ already do cross-token, what's left?
   (b) Tight-bandwidth regime: at 5GB/s a 17MB expert needs ~8.5 layers lead; within-token horizon is bounded by depth (24 layers). Contribution = make offloaded MoE viable at very low bandwidth (NVMe) where even perfect within-token prefetch cannot hide the transfer. Via what mechanism that is NOT HOBBIT (low-precision) or just bigger cache?
   (c) Something else you see in the data.
5. If the honest answer is "this whole line is incremental over SPICE+SpecMD", say so plainly and name what, if anything, could still be a real contribution given everything measured.

Cite mechanisms. Do not be agreeable.
