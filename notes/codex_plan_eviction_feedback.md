```text
exposed_cost_h = fetch_ms(r) * 1[time_to_use_h < fetch_ms(r)]
```
Evict the resident with minimum `score`. Tie-break by LRU.
Calibrate `calib[h, bucket(conf)]` on held-out traces. Do not trust raw draft confidence.
**Q4**
Biggest collapse risk: all gains come from better prefetch admission/protection, not from eviction.
Pre-empting experiment:
Run a 2x2 ablation with identical forecast and identical H2D budget:
```text
prefetch: off / fixed-draft
eviction: LRU / LS / forecast-Belady / oracle-Belady
```
Key reviewer-proof case:
```text
fixed-draft prefetch + LRU
vs
fixed-draft prefetch + forecast-Belady
```
If forecast eviction does not win there, the contribution is just “better prefetcher.”
Also report collision misses: prefetched expert later evicted before use.
**Minimal Viable Sequence**
1. Add forecast trace format and `forecast_belady` in demand-only simulator.
2. Test oracle forecast, noisy-oracle forecasts, anchor/layer_prior forecasts. This gives required forecast quality before touching Qwen training.
3. Implement training-free Qwen draft rollout, anchored at each real layer state. Measure recall/AUC and run it through `forecast_belady`.
4. Only if it passes the gates, train Qwen LoRE draft.
5. Add WITH-prefetch DMA-deadline simulator and 2x2 ablations.
6. Compare LRU / LS / FineMoE-policy diagnostic / forecast-Belady / oracle-Belady.
7. Runtime prototype only after sim captures `>=50%` of oracle gap on 10-50% cache and shows eviction-only benefit with prefetch held fixed.
tokens used
78,606
**Verdict**
Plan is directionally sound, but Step 1 is out of order. Do not train a Qwen LoRE draft first. First prove that any forecast of the shape you can cheaply produce can drive eviction better than LRU/LS in the fixed simulator.
**Bad / Wasted / Out Of Order**
1. **Step 1 full training first: wasted.**
   Measure a training-free Qwen draft first. A failed training-free draft does not kill LoRE, but it tells you whether hidden-state rollout has usable routing signal before spending GPU time.
2. **Raw top-K accuracy is the wrong gate.**
   Eviction needs resident next-use ranking, not just future expert-set overlap. Use both:
   `deadline-weighted slot recall@4` and `resident next-use AUC`.
3. **Bandwidth sweep is wasted in demand-only mode.**
   Miss count is bandwidth-independent there. Sweep bandwidth only after WITH-prefetch / DMA-deadline scheduling exists.
4. **Cache 1-2% is not a decision region.**
   You already showed it is cold-miss bound. Keep it as context, not a go/no-go gate.
5. **Step 4 threshold is too blunt.**
   “beats LRU by >=25% in 5-50%” is inconsistent with your own 5% oracle gap. Gate on 10-50%, and use oracle-gap capture.
**Q1**
Yes: training-free first is the cheapest correct gate.
Metric:
`deadline_weighted_slot_recall@4(h=1..6)`  
where nearer deadlines get higher weight, and target is real Qwen top-4 experts.
Proceed to LoRE training if:
- `deadline_weighted_slot_recall@4 >= 0.50`
- and `resident_next_use_AUC >= 0.70`
- and at least `+0.10 absolute` over `layer_prior` / `anchor_repeat`.
Kill or rethink if:
- recall `< 0.45`, or
- AUC `< 0.65`, or
- forecast-Belady with this forecast captures `< 20%` of oracle-vs-LRU gap.
Do not use exact-set match as the gate. It is too harsh for top-4 of 60.
**Q2**
Same forecast for prefetch + eviction is legitimate, but not novel by itself.
FineMoE already uses expert probability maps for both prefetching and eviction priority: prefetch priority uses probability over layer distance; eviction uses probability times cache frequency. MoE-SpAc is even closer in framing: it explicitly says it unifies prefetching and eviction in the same utility space.
Differentiate on:
- online SPICE draft lookahead, not historical expert-map retrieval;
- stochastic Belady / deadline objective, not LFU-probability heuristic;
- lossless verified demand path;
- demand-priority DMA and hard protection of current / in-flight demand;
- ablations proving eviction benefit with prefetch held fixed.
Sources: [FineMoE PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf), [MoE-SpAc arXiv](https://arxiv.org/abs/2603.09983).
**Q3**
Implement this score:
```text
score(r) =
  INF, if r is current demand or in-flight demand
  (1 / bytes(r)) * Σ_h first_h(r) * exposed_cost_h(r)
```
For resident expert `r=(layer, expert)`:
```text
p_h(r) =
  calib[h, bucket(conf_h)]        if expert in predicted_topk_h and layer matches
  eps_layer_prior[h, expert]      otherwise
```
Then:
```text
first_h(r) = p_h(r) * Π_{i<h}(1 - p_i(r))
```
And:
```text
exposed_cost_h(r) = max(0, fetch_ms(r) - slack_to_deadline_h)
```
If the simulator does not yet model DMA queue slack, use the simpler binary form:
```text
exposed_cost_h = fetch_ms(r) * 1[time_to_use_h < fetch_ms(r)]
```
Evict the resident with minimum `score`. Tie-break by LRU.
Calibrate `calib[h, bucket(conf)]` on held-out traces. Do not trust raw draft confidence.
**Q4**
Biggest collapse risk: all gains come from better prefetch admission/protection, not from eviction.
Pre-empting experiment:
Run a 2x2 ablation with identical forecast and identical H2D budget:
```text
prefetch: off / fixed-draft
eviction: LRU / LS / forecast-Belady / oracle-Belady
```
Key reviewer-proof case:
```text
fixed-draft prefetch + LRU
vs
fixed-draft prefetch + forecast-Belady
```
If forecast eviction does not win there, the contribution is just “better prefetcher.”
Also report collision misses: prefetched expert later evicted before use.
**Minimal Viable Sequence**
1. Add forecast trace format and `forecast_belady` in demand-only simulator.
2. Test oracle forecast, noisy-oracle forecasts, anchor/layer_prior forecasts. This gives required forecast quality before touching Qwen training.
3. Implement training-free Qwen draft rollout, anchored at each real layer state. Measure recall/AUC and run it through `forecast_belady`.
4. Only if it passes the gates, train Qwen LoRE draft.
5. Add WITH-prefetch DMA-deadline simulator and 2x2 ablations.
6. Compare LRU / LS / FineMoE-policy diagnostic / forecast-Belady / oracle-Belady.
7. Runtime prototype only after sim captures `>=50%` of oracle gap on 10-50% cache and shows eviction-only benefit with prefetch held fixed.
