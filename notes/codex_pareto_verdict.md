**Verdict:** fix it. As written, Part A+B is an **off-policy replay approximation**, not the true Pareto.

1. Decoupling is only sound for “fixed clean-trace intervention PPL.”
   Part B measures the PPL impact of the selected drop set, but Part A chose that drop set from clean routing. Once you drop at token/layer `t`, hidden state and KV change, so later routers, gate weights, cache hits/misses, and fetch/drop decisions can change. Error sign is unknown: you may understate PPL, overstate PPL, understate stalls, or miss new high-importance misses.

   Fix: run an **online teacher-forced replay** for sentinel points. No real DMA needed. Recompute routing with the degraded live KV, apply the controller, and run the latency simulator in lockstep. Compare decoupled vs online on:
   `top-k route mismatch`, `gate TV/KL`, `fetch/drop disagreement`, `hit/miss disagreement`, `dropped gate mass`, `TPOT error`, `PPL error`.

2. Single rejection pitfall: claiming the clean-trace Pareto is the real policy Pareto.
   The “verified true gate” claim collapses after the first drop unless routing is recomputed on the perturbed trajectory. Close second: modeling HOBBIT as “reduced fetch_ms + guessed PPL penalty” instead of measuring actual quantized experts under equal cache/bandwidth.

3. Minimal correct version:
   Do exact online teacher-forced decode first, not just a few toy tokens. Use enough length for cache warmup and miss bursts, e.g. 128-512 decode tokens per text, multiple texts, representative thresholds, and at least the hardest cache/BW points. Then use decoupled replay for the large sweep only after reporting its error envelope. If the envelope is large, decoupling cannot be the main curve.

4. Claim wording:
   Defensible: “Fetched miss experts are executed exactly; only misses with verified gate weight below threshold/slack policy are dropped; under this controller, observed PPL/latency tradeoff is bounded empirically on the evaluation set.”
   Not defensible: “bounded quality loss” as a formal model guarantee from gate weight alone. Gate weight is router importance, not causal loss sensitivity. You can bound omitted weighted expert contribution, e.g. dropped gate mass or `sum w_e ||E_e(x)||`; you cannot honestly bound PPL without strong norm/Lipschitz assumptions.

5. Differentiator:
   SpecMD gives the cache/eviction/prefetch frame; Least-Stale exploits predictable expert access to reduce collision misses, but it is not a verified lossy miss controller. Use it as eviction, not novelty. ([arxiv.org](https://arxiv.org/abs/2602.03921))  
   HOBBIT’s mechanism is mixed-precision replacement of less-critical cache-miss experts plus token/layer/sequence cache management, not exact high-weight fetch vs low-weight drop. ([arxiv.org](https://arxiv.org/abs/2411.01433))  
   AdapMoE changes active expert count using sensitivity-based gating/management and combines that with prefetch/cache management; it is not post-router verified miss admission. ([arxiv.org](https://arxiv.org/abs/2408.10284))  

Crisp novelty statement: **online SLO-constrained miss admission using verified post-router gate weight: spend DMA only on high verified gate mass, drop low verified gate mass, with SpecMD-LS eviction held fixed.**

Do not claim global Pareto-optimality. Claim Pareto improvement within a controlled policy class, and prove it by equal-resource ablations: gate-weight threshold vs rank threshold, verified gate vs predicted/draft importance, fetch/drop vs substitution, measured HOBBIT quantized fetch, with identical cache size, bandwidth, prefetch, expert precision, and workload.
tokens used
29,251
**Verdict:** fix it. As written, Part A+B is an **off-policy replay approximation**, not the true Pareto.

1. Decoupling is only sound for “fixed clean-trace intervention PPL.”
   Part B measures the PPL impact of the selected drop set, but Part A chose that drop set from clean routing. Once you drop at token/layer `t`, hidden state and KV change, so later routers, gate weights, cache hits/misses, and fetch/drop decisions can change. Error sign is unknown: you may understate PPL, overstate PPL, understate stalls, or miss new high-importance misses.

   Fix: run an **online teacher-forced replay** for sentinel points. No real DMA needed. Recompute routing with the degraded live KV, apply the controller, and run the latency simulator in lockstep. Compare decoupled vs online on:
   `top-k route mismatch`, `gate TV/KL`, `fetch/drop disagreement`, `hit/miss disagreement`, `dropped gate mass`, `TPOT error`, `PPL error`.

2. Single rejection pitfall: claiming the clean-trace Pareto is the real policy Pareto.
   The “verified true gate” claim collapses after the first drop unless routing is recomputed on the perturbed trajectory. Close second: modeling HOBBIT as “reduced fetch_ms + guessed PPL penalty” instead of measuring actual quantized experts under equal cache/bandwidth.

3. Minimal correct version:
   Do exact online teacher-forced decode first, not just a few toy tokens. Use enough length for cache warmup and miss bursts, e.g. 128-512 decode tokens per text, multiple texts, representative thresholds, and at least the hardest cache/BW points. Then use decoupled replay for the large sweep only after reporting its error envelope. If the envelope is large, decoupling cannot be the main curve.

4. Claim wording:
   Defensible: “Fetched miss experts are executed exactly; only misses with verified gate weight below threshold/slack policy are dropped; under this controller, observed PPL/latency tradeoff is bounded empirically on the evaluation set.”
   Not defensible: “bounded quality loss” as a formal model guarantee from gate weight alone. Gate weight is router importance, not causal loss sensitivity. You can bound omitted weighted expert contribution, e.g. dropped gate mass or `sum w_e ||E_e(x)||`; you cannot honestly bound PPL without strong norm/Lipschitz assumptions.
