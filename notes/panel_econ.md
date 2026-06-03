PEER REVIEW (harsh, quantitative). Rule on the ECONOMIC MODEL and the DECISIVE EXPERIMENT design for "SPICE-X selective residency / admission" (offloaded MoE, single GPU, batch=1 decode).

Measured real A800 components: C_cpu=0.18ms (one expert, intra-op threaded), C_gpu=0.079ms (resident expert), C_fetch=0.78ms (17MB@22GB/s pinned async), CPU computes 4 misses in ~1.5-2.2ms (intra-op contention, not free parallel). Access near-uniform/memoryless (Gini 0.17, autocorr Jaccard lag1 0.13).

Admission rule: fetch-and-cache iff R_e*(C_cpu - C_gpu) > C_fetch + V_victim, where R_e = expected future reuse before eviction. With numbers: R_e > (0.78 + V_victim)/0.10 >= ~8.

QUESTIONS:
1. Is this break-even derivation correct? Does it correctly account for: the CURRENT use (served either way), the one-time fetch cost amortized over R_e future uses, the victim's lost value V_victim, and the fact that CPU-serve has exposed-stall risk (CPU contention at burst) while GPU-resident does not? What is wrong or missing (e.g. should it be exposed-stall not raw C_cpu? bandwidth contention from fetches? deadline?)
2. Given near-memoryless access + tight cache, is the claim "most experts have R_e < 8 so should be CPU-served" likely TRUE, and is it actually ACTIONABLE (does admitting fewer experts measurably cut TPOT, or does CPU-serving them just move the cost)?
3. The DECISIVE experiment: I propose a wall-clock or deadline-replay TPOT comparison: SPICE-default(cache-all-fetched) vs SPICE-X(selective admit + CPU-serve-once) vs Fiddler(CPU/GPU load-balance) vs LS-eviction, on Qwen+DeepSeek at residency 5/10/20%, metrics TPOT + H2D bytes/token + cache-pollution(admitted-but-never-reused fraction). Is this the right experiment? What baseline/metric am I missing? What is the single most likely way this shows no TPOT win?
4. Should the R_e predictor be the hit-rate value V(j,e) I already built (within-token A + transition B, beats LS by +0.048 at 5% residency on hit-rate), or does admission need a different target (reuse COUNT not next-use probability)?
GIVE: economic-model verdict (correct/flawed-fix-it), and the minimal decisive experiment with exact GO/NO-GO thresholds.
