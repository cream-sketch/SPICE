Harsh review of a cache-value design for a SPICE paper component (batch=1 decode, offloaded fine-grained MoE Qwen1.5-MoE 60-expert top4 / DeepSeek-V2-Lite). Context: prior experiments on this branch found (a) within-token-draft-only forecast-eviction LOSES to Least-Stale(LS); (b) a crude token->expert table used for cache ~= LS (no utility win); (c) token->expert recall=0.58 (real signal but didn't convert). kill-shot just measured real components: t_expert_cpu=0.18ms, t_copy_h2d(17MB)=0.78ms (PCIe 22GB/s), CPU computes N misses with intra-op contention (4 misses=1.5-2.2ms), exposed stall exists ONLY at tight cache (5-10% residency).

The new design (below) fuses within-token SPICE draft prob and cross-token token/LM-head prior at the VALUE layer via a survival form, NOT by biasing draft logits. Decisive cheap pre-check = pure cache hit-rate of V-eviction vs LS vs Belady at tight cache.

QUESTIONS (be blunt):
1. Is the survival/hazard fusion P_use = 1 - prod_s(1-q_s) the right way to combine within-token (current token future layers) and cross-token (future tokens same layer) use-events for an EVICTION value? Independence is violated (cross-layer within token correlated; consecutive tokens same layer correlated). Does this break it or is it an acceptable first-order model? Any cleaner correct form?
2. Given my prior negatives (forecast-eviction<LS, token_table~=LS), what is the SINGLE most likely reason V will ALSO tie LS, and what specific design choice would prevent that? Is the cross-token A[j,v*,e] term (conditioned on the KNOWN greedy next token) genuinely more than what LS captures for near-memoryless access?
3. Is pure cache HIT-RATE at tight cache the right cheap pre-check, or can a policy tie on hit-rate but win on TPOT/bytes-per-useful-hit (i.e. is hit-rate a valid kill gate)?
4. Leakage/fairness traps in: estimating A[j,v,e] from traces, using v_{t+1}, equal-budget vs LS, Belady computed per-(layer,expert) at tight cache. What will I get wrong?
5. Verdict: is this worth ONE decisive hit-rate experiment, or does it inherit the token_table~=LS death and should fold into paper I now? Give: defensible-component / appendix / dead-now.
# SPICE-X: target-conditioned expert value estimation (cache-value layer on top of SPICE)

## 核心 (user-refined, value-layer fusion 非 logit-bias)
不改 SPICE draft。外挂一个 next-use hazard / cache-value 层，融合两个不同随机变量:
- within-token (SPICE draft): q_s = pi_hat_spice[t,j,e] for same token, j>current_layer
- cross-token (LM-head/token prior): q_s = sum_v P_LM(x_{t+1}=v) A[j,v,e]; greedy => A[j,v*,e]
  A[j,v,e] = P(expert e at layer j | token id v), 从 train trace 统计 (Laplace alpha 平滑)
survival 融合:
  P_use_before_deadline(j,e) = 1 - prod_s (1 - q_s(j,e))
  V(j,e) = P_use * T_refetch(j,e) * I(j,e) - C_cache(j,e)
eviction: evict argmin V; miss-handling: argmin{fetch_cache: T_fetch - V_future, cpu_once: T_cpu, drop: qloss}

## 修正 (来自本分支硬先验)
1. baseline = LS + Belady-oracle 上界 (forecast-eviction 死于无 cross-token 项输 LS; token_table cross-token cache ~= LS)
2. survival 独立性假设要标注 (同token跨层/相邻token同层相关)
3. T_refetch 单模型内常数(专家都~17MB), 不区分同模型专家; 真正区分项 I(j,e)=gate weight
4. 收益窗口=紧cache 5-10% (kill-shot 证 30% Fiddler已最优)
5. timing: greedy 下 v_{t+1} 仅在 token t forward 完成后可知 -> A[j,v_{t+1},e] 只在 token t+1 的 forward 内用 (eviction 逐层连续做, timely 成立)

## 决定性预检 (先跑, 纯命中率, 复用 eval_real_trace_eviction.py)
加 V-policy, 紧cache(5/10/20%)比 hit-rate: V>LS 则进 replay; V~=LS 则死->paper I.
泄漏防护: A[j,v,e] 必须 train/test 分开统计(不能用测试序列自身); within-token draft 用真实 draft 或 oracle-within-token 上界(分离两项贡献).
