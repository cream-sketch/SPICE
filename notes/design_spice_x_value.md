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
