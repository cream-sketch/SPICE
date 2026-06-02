# >>> v3 更正 (2026-06-02, 读完 SPICE 原论文后) <<<
# 重大更正: SPICE 的 LoRE 是路由预测的"低秩状态转移" E(z)=z+BAz, 不是近似专家输出, 也不是专家权重低秩.
# 因此: (a) 实验0a(专家权重满秩)对 SPICE 无杀伤, 仅排除"近似专家输出"误读; (b) codex 重构里"代理只调度路由不近似输出"SPICE 已实现, 非新贡献.
# 新的非增量靶子 = SPICE 自己暴露的弱点 (论文 Table VI/III): 小 cache 预算下输给 LRU; H2D 198GB 极吃带宽; 无 eviction 设计; miss 只阻塞补取; batch>1 缺失; 漏比 FineMoE/HOBBIT.
# 拟定主张: 让 speculative prefetch 在 "紧内存+紧带宽" 真实 offloading regime 可用 (predictor 预测驱动的 eviction + 带宽感知 speculation throttle + miss-window 调度), 保留 lossless verification.
# 修正前提后已重新咨询 codex: notes/codex_query_direction_v2.md -> notes/codex_run_direction_v3.log. 待其结论再定稿.
# 以下 §2-§6 中关于 (B) "低秩近似输出" 的部分按上面更正理解, 不再作为路线.

# SPICE 非增量方向与实验路线（v2, 2026-06-02，codex 复核后）

依据：6 篇 paper + SOTA SpecMD(Apple 2602.03921) + codex 独立评估(`notes/codex_direction_assessment_v1.md`) + 实验0a 实测。
原则：在现有 SPICE 代码基础上改；honest negative；先定 go/no-go 再跑；关键代码上新分支并打 tag；代码过 codex。

## 1. 任务定位
offloaded MoE 推理的 **专家路由→预取→缓存/驱逐→miss 处理** I/O 关键路径（与 SparMoE 的 compute/gather-tax 正交）。

## 2. 两个决定性的负结果（已确认，省去弯路）
- **实验0a（已跑）**：真实 Qwen1.5-MoE 专家权重近乎满秩——effective_rank≈847-903/1408；rank-64(≈4.5%参数)仅保留 18-21% 能量；90%能量需 rank≈780。
  → **"低秩代理做专家输出近似替代"不可行**。代理不能当 output substitute。
- **codex 判定 (A) 已被 FineMoE/fMoE subsume**（它已用 gate 概率分布做 prefetch+eviction priority）。
  → uncertainty-aware eviction **不能作为主 thesis**，只能降为 scheduler 组件。

## 3. 唯一存活的非增量 thesis（codex 重构，已采纳）
> **Verified expert-level miss recovery**：用常驻低秩代理（LoRE）把一次不可避免的 miss stall **转成 downstream 预取窗口**——代理只用于**预测/调度下游若干层的精确专家搬运**，真实专家到达后**精确重放**，logits 完全不变。
> 区别于"更好的预测器"：收益不是更高命中率，而是**把 miss 等待时间用于下游精确预取**，恢复关键路径 stall。

与最近邻 prior art 的边界：
- HOBBIT=miss 时换低精度（lossy，无精确纠正）；MiLo=量化低秩补偿（非 runtime miss recovery）；Speculative MoE=分布式 token/expert 预调度（非单卡 offload）；FineMoE=概率制导缓存（无近似执行）。**没有一篇做"miss 时低秩代理驱动下游精确预取+提交前精确重放"**。

## 4. Fatal flaws（codex）与对策
1. 精确 logits 会抹掉朴素近似收益 → 收益**重定义为下游预取调度**，不是复用近似激活。
2. transformer 尾部非线性(RMSNorm/attn/top-k/SwiGLU)，delta 无法廉价精确传播，top-k 会翻 → **先做 exact replay（零 logit 误差）**，bounded-error 仅作可选第二模式。
3. 误差界大概率 vacuous → 不把 certification 当首要贡献。
4. (A) 被强 paper 覆盖 → 降级为组件。
5. **真实 trace 预测质量可能直接杀死 (B)**：合成 0.99 命中无意义；真实 Qwen 简单预测器仅 0.34-0.39。若代理无法预测 miss 之后下游精确专家 → wrong-prefetch 浪费带宽、挤掉有用专家，(B) 死。**必须最先测。**

## 5. 实验路线（go/no-go）

### 实验1（下一步，最便宜的生死门）— 真实 trace 下游路由可预测性
- 数据：在 server 用现有 `collect_hf_moe_traces.py` 采集 Qwen1.5-MoE-A2.7B 真实 prompt（ShareGPT/WikiText）的 hidden states + router logits + 选中专家。
- 测：给定第 l 层状态，代理(draft rollout / LoRE)预测 l+1..l+K 层精确专家集合的 top-k 重叠 vs lookahead 距离 K；对比现有 history 预测器(anchor_repeat 0.34 / layer_prior 0.39)。
- **go/no-go**：若 hidden-state 代理在 K≥2..4 上显著超过 history baseline 且 set-overlap 足以支撑有用预取（待定阈值，参考 oracle gap）→ 进实验2；若代理深度预测崩塌 → (B) 死，转 fallback（见 §6）。

### 实验2 — real-trace miss-shadow replay（codex 推荐的核心实验）
- 受限 expert cache + pinned CPU 专家 + 实测 A800 PCIe H2D 时延 + exact target 执行。
- baseline miss 时，SPICE-B 不提交近似 logits，只用代理预测下游专家并发起 l+1..L 的 H2D 预取；真专家到达后精确重放，测下游 H2D 等待是否被消除。
- 主指标 `RecoveredStall = (Stall_Aonly - Stall_AplusB) / Stall_Aonly`（关键路径 H2D 等待）。
- **go**：exact-replay 恢复 ≥30% 关键路径 stall 且 TPOT ≥1.25× 优于最强 A-only baseline（Qwen+DeepSeek 同 cache budget），额外 H2D 流量 ≤25%，logits 完全一致。
- **no-go**：stall 恢复 <15% 或 TPOT <1.10× 或流量 >1.5×。

### 必须打的 baseline（codex）
lossless：on-demand、LRU/LFU、Mixtral-Offloading、MoE-Infinity、ProMoE、**FineMoE/fMoE(必须，直击A)**、HybriMoE；近似：HOBBIT、Cache-Conditional Experts、MiLo 低秩补偿；speculative：SP-MoE/Pre-Attention/MoE-SpeQ/EARTH/Speculative MoE；oracle：downstream-prefetch 上界 + zero-cost correction 下界（连下界都弱则 B 死）。
baseline 用上游原始实现，自写 wrapper 仅 diagnostic（codex moe-baseline-integrity 规则）。

## 6. 若 (B) 也死的 fallback
转向 §2 已排除主张之外仍开放的：**verified deep-lookahead 调度 + 与代理耦合的"哪些 miss 值得开窗、哪些直接阻塞"的 cache 决策**（SpecMD 只 lookahead=1 且启发式 LS）——但需重新论证非增量性。
