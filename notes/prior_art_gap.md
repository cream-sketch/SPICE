# SPICE 先验技术 / 新颖性差距分析 (Prior-Art / Novelty-Gap Analysis)

日期 (Date): 2026-06-03
范围 (Scope): offloaded-MoE，单卡 A800，batch=1，EXACT 同精度，experts 在 CPU DRAM，
按需经 PCIe (~21.6 GB/s) 取到 HBM 或在 CPU (DRAM ~200 GB/s) 上计算 (Fiddler 风格)。

本文针对 SPICE 的两条候选贡献方向做诚实的覆盖度判定 (YES/PARTIAL/NO)：

- **DIRECTION A "SPICE-W verified-window grouped CPU service"**:
  verified draft 形成 K 个推测 token 的窗口；逐层收集 K 个位置的 routed experts；
  每个 UNIQUE 的 miss expert 在 CPU 上对所有路由到它的位置一次性计算 (grouped GEMM，many-GEMV->one-GEMM)；
  只有被接受/verified-prefix 的 experts 进入 HBM admission（rejected 后缀不污染缓存）。
  SPICE = verified expert-demand oracle。
- **DIRECTION B "partial-expert CPU/GPU split"**:
  SwiGLU expert (a=h@Wg, b=h@Wu, m=silu(a)*b, y=m@Wd)，miss 时把 NON-resident 矩阵段
  在 CPU (DRAM 带宽) 上计算，而不是经 PCIe FETCH。例：Wg,Wu 常驻 GPU (算 m)，CPU 算 y=m@Wd 只读 Wd。
  关键区别 vs MoEpic：MoEpic 缓存 top 段 + FETCH bottom 段；我们 CPU-COMPUTE 非常驻段（无 PCIe）。

---

## (a) 每篇论文的精确机制一行总结 (One-line exact-mechanism summary)

| # | 论文 | 精确机制 (exact mechanism) |
|---|------|----------------------------|
| 1 | **SP-MoE** (2510.10302) | SD + 预取：drafting 阶段用 draft-model 每层 attention 输出喂入 target-model 同层 gating 预测 critical experts，沿 CUDA worker stream 在 cutoff layer 0..L 之前异步预取到 HBM；experts 全部 FETCH 到 GPU 计算，LRU 缓存。无 CPU 计算，无 grouped GEMM，无 accept-aware admission。 |
| 2 | **MoE-SpeQ** (2511.14102) | SD + 预取：用一个 **4-bit 量化** draft model 做多步 lookahead，建 Expert Lookahead Buffer (ELB, k×L)，由 Amortization Roofline 自适应选 draft 长度 k，分三阶段预取把非常驻 expert 经 PCIe **取到 HBM**；hybrid-precision，KV/non-expert 共享。无 CPU expert 计算，无 per-expert grouped GEMM，无 accept-aware admission。 |
| 3 | **Pre-Attention Expert Prediction** (2511.10676) | 纯预测器：用 attention **之前**的 pre-attention normalized 权重 + 2 层线性 (ranking-aware loss) 预测本层 top-k experts（解决首层冷启动），93-97.6% 准确，给预取/缓存提供 oracle。只是预测信号，不涉及 CPU/GPU 拆分、SD、grouped GEMM。 |
| 4 | **SpecMD (A Comprehensive Study...)** (2602.03921) | 缓存策略 benchmark 框架 + 新 eviction 策略 **Least-Stale**（按 forward-pass staleness+layer position 排序逐出，避免 collision miss）；发现 "score-based prefetch > top-k"、"LRU/LFU 不符合 MoE 访问模式"。仅缓存/逐出/预取策略层面，无 CPU 计算、无 SD-window grouped service、无 partial-expert split。 |
| 5 | **MoEpic** (2509.08342) | 把每个 expert **纵向**切成 top/bottom 两段，top 段常驻 GPU（hot experts），下一层预测并预取激活 experts 的 **bottom 段经 PCIe FETCH 到 GPU**（或整 expert if miss），与 GPU 计算 overlap。**bottom 段是 FETCH 不是 CPU 计算**；不用 SD。 |
| 6 | **SpecMoEOff** (2508.21706) | SD **放大每个 expert 的 workload**：draft 生成多 token，target 一次 verify，多个 token 路由到同 expert 即把 per-token GEMV 合成 batch GEMM，从而 **摊薄 PCIe 取 expert 的成本**；experts **FETCH 到 GPU** 计算（CPU 只做 chunked attention verification kernel）；需 **大 batch (515-824)** 才有效；按 batch 维切 KV。无 per-unique-expert 分段 GEMM，无 accept-aware admission，非 batch-1。 |
| 7 | **HybriMoE** (2504.05897) | hybrid CPU-GPU：**whole expert** 动态分配到 CPU 或 GPU（intra-layer 调度均衡负载）+ impact-driven inter-layer 预取 + score-based 缓存。CPU 计算的是整 expert，不拆单 expert 矩阵；无 SD。 |
| 8 | **Fiddler** (2402.07033) | miss 时把 activation 拷到 CPU，在 CPU 上算**整个 expert**（up+down 全部），小输出拷回 GPU，避免搬 expert 权重过 PCIe；按 latency model `cpu_lat(s)` vs `gpu_lat(s)+transfer_lat()` 决定每个 expert 在 CPU 还是 GPU。**不拆单 expert**；无 SD。 |
| 9 | **KTransformers** (SOSP25) | hybrid CPU-GPU：AMX 专用 kernel 在 **CPU 上算 expert**（整 expert）并把结果传回 GPU；新 **Expert Deferral** 机制延迟 expert 计算以最大化 CPU/GPU overlap（CPU 利用率 ->~100%）。whole-expert 粒度；无 SD（与 SD 正交）。 |
| 10 | **DALI** (2602.03495) | workload-aware offloading：把 expert->CPU/GPU 分配建成 0-1 整数规划用 greedy 求解 + residual-based 预取（用层间 residual 预测高 workload expert）+ workload-aware 缓存替换。**whole-expert** 分配；无 SD，无 partial-expert split。 |
| 11 | **PowerInfer** (2312.12456) | dense LLM：按 power-law，**hot neuron 常驻 GPU、cold neuron 在 CPU 算**（whole-neuron 粒度）+ online predictor + sparse operator。神经元（非 expert）粒度；不拆单矩阵段；为 dense 模型，非 MoE-offload，非 SD。 |
| 12 | **llama.cpp 实践** | 把整层/整 expert 的 FFN tensor (gate/up/down) 整体放 CPU，miss 时 activation 过 PCIe 到 CPU，CPU 用 RAM 中权重算**整 expert** 再传回。whole-expert/whole-layer 粒度；非 intra-expert 段拆分；无 SD。 |

---

## (b) 方向覆盖度判定表 (Direction coverage tables)

### DIRECTION A — verified-window grouped CPU service (SD oracle + per-unique-expert grouped GEMM on CPU + accept-aware HBM admission)

A 的机制由四个原子组成，全部满足才算 COVERED：
(A1) SD verified window 作为 expert-demand oracle；
(A2) 逐层把 K 个位置的 demand 收集，每个 UNIQUE miss expert **一次** grouped GEMM；
(A3) 该 grouped GEMM 在 **CPU**（DRAM 带宽）上算，而非 FETCH 到 GPU；
(A4) **只有 accepted/verified-prefix** 的 expert 进 HBM admission，rejected 后缀不污染缓存。

| 论文 | 覆盖? | 具体原因 (cite exact mechanism) |
|------|-------|--------------------------------|
| SpecMoEOff | **PARTIAL** | 有 A1 (SD window) + A2 的"合并到同 expert 摊薄"思想；但 **A3 失败**（expert FETCH 到 GPU，CPU 只做 attention verify，不在 CPU 算 expert GEMM）；**A4 失败**（verify 在 load 之后，不按 accept 决定 admission）；且 **需大 batch (515-824)，非 batch-1**，整窗摊薄而非 per-unique-expert 分段 GEMM。 |
| SP-MoE | PARTIAL | 有 A1（SD 的 verified/critical-expert 概念）；A2/A3/A4 全无：experts 全 FETCH 到 GPU，无 CPU grouped GEMM，rejected 后缀照样按预测预取（明确指出低接受率时为 rejected token 取 expert 浪费 I/O，但其方案是减少预取而非 accept-aware admission）。 |
| MoE-SpeQ | PARTIAL | 有 A1（SD lookahead oracle = ELB）；A3/A4 无（FETCH 到 HBM，量化 draft，无 CPU expert 计算，无 accept-gated admission）。A2 无 per-unique-expert CPU 分段。 |
| Fiddler / KTransformers / HybriMoE / DALI | NO | 它们做 CPU expert 计算（满足"CPU 算 expert"），但 **没有 SD verified window (A1 无)**，按 latency/ILP 在单 token 粒度分配整 expert，没有 A2 的窗口内 per-unique-expert grouping，没有 A4 accept-aware。 |
| MoEpic / SpecMD / Pre-Attn / PowerInfer / llama.cpp | NO | 无 SD verified window；无 CPU grouped expert service；无 accept-aware admission。 |

### DIRECTION B — partial-expert CPU/GPU split (resident 段算 m，非常驻段在 CPU 算，避免 PCIe fetch)

B 的机制由三个原子组成：
(B1) **单个 expert** 被拆成常驻 GPU 段 + 非常驻段（intra-expert，沿矩阵）；
(B2) 非常驻段在 miss 时 **在 CPU 上计算**（读 DRAM 中该段权重），而非 FETCH 到 GPU；
(B3) 据此把一次 expert 服务拆成 GPU 算前半 (m=silu(h@Wg)*(h@Wu)) + CPU 算后半 (y=m@Wd)，PCIe 上只走小激活。

| 论文 | 覆盖? | 具体原因 (cite exact mechanism) |
|------|-------|--------------------------------|
| MoEpic | **PARTIAL** | 满足 B1（单 expert 纵向切 top/bottom，top 常驻 GPU）；但 **B2 失败**（bottom 段是 **FETCH 经 PCIe 到 GPU** 计算，不在 CPU 算）。即 B 的"段拆分"已被 MoEpic 占住，但"非常驻段 CPU 计算免 PCIe"这一条 MoEpic 没有。 |
| Fiddler | PARTIAL | 满足 B2 的精神（activation 到 CPU，CPU 算 expert，免搬权重）；但 **B1 失败**（算的是**整 expert**，不拆单 expert 矩阵段）；不做 GPU 算前半/CPU 算后半的混合单 expert 流水。 |
| KTransformers | PARTIAL | 同 Fiddler：CPU 算整 expert (AMX)，Expert Deferral 做 overlap；不拆单 expert 矩阵段 (B1 无)。 |
| HybriMoE / DALI / llama.cpp | PARTIAL | whole-expert 在 CPU 或 GPU（B2 有 CPU 计算思路），但 **whole-expert 粒度，B1 无**。llama.cpp 可按 tensor-type(gate/up/down) 整体分设备，但仍是"全部 expert 的某一 tensor-type"整层分配，不是单 expert 内 GPU 算前半 + CPU 算后半的拆分。 |
| PowerInfer | PARTIAL | 有"部分行/神经元在 CPU 算"的精神，但 dense 模型、neuron 粒度、非 MoE expert 内段拆分，非 SwiGLU 前半/后半流水。 |
| SP-MoE / MoE-SpeQ / SpecMoEOff / SpecMD / Pre-Attn | NO | 全部 FETCH 到 GPU 或仅预测/缓存，无任何 expert 内段拆分 + 段级 CPU 计算。 |

---

## (c) 剩余的非增量差距 (Precise remaining non-incremental gap)

### DIRECTION A — 差距 (GAP，可主张的精确句子)

> **可主张**: 据我们所知，SPICE 是第一个在 **单卡 batch=1** 下，用 verified speculative window 作为
> **exact expert-demand oracle**，逐层把窗口内路由到**同一 unique miss expert** 的多个位置聚成
> **一次 CPU grouped GEMM 服务**（many-GEMV->one-GEMM 在 DRAM 带宽上算，免 PCIe fetch），
> 并且**仅把 verified-prefix（被接受）的 expert 纳入 HBM admission**（rejected 后缀不污染缓存）的系统。

为什么是非增量：现有最接近者 **SpecMoEOff** 用 SD 摊薄 expert 成本的方向相同，但 (1) 它 FETCH-to-GPU 而非 CPU 计算，(2) 它靠 **大 batch (515-824)** 提供 token 数来摊薄——在 **batch=1** 下 SpecMoEOff 的核心假设崩溃；SPICE 用**推测窗口的 K 个 token**在 batch=1 下提供同样的"多 token -> 一个 expert"批量度，且把它落到 **CPU grouped GEMM + accept-aware admission**。这三点组合（batch-1 窗口批量度 + CPU 段服务 + accept-gated admission）没有任何单篇覆盖。

注意 (诚实): A1(SD oracle)、A2(摊薄思想) 已分散存在；A 的新颖性 **不在任一原子**，而在 "**batch-1 下用窗口替代 batch 来喂 CPU grouped GEMM，并 accept-gate admission**" 的组合 + 把它做成 verified（exact，非预测概率）oracle。若审稿人把 SpecMoEOff 的 batch 维 grouping 等同于窗口维 grouping，A 会被压成增量——必须用 batch-1 实测把差异打出来。

### DIRECTION B — 差距 (GAP，可主张的精确句子)

> **可主张**: 据我们所知，SPICE 是第一个对 SwiGLU expert 做 **intra-expert 矩阵段拆分**，
> 使常驻 GPU 段计算前半 (m = silu(h@Wg) ⊙ (h@Wu))、而把**非常驻段 (Wd) 在 CPU 上计算** (y = m@Wd)，
> 从而 miss 时 **PCIe 上只走小激活 m，完全不 FETCH 任何 expert 权重段**的系统。

为什么是非增量：两个最接近者各占一半——**MoEpic** 占了 "单 expert 段拆分 (B1)" 但非常驻段 **FETCH 到 GPU**；**Fiddler/KTransformers** 占了 "CPU 算 expert 免搬权重 (B2)" 但只算**整 expert**。SPICE = MoEpic 的段拆分 ∩ Fiddler 的 CPU 计算 = "**段拆分 + 非常驻段 CPU 计算**"，这个交集是空的（无人占据）。其物理论据明确：DRAM ~200 GB/s vs PCIe ~21.6 GB/s，对非常驻段 CPU 计算（读 DRAM）比 FETCH（过 PCIe）快约一个量级。

注意 (诚实): B 的风险是"增量感"——MoEpic 只需把 bottom 段改成 CPU 计算即落到 SPICE。必须证明这是**有意义的非平凡设计点**：即对哪些段 (Wd 后半 vs Wg/Wu 前半)、在什么 token/段尺寸下，CPU-compute-segment 严格优于 fetch-segment，并给出 roofline/交叉点，否则会被视为 MoEpic 的工程变体。

---

## (d) 每方向最近先验 + 击败它必须证明的一件事 (Closest prior + the one thing to show)

### DIRECTION A
- **最近先验 (closest prior work)**: **SpecMoEOff** (2508.21706) — 同样用 SD 把多 token 摊薄到同 expert 以降 PCIe 单位成本。
- **击败它必须证明的一件事 (the one thing)**: 在 **batch=1 单卡** 下，用 verified **窗口**（而非 SpecMoEOff 所需的 515-824 大 batch）提供的 per-unique-expert 批量度，配合 **CPU grouped GEMM + accept-gated admission**，达到 SpecMoEOff 在 batch=1 时**达不到**的 decode 时延/吞吐；即给出 "batch=1 时 SpecMoEOff 退化、SPICE 不退化" 的同卡对照曲线。

### DIRECTION B
- **最近先验 (closest prior work)**: **MoEpic** (2509.08342) — 唯一已做 **intra-expert 段拆分**者（top 常驻、bottom 经 PCIe fetch）。
- **击败它必须证明的一件事 (the one thing)**: 对非常驻段把 **MoEpic 的 PCIe-FETCH 替换为 CPU-COMPUTE**，并给出 roofline 交叉点 + 端到端实测，证明在目标段（如 Wd）与 batch-1 token 尺度下，**CPU 段计算 (读 DRAM ~200GB/s)** 的有效服务时延严格低于 **段 FETCH (PCIe ~21.6GB/s) + GPU 计算**，量化 PCIe 流量从 "段权重" 降到 "小激活" 的收益。

---

## 总体诚实结论 (Honest bottom line)

- **A**: GAP 存在但 **窄**，且**完全依赖 batch=1 这一约束**来与 SpecMoEOff 区分。novelty 在组合（窗口批量度替代 batch + CPU grouped GEMM + accept-gated admission），不在单一原子。若不能用 batch-1 实测打出差异，A 退化为 SpecMoEOff 的 batch-1 特例 = 增量。
- **B**: GAP 存在且**论据干净**（MoEpic 段拆分 ∩ Fiddler CPU 计算 = 空集，物理上 DRAM>PCIe 一个量级）。风险是"MoEpic 的小改动"观感，必须用 roofline 交叉点 + 段选择分析把它变成有意义的设计点，而非一行代码替换。
