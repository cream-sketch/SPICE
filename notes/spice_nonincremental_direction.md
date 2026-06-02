# SPICE 非增量方向与实验路线（v1, 2026-06-02）

状态：方向草案，待 codex prior-art 复核（`notes/codex_run_direction_v2.log`）。
语言：中文（专业术语保留英文）。

## 1. 任务定位（routing 里的哪个任务）

SPICE 做的是 **offloaded MoE 推理中的专家路由→预取→缓存/驱逐→miss 处理** 这条 I/O 关键路径。
不是 SparMoE 的 compute/kernel（gather tax）那条路；两者正交。

## 2. 领域 SOTA 与共同空白（来自 6 篇 paper）

SOTA = **SpecMD (Apple, arXiv 2602.03921, 2026-02, under review)**：speculative expert prefetching 的综合 benchmark + 新驱逐策略 Least-Stale。
- 第一性发现：**MoE 专家访问不满足时间局部性，LRU/LFU 是错的**；提出 collision miss（同一 forward pass 内驱逐又立刻要用）。
- LS 驱逐：1% 容量下 collision 比 LRU 降最多 85×，命中率 88-92%，TTFT -34.7%。

所有 6 篇（SpecMD / SP-MoE / Pre-Attention / MoE-SpeQ / EARTH / SpecBranch）的共同空白，也是 SpecMD **亲口承认**的：
1. miss/misprediction 只能 **同步 fetch（stall） / drop（掉质量） / substitution（换不同专家——SpecMD 实测全面失败、不可靠）**；
2. **名为 speculative 却没有 verification / 纠错环**；
3. **uncertainty-aware / approximate-then-correct 缓存完全缺席**；
4. 只预取下一层（lookahead=1）；无 oracle 上界 gap 分析。

## 3. SPICE 的非增量主张（thesis）

> **Verified low-rank expert approximation as miss recovery for offloaded MoE.**
> 专家 cache miss 时，不 stall、不 drop、不换别的专家，而是执行**同一缺失专家的常驻低秩代理（LoRE）**给出近似输出让 decode 继续，并在真实专家经 PCIe 流入时**验证/纠正**；配合 **uncertainty/staleness-aware 驱逐**（超越 LRU，融合 SpecMD-LS 与 draft 置信度）。

与 SpecMD 的本质差异（被其 negative result 背书）：
- SpecMD Substitution：换**另一个不同的已缓存专家** → 失败。
- 本方案：用**同一专家的低秩近似** + 纠错 → 正好补 SpecMD 留白。

SPICE 已有的 LoRE 低秩专家代理 + verified fallback，是天然载体（目前只在合成 target 上、且 miss 仍是 naive stall）。

## 4. 必须先证伪的地基假设（决定 thesis 生死）

- **H0（低秩可近似性）**：真实 MoE 专家权重在低秩 r ≪ full 下能保留绝大部分能量/输出。若满秩 → (B) 死。
- **H1（近似优于 drop/substitute）**：低秩代理做 miss recovery 的端到端质量损失显著小于 drop 与 substitute，且存储成本 ≪ 一个完整专家。
- **H2（系统净收益）**：把近似计算与 PCIe 传输重叠，端到端 TTFT/吞吐净正，且纠错保证质量可控。

## 5. 实验路线（go/no-go）

### 实验0a（本轮启动）— 真实 Qwen 专家 SVD 谱
- 输入：`~/sparmoe/models/Qwen1.5-MoE-A2.7B-Chat`（24 层 ×60 专家，gate/up: 2048×1408, down: 1408×2048）。
- 测：跨层/跨专家抽样，gate_proj/up_proj/down_proj 的奇异值能量谱；达到 90/95/99% Frobenius 能量所需 rank；rank∈{8,16,32,64,128} 的能量占比。
- **go/no-go**：若多数专家在 rank ≤ ~64（≈4.5% 参数）保留 ≥90% 能量 → H0 初步成立，进实验0b；若需要 rank≈full → H0 失败，(B) 改道（转向纯 eviction/lookahead 贡献）。

### 实验0b — miss recovery 质量对比（真实权重 + 真实路由）
- 真实文本（WikiText / GSM8K）跑 Qwen forward，记录每层 top-4 路由。
- 模拟 miss，对 miss 专家比较：(a) 真实专家(oracle) / (b) drop / (c) substitute 最近缓存专家 / (d) 低秩代理(ours)。
- 测：每专家输出 cosine/相对误差 + 端到端 PPL 退化 vs 存储成本。
- **go/no-go**：(d) 在 rank≪full 下 PPL 退化显著优于 (b)(c)，接近 (a) → H1 成立。

### 实验1 — eviction（与 (B) 并行，独立可发）
- 在真实 trace 上复现 SpecMD：LRU vs LS vs (predictor-confidence + staleness) 融合驱逐；测 collision/cold miss、命中率、oracle gap。

## 6. 纪律
- 每个实验先写 go/no-go 阈值，再跑；honest negative 照实记录。
- 关键代码上 git（新 branch，不直接推 main），重大结果打 tag。
- baseline 用上游原始实现（见 codex moe-baseline-integrity 规则），自写 wrapper 仅作 diagnostic。
- 代码每次过 codex review。
