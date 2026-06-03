# SPICE 信息到动作分析: 现在不应再做普通 prefetch

日期: 2026-06-03

## 0. 结论先行

你现在**不应该把新贡献说成 "做 prefetch"**。SPICE 的 draft model 已经实现了 verified speculative expert prefetch:

```text
draft predicts future experts
runtime prefetches them
target router verifies
miss falls back to exact fetch
logits remain exact
```

所以继续说 "我们也 prefetch" 会被 reviewer 直接归入 SPICE / SP-MoE / MoE-SpeQ / FineMoE / ExpertFlow 这一类。

现在真正的问题应该改写成:

```text
SPICE 已经有 draft prefetch。
下一步不是再证明能预取, 而是研究:
哪些信息能在资源受限时转成正确的 expert movement 动作?
```

也就是:

```text
Information -> Action -> Utility
```

不是:

```text
Signal -> better top-k recall
```

这一区别很重要。一个信号只有在以下四个条件同时成立时才是研究上有价值的信息:

1. **Timely**: 在决策 deadline 前可得。
2. **Reliable / calibrated**: 能转成概率、重要度或置信度。
3. **Actionable**: 能改变一个具体动作, 如 admit/fetch/drop/protect/evict/group/throttle。
4. **Resource-valued**: 能降低 `exposed stall`, `H2D bytes`, `PPL cost`, `queue delay` 中至少一个真实目标。

目前最清楚的经验教训是:

```text
有信息 != 有用。
信息必须和动作匹配。
```

例如 SPICE draft 的 within-token future-layer routing 信息很强, 但拿它做 resident eviction 效果不好, 因为 resident `(layer, expert)` 的下次复用主要来自 future tokens, 而不是当前 token 的未来层。这是 information/action mismatch, 不是 draft 没信息。

## 1. 数据流: SPICE decode 中信息何时出现

先把单个 decode token、单个 MoE layer 的数据流摊开。否则 miss-handling 和 eviction 会被混成一个问题。

### 1.1 单层执行时间线

对 token `t`、layer `l`:

```text
input token id / previous generated token
        |
        v
residual hidden h_l_in
        |
        |  (A) pre-attention hidden 可见
        v
self-attention / RMSNorm
        |
        |  (B) attention weights, post-attention hidden 可见
        v
router / gate
        |
        |  (C) true top-k experts, gate weights, router margins 可见
        v
cache lookup for selected experts
        |
        +-- hit: execute exact expert from GPU cache
        |
        +-- miss:
              |
              |  (D) miss-handling decision: fetch / drop / low-precision / substitute
              v
            H2D fetch if admitted
        |
        v
expert outputs + shared expert
        |
        |  (E) actual contribution / hidden delta 可见
        v
h_{l+1}_in
```

SPICE draft 的并行数据流:

```text
anchor hidden h_l_in or h_l_after_attn
        |
        v
draft rollout / LoRE / shared-only surrogate
        |
        v
predicted future-layer expert candidates
        |
        v
prefetch candidate queue
```

硬件数据流:

```text
CPU DRAM / pinned host expert weights
        |
        | H2D DMA, limited PCIe bandwidth
        v
GPU expert cache slots
        |
        v
expert GEMM execution
```

cache 状态:

```text
resident       已在 GPU cache
in_flight      已发起 H2D, 尚未 ready
reserved       计划保留给即将到达/即将使用的 expert
stale          上一个 pass/token 留下, 当前价值不明
current/future 当前 token 后续层预计会用
```

### 1.2 三类动作的不同信息需求

不要把 prefetch、eviction、miss-handling 混成一个 "scheduler"。它们需要的信息不同。

#### Prefetch / admission

发生时间:

```text
expert 真实需要之前
```

它问的问题:

```text
这个 candidate 值不值得提前搬?
能不能在 deadline 前 ready?
会不会污染 cache / 挤掉 demand?
```

需要的信息:

```text
P(expert will be used before deadline)
deadline slack
copy cost / bytes
cache pressure
DMA queue state
wrong-prefetch cost
```

SPICE draft 已经负责生成 future-layer candidates。新工作若碰 prefetch, 应该说 admission/budgeting, 不应说 "我们实现 prefetch"。

#### Eviction

发生时间:

```text
cache 满, 必须给新 expert 腾 slot
```

它问的问题:

```text
resident expert 中谁未来价值最低?
踢掉谁最不可能造成未来 exposed stall?
```

需要的信息:

```text
resident expert 的 next-use probability
next-use deadline
refetch cost
当前 token 后续层是否会用
未来 token 是否会复用
expert 是否 in-flight/demand/current protected
```

关键点:

```text
eviction 的对象是 resident (layer, expert)。
一旦当前 token 已经过了 layer l, 这个 (l,e) 的下次复用通常来自 future token, 不是当前 token 的后续 layer。
```

所以 within-token SPICE draft 不天然适合 eviction。eviction 更需要 cross-token 信息:

```text
token-id prior
LM-head next-token prior
request pool / batch reuse
historical frequency with context
```

SpecMD Least-Stale 已经捕获了最基本的 layer-order/staleness 信息。要超过它, 必须提供它没有的 **concrete future-token identity signal**。

#### Miss-handling

发生时间:

```text
target router 已经选出 true top-k, cache lookup 后发现 miss
```

它问的问题:

```text
这个 missing expert 是否值得现在花 H2D stall 服务?
如果不服务, 质量代价多大?
如果低精度/替代/丢弃, 是否满足 SLO?
```

需要的信息:

```text
verified true gate weight
rank within top-k
router margin / entropy
estimated contribution ||w_e E_e(x)||
drop-induced hidden delta estimate
current latency/SLO pressure
fetch cost and queue delay
```

关键点:

```text
miss-handling 不需要预测 "会不会用"。
它已经知道 expert 被 target router 选中了。
它需要判断 "这个 miss 值不值得服务"。
```

这就是为什么 verified gate weight 对 miss-handling 有用, 但对 prefetch/eviction 不一定足够。

### 1.3 信息早晚表

| 信息 | 出现时间 | 可用于 prefetch/admission | 可用于 eviction | 可用于 miss-handling |
|---|---|---:|---:|---:|
| token id | token step 开始 | 是, token prior | 是, future-token reuse prior | 间接 |
| LM-head next-token distribution | 上一 token logits 后 | 是, next-token prior | 是, next-token cache protection | 间接 |
| pre-attention hidden | layer attention 前 | 可能, 当前层早期预测 | 弱 | 弱 |
| attention weights | attention 中/后 | 可能, context prior | 可能, context reuse | 弱 |
| post-attention hidden | router 前 | 当前层太晚; 可做后续层 anchor | 弱 | 间接 |
| router top-k ids | router 后 | 当前层太晚; 可校准后续 | 弱 | 是, known demand |
| gate weights / margins | router 后 | 当前层太晚; 可校准 | 弱 | 强 |
| actual expert contribution | expert 执行后 | 太晚 | 太晚 | 可训练/解释 sensitivity |
| cache state | runtime 持续可见 | 强 | 强 | 强 |
| DMA queue state | runtime 持续可见 | 强 | 中 | 强 |
| request pool | serving runtime | 强 | 强 | 中 |

### 1.4 两条核心判断

第一:

```text
eviction 依赖 future reuse 信息, 尤其是 future-token reuse。
miss-handling 依赖 current miss value 信息, 尤其是 quality/stall tradeoff。
```

第二:

```text
SPICE draft 主要服务 prefetch/admission。
verified router/gate 主要服务 miss-handling。
token/LM-head/request-pool 主要服务 eviction 和 cross-token movement。
```

因此如果你要主攻 miss-handling:

```text
重点信息 = gate weight / margin / contribution / SLO / queue delay
```

如果你要主攻 eviction:

```text
重点信息 = next-token/token-id prior / request pool / future reuse probability / staleness
```

## 1. 当前已知实验状态

### 1.1 已经基本杀掉的方向

#### A. draft within-token forecast -> eviction

结论: **作为主贡献应杀掉**。

原因:

```text
SPICE draft: predicts current token's future layers
eviction need: predicts future-token same-layer reuse
```

两者不匹配。实验中 forecast eviction 打不过 SpecMD Least-Stale, 所以不能写成 "SPICE forecast makes eviction better"。

可保留用途:

```text
protect already-prefetched future-layer experts
prefetch admission confidence
deadline-aware scheduling diagnostics
```

但不要作为主贡献。

#### B. lossless miss-shadow at batch=1 / single PCIe

结论: **batch=1 单 PCIe 下作为主贡献应杀掉**。

原因:

```text
miss happens
single H2D channel is busy fetching the missing expert
shadow-issued downstream prefetch must wait behind demand fetch
```

所以 "miss 期间同时预取 downstream experts" 在物理上没有免费带宽。除非改成:

```text
batch > 1 request scheduling
multiple DMA channels
chunked/preemptible H2D
serving-level cache-aware batching
```

否则 batch=1 下没有一阶收益。

#### C. gate/rank/mass drop policy superiority

结论: **drop 这个 lever 是真实的, 但单个 policy 的优越性不稳**。

已验证:

```text
Qwen 10% cache, bw12, online teacher-forced:
threshold=0.05 -> PPL +1.8%, stall -33%, drop_rate 27%
threshold=0.10 -> PPL +10%, stall -70%, drop_rate 58%
```

这是很强的 SPICE SLO mode 证据。但 cross-model 上 gate threshold / fixed rank / cumulative mass 没有稳定支配关系。因此不要写:

```text
our gate-threshold policy dominates all miss policies
```

应写:

```text
SPICE exact mode fetches all verified misses.
SPICE SLO mode exposes a latency-quality Pareto by admitting only important misses.
```

#### D. bandit / adaptive prefetch depth

结论: **depth adaptation 被 kill-shot 杀掉**。

`notes/killshot_verdict.md` 显示:

```text
Qwen, bw {5,12,24}, cache {72,144,288}
oracle-best prefetch depth = 2 in all 9 regimes
static depth=2 regret = 0%
roofline depth over-prefetches and has 4-15% regret
```

所以至少在当前建模下, "learned adaptive depth" 没有空间。Bandit 不能当主线, 只能作为 appendix 级别的 controller ablation。

### 1.2 被重新打开的方向

#### token identity -> expert prior

这个方向被重新打开。之前 token-conditional consistency 的指标有缩放 bug, 修正后:

```text
Qwen token-id -> top-k expert recall: 0.582 vs random 0.067
DeepSeek token-id -> top-k expert recall: 0.595 vs random 0.094
```

这说明 routing 不是简单 memoryless。更准确地说:

```text
consecutive tokens are weakly autocorrelated,
but the same token id has a stable expert preference.
```

这很重要, 因为它可能补上 SPICE draft 的短板:

```text
SPICE draft: within-token future-layer structure
token prior: cross-token / next-token expert structure
```

但目前还不能 claim。`experiments/token_table_replay.py` 的 review 指出 DMA backlog、oracle position、LFU baseline 等问题, 当前 token-table utility 结果不能作为结论。必须先修 replay。

## 2. 重新定义: 什么不是 prefetch, 什么仍然值得做

用户提醒是对的:

```text
prefetch 已经通过 SPICE draft model 实现了。
```

因此新工作不能说 "我们做了 prefetch"。但是以下动作仍然不是简单重复 SPICE:

### 2.1 Prefetch admission, not prefetch

SPICE 已经能提出 candidates。问题是:

```text
哪些 candidate 不值得搬?
哪些 candidate 会污染 cache?
哪些 candidate 在 deadline 前根本搬不到?
```

这是 admission / budget allocation, 不是 prefetch 本身。

可用信息:

```text
draft confidence
router margin
LM-head next-token distribution
token prior
DMA queue slack
cache pressure
```

成功标准:

```text
same or lower H2D bytes
same or lower exposed stall
lower wrong-prefetch bytes
no lossless correctness regression
```

### 2.2 Cross-token expert movement, not within-token SPICE prefetch

SPICE draft 的强项是当前 token 的未来层。它不天然解决 future token 的同层 expert reuse。

如果 token-id / LM-head / attention context 能预测下一个 token 的 expert demand, 那是一个不同维度:

```text
within-token: layer l -> l+1...l+K
cross-token: token t -> token t+1, same/all layers
```

这仍然可能会被 SP-MoE / MoE-SpeQ 接近, 所以必须强调区别:

```text
not speculative decoding expert prefetch;
not a second draft model;
use the target model's own next-token distribution / token-conditioned routing prior
to construct expected expert demand under exact target decoding.
```

### 2.3 Miss admission, not prefetch

SPICE exact fallback:

```text
miss -> fetch all true experts
```

SLO mode:

```text
miss -> fetch high-importance experts, drop/low-precision low-importance ones
```

这不是 prefetch, 而是 miss handling。它会牺牲 exactness, 所以必须分模式:

```text
Exact mode: max logit diff = 0
SLO mode: empirical PPL/quality budget
```

### 2.4 Request grouping / serving scheduling, not prefetch

如果有多个 concurrent requests, 可利用信息是:

```text
predicted expert-set overlap across requests
```

动作是:

```text
group tokens/requests by expert overlap
load one expert once, serve multiple tokens
```

这不是单请求 prefetch。它是 serving scheduler, 与 batch=1 单流问题不同。

## 2.5 如果主攻 miss-handling: 信息、动作、数据流

miss-handling 的决策点在 **router 之后、expert 执行之前**。这时系统已经知道:

```text
target-selected experts = true top-k
gate weights = verified importance proxy
cache hit/miss = exact
fetch cost = known from expert size + DMA queue
latency pressure / SLO = runtime state
```

所以 miss-handling 的本质不是预测, 而是 **service admission**:

```text
For each missing selected expert:
  should we pay H2D stall to serve it exactly?
  or degrade it under an SLO?
```

### 2.5.1 miss-handling 输入数据流

```text
post-attention hidden h
    |
    v
router logits z = W_r h
    |
    v
top-k experts E = {e_1...e_k}
gate weights w = {w_1...w_k}
router margin / entropy
    |
    v
cache lookup
    |
    v
for each missing e_i:
    features:
      layer l
      expert id e_i
      gate weight w_i
      rank i
      margin / entropy
      estimated contribution proxy
      fetch_ms(e_i)
      DMA queue delay
      current cache pressure
      target latency/PPL SLO
```

### 2.5.2 miss-handling 可选动作

按保守到激进排序:

```text
FETCH_EXACT
  H2D fetch full expert, exact output, higher stall.

FETCH_LOW_PRECISION
  H2D lower-bit expert, lower stall, approximate output.

SUBSTITUTE
  use similar resident expert, no H2D or less H2D, approximate output.

DROP
  zero this expert contribution, no H2D, approximate output.
```

SPICE 原始 exact fallback 是:

```text
always FETCH_EXACT
```

你的 SLO miss-handling 是:

```text
if expected quality value >= latency cost:
    FETCH_EXACT or FETCH_LOW_PRECISION
else:
    DROP / SUBSTITUTE
```

### 2.5.3 miss-handling 最有用的信息

#### Verified gate weight / rank

优点:

```text
free: router 已经算出
verified: 来自 target router, 不是 draft guess
timely: 正好在 miss decision 前出现
actionable: fetch/drop threshold
```

缺点:

```text
gate weight 是 routing importance, 不是 causal PPL sensitivity
cross-model 上 gate/rank/mass 未稳定支配
```

当前定位:

```text
SLO mode backbone, but not "new superior policy" alone.
```

#### Router margin / entropy

直觉:

```text
router 分布尖锐 -> top expert 更关键, drop 风险高
router 分布平坦 -> 多 expert 可替代性更强, drop 风险低
```

动作:

```text
adaptive threshold:
  high entropy -> allow more drop
  low entropy / high margin -> fetch exact
```

需要实验:

```text
gate-only vs gate+entropy/margin Pareto
```

如果不支配 gate/rank, 只能做解释。

#### Expert contribution estimate

真实贡献是:

```text
|| w_i * E_i(h) ||
```

但真实贡献需要执行 expert, 对 miss 来说太晚。可做近似:

```text
gate weight * expected expert norm by layer/expert
gate weight * small proxy expert output norm
hidden norm / router margin features -> contribution regressor
```

动作:

```text
fetch/drop/low-precision based on estimated delta hidden
```

go/no-go:

```text
must Pareto-dominate gate/rank/mass on Qwen + DeepSeek
```

否则不值得复杂化。

#### Runtime SLO / queue delay

这个信息很关键, 因为 miss-handling 不是只看质量:

```text
same expert miss:
  if DMA queue empty and latency slack exists -> fetch
  if queue long and SLO violation imminent -> drop/low precision
```

这可以把 policy 从固定 threshold 变成 SLO controller:

```text
fetch if quality_value / expected_stall >= current_slo_price
```

这比单纯 gate threshold 更像系统贡献。

### 2.5.4 推荐 miss-handling 公式

不要写成硬阈值, 可以写成 utility:

```text
serve_value(e_i) =
    quality_value(e_i)
    - lambda_slo * expected_stall(e_i)
    - lambda_queue * demand_queue_delay

quality_value(e_i) =
    f(gate_weight, rank, margin, entropy, contribution_proxy)
```

动作:

```text
if serve_value >= 0:
    FETCH_EXACT
else if approximate mode enabled and value_low_precision >= 0:
    FETCH_LOW_PRECISION
else:
    DROP
```

论文表述:

```text
verified SLO-constrained miss admission
```

不要表述为:

```text
better prefetch
```

## 2.6 如果主攻 eviction: 信息、动作、数据流

eviction 的决策点在 **cache pressure** 出现时:

```text
new expert needs a slot
cache is full
choose a resident victim
```

这时被选中的 victim 未必和当前 layer 有直接关系。eviction 的核心是:

```text
Which resident expert has the lowest future value?
```

### 2.6.1 eviction 输入数据流

```text
cache state:
  resident keys (layer, expert)
  last used time
  frequency
  stale/current flag
  ready/in-flight flag
  protected demand/current entries

current execution state:
  token id
  layer l
  target selected experts
  draft future-layer predictions
  LM-head/next-token prior if available
  request pool if serving

hardware state:
  expert bytes
  expected refetch time
  DMA queue
  deadline slack
```

### 2.6.2 eviction 可选动作

```text
evict stale resident
evict low next-use-prob resident
evict resident whose next-use deadline is far
evict resident cheap to refetch
protect in-flight / demand / near-deadline experts
```

### 2.6.3 eviction 最有用的信息

#### Staleness / layer position

这是 SpecMD-LS 的核心:

```text
current forward pass 的未来层专家应该保护
已经过了的 stale experts 优先驱逐
```

它是强 baseline。不要轻易宣称超过它。

#### Token-id / LM-head future-token prior

这是最可能补 LS 的信息。

对于 resident `(layer l, expert e)`:

```text
P(next token uses e at layer l)
= sum_v P(next_token=v) * P(e | layer l, token=v)
```

如果 greedy token 已知:

```text
P(e | layer l, actual_next_token)
```

这正好是 eviction 需要的 future-token reuse 信息。

动作:

```text
protect residents likely needed by next token
evict residents unlikely under next-token prior
```

这比 within-token draft 更匹配 eviction。

#### Request pool / microbatch overlap

serving 时可看多个请求:

```text
P(any pending request soon uses resident e)
```

动作:

```text
protect experts shared by multiple pending requests
evict experts useful to only one low-priority request
```

这可能比 batch=1 eviction 更有收益, 但会进入 serving scheduler 论文。

#### Full router distribution / hidden-state predictor

可能提供未来 token 或当前 context 的 expert prior, 但必须证明比 token/LM-head prior 强。

### 2.6.4 推荐 eviction 公式

对 resident `r=(layer, expert)`:

```text
future_value(r) =
    P_use_before_deadline(r)
    * expected_stall_saved_if_kept(r)
    - cache_opportunity_cost(r)
```

其中:

```text
P_use_before_deadline =
    LS/staleness prior
    + token/LM-head next-token prior
    + request-pool prior
    + draft future-layer prior
```

驱逐:

```text
evict argmin future_value(r)
```

硬保护:

```text
never evict current demand
never evict in-flight demand
protect ready experts with immediate deadline
```

关键提醒:

```text
如果没有 token/LM-head/request-pool 这类 future-token signal,
eviction 很难超过 SpecMD-LS。
```

## 2.7 miss-handling vs eviction 的根本区别

| 维度 | Miss-handling | Eviction |
|---|---|---|
| 决策对象 | 当前 missing selected expert | cache 中某个 resident expert |
| 是否知道当前会用 | 已知, target router 已选择 | 未知, 关心未来是否再用 |
| 最关键问题 | 值不值得服务这个 miss | 谁未来价值最低 |
| 最关键质量信息 | gate weight, rank, margin, contribution | next-use probability, deadline |
| 最关键硬件信息 | fetch delay, queue, SLO pressure | refetch cost, cache pressure, deadline |
| 最匹配的信息 | verified router/gate/current hidden | token/LM-head/request pool/staleness |
| SPICE draft 作用 | 间接, 可估计后续影响 | 只对当前 token 未来层有帮助 |
| 主要 baseline | fetch-all, HOBBIT, rank/drop | LRU/LFU, SpecMD-LS, Belady |

因此:

```text
如果你想做 miss-handling:
  重点不是未来预测, 而是当前 miss 的 quality-vs-stall utility。

如果你想做 eviction:
  重点不是 gate weight, 而是 resident 的 future-token next-use signal。
```

## 2.8 候选: compute-placement miss-handling

这个候选很强, 因为它不是在改 prefetch, 而是在改 miss 的执行位置:

```text
miss 时不搬 17MB expert weights 到 GPU,
而是把几 KB hidden activation 送到 CPU,
在权重所在地用 CPU 精确计算 expert,
再把几 KB 输出/partial sum 回传 GPU。
```

传统 offloading miss:

```text
expert miss
  -> CPU DRAM --(17MB weight H2D)--> GPU cache
  -> GPU executes expert
  -> maybe keep expert resident
```

compute-placement miss:

```text
expert miss
  -> GPU --(hidden activation)--> CPU
  -> CPU executes expert where weights already reside
  -> CPU --(partial output)--> GPU
  -> no GPU cache pollution
```

### 2.8.1 为什么它看起来很有力

以 Qwen1.5-MoE-A2.7B 的 routed expert 形状估算:

```text
expert weights ~= 3 * 2048 * 1408 * 2 bytes ~= 16.5 MiB
hidden activation ~= 2048 * 2 bytes ~= 4 KiB
input + output ~= 8 KiB
```

权重 H2D fetch:

```text
17MB @ 5GB/s  ~= 3.3 ms
17MB @ 12GB/s ~= 1.4 ms
17MB @ 24GB/s ~= 0.7 ms
```

CPU exact expert 的 batch=1 下界主要是读一遍 weights:

```text
17MB / 100GB/s DRAM ~= 0.17 ms
17MB / 50GB/s DRAM  ~= 0.34 ms
```

所以对于 one-shot miss、小 batch、低 PCIe 带宽, CPU compute 可能比 fetch-to-GPU 快 5-20x, 并且不占 cache slot。

### 2.8.2 正确的数据流: 一层传一次 activation, 不是每个 expert 传一次

更合理的数据流是 layer-level partial sum:

```text
GPU:
  router selects top-k experts and gate weights
  partition selected experts:
    GPU-resident hits
    CPU-placement misses
    GPU-admitted misses

  execute GPU-resident experts on GPU
  if any CPU-placement miss:
      send h_l once to CPU with selected expert ids + weights

CPU:
  for all CPU-placement selected experts:
      compute w_i * E_i(h_l)
  sum CPU-side partial routed output
  return one partial vector to GPU

GPU:
  add GPU partial + CPU partial + shared expert
```

这比 "每个 miss expert 都传一次 8KB" 更好。实际 PCIe traffic 应接近:

```text
one hidden vector out + one partial output back + metadata per layer
```

### 2.8.3 它依赖什么信息

compute-placement 的决策不是 "会不会用"。miss 已经证明它会用。它问的是:

```text
这个 miss 应该在哪里服务?
CPU compute once, 还是 GPU fetch/cache?
```

需要的信息:

```text
current miss features:
  selected expert id
  layer
  gate weight / rank / margin
  number of CPU-placement misses in this layer

hardware features:
  CPU exact GEMV latency for this expert shape
  D2H/H2D activation roundtrip latency
  weight H2D latency
  GPU cache pressure
  CPU thread/NUMA queue
  DMA queue

future value features:
  probability this expert is reused soon
  expected saved refetch if admitted to GPU
  token/LM-head next-token prior
  request-pool reuse
  SpecMD-LS staleness
```

核心 utility:

```text
cost_cpu(e) =
    D2H(h) + CPU_GEMV(e, h) + H2D(partial) + sync

cost_fetch_gpu(e) =
    H2D(weights_e) + GPU_compute(e)
    + cache_opportunity_cost(e)
    - expected_future_refetch_saved(e)

cost_drop(e) =
    quality_penalty(e) * slo_price
```

动作:

```text
choose min(cost_cpu, cost_fetch_gpu, cost_drop/low_precision)
```

这就把 miss-handling 和 eviction 连接起来了:

```text
如果未来复用概率低 -> CPU compute once, 不污染 cache
如果未来复用概率高 -> fetch to GPU and admit/cache
如果重要度低且 SLO 紧 -> drop/low precision
```

### 2.8.4 和 Fiddler / HybriMoE 的关系

这条机制本身**不新**。

Fiddler 已经明确提出:

```text
when expert weights are missing on GPU:
  option B: copy weights to GPU and execute
  option C: copy activations to CPU, execute on CPU, copy outputs back
```

Fiddler 的关键判断是:

```text
CPU path 对小 input size 有利;
GPU weight-transfer path 对大 batch/input size 有利。
```

HybriMoE 也在 hybrid CPU-GPU scheduling/cache management 方向, 包括动态 intra-layer scheduling、prefetch/cache 策略。

所以不能把 novelty 写成:

```text
we propose CPU computation for missed experts
```

这会撞 Fiddler。

SPICE 可能的差异必须写成:

```text
Fiddler decides CPU vs GPU mainly from input size and placement latency.
SPICE placement decides CPU vs GPU vs cache admission vs SLO drop using
verified routing, draft/token future-reuse signals, and cache/DMA state.
```

也就是:

```text
not CPU compute itself,
but information-driven miss placement and cache admission.
```

### 2.8.5 什么时候 CPU compute 应该赢

CPU compute 适合:

```text
batch=1 / small per-expert token count
low PCIe bandwidth
large expert weights
low future reuse probability
high GPU cache pressure
miss is likely one-shot
```

GPU fetch/cache 适合:

```text
expert likely reused soon
many tokens route to same expert
CPU queue saturated
cache has low-value victim
PCIe bandwidth high enough
```

drop/low precision 适合:

```text
low gate/contribution value
strict latency SLO
quality budget available
```

### 2.8.6 风险

1. **0.2ms 需要实测。**

   理论 DRAM 带宽下界不等于真实 CPU kernel latency。需要 packed weights、NUMA pinning、AVX512/BF16 或高效 GEMV kernel、线程池、pinned transfer buffers。Python/torch per-expert 调用会把收益吃掉。

2. **小传输不是零开销。**

   8KB bandwidth 成本很小, 但 D2H/H2D roundtrip、同步、CPU dispatch 可能几十微秒到更高。

3. **"精确无损"要小心。**

   CPU BF16/FP32 accumulation 和 GPU BF16 GEMM 可能不 bit-identical。更诚实的说法是:

   ```text
   same expert, same weights, no algorithmic approximation;
   numerical difference must be measured.
   ```

   如果 SPICE exact mode 要求 `max_logit_diff=0`, CPU path 可能只能称为 numerically equivalent within tolerance。

4. **CPU 可能成为新瓶颈。**

   Qwen 每 token 最多 `24 layers * top4 = 96` routed expert calls。如果都走 CPU, 即使 0.2ms/expert 串行也是 19ms/token。必须按 layer 并行多个 experts, 并和 GPU hits/shared compute overlap。

5. **已有工作 baseline 很强。**

   Fiddler/HybriMoE/KTransformers 类系统必须作为 baseline 或至少清楚说明边界。

### 2.8.7 最小验证实验

先做 microbenchmark, 不要先接完整 runtime:

```text
Measure:
  T_cpu_exact(m experts in one layer)
  T_d2h_hidden + T_h2d_partial
  T_fetch_weight_to_gpu + T_gpu_expert
  CPU/GPU parallel partial-sum path

Sweep:
  m = 1,2,4,8 CPU experts per layer
  Qwen and DeepSeek expert shapes
  CPU threads / NUMA / precision
  bw = 5/12/24GB/s equivalent if simulating transfer
```

Go/no-go:

```text
T_cpu_layer(m_miss) <= 0.5 * T_fetch_layer(m_miss)
for realistic m_miss,
and numerical diff is acceptable.
```

然后做 placement replay:

```text
policies:
  SPICE fetch-all
  Fiddler-style input-size threshold
  CPU-always-on-miss
  SPICE-informed placement:
      CPU if one-shot / low reuse
      GPU fetch/cache if high future reuse
      drop/low precision if low SLO value

metrics:
  TPOT
  exposed stall
  CPU busy time
  H2D bytes
  cache pollution
  PPL / numerical diff
```

### 2.8.8 判断

这是**非常强的工程候选**, 甚至可能比继续挖 gate/drop/eviction policy 更有实际收益。

但它不是干净的新 primitive。单独说 "miss 时 CPU 算 expert" 会被 Fiddler 覆盖。

更好的 SPICE 版本是:

```text
SPICE-Placement:
verified expert movement with execution placement.

On a miss, SPICE decides whether to:
  CPU-compute once,
  GPU-fetch-and-cache,
  or degrade under an SLO,
using future-reuse and quality information.
```

如果能证明 SPICE-informed placement 超过 Fiddler-style input-size threshold 和 HybriMoE-style hybrid scheduling, 这可以成为很强的 revision contribution。

## 3. 信息源深分析

下面按信息源逐一分析。

### 3.1 Token ID

#### 信息内容

已验证强信号:

```text
P(expert | layer, token_id)
```

同一个 token id 在不同上下文里有相当稳定的 expert preference。

#### 可得时间

当前 token id 在 decode step 开始时已知。下一个 token id 在 sampling/greedy 之后已知。LM-head distribution 在 sampling 前可得。

#### 可驱动动作

```text
current-token expert prior
next-token expert prewarming
cache protection
cross-token admission
request grouping
```

#### 关键实验

修复 `token_table_replay.py`, 跑:

```text
train prompt split builds token->expert table
eval prompt split uses table only
policies: LRU / LFU(train-static) / SpecMD-LS / token-table / oracle
metrics: exposed_stall, H2D bytes, wrong-prefetch bytes, hit rate, LS->oracle gap closed
```

Go/no-go:

```text
token-table must beat best(LFU, LS) by >=10-15% exposed stall
and close >=25% of LS->oracle gap
without materially increasing H2D bytes.
```

#### 风险

如果它只提高 recall 但增加 H2D 或 cache pollution, 则只是 "LFU with token keys"。

### 3.2 LM-head next-token distribution

#### 信息内容

target LM head 给出:

```text
P(next token = v | current context)
```

结合 token-expert table:

```text
P(expert e at layer l for next token)
= sum_v P(v | context) * P(e | l, token=v)
```

#### 可得时间

当前 token 的 logits 出来后、下一 token decode 开始前可得。对于 greedy decoding, next token id 直接已知。对于 sampling, top-M token distribution 可用于 expected expert demand。

#### 可驱动动作

```text
next-token expert movement plan
cache reservation/protection
drop/admission threshold selection for next step
```

#### 为什么可能比 SPICE prefetch 新

SPICE draft 预测的是当前 token 的后续层 routing。LM-head+token prior 预测的是下一 token 的 expert demand。这是 cross-token dimension。

但要警惕:

```text
SP-MoE / MoE-SpeQ already use speculative decoding and draft tokens for expert prefetch.
```

区别必须是:

```text
no extra draft model required;
uses target LM-head distribution and token-conditioned expert prior;
focus is resource admission/cache protection, not the existence of prefetch.
```

#### 关键实验

在真实 decode trace 上对比:

```text
token-known oracle: use actual next token id
LM-head top-M expectation: use predicted distribution
layer prior / LFU / LS
```

如果 actual-token table 很强而 LM-head top-M 弱, 说明 sampling uncertainty 吃掉了收益。

### 3.3 Hidden state

hidden state 不是一个信息, 至少要分三种。

#### 3.3.1 Post-attention hidden

这是 router 的直接输入:

```text
h_l_after_attn -> router -> true experts
```

信息最强, 但时间最晚。对当前 layer prefetch 太晚, 因为 router 后马上需要 expert。

适合动作:

```text
miss admission
drop/precision selection
current-layer contribution estimation
anchor for draft rollout
```

不适合作为新 prefetch 主线。

#### 3.3.2 Pre-attention hidden

这是更早的当前层信息:

```text
h_l_before_attn -> predict layer-l experts before attention finishes
```

如果能在 attention compute window 内提前发起 H2D, 可能有真实收益。

但窗口很小:

```text
attention latency maybe << expert H2D time
17MB at 12GB/s ~= 1.38 ms
17MB at 5GB/s ~= 3.32 ms
```

所以需要 deadline replay, 不能只报 recall。

关键实验:

```text
train lightweight predictor: pre-attn hidden -> current-layer top-k
simulate copy starts at attention start
compare to router-time demand fetch
metric: exposed stall saved per token
```

与 prior art 风险:

```text
Pre-gated MoE and hidden-state expert predictors already exist nearby.
```

除非这个 predictor 明确服务于 SPICE 的 verified movement/admission, 否则很容易变成又一个 predictor paper。

#### 3.3.3 Previous-layer / draft hidden

这基本就是 SPICE 的已有区域:

```text
h_{l-1}, draft rollout -> future-layer experts
```

不要重新包装成新 prefetch。可做的是:

```text
confidence calibration
admission / budget
wrong-prefetch avoidance
integration with token prior
```

### 3.4 Attention information

#### 信息内容

attention 权重告诉当前 token 从哪些历史 token 读取信息。既然 token id 对 experts 有强 prior, attention 可以把 token prior 变成 context prior:

```text
P(e | current context, layer)
~= sum_i attention_weight(current -> token_i) * P(e | token_i, layer)
```

这比单 token table 更细, 因为它利用上下文。

#### 可得时间

attention 计算过程中或完成后可得。对同层 MLP 来说可能偏晚, 但对后续层、下一 token、cache protection 有用。

#### 可驱动动作

```text
context-conditioned expert prior
confidence weighting for token-table predictions
request grouping by attended semantic tokens
```

#### 关键实验

构造三个 prior:

```text
layer prior
current-token table
attention-weighted history-token table
```

比较:

```text
routing recall
calibration
deadline-aware utility replay
```

Go/no-go:

```text
attention-weighted prior must improve token-table utility, not just recall.
```

#### 风险

attention 本身需要计算完成。如果信息出现太晚, 只能用于后续层/下一 token, 不能用于当前层。

### 3.5 Full router distribution and margin

#### 信息内容

不仅是 top-k expert ids, 而是完整 router distribution:

```text
router_probs over all experts
top-k margin
entropy
tail mass
near-miss experts
```

#### 可得时间

router 后可得。对当前层 prefetch 太晚, 但对 miss/drop/precision 和后续动作有用。

#### 可驱动动作

```text
fetch/drop/low-precision admission
confidence for approximate SLO mode
cache protection of near-boundary experts
future same-layer reuse diagnostic
```

#### 已有风险

`experiments/cross_token_info.py` 曾测 full-prob AUC, 但 notes 里已经记录了潜在问题:

```text
AUC ties handling
Qwen router probs double-softmax
cross-token signal choice possibly too weak
```

所以负结果不能完全当最终结论。

#### 关键实验

修正 full-prob AUC 后重跑:

```text
router full distribution at t,l -> selected experts at t+1,l
compare selected recency / frequency / token-id prior
```

如果 full distribution 明显强于 token prior, 它可能成为 cross-token cache signal。若不强, 降级为 miss admission confidence。

### 3.6 Expert output contribution and sensitivity

#### 信息内容

真实专家贡献:

```text
||w_e E_e(x)||
hidden delta after dropping e
router/logit margin after dropping e
estimated delta NLL
```

已有 Qwen 观察:

```text
routed contribution is about 10% of hidden norm
shared expert larger than routed sum
gate bottom matches contribution bottom ~0.96
```

这解释了 low-importance drop 为何有效。

#### 可驱动动作

```text
miss drop
low-precision fetch
expert substitution
SLO quality controller
```

#### 风险

如果 gate/rank 已经很好地代表 contribution, 那 contribution predictor 不会明显支配已有 policy。

Go/no-go:

```text
contribution/sensitivity predictor must Pareto-dominate gate/rank/mass across Qwen + DeepSeek.
```

否则它只能作为解释性分析。

### 3.7 KV/request pool / serving-level information

#### 信息内容

单请求 batch=1 中很多动作没有空间。但 serving runtime 有:

```text
pending requests
their next token distributions
predicted expert sets
deadline/SLO classes
cache state shared across requests
```

#### 可驱动动作

```text
cache-aware request grouping
microbatch expert-set multiplexing
serve tokens with overlapping experts together
```

已有 multiplex 结果:

```text
Qwen B=32 -> byte/token reduction 2.46x
DeepSeek B=32 -> byte/token reduction 3.31x
```

这是强系统杠杆, 但已经超出原 batch=1 SPICE 边界。它适合另一个方向:

```text
SPICE-Serve: token/expert-aware request scheduling for offloaded MoE serving
```

#### 风险

这会改变 latency/SLO 模型, 需要 queueing analysis, 不能简单塞进原 SPICE paper。

## 4. 和相关工作的边界

### 4.1 SpecMD

SpecMD 系统研究 MoE caching policy, 指出 LRU/LFU 不适合 MoE expert access, 并提出 Least-Stale 减少 collision miss。

参考: https://arxiv.org/abs/2602.03921

边界:

```text
不要和 SpecMD 抢 eviction-only。
可采用 LS 作为 cache baseline/component。
新工作必须利用 SPICE 特有的信息, 如 draft signal、verified gate、LM-head/token prior。
```

### 4.2 FineMoE / fMoE

FineMoE 使用 expert selection patterns 和 prompt semantic hints 指导 prefetch/caching/offloading。

参考: https://arxiv.org/abs/2502.05370

边界:

```text
semantic/prompt/expert-trajectory signal 已经接近 FineMoE。
如果做 hidden/attention/token context, 必须证明它不是简单 semantic prefetch,
而是 target-verified, deadline-aware expert movement/admission。
```

### 4.3 HOBBIT

HOBBIT 做 mixed-precision expert offloading, 对 less-critical cache-miss experts 使用低精度替代以减少 miss latency。

参考: https://arxiv.org/abs/2411.01433

边界:

```text
drop/low-precision miss handling 很接近 HOBBIT。
SPICE 的区别只能是 verified post-router importance + exact/SLO dual mode + on-policy Pareto,
不能声称 "miss approximation" 本身是新。
```

### 4.4 SP-MoE and MoE-SpeQ

SP-MoE 和 MoE-SpeQ 都使用 speculative/draft token 路径做 expert prefetch, 并带有 cutoff/governor/async prefetch runtime。

参考:

- https://arxiv.org/abs/2510.10302
- https://arxiv.org/abs/2511.14102

边界:

```text
不能把 "future-token expert prefetch" 作为孤立贡献。
如果使用 LM-head/token prior, 贡献必须是:
target distribution / token-conditioned table -> resource admission/cache protection,
而不是又一个 speculative decoding prefetcher。
```

### 4.5 MoE-Infinity / activation-aware offloading

MoE-Infinity 使用 activation-aware expert offloading, 关注 expert activation traces 和 memory hierarchy。

参考: https://arxiv.org/abs/2401.14361

边界:

```text
activation trace / frequency / hot-set caching 已经很近。
token-conditioned prior 必须证明比 LFU/static hotset/activation trace 更有 decision utility。
```

### 4.6 Pre-gated MoE / hidden-state prediction

Pre-gated MoE 类工作使用更早的信号预测 active experts, 以便提前准备专家。

参考: https://www.microsoft.com/en-us/research/uploads/prod/2024/05/isca24_pregated_moe_camera_ready.pdf

边界:

```text
pre-attention hidden -> current-layer expert prediction 很容易撞这一类。
除非它被定位为 SPICE verified resource movement 的 admission signal,
否则 novelty 弱。
```

### 4.7 SiDA-MoE / serving-level data-aware scheduling

SiDA-MoE 这类工作已经把 data-aware serving、hash/predictor、expert placement 联系起来。

参考: https://proceedings.mlsys.org/paper_files/paper/2024/file/698cfaf72a208aef2e78bcac55b74328-Paper-Conference.pdf

边界:

```text
request-level scheduling / batching 需要非常清楚的 SLO 和 baseline。
它可能是新方向, 但不是原 SPICE batch=1 prefetch 的小补丁。
```

## 5. 三个候选非增量方向

### 方向 A: SPICE-T, token/LM-head conditioned expert movement

#### 核心思想

SPICE draft 已经覆盖 within-token future-layer。新增 token/LM-head prior 覆盖 cross-token expert demand:

```text
P(expert | next token, layer)
```

动作不是普通 prefetch, 而是:

```text
cross-token cache admission
expert protection
next-token movement budgeting
```

#### 为什么可能非增量

它补的是 SPICE draft 结构性缺口:

```text
draft: within-token
token/LM-head: cross-token
```

如果 utility replay 过关, 这是目前最有希望的新 coupling。

#### 失败模式

```text
token prior recall high but H2D utility low
prefetch too much, cache pollution
LM-head distribution too diffuse
similar to SP-MoE/MoE-SpeQ future-token speculation
```

#### 决定性实验

先修 `token_table_replay.py`:

```text
explicit demand-priority DMA queue
per-sequence oracle
train-static LFU
correct LRU
expert-level coverage
```

然后跑:

```text
token-known table
LM-head expected table
LFU/static
SpecMD-LS
oracle
```

Go/no-go:

```text
>=10-15% exposed stall reduction vs best(LFU, LS)
>=25% LS->oracle gap closed
no H2D byte explosion
cross-model at least Qwen + DeepSeek
```

### 方向 B: attention/hidden-conditioned admission, not prediction

#### 核心思想

不要说 "hidden predicts experts for prefetch"。说:

```text
hidden/attention signal calibrates which SPICE candidates deserve scarce DMA/cache.
```

候选:

```text
pre-attention hidden -> current-layer early confidence
attention-weighted token prior -> context-conditioned cross-token prior
full router margin -> miss/SLO admission
```

#### 为什么可能非增量

如果能证明 hidden/attention 不只是提高 recall, 而是显著减少 wrong-prefetch H2D / cache pollution, 它是 "information quality" 贡献。

#### 失败模式

```text
information too late
predictor too costly
gain absorbed by existing SPICE draft
close to Pre-gated/FineMoE/ExpertFlow
```

#### 决定性实验

统一评价:

```text
same candidate budget
same H2D bytes
compare signal-specific admission:
  draft confidence
  token prior
  attention-weighted token prior
  hidden predictor
  router margin
```

metric:

```text
on-time hit
wrong-prefetch bytes
exposed stall
cache collision
PPL if lossy mode
```

### 方向 C: SPICE as a verified resource-constrained scheduler

#### 核心思想

如果 A/B 不过 utility gate, 就诚实收敛为 systems paper:

```text
SPICE is not only a prefetch method.
It is a framework that exposes which routing signals are actionable under cache/bandwidth/quality budgets.
```

贡献:

```text
1. rigorous information-to-action characterization
2. verified exact mode + SLO miss-admission mode
3. equal-budget source-only baselines
4. corrected negative results showing which intuitive ideas fail
```

#### 为什么仍可发表

这不是单个新 primitive, 而是可靠系统工作:

```text
clear pruning log
on-policy teacher-forced quality/latency Pareto
deadline-aware replay
source-only baselines
hardware/runtime accounting
```

#### 风险

顶会 novelty 会弱于一个真正的新 coupling。必须把 characterization 做扎实, 否则像 "一堆 ablations"。

## 6. 当前最该做的事

### Step 1: 修 token-table utility replay

这是唯一真正重新打开的方向。当前 replay 有 correctness risk, 不修不能判断。

必须修:

```text
1. demand miss must respect DMA backlog or explicit demand-priority queue
2. pending prefetch promoted/delayed correctly under demand priority
3. oracle gpos off-by-one
4. oracle per sequence
5. LFU/static built from train only
6. LRU really LRU
7. coverage = expert-level recall, not token-level has-entry
```

### Step 2: 如果 token-table 过关, 加 LM-head distribution

先做 upper bound:

```text
actual next-token id -> token table expert prior
```

再做 realizable:

```text
LM-head top-M distribution -> expected expert prior
```

如果 actual token strong 但 LM-head weak, 说明收益被 sampling uncertainty 限制。

### Step 3: 如果 token-table 不过关, 转 attention-weighted token prior

因为 token id 已经有信息, attention 是自然的 context 加权方式:

```text
attention over history token ids -> weighted expert prior
```

不要先训练复杂 hidden predictor。先用 attention+token table 这种 training-free signal 看 utility。

### Step 4: hidden-state predictor 最后做

hidden predictor 成本最高、prior art 最近。只有前面 training-free signals 过不了但显示信息缺口时再做。

## 7. 最推荐的论文叙事

如果 token/LM-head utility 成立, 最强叙事是:

```text
SPICE draft solved within-token expert movement.
But offloaded MoE also has a cross-token resource problem:
which experts should survive or be prepared for the next token?
We show routing is strongly token-conditioned, and convert the target LM's own token distribution into a cross-token expert movement prior.
Combined with SPICE's verified draft and exact fallback, this yields a resource-aware expert movement runtime rather than another prefetch predictor.
```

如果 token/LM-head utility 不成立, 最强诚实叙事是:

```text
SPICE as information-to-action characterization:
we systematically test which MoE routing signals are actionable under real cache/DMA constraints.
Most apparent information does not survive the action/utility gate.
The robust positive is a verified exact/SLO dual-mode miss handling Pareto.
```

## 8. 回答用户的关键问题

> 我现在应该做的不是 prefetch 吧, prefetch 不是通过 SPICE 的 draft model 已经实现了吗?

是的。现在不应该再把工作说成普通 prefetch。

更精确的说法是:

```text
SPICE already implements within-token verified prefetch.
The next work must decide how to spend scarce cache/DMA resources around that prefetch:
  admit or reject a predicted expert,
  protect or evict a resident expert,
  fetch or drop a miss under an SLO,
  use token/LM-head information for cross-token movement,
  group requests when serving concurrency exists.
```

所以新工作不是:

```text
more prefetch
```

而是:

```text
resource-constrained expert movement from richer information sources
```

其中最值得优先验证的是:

```text
token/LM-head conditioned cross-token expert movement
```

因为它最明确地补 SPICE draft 的结构性短板。

## 9. 2026-06-03 基于 SPICE PDF 的收束判断

### 9.1 SPICE 原文到底留下了什么洞

读完 `SPICE.pdf` 后, 原文的真实边界可以更精确地写成:

```text
SPICE = within-token, cross-layer, verified expert prefetch.
```

它做得很清楚:

```text
draft predicts future-layer experts
asynchronous H2D prefetch fills GPU expert cache
target router verifies actual top-k
missing selected expert -> synchronous fallback fetch
```

但它没有解决三个资源决策:

```text
1. finite cache 下, 哪些 prefetched/resident experts 值得保留?
2. miss 出现后, 是否必须无条件 fetch exact?
3. H2D budget 紧张时, speculative traffic 和 demand traffic 如何分配?
```

这不是外加问题。SPICE 自己的实验已经暴露边界:

```text
Table I: SPICE H2D 198.80GB, LRU 57.44GB
Table VI: 128 cache slots 下 SPICE 62.07ms 输给 LRU 58.95ms
Table III: Top-K 增大后 fallback 和 PCIe active 很快饱和
```

所以 revision 的切入口应该是:

```text
SPICE predicts what may be useful.
New work decides what is worth serving under cache/bandwidth/quality constraints.
```

而不是:

```text
we prefetch better.
```

### 9.2 batch 设定

SPICE 原文没有把 multi-request batch serving 作为问题设定。唯一明确的 `batch size 8` 是 draft model training, 不是 inference serving。

原文 inference 叙事是:

```text
per-token TPOT/TTFT
per-layer expert movement
single MoE layer timeline
router mismatch -> synchronous fallback
replayed decoding steps
```

因此可以说:

```text
SPICE is written as implicit single-stream decode.
It does not model concurrent request scheduling, queueing delay, or expert sharing across requests.
```

这决定了后续方向:

```text
batch=1 / single-stream:
  miss-handling and cache/bandwidth admission 是自然补强.

batch>1 / serving:
  cross-request multiplexing 是另一个论文 scope, 不是 SPICE 小修.
```

### 9.3 相关工作压力比之前更大

2026 年的新工作让 "utility cache policy" 这条路更拥挤:

```text
SpecMD: standardized cache-policy benchmark + Least-Stale eviction.
MoE-SpAc: speculative activation utility, workload balancer, unified prefetch+eviction utility.
ActiveEvict: proactive eviction + budget-aware routing.
FluxMoE: decouples expert residency, streams experts transiently to prioritize KV/runtime state.
Alloc-MoE: activation budget allocation across layer/token.
```

再加上已有的:

```text
Fiddler: CPU/GPU execution placement for MoE experts.
HybriMoE: hybrid CPU-GPU scheduling + cache management.
HOBBIT: low-precision cache-miss experts.
FineMoE: fine-grained expert patterns + semantic hints for prefetch/caching/offloading.
```

这意味着以下 claim 都很危险:

```text
"we propose eviction utility"              -> MoE-SpAc / SpecMD / ActiveEvict
"we compute missed experts on CPU"         -> Fiddler / HybriMoE
"we approximate low-value miss experts"    -> HOBBIT / Alloc-MoE / AdapMoE
"we use semantic/context routing patterns" -> FineMoE
"we unify prefetch and eviction"           -> MoE-SpAc
```

要保持 SPICE 主线, 新 claim 必须更窄也更硬:

```text
SPICE exposes verified routing information before commitment.
We study which verified or target-derived signals are actionable for
miss service and cache admission under exact/SLO modes.
```

### 9.4 三个真正不同的故事

#### Story A: Verified miss admission for exact/SLO dual-mode SPICE

核心:

```text
SPICE exact mode: fetch every verified miss.
SLO mode: admit/drop/approximate each verified miss by quality-vs-stall utility.
```

最强证据:

```text
Qwen 10% cache, bw12:
drop 27% low-importance miss -> stall -33%, PPL +1.8%
drop 58% -> stall -70%, PPL +10%
```

优点:

```text
和 SPICE 原文关系最紧.
数据已经有 verified on-policy Pareto.
不需要声称新的 predictor.
```

致命弱点:

```text
drop/low-precision miss handling 已经接近 HOBBIT/AdapMoE/Alloc-MoE.
gate/rank/mass policy 轴已验证不稳定支配, 不能当非增量 novelty.
```

适合定位:

```text
practical revision contribution / SLO mode,
not standalone non-incremental mechanism.
```

#### Story B: SPICE-informed miss placement

核心:

```text
On a verified miss:
  CPU-compute once if reuse value is low;
  GPU-fetch-and-cache if future reuse value is high;
  drop/low precision if SLO value is low.
```

数据流:

```text
router verifies selected experts
cache lookup reveals miss
runtime estimates:
  CPU compute cost
  GPU fetch/cache cost
  future reuse value
  SLO quality value
choose execution placement and cache admission
```

优点:

```text
实际系统收益可能最大.
把 miss-handling 和 eviction 统一成 "serve once vs cache for reuse".
自然解释为什么信息有用: information changes placement action.
```

致命弱点:

```text
CPU compute itself不是新机制, Fiddler/HybriMoE 已经覆盖.
必须证明 SPICE-informed utility 超过 Fiddler-style input-size/hardware threshold.
```

最小 kill test:

```text
microbenchmark Qwen/DeepSeek:
  T_cpu_layer(m misses) + activation roundtrip
  vs T_fetch_weights(m misses) + GPU compute

go if:
  CPU placement reduces exposed miss latency by >=2x
  and SPICE-informed placement beats CPU-always/Fiddler-threshold by >=15-20%.
```

适合定位:

```text
strong engineering extension if measured wins are large.
Incremental risk high unless baseline comparison is clean.
```

#### Story C: Cross-token expert movement from target LM-head/token prior

核心:

```text
SPICE draft predicts current token's future layers.
It does not predict which same-layer experts future tokens will reuse.

Use target LM-head / actual next-token id / token-conditioned expert prior
to estimate future-token expert demand, then drive cache protection/admission.
```

为什么这是最可能的非增量点:

```text
It attacks an information/action mismatch in SPICE:
  SPICE information = within-token future-layer
  eviction need = future-token same-layer reuse
```

动作不是普通 prefetch:

```text
protect resident experts likely useful for next token
avoid admitting one-shot misses into GPU cache
reserve cache for predicted cross-token reuse
```

致命弱点:

```text
FineMoE/SpecMD/MoE-SpAc 已经覆盖部分 pattern/utility/cache 空间.
token prior recall 高不等于 utility 高.
当前 token_table_replay 还有正确性 bug, 不能直接用结论.
```

最小 kill test:

```text
fix token_table_replay correctness first:
  demand-priority DMA queue
  per-sequence oracle
  train-static LFU
  correct LRU
  no cross-sequence oracle leakage

then compare:
  LRU / SpecMD-LS / train-LFU / token-table / LM-head top-M / oracle

go if:
  token/LM-head policy beats best(SpecMD-LS, LFU) by >=10-15% exposed stall
  and closes >=25% of LS->oracle gap
  without H2D byte explosion
  on Qwen and DeepSeek.
```

适合定位:

```text
best non-incremental candidate if utility passes.
It gives a real new information-action coupling on top of SPICE.
```

### 9.5 我的推荐

当前不要押 Story A 单独成主线。它是有用的, 但已有工作太近, 且 policy superiority 已经被实验削弱。

当前也不要先押 Story B。它可能很实用, 但 Fiddler/HybriMoE 压力很大, 需要系统实现和 baseline 工作量。

最应该先做的是 Story C 的 kill test:

```text
fix token_table_replay -> run token/LM-head cross-token utility.
```

原因:

```text
1. 它最直接补 SPICE 的结构性缺口.
2. 它不是 "more prefetch", 而是 cross-token cache/admission.
3. 它能回答用户最关心的问题: 有效信息如何真正被踢出来并用于动作.
4. 如果失败, 失败本身也支持 characterization story:
   apparent routing information does not survive the utility gate.
```

最终论文可以有两个层次:

```text
Main contribution if Story C passes:
  SPICE-X: cross-token target-derived expert movement for cache admission.

Support contribution:
  verified SLO miss admission Pareto.

Characterization:
  why within-token draft eviction, miss-shadow, and bandit depth fail.
```

如果 Story C 失败:

```text
Do not force novelty.
Write the paper as a rigorous SPICE characterization + exact/SLO resource scheduler,
or pivot to SPICE-informed placement with Fiddler/HybriMoE baselines.
```

### 9.6 最短行动计划

```text
Day 1:
  fix token_table_replay correctness.

Day 2:
  run Qwen:
    LRU / SpecMD-LS / train-LFU / token-known / LM-head top-M / oracle.

Day 3:
  run DeepSeek same table.

Decision:
  if token/LM-head wins -> build SPICE-X story.
  if not -> run CPU miss-placement microbench.
  if both fail -> write characterization + SLO miss-admission revision.
```

### 9.7 SPICE-REC corrected gate: CPU miss service as recovery window

SPICE-REC 的更准确表述不是:

```text
miss becomes zero PCIe.
```

而是:

```text
miss no longer consumes weight-scale demand H2D.
CPU exact service frees a bounded PCIe window that may be used for downstream verified prefetch.
```

CPU 服务仍然有 KB 级 activation D2H / output H2D, 且有两个必须建模的 contention:

```text
1. CPU output H2D must preempt or bypass low-priority prefetch.
   If the tiny output sits behind a non-preemptible 17MB prefetch, REC loses.

2. CPU expert compute and PCIe DMA both read host DRAM.
   If CPU GEMV is memory-bound, the "free PCIe window" is not actually free.
```

因此 runtime 必须是:

```text
high-priority D2H activation
CPU exact missed expert compute
low-priority chunked downstream prefetch into shadow/protected buffers
high-priority H2D CPU partial output
resume exact target
```

关键不是 issued prefetch bytes, 而是:

```text
timely_useful_prefetch_bytes
```

定义:

```text
timely_useful_prefetch_bytes =
  bytes of REC-issued experts that
    (a) are actually selected by later target routers,
    (b) arrive before their demand deadline,
    (c) would otherwise have caused exposed miss stall,
    (d) are not evicted/polluting main cache before use.
```

窗口效率:

```text
eta_rec =
  timely_useful_prefetch_bytes
  / (B_pcie_under_cpu_compute * T_cpu_service_window)
```

其中 `B_pcie_under_cpu_compute` 必须用并发 CPU expert compute 时测得的 PCIe 带宽, 不能用 standalone copy bandwidth。

#### Corrected kill test

最小对比组:

```text
1. SPICE_fetch_fallback
   miss -> demand H2D weight -> GPU compute.

2. CPU_only_Fiddler
   miss -> CPU exact service, no recovery prefetch.

3. SPICE_REC
   CPU exact service + chunked low-priority downstream shadow prefetch.

4. random_REC
   same REC byte budget, random future experts.

5. dummy_prefetch
   same bytes copied to unused buffers, isolates contention overhead.

6. oracle_shadow_no_pollution
   oracle future experts, protected shadow buffer, no main-cache pollution.

7. spice_rec_no_evict
   REC prefetch only into free/protected shadow slots, no eviction of main cache.
```

Go:

```text
SPICE_REC vs CPU_only_Fiddler:
  exposed stall improves >=15-20%
  wall-clock improves >=8-10%

eta_rec is non-trivial:
  timely_useful_prefetch_bytes / theoretical_window_bytes >= ~20-30%

oracle_shadow_no_pollution upper bound is large:
  otherwise REC has no headroom.
```

No-go:

```text
eta_rec < ~15-20%,
or oracle_shadow_no_pollution barely beats CPU_only,
or dummy_prefetch slowdown cancels the useful prefetch gain,
or CPU output H2D is delayed by low-priority prefetch chunks.
```

#### Defensible novelty boundary

This is not:

```text
we combine CPU execution and prefetch.
```

That is too close to Fiddler / HybriMoE / MoE-SpAc.

The defensible claim, only if the kill test passes, is:

```text
SPICE-REC uses exact CPU miss service as an active bandwidth-decoupling primitive:
the exposed CPU service latency is converted into timely verified downstream expert movement,
under chunked/preemptible DMA and shadow-cache protection.
```

If `SPICE_REC` cannot beat `CPU_only_Fiddler`, the idea collapses to Fiddler plus incidental prefetch and should not be a main contribution.
