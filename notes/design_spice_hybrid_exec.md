# SPICE-HX: 基于 SPICE draft 的细粒度 CPU-GPU 专家协同执行（增强 Fiddler）

## 1. 定位 Positioning
- 主线是 SPICE（verified speculative expert prefetch）。本组件是其上的一个**系统执行层补充实验**。
- 不替换 SPICE；不声称新机制原理（CPU-compute-on-miss 原理承袭 Fiddler 2402.07033）。
- 贡献定位：在**细粒度 MoE（Qwen1.5-MoE 60专家top4 / DeepSeek-V2-Lite 64专家top6）+ 资源受限单卡**下，用 SPICE 的 verified draft 把 Fiddler 的"反应式、串行 CPU/GPU 交接"升级为 **draft 驱动的细粒度重叠流水线**，给出真实 TPOT 收益。精确无损（logits 不变）。

## 2. Fiddler 实测弱点（源码 ref_repo/fiddler/src/fiddler/mixtral.py）
- W1 无重叠：mixtral_forward 内每层先跑 gpu_experts 循环再跑 cpu_experts 循环，串行；CPU 算时 GPU 闲、反之亦然；下一层等本层。
- W2 纯反应式：仅在每层 router 后决策，无 lookahead，不能预取/预热下一层。
- W3 静态全局热度放置：set_expert_loc 一次性按 popularity 放 GPU 永不变；细粒度 MoE popularity Gini~0.17（无热集）→ 近乎失效。
- W4 粗糙静态代价：latency_cpu=7/latency_gpu=70 硬编码；miss 走 expert_placeholder.load_state_dict 同步拷权重、单缓冲、无 pinned/双缓冲/异步。

## 3. 精确性边界（先钉死，避免假重叠）
batch=1 decode 的真实依赖链：attn(l)->gate(l)->experts(l)->+residual->attn(l+1)。
- 不可重叠：experts(l) 与 attn(l+1)（真依赖）。speculative CPU 预计算专家需要 attn(l) 后的真输入，故**不做投机预计算**（否则不精确）。
- 可重叠（全部精确，不改 logits）：
  1. **Prefetch(l+1 预测专家权重) ‖ Compute(l)**：draft 预测下一层 top-k，提前异步 CPU->GPU 预取权重（双缓冲、pinned）。真 router 仍决定；错预测=浪费字节，非错 logits。【SPICE 独有，Fiddler/无 draft 做不到】
  2. **层内 GPU{resident routed + shared expert} ‖ CPU{missed routed}**：独立 CUDA stream + CPU 线程池并发。【Fiddler 此处串行】
  3. **激活 D2H/H2D ‖ 计算**：pinned memory + stream 重叠。

## 4. 与最近工作的差异（codex 重点审）
- vs Fiddler：Fiddler 无预取（反应式）、CPU/GPU 串行、静态热度放置、常数代价。本工作加 draft 预取 + 并发 CPU‖GPU + 重叠传输 + deadline 代价。
- vs HybriMoE(2504.05897)：HybriMoE 也有预取+动态调度，但用其自带（较弱）预测器。差异=SPICE 的 verified frozen-router-rollout draft（recall@4 ~0.7+，远高于 history）作为唯一 lookahead 源 + verified/exact 保证。**最薄环节，需 codex 裁决是否足以区分。**

## 5. 模块设计（函数级 / IO / 数据流）
新增 experiments/spice_hybrid_exec.py（复用 qwen_spice_draft 的 draft、Fiddler 的 run_expert_at_cpu 模式）。

### 5.1 数据结构
- ExpertLoc: dict[(layer,expert)] -> "gpu_resident" | "cpu"。GPU 容量 n_expert_on_gpu。
- PrefetchBuffer: 双缓冲 pinned tensor，承载下一层预测专家权重。

### 5.2 核心函数
- load_qwen_offloaded(model_dir, gpu, n_gpu_expert_ratio) -> (model, expert_loc)
  注意/router/shared expert/embed/norm 常驻 GPU；routed experts 按 ratio 放 GPU 其余留 CPU pinned。
- draft_predict_next_layer(model, hidden_l, attn_mask, top_k) -> List[expert_id]
  复用 qwen_spice_draft.draft_rollout_predict 的单层 horizon=1 版本（frozen attn+router, shared-only 传播）。
- prefetch_experts_async(model, layer, expert_ids, stream) -> handles
  对预测且非 resident 的专家发起 CPU->GPU 异步拷贝（non_blocking, 独立 copy stream）。
- run_layer_hybrid(model, layer, hidden, expert_loc, prefetched, cpu_pool, compute_stream, copy_stream)
    -> hidden_out
  1) attn+gate(GPU) -> 真 top-k + gate weight；2) 划分 resident/prefetched(GPU) vs missed(CPU)；
  3) 并发：GPU stream 跑 {shared + resident/prefetched routed}，CPU 线程池跑 {missed routed}（run_expert_at_cpu）；
  4) 同步合并 index_add；5) 触发 draft_predict + prefetch(l+1)（与本层 3 重叠）。
- generate_hybrid(model, input_ids, n_tokens, expert_loc, ...) -> (output_ids, per_token_latency)
  逐 token 自回归；逐层调 run_layer_hybrid；记录 TPOT。
- verify_exact(model, input_ids): 对比 hybrid logits 与 HF 全 GPU forward，max_logit_diff 须 ~0。

### 5.3 baselines（同一 harness，等资源）
- B0 SPICE-fetch-all：miss 同步拷权重到 GPU 再算（无 CPU-compute，无重叠）= 现状。
- B1 Fiddler-port：CPU-compute miss + Fiddler 的 per-layer 串行 partition（无 draft 预取、无并发）。
- B2 (ours) SPICE-HX：draft 预取 + 并发 CPU‖GPU + 重叠传输。
- (可选) B3 ours 去掉 draft 预取（ablation，证明预取贡献）。

## 6. 评测与方向更新规则（DIRECTION-UPDATE）[codex-revised 2026-06-03]
codex 修订：B0(fetch-all) 降为 sanity baseline；真实基线是 **B1-parallel(CPU-miss 计算 ‖ GPU 工作, 无预取)**；预取必须 l+h(h=8~10) 带宽感知队列，非 l+1；正确性指标改 **routing-exact + 数值容差**（不承诺 max_logit_diff=0）；新增 **oracle-prefetch 上界**。
- 指标：真实 wall-clock TPOT（ms/token），exposed stall，bytes moved/token，routing-exact 率 + logit 数值误差。
- 模型：Qwen1.5-MoE-A2.7B + DeepSeek-V2-Lite，GPU routed-expert 容量 10%/30%。
- 决定性实验(先跑, codex D)：组件标定 + trace deadline replay，对比 B1-parallel / oracle-prefetch / SPICE-prefetch / HybriMoE-style-predictor。
- GO：oracle-prefetch 比 B1-parallel >=10-15% 且 SPICE 捕获 >=60-70% oracle exposed-stall 削减；论文级再要 SPICE 比 HybriMoE-predictor >=5-10% TPOT。
- NO-GO（诚实负结果, 写入 paper I characterization）：若 batch=1 下 CPU-miss 计算已被 GPU 工作藏住（exposed stall ~0）→ "batch=1 CPU-compute 已近最优, 无调度空间"；或 oracle <10% → 预取无价值；或重叠被 GIL/PCIe 吃掉。

## 6.1 KILL-SHOT 先测（最便宜, 决定一切）
真实 Qwen batch=1 decode 单层组件时间：t_attn, t_shared, t_gate, t_expert_gpu, t_expert_cpu(0.32ms 已测), t_copy_h2d(17MB pinned async)；real-trace 在 10%/30% residency 的每层 miss 数分布；CPU‖GPU 真实 overlap_factor。
判据：若 t_expert_cpu * n_miss_per_layer <= hideable_gpu(=t_attn+t_shared+t_gate+t_expert_gpu*n_resident) → CPU miss 全藏 → 无收益(NO-GO)。否则有 exposed stall → 进 deadline replay。

## 7. 复用清单
- ref_repo/fiddler: run_expert_at_cpu 模式、CPU/GPU partition、set_expert_loc、expert_placeholder 双缓冲思路（改成异步双缓冲）。
- experiments/qwen_spice_draft.py: draft_rollout_predict / shared_only_mlp_forward / true_forward。
- experiments/cpu_expert_bench.py: 已测 CPU 专家 0.32ms vs fetch 3.22ms@5GB/s 的标定。
