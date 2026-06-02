# SPICE 仓库实验复现记录 (2026-06-02)

环境: moe-server-248 (A800), conda env sparmoe (torch2.5.1/cu124, tf4.46). 真实模型 Qwen1.5-MoE-A2.7B-Chat.

## 1. 合成 draft 路径 (train_draft_model + eval_draft_prefetch, layers8/experts16/topk2/rank16, 800 steps)
- verified prefetch: hit=0.9965, fallback=0.0035, wrong_prefetch=0.244, avg_lookahead_depth=2.35
- 静态 draft slot_hit≈0.50, exact_set≈0.27
- 对照 validation/anchored_eval (hit 0.9924/fb 0.0076): 复现一致 (PASS).

## 2. 合成 system sim (prefetch_system_sim --mode main, steps512, expert_mb8)
| policy | hit | fallback | h2d_gb | tpot_ms | pcie_active |
|---|---|---|---|---|---|
| naive | 0.000 | 1.000 | 384.0 | 71.25 | 0.44 |
| lru / moe_offloading | 0.850 | 0.150 | 57.44 | 44.67 | 0.105 |
| pregated | 0.940 | 0.060 | 149.8 | 41.86 | 0.291 |
| spice | 0.972 | 0.028 | 105.0 | 41.84 | 0.204 |
- LRU 57.44GB 与 validation 完全一致; SPICE 最高 hit 最低 tpot. 复现 (PASS). 注: 这是合成 trace.

## 3. 真实 Qwen traces (eval_hf_trace_prefetch, 我采集的 160 条 WikiText, 27111 tokens, top_k4)
- oracle: hit=1.0000
- anchor_repeat: hit=0.3234, fallback=0.6766, wrong_prefetch=0.929, depth=5.38
- 对照 validation/qwen_moe_trace (anchor_repeat 0.342): 复现一致 (PASS).

## 关键结论 (改进的出发点)
- SPICE 的 99.76% 高命中来自**合成 target 的 draft 路径** (以及论文的真实模型 wrapper).
- 仓库内能在**真实 Qwen 路由**上运行的只有简单 history 预测器 -> hit 仅 0.32 (oracle 1.0).
- 真实模型的 SPICE draft predictor (论文宣称, 仓库未完整提供 model-specific wrapper) 是补 gap 的关键缺口.
- 改进方向 (codex 定): 在此 baseline 上, 把 cache 调度从 hit-rate 最大化改为 "cache+DMA deadline 下最小化暴露 demand bytes" 的 verified Belady 调度器.
