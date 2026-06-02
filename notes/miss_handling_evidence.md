# Miss-handling 方向的实证证据 (2026-06-02)

方向 (用户+codex 一致): 完善 SPICE = 资源受限 GPU 上 verified、importance-aware 的 miss-handling
(+ 采用 SpecMD-LS 作为 eviction 组件, 自研 forecast 驱逐已证伪 notes/exp1_eviction_verdict.md).
对齐 SpecMD 四要素: routing/prefetch(SPICE/SOTA已占) | eviction(采用LS) | **miss-handling(主贡献)**.

## importance 偏斜 (per-rank gate 权重)
- Qwen1.5-MoE (top-4, 650k token-layer): rank 0.157/0.092/0.062/0.045; top-4 仅占 35.6% softmax 质量 (1 shared expert backbone).
- DeepSeek-V2-Lite (top-6, 524k, 2 shared experts): 路由极偏斜, rank-1≈0.957 (raw softmax over 64). 注意: 权重≠输出贡献, 见下 PPL.

## drop-quality PPL (实测, 真实权重) —— miss-handling 的 task-performance 轴
| drop 最低 n 个 routed | Qwen ΔPPL | DeepSeek ΔPPL |
|---|---|---|
| 1 | +1.4% | +0.88% |
| 2 | +6.3% | +4.5% |
| 3 | +37%(到rank1) | +9.1% |
注: Qwen top-4 drop3=只剩rank1; DeepSeek top-6 drop4=+27%, drop5(剩rank1)=+102%.

## 结论
- 跨 2 个模型: **miss 最低 1 个 routed 专家时 drop ≈ +1% PPL**, 省掉一次 fetch (5GB/s 下 ~3.4ms). 可随延迟压力 drop 更多, 质量优雅降级 = latency-quality Pareto.
- 这是 SpecMD 点名的 miss-handling "balance user-experience vs task-performance" 的真实杠杆.
- 关键: SPICE 的 verification 顺带给出**真实 gate 权重** -> 可做 per-miss importance 决策 (高 rank fetch / 低 rank drop), lossless-where-it-matters.

## 下一步
1. 把 miss-handling 策略 (importance 阈值 + latency SLO) 接入 deadline-aware 模拟器, 对照 SPICE(fetch-all) / SpecMD(fetch/drop/substitute) 的 latency-quality Pareto, 资源受限 regime.
2. CUDA demand-priority DMA 调度器 (真实 wall-clock).
