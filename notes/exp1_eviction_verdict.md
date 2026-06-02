# 实验1 (eviction headroom kill test) 结论 — 2026-06-02

代码已过 codex review 修 5 bug; 真实 Qwen WikiText traces (6000 tokens, top_k4), demand-only, bw=5GB/s, expert 17MB. 证据: notes/evidence/exp1_eviction_headroom_v2.json.

## 命中率 / 暴露 stall (ms/token)
| cache | LRU hit | LS hit | LFU hit | Belady hit | Belady vs LRU 暴露stall头room |
|---|---|---|---|---|---|
| 1% (14)  | 0.000 | 0.009 | 0.000 | 0.068 | ~6% |
| 5% (72)  | 0.000 | 0.059 | 0.000 | 0.270 | 巨大(LRU=0) |
| 10% (144)| 0.191 | 0.119 | 0.100 | 0.402 | 26.1% |
| 20% (288)| 0.318 | 0.235 | 0.219 | 0.573 | 37.4% |
| 50% (720)| 0.621 | 0.550 | 0.546 | 0.843 | 58.5% |

## 判定: 有条件 GO
- oracle 驱逐在 5-50% cache 比 SPICE 的 LRU 驱逐降暴露 stall 26-58% -> 存在 forecast 可填的真实头room.
- tight regime (1-2%) cold-miss 主导, 谁都救不了 (头room 仅 6-12%); 收益在中等 cache.

## 诚实 caveat (不可隐瞒)
1. 该 demand-only 测试对 SpecMD-LS 不公平: LS 的优势在"保护已预取未来专家免被驱逐(collision miss)", 本测试无预取, 故 LS 退化(甚至输 LRU). 公平对比需 WITH-prefetch 版 (后续补).
2. 廉价静态代理抓不住 oracle 头room: LFU(0.100) < LRU(0.191); 纯频率无效. 说明捕获头room需要 per-token 多层动态 forecast.

## 下一步 (改进 SPICE 的核心)
把真实 Qwen 的 SPICE draft predictor 跑通 (仓库缺失/论文宣称的 model-specific wrapper), 用其 forecast 驱动 Belady 式驱逐 (value(e)=P(deadline前再用)*节省stall/bytes), 对照 oracle 上界 + LRU/LS. 这是 codex 的 "forecast gap" gate: 可实现 forecast 须抓住 oracle 头room 的大部分.
