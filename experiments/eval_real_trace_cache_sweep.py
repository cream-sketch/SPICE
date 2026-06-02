"""Experiment 1: real-trace cache-budget x bandwidth sweep on offloaded MoE.

实验1：真实 trace 上 cache 预算 x PCIe 带宽 的 sweep。

Goal / 目标:
  Confirm SPICE's own admitted weakness (paper Table VI/III): aggressive
  speculative prefetch trades a lot of H2D bandwidth for latency and LOSES to
  plain LRU under tight cache budget / constrained bandwidth -- exactly the
  regime where offloading matters most. This characterizes the attackable gap
  for a non-incremental SPICE optimization.
  确认 SPICE 自曝弱点: 激进 speculative prefetch 用大量 H2D 带宽换延迟,
  在紧 cache / 紧带宽下输给纯 LRU. 这是非增量优化要攻的靶子.

It replays REAL per-token per-layer routing (from collect_hf_moe_traces.py
traces), simulating a GPU expert cache backed by CPU DRAM over PCIe, and
compares policies on hit rate, H2D traffic, and critical-path stall.
只读重放真实路由, 比较各策略的命中率/H2D流量/关键路径 stall.

Time model (per decode token, per layer), mirroring prefetch_system_sim.py:
  - each layer has a compute window T_comp (attn + K experts);
  - a prefetcher overlaps H2D transfers with compute, bounded by
    bandwidth * T_comp bytes per layer window (overlap budget);
  - a demand miss (needed expert not resident when its layer starts and not
    yet covered by overlapped prefetch) stalls on the critical path for
    expert_bytes / bandwidth.
时间模型: 预取与计算重叠(受带宽窗口约束); 未覆盖的 demand miss 落关键路径 stall.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from common import ensure_dir, write_json


class Cache:
    """Expert cache keyed by (layer, expert) with LRU or staleness eviction.

    专家缓存, 键为 (layer, expert), 支持 LRU 或 staleness 驱逐.
    """

    def __init__(self, capacity: int, policy: str = "lru"):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.policy = policy
        self.store: "OrderedDict[tuple[int, int], int]" = OrderedDict()  # key -> last_touch_step

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self.store

    def touch(self, key: tuple[int, int], step: int) -> None:
        if key in self.store:
            self.store[key] = step
            self.store.move_to_end(key)

    def admit(self, key: tuple[int, int], step: int) -> None:
        """Insert key, evicting per policy when over capacity. 插入并按策略驱逐."""
        if key in self.store:
            self.store[key] = step
            self.store.move_to_end(key)
            return
        self.store[key] = step
        self.store.move_to_end(key)
        while len(self.store) > self.capacity:
            # LRU: evict oldest-touched (front). 同 staleness 的简化退化.
            self.store.popitem(last=False)


def load_token_routes(trace_dir: Path, top_k: int) -> tuple[list[list[list[int]]], int, int]:
    """Load real routing as routes[seq_index] -> list-over-tokens of per-layer expert sets.

    返回 routes[trace] = [ per-token [ per-layer [expert ids] ] ], 以及层数/专家数.
    Each trace file is one sequence (decode stream over its tokens).
    """
    with (trace_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    trace_files = manifest.get("trace_files", [])
    sequences: list[list[list[list[int]]]] = []
    num_layers = 0
    num_experts = 0
    for name in trace_files:
        payload = torch.load(trace_dir / name, map_location="cpu", weights_only=False)
        probs = payload["router_probs"]  # list over layers, each [1, seq, E] or [seq, E]
        if not probs:
            continue
        layer_topk: list[torch.Tensor] = []
        for p in probs:
            p2 = p.float()
            if p2.ndim == 3:
                p2 = p2.reshape(-1, p2.shape[-1])  # [seq, E]
            idx = torch.topk(p2, k=top_k, dim=-1).indices  # [seq, K]
            layer_topk.append(idx)
            num_experts = max(num_experts, int(p2.shape[-1]))
        num_layers = max(num_layers, len(layer_topk))
        token_count = min(t.shape[0] for t in layer_topk)
        # per-token: [layer][token] -> reorganize to [token][layer]
        seq_tokens: list[list[list[int]]] = []
        for tok in range(token_count):
            per_layer = [layer_topk[l][tok].tolist() for l in range(len(layer_topk))]
            seq_tokens.append(per_layer)
        sequences.append(seq_tokens)
    if not sequences:
        raise ValueError(f"no usable traces in {trace_dir}")
    return sequences, num_layers, num_experts


def predict_future_layer(
    mode: str,
    seq_tokens: list[list[list[int]]],
    token: int,
    future_layer: int,
    top_k: int,
    layer_freq: list[dict[int, int]],
) -> list[int]:
    """Predict the expert set for a future layer (used by speculative prefetch).

    预测下游某层的专家集合(供 speculative prefetch 使用).
    - oracle: 真实未来路由(预测上界, 隔离带宽效应);
    - layer_prior: 该层历史频次 top-k(可实现, 无 hidden state).
    """
    if mode == "oracle":
        return seq_tokens[token][future_layer]
    if mode == "layer_prior":
        freq = layer_freq[future_layer]
        if not freq:
            return seq_tokens[token][future_layer]
        return [e for e, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    raise ValueError(f"unknown predict mode: {mode}")


def simulate(
    sequences: list[list[list[list[int]]]],
    num_layers: int,
    policy: str,
    capacity: int,
    expert_bytes: int,
    bandwidth_gbps: float,
    t_attn_ms: float,
    t_expert_ms: float,
    top_k: int,
    lookahead: int,
    predict_mode: str,
) -> dict:
    """Replay all sequences under one policy/budget/bandwidth and return metrics.

    在一种 策略/预算/带宽 下重放全部序列并返回指标.
    policy in {on_demand, lru, spec}. spec = LRU cache + speculative prefetch.
    """
    cache = Cache(capacity, policy="lru")
    bytes_per_ms = bandwidth_gbps * (1024 ** 3) / 1000.0
    t_comp_ms = t_attn_ms + top_k * t_expert_ms  # per-layer compute window 单层计算窗口
    overlap_budget_bytes = bytes_per_ms * t_comp_ms  # H2D bytes that hide under one layer compute

    total_slots = 0
    hit_slots = 0
    demand_miss_slots = 0
    h2d_bytes_prefetch = 0
    h2d_bytes_demand = 0
    stall_ms = 0.0
    wrong_prefetch = 0
    prefetch_issued = 0

    step = 0  # global decode step (token) counter for LRU recency
    layer_freq: list[dict[int, int]] = [dict() for _ in range(num_layers)]

    for seq in sequences:
        for token_idx, per_layer in enumerate(seq):
            for l in range(num_layers):
                # --- speculative prefetch for future layers, overlapped under this layer's compute ---
                if policy == "spec":
                    budget = overlap_budget_bytes
                    for d in range(1, lookahead + 1):
                        fl = l + d
                        if fl >= num_layers:
                            break
                        pred = predict_future_layer(predict_mode, seq, token_idx, fl, top_k, layer_freq)
                        actual = set(per_layer[fl]) if fl < len(per_layer) else set()
                        for e in pred:
                            key = (fl, e)
                            if key in cache:
                                continue
                            if budget < expert_bytes:
                                break
                            cache.admit(key, step)
                            budget -= expert_bytes
                            h2d_bytes_prefetch += expert_bytes
                            prefetch_issued += 1
                            if e not in actual:
                                wrong_prefetch += 1
                        if budget < expert_bytes:
                            break

                # --- demand: experts actually needed at layer l ---
                for e in per_layer[l]:
                    key = (l, e)
                    total_slots += 1
                    if key in cache:
                        hit_slots += 1
                        cache.touch(key, step)
                    else:
                        demand_miss_slots += 1
                        h2d_bytes_demand += expert_bytes
                        stall_ms += expert_bytes / bytes_per_ms  # blocking fetch on critical path
                        cache.admit(key, step)
                    layer_freq[l][e] = layer_freq[l].get(e, 0) + 1
            step += 1

    compute_ms = step * num_layers * t_comp_ms
    total_ms = compute_ms + stall_ms
    h2d_total_gb = (h2d_bytes_prefetch + h2d_bytes_demand) / (1024 ** 3)
    return {
        "policy": policy,
        "predict_mode": predict_mode if policy == "spec" else None,
        "lookahead": lookahead if policy == "spec" else 0,
        "capacity": capacity,
        "bandwidth_gbps": bandwidth_gbps,
        "expert_mb": expert_bytes / (1024 ** 2),
        "total_slots": total_slots,
        "hit_rate": hit_slots / max(1, total_slots),
        "demand_miss_rate": demand_miss_slots / max(1, total_slots),
        "h2d_gb": h2d_total_gb,
        "h2d_prefetch_gb": h2d_bytes_prefetch / (1024 ** 3),
        "h2d_demand_gb": h2d_bytes_demand / (1024 ** 3),
        "wrong_prefetch": wrong_prefetch,
        "prefetch_issued": prefetch_issued,
        "wrong_prefetch_rate": wrong_prefetch / max(1, prefetch_issued),
        "compute_ms": compute_ms,
        "stall_ms": stall_ms,
        "total_ms": total_ms,
        "tpot_ms": total_ms / max(1, step * 1),  # per decode token (across layers)
        "decode_tokens": step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: real-trace cache x bandwidth sweep")
    parser.add_argument("--trace_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--expert_mb", type=float, default=17.0, help="bytes per expert (Qwen MoE ~17MB bf16)")
    parser.add_argument("--t_attn_ms", type=float, default=0.20)
    parser.add_argument("--t_expert_ms", type=float, default=0.05)
    parser.add_argument("--lookahead", type=int, default=6)
    parser.add_argument("--cache_slots", type=str, default="32,64,128,256,512,1024")
    parser.add_argument("--bandwidths_gbps", type=str, default="4,8,16,24")
    parser.add_argument("--policies", type=str, default="on_demand,lru,spec_oracle,spec_prior")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    sequences, num_layers, num_experts = load_token_routes(trace_dir, args.top_k)
    expert_bytes = int(args.expert_mb * 1024 * 1024)
    cache_slots = [int(x) for x in args.cache_slots.split(",") if x.strip()]
    bandwidths = [float(x) for x in args.bandwidths_gbps.split(",") if x.strip()]

    policy_spec = {
        "on_demand": ("on_demand", None),
        "lru": ("lru", None),
        "spec_oracle": ("spec", "oracle"),
        "spec_prior": ("spec", "layer_prior"),
    }
    requested = [p.strip() for p in args.policies.split(",") if p.strip()]

    rows = []
    for bw in bandwidths:
        for cap in cache_slots:
            for name in requested:
                base_policy, pmode = policy_spec[name]
                row = simulate(
                    sequences=sequences,
                    num_layers=num_layers,
                    policy=base_policy,
                    capacity=cap,
                    expert_bytes=expert_bytes,
                    bandwidth_gbps=bw,
                    t_attn_ms=args.t_attn_ms,
                    t_expert_ms=args.t_expert_ms,
                    top_k=args.top_k,
                    lookahead=args.lookahead,
                    predict_mode=pmode or "oracle",
                )
                row["policy_name"] = name
                rows.append(row)

    out = {
        "experiment": "real_trace_cache_sweep_1",
        "trace_dir": str(trace_dir),
        "num_sequences": len(sequences),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "total_routed_slots_per_policy": rows[0]["total_slots"] if rows else 0,
        "config": {
            "top_k": args.top_k, "expert_mb": args.expert_mb,
            "t_attn_ms": args.t_attn_ms, "t_expert_ms": args.t_expert_ms,
            "lookahead": args.lookahead,
        },
        "rows": rows,
    }
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    write_json(out_path, out)
    # 简明表格打印
    print(f"layers={num_layers} experts={num_experts} seqs={len(sequences)} tokens={rows[0]['decode_tokens'] if rows else 0}")
    print(f"{'bw':>4} {'cap':>5} {'policy':>12} {'hit':>7} {'miss':>7} {'h2d_gb':>8} {'stall_ms':>9} {'tpot_ms':>8}")
    for r in rows:
        print(f"{r['bandwidth_gbps']:>4.0f} {r['capacity']:>5} {r['policy_name']:>12} "
              f"{r['hit_rate']:>7.3f} {r['demand_miss_rate']:>7.3f} {r['h2d_gb']:>8.2f} "
              f"{r['stall_ms']:>9.1f} {r['tpot_ms']:>8.3f}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
