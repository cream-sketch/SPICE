"""Experiment 1: eviction-policy headroom on real MoE routing traces.

实验1：真实 MoE 路由 trace 上, 驱逐策略的头room (oracle-Belady vs SpecMD-LS vs LRU).

Codex-specified kill test (does NOT need the SPICE draft model):
the non-incremental thesis is "SPICE turns a verified routing draft into an
online stochastic Belady scheduler under hard cache + PCIe budgets". The very
first go/no-go is whether there is ANY headroom over SpecMD's Least-Stale (LS)
eviction at all. If perfect-future Belady eviction barely beats LS under a tight
cache, a forecast-driven cache signal is worthless and the thesis dies.
codex 指定的 kill test (不需要 draft 模型): 若完美未来的 Belady 驱逐在紧 cache 下
也只能微弱胜过 LS, 则 forecast 驱动的 cache 信号无价值, thesis 死.

Model: replay the REAL per-token per-layer top-K routing as a sequential decode
stream. Demand-only fetch over a single PCIe channel; a demand miss stalls the
critical path for expert_bytes / bandwidth (deadline-aware exposed stall, no
speculative overlap in this isolation test). Compare eviction policies on
exposed_H2D_stall_ms_per_token and hit rate, across cache budgets x bandwidth.
模型: 顺序重放真实路由; demand miss 在单 PCIe 通道上阻塞 critical path;
对比各驱逐策略的 exposed_H2D_stall_ms_per_token 与命中率.

Eviction policies / 驱逐策略:
  lru          - least recently used (SPICE/SpecMD baseline)
  lfu          - least frequently used
  specmd_ls    - Least-Stale: assume cyclic layer reuse; evict the resident
                 whose layer is FARTHEST in the forward cycle (max cyclic
                 distance to next expected use). SpecMD's key idea.
  oracle_belady- evict the resident whose ACTUAL next use (from the real future
                 stream) is farthest; never-again -> evict first. Optimal,
                 upper bound that the SPICE forecast would try to approximate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import ensure_dir, write_json


def load_token_routes(trace_dir: Path, top_k: int) -> tuple[list[list[list[int]]], int, int]:
    """Load real routing as sequences[seq][token][layer] = list of expert ids.

    加载真实路由为 sequences[序列][token][层] = 专家id列表.
    """
    with (trace_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    sequences: list[list[list[list[int]]]] = []
    num_layers = 0
    num_experts = 0
    for name in manifest.get("trace_files", []):
        payload = torch.load(trace_dir / name, map_location="cpu", weights_only=False)
        probs = payload["router_probs"]
        if not probs:
            continue
        layer_topk: list[torch.Tensor] = []
        for p in probs:
            p2 = p.float()
            if p2.ndim == 3:
                p2 = p2.reshape(-1, p2.shape[-1])
            layer_topk.append(torch.topk(p2, k=top_k, dim=-1).indices)
            num_experts = max(num_experts, int(p2.shape[-1]))
        num_layers = max(num_layers, len(layer_topk))
        token_count = min(t.shape[0] for t in layer_topk)
        sequences.append([[layer_topk[l][tok].tolist() for l in range(len(layer_topk))]
                          for tok in range(token_count)])
    if not sequences:
        raise ValueError(f"no usable traces in {trace_dir}")
    return sequences, num_layers, num_experts


def build_access_stream(
    sequences: list[list[list[list[int]]]], num_layers: int, max_tokens: int
) -> list[tuple[int, list[int]]]:
    """Flatten to an ordered decode stream of (layer, expert_ids) accesses.

    展平为有序 decode 流: 每个元素 = (层, 专家id列表). 序列内 token 顺序保留;
    序列之间视为独立 decode 流首尾相接 (近似; 跨序列复用偏少, 不影响紧 cache 结论).
    max_tokens 限制总 decode token 数, 控制纯 Python 重放时间.
    """
    stream: list[tuple[int, list[int]]] = []
    tokens = 0
    for seq in sequences:
        for per_layer in seq:
            for l in range(num_layers):
                stream.append((l, per_layer[l]))
            tokens += 1
            if tokens >= max_tokens:
                return stream
    return stream


def simulate(
    stream: list[tuple[int, list[int]]],
    num_layers: int,
    policy: str,
    capacity: int,
    expert_bytes: int,
    bandwidth_gbps: float,
    t_layer_ms: float,
) -> dict:
    """Replay the demand-only stream under one eviction policy and budget.

    在一种驱逐策略与预算下重放 demand-only 流. 返回命中率与 exposed stall.
    """
    bytes_per_ms = bandwidth_gbps * (1024 ** 3) / 1000.0
    fetch_ms = expert_bytes / bytes_per_ms  # 单专家同步取的时间

    # Belady: 预计算每个 key 的出现位置列表 + 指针
    occ: dict[tuple[int, int], list[int]] = {}
    if policy == "oracle_belady":
        for i, (l, experts) in enumerate(stream):
            for e in experts:
                occ.setdefault((l, e), []).append(i)
    occ_ptr: dict[tuple[int, int], int] = {k: 0 for k in occ}

    cache: set[tuple[int, int]] = set()
    last_used: dict[tuple[int, int], int] = {}   # for LRU (global access counter)
    freq: dict[tuple[int, int], int] = {}        # for LFU
    key_layer: dict[tuple[int, int], int] = {}   # for LS (cyclic distance)

    total_slots = 0
    hit_slots = 0
    miss_slots = 0
    counter = 0
    INF = len(stream) + 10

    def next_use(key: tuple[int, int], cur_pos: int) -> int:
        positions = occ.get(key, ())
        ptr = occ_ptr.get(key, 0)
        while ptr < len(positions) and positions[ptr] <= cur_pos:
            ptr += 1
        occ_ptr[key] = ptr
        return positions[ptr] if ptr < len(positions) else INF

    def evict_one(cur_layer: int, cur_pos: int) -> None:
        if policy == "lru":
            victim = min(cache, key=lambda k: last_used[k])
        elif policy == "lfu":
            victim = min(cache, key=lambda k: (freq[k], last_used[k]))
        elif policy == "specmd_ls":
            # 假设每层每 token 周期复用: 下一次使用的 cyclic 距离 = (key_layer - cur_layer) mod L
            # 距离最大者最久才会再用 -> 驱逐 (Least-Stale 核心)
            victim = max(cache, key=lambda k: ((key_layer[k] - cur_layer) % num_layers, last_used[k]))
        elif policy == "oracle_belady":
            victim = max(cache, key=lambda k: next_use(k, cur_pos))
        else:
            raise ValueError(f"unknown policy {policy}")
        cache.discard(victim)
        last_used.pop(victim, None); freq.pop(victim, None); key_layer.pop(victim, None)

    exposed_stall_ms = 0.0
    decode_tokens = len(stream) // max(1, num_layers)

    for pos, (l, experts) in enumerate(stream):
        for e in experts:
            key = (l, e)
            total_slots += 1
            counter += 1
            if key in cache:
                hit_slots += 1
                last_used[key] = counter
                freq[key] = freq.get(key, 0) + 1
            else:
                miss_slots += 1
                exposed_stall_ms += fetch_ms  # demand miss 同步取, 落 critical path
                while len(cache) >= capacity and cache:
                    evict_one(l, pos)
                cache.add(key)
                last_used[key] = counter
                freq[key] = freq.get(key, 0) + 1
                key_layer[key] = l

    compute_ms = decode_tokens * num_layers * t_layer_ms
    total_ms = compute_ms + exposed_stall_ms
    return {
        "policy": policy,
        "capacity": capacity,
        "bandwidth_gbps": bandwidth_gbps,
        "expert_mb": expert_bytes / (1024 ** 2),
        "total_slots": total_slots,
        "hit_rate": hit_slots / max(1, total_slots),
        "miss_rate": miss_slots / max(1, total_slots),
        "exposed_stall_ms": exposed_stall_ms,
        "exposed_stall_ms_per_token": exposed_stall_ms / max(1, decode_tokens),
        "compute_ms": compute_ms,
        "tpot_ms": total_ms / max(1, decode_tokens),
        "decode_tokens": decode_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: eviction headroom on real MoE traces")
    parser.add_argument("--trace_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--expert_mb", type=float, default=17.0)
    parser.add_argument("--t_layer_ms", type=float, default=0.40, help="per-layer compute time (attn+experts)")
    parser.add_argument("--cache_slots", type=str, default="14,29,72,144,288,720")
    # demand-only 下 miss 数与带宽无关(带宽只线性缩放 stall), 故单带宽足以判定头room
    parser.add_argument("--bandwidths_gbps", type=str, default="5")
    parser.add_argument("--policies", type=str, default="lru,lfu,specmd_ls,oracle_belady")
    parser.add_argument("--max_decode_tokens", type=int, default=8000)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    sequences, num_layers, num_experts = load_token_routes(trace_dir, args.top_k)
    stream = build_access_stream(sequences, num_layers, args.max_decode_tokens)
    expert_bytes = int(args.expert_mb * 1024 * 1024)
    cache_slots = [int(x) for x in args.cache_slots.split(",") if x.strip()]
    bandwidths = [float(x) for x in args.bandwidths_gbps.split(",") if x.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    total_distinct = num_layers * num_experts

    rows = []
    for bw in bandwidths:
        for cap in cache_slots:
            for pol in policies:
                rows.append(simulate(stream, num_layers, pol, cap, expert_bytes, bw, args.t_layer_ms))

    out = {
        "experiment": "real_trace_eviction_headroom_1",
        "trace_dir": str(trace_dir),
        "num_sequences": len(sequences),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "total_distinct_layer_experts": total_distinct,
        "stream_len": len(stream),
        "config": {"top_k": args.top_k, "expert_mb": args.expert_mb, "t_layer_ms": args.t_layer_ms},
        "rows": rows,
    }
    ensure_dir(Path(args.out).parent)
    write_json(Path(args.out), out)

    # 打印 + Belady vs LS 头room
    print(f"layers={num_layers} experts={num_experts} distinct={total_distinct} "
          f"stream={len(stream)} tokens={rows[0]['decode_tokens']}")
    print(f"{'bw':>4} {'cap':>5} {'%':>5} {'policy':>13} {'hit':>7} {'stall/tok(ms)':>13}")
    by_key: dict[tuple[float, int], dict[str, float]] = {}
    for r in rows:
        pct = 100.0 * r["capacity"] / max(1, total_distinct)
        print(f"{r['bandwidth_gbps']:>4.0f} {r['capacity']:>5} {pct:>4.1f}% {r['policy']:>13} "
              f"{r['hit_rate']:>7.3f} {r['exposed_stall_ms_per_token']:>13.3f}")
        by_key.setdefault((r["bandwidth_gbps"], r["capacity"]), {})[r["policy"]] = r["exposed_stall_ms_per_token"]
    print("\n[Belady vs Least-Stale headroom] (go if >=25%)")
    for (bw, cap), d in sorted(by_key.items()):
        if "oracle_belady" in d and "specmd_ls" in d and d["specmd_ls"] > 0:
            head = (d["specmd_ls"] - d["oracle_belady"]) / d["specmd_ls"]
            print(f"  bw={bw:.0f} cap={cap}: LS={d['specmd_ls']:.3f} Belady={d['oracle_belady']:.3f} "
                  f"headroom={head*100:.1f}%")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
