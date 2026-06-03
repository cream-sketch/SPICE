"""Experiment 1: eviction-policy headroom on real MoE routing traces.

实验1：真实 MoE 路由 trace 上, 驱逐策略的头room (oracle-Belady vs SpecMD-LS vs LRU).

Codex-specified kill test (does NOT need the SPICE draft model):
the non-incremental thesis is "SPICE turns a verified routing draft into an
online stochastic Belady scheduler under hard cache + PCIe budgets". The very
first go/no-go is whether there is ANY headroom over SpecMD's Least-Stale (LS)
eviction at all. If perfect-future Belady eviction barely beats LS under a tight
cache, a forecast-driven cache signal is worthless and the thesis dies.
codex 指定 kill test(不需 draft): 若完美未来 Belady 驱逐在紧 cache 下也只微弱胜 LS,
则 forecast 驱动的 cache 信号无价值, thesis 死.

Model (demand-only, deadline-aware): replay REAL per-token per-layer top-K
routing as a sequential decode stream, flattened to PER-EXPERT accesses so the
Belady oracle and the cache operate at expert granularity (codex review fix).
Each sequence (prompt) is simulated INDEPENDENTLY with a cold cache so the
oracle cannot see across sequences. A demand miss is a synchronous PCIe fetch
that stalls the critical path for expert_bytes / bandwidth (no speculative
overlap in this isolation test). Miss count -> exposed stall is linear in
bandwidth, so one bandwidth suffices for the headroom ratio.
模型(demand-only, 按专家粒度, 按序列独立, 冷启动): 真实路由展平为 per-expert 访问;
demand miss 同步取并阻塞 critical path; 各驱逐策略对比命中率与 exposed stall.

Eviction policies / 驱逐策略:
  lru          - least recently used (SPICE/SpecMD baseline)
  lfu          - least frequently used
  specmd_ls    - Least-Stale: cyclic layer reuse; evict resident whose layer is
                 FARTHEST in the forward cycle (same-layer => a full cycle away),
                 LRU fallback. SpecMD's key idea.
  oracle_belady- evict resident whose ACTUAL next use (real future of THIS
                 sequence) is farthest; never-again -> evict first. Optimal upper
                 bound the SPICE forecast would try to approximate.
"""

from __future__ import annotations

import sys, pathlib  # bootstrap: resolve core/common (lives in ../core) regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "core"))

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from common import ensure_dir, write_json


def load_token_routes(trace_dir: Path, top_k: int) -> tuple[list[list[list[list[int]]]], int, int]:
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


def flatten_sequence(seq: list[list[list[int]]], num_layers: int) -> list[tuple[int, int]]:
    """One sequence -> ordered PER-EXPERT access list [(layer, expert), ...].

    单序列展平为按专家粒度的有序访问列表 (codex review 修复: Belady/cache 按专家粒度).
    """
    flat: list[tuple[int, int]] = []
    for per_layer in seq:
        for l in range(num_layers):
            for e in per_layer[l]:
                flat.append((l, e))
    return flat


def simulate_sequence(flat: list[tuple[int, int]], num_layers: int, policy: str, capacity: int) -> tuple[int, int]:
    """Replay one sequence's per-expert stream under one eviction policy.

    在一种驱逐策略下重放单序列(冷启动). 返回 (hits, misses).
    """
    occ: dict[tuple[int, int], list[int]] = {}
    if policy == "oracle_belady":
        for i, key in enumerate(flat):
            occ.setdefault(key, []).append(i)
    occ_ptr: dict[tuple[int, int], int] = {k: 0 for k in occ}
    INF = len(flat) + 10

    cache: set[tuple[int, int]] = set()
    last_used: dict[tuple[int, int], int] = {}
    freq: dict[tuple[int, int], int] = {}
    key_layer: dict[tuple[int, int], int] = {}

    hits = 0
    misses = 0

    def next_use(key: tuple[int, int], cur_pos: int) -> int:
        positions = occ.get(key, ())
        ptr = occ_ptr.get(key, 0)
        while ptr < len(positions) and positions[ptr] <= cur_pos:
            ptr += 1
        occ_ptr[key] = ptr
        return positions[ptr] if ptr < len(positions) else INF

    def ls_dist(k: tuple[int, int], cur_layer: int) -> int:
        # same-layer residents next recur a FULL cycle later (codex fix #2)
        d = (key_layer[k] - cur_layer) % num_layers
        return num_layers if d == 0 else d

    def evict_one(cur_layer: int, cur_pos: int) -> None:
        if policy == "lru":
            victim = min(cache, key=lambda k: last_used[k])
        elif policy == "lfu":
            victim = min(cache, key=lambda k: (freq[k], last_used[k]))
        elif policy == "specmd_ls":
            # farthest cyclic reuse first; LRU fallback (codex fix #2,#3)
            victim = max(cache, key=lambda k: (ls_dist(k, cur_layer), -last_used[k]))
        elif policy == "oracle_belady":
            victim = max(cache, key=lambda k: next_use(k, cur_pos))
        else:
            raise ValueError(f"unknown policy {policy}")
        cache.discard(victim)
        last_used.pop(victim, None)
        freq.pop(victim, None)
        key_layer.pop(victim, None)

    for pos, (l, e) in enumerate(flat):
        key = (l, e)
        if key in cache:
            hits += 1
            last_used[key] = pos
            freq[key] = freq.get(key, 0) + 1
        else:
            misses += 1
            while len(cache) >= capacity and cache:
                evict_one(l, pos)
            if capacity >= 1:
                cache.add(key)
                last_used[key] = pos
                freq[key] = freq.get(key, 0) + 1
                key_layer[key] = l
    return hits, misses


def run_policy(
    flats: list[list[tuple[int, int]]],
    token_counts: list[int],
    num_layers: int,
    policy: str,
    capacity: int,
    expert_bytes: int,
    bandwidth_gbps: float,
    t_layer_ms: float,
    top_k: int,
) -> dict:
    """Aggregate one policy/budget over all (independent) sequences.

    在所有(独立)序列上聚合一种策略/预算的结果.
    """
    bytes_per_ms = bandwidth_gbps * (1024 ** 3) / 1000.0
    fetch_ms = expert_bytes / bytes_per_ms

    total_hits = 0
    total_misses = 0
    for flat in flats:
        h, m = simulate_sequence(flat, num_layers, policy, capacity)
        total_hits += h
        total_misses += m
    total_slots = total_hits + total_misses
    total_tokens = sum(token_counts)
    exposed_stall_ms = total_misses * fetch_ms
    compute_ms = total_tokens * num_layers * t_layer_ms
    return {
        "policy": policy,
        "capacity": capacity,
        "bandwidth_gbps": bandwidth_gbps,
        "expert_mb": expert_bytes / (1024 ** 2),
        "total_slots": total_slots,
        "hit_rate": total_hits / max(1, total_slots),
        "miss_rate": total_misses / max(1, total_slots),
        "exposed_stall_ms": exposed_stall_ms,
        "exposed_stall_ms_per_token": exposed_stall_ms / max(1, total_tokens),
        "compute_ms": compute_ms,
        "tpot_ms": (compute_ms + exposed_stall_ms) / max(1, total_tokens),
        "decode_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: eviction headroom on real MoE traces")
    parser.add_argument("--trace_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--expert_mb", type=float, default=17.0)
    parser.add_argument("--t_layer_ms", type=float, default=0.40)
    parser.add_argument("--cache_slots", type=str, default="14,29,72,144,288,720")
    # demand-only 下 miss 数与带宽无关(带宽只线性缩放 stall), 单带宽足以判定头room
    parser.add_argument("--bandwidths_gbps", type=str, default="5")
    parser.add_argument("--policies", type=str, default="lru,lfu,specmd_ls,oracle_belady")
    parser.add_argument("--max_decode_tokens", type=int, default=6000)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    sequences, num_layers, num_experts = load_token_routes(trace_dir, args.top_k)

    # 按序列独立(codex fix #5): 截断到 max_decode_tokens 个 token 总量
    flats: list[list[tuple[int, int]]] = []
    token_counts: list[int] = []
    used = 0
    for seq in sequences:
        if used >= args.max_decode_tokens:
            break
        take = min(len(seq), args.max_decode_tokens - used)
        flats.append(flatten_sequence(seq[:take], num_layers))
        token_counts.append(take)
        used += take

    expert_bytes = int(args.expert_mb * 1024 * 1024)
    cache_slots = [int(x) for x in args.cache_slots.split(",") if x.strip() and int(x) >= 1]
    bandwidths = [float(x) for x in args.bandwidths_gbps.split(",") if x.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    total_distinct = num_layers * num_experts

    rows = []
    for bw in bandwidths:
        for cap in cache_slots:
            for pol in policies:
                t0 = time.time()
                rows.append(run_policy(flats, token_counts, num_layers, pol, cap,
                                       expert_bytes, bw, args.t_layer_ms, args.top_k))
                print(f"[cell] bw={bw} cap={cap} {pol} done in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    out = {
        "experiment": "real_trace_eviction_headroom_1",
        "trace_dir": str(trace_dir),
        "num_sequences": len(flats),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "total_distinct_layer_experts": total_distinct,
        "decode_tokens": sum(token_counts),
        "config": {"top_k": args.top_k, "expert_mb": args.expert_mb, "t_layer_ms": args.t_layer_ms,
                   "per_sequence_independent": True, "expert_granularity": True},
        "rows": rows,
    }
    ensure_dir(Path(args.out).parent)
    write_json(Path(args.out), out)

    print(f"layers={num_layers} experts={num_experts} distinct={total_distinct} "
          f"seqs={len(flats)} tokens={sum(token_counts)}")
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
            best_heur = min(d.get(p, float("inf")) for p in ("lru", "lfu", "specmd_ls"))
            head_best = (best_heur - d["oracle_belady"]) / best_heur if best_heur > 0 else 0.0
            print(f"  bw={bw:.0f} cap={cap}: LS={d['specmd_ls']:.3f} Belady={d['oracle_belady']:.3f} "
                  f"headroom_vs_LS={head*100:.1f}%  headroom_vs_best_heuristic={head_best*100:.1f}%")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
