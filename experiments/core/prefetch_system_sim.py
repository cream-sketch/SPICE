from __future__ import annotations

import sys, pathlib  # bootstrap: resolve sibling core modules regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import itertools
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from common import (
    build_arg_parser,
    bytes_to_gb,
    bytes_to_mb,
    device_from_arg,
    ensure_dir,
    measure_gpu_power_watts,
    set_seed,
    write_json,
)


@dataclass
class SimConfig:
    layers: int = 16
    steps: int = 192
    experts: int = 64
    top_k: int = 6
    cache_capacity: int = 512
    expert_mb: int = 8
    compute_ms: float = 2.5
    predictor_acc: float = 0.78
    static_predictor_acc: float = 0.58
    confidence_threshold: float = 0.70
    lookahead_min: int = 1
    lookahead_max: int = 6
    draft_ms: float = 0.06
    online_ms: float = 0.0
    prefetch_util_target: float = 0.82


def make_trace(cfg: SimConfig, seed: int) -> list[list[tuple[int, ...]]]:
    rng = np.random.default_rng(seed)
    hot = [rng.choice(cfg.experts, size=min(10, cfg.experts), replace=False) for _ in range(cfg.layers)]
    trace: list[list[tuple[int, ...]]] = []
    prev = [tuple(rng.choice(cfg.experts, size=cfg.top_k, replace=False).tolist()) for _ in range(cfg.layers)]
    for _ in range(cfg.steps):
        step = []
        for l in range(cfg.layers):
            if rng.random() < 0.72:
                pool = np.unique(np.concatenate([np.array(prev[l]), hot[l]]))
            else:
                pool = np.arange(cfg.experts)
            if len(pool) < cfg.top_k:
                pool = np.arange(cfg.experts)
            sel = tuple(sorted(rng.choice(pool, size=cfg.top_k, replace=False).tolist()))
            step.append(sel)
            prev[l] = sel
        trace.append(step)
    return trace


def predict_set(actual: tuple[int, ...], cfg: SimConfig, rng: np.random.Generator) -> tuple[int, ...]:
    pred: set[int] = set()
    for e in actual:
        if rng.random() < cfg.predictor_acc:
            pred.add(e)
        else:
            pred.add(int(rng.integers(0, cfg.experts)))
    while len(pred) < cfg.top_k:
        pred.add(int(rng.integers(0, cfg.experts)))
    return tuple(sorted(pred))


def stable_policy_seed_offset(name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name))


def parse_int_list(raw: str) -> list[int]:
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    if not vals:
        raise ValueError("expected at least one integer")
    return vals


def parse_policy_list(raw: str) -> list[str]:
    policies = []
    valid = {"naive", "lru", "moe_offloading", "pregated", "spice"}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in valid:
            raise ValueError(f"unknown policy {item!r}; valid policies: {sorted(valid)}")
        policies.append(item)
    if not policies:
        raise ValueError("expected at least one policy")
    return policies


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.store: OrderedDict[tuple[int, int], None] = OrderedDict()

    def contains(self, key: tuple[int, int]) -> bool:
        ok = key in self.store
        if ok:
            self.store.move_to_end(key)
        return ok

    def add(self, key: tuple[int, int]) -> bool:
        if key in self.store:
            self.store.move_to_end(key)
            return False
        self.store[key] = None
        while len(self.store) > self.capacity:
            self.store.popitem(last=False)
        return True


def simulate_policy(name: str, trace: list[list[tuple[int, ...]]], cfg: SimConfig, seed: int) -> dict:
    rng = np.random.default_rng(seed + stable_policy_seed_offset(name) % 10000)
    cache = LRUCache(cfg.cache_capacity)
    bytes_per_expert = cfg.expert_mb * 1024 * 1024
    h2d_bytes = 0
    target_slots = 0
    hits = 0
    fallbacks = 0
    wrong_prefetches = 0
    draft_overhead_ms = 0.0
    online_overhead_ms = 0.0
    stall_ms = 0.0
    overlapped_ms = 0.0
    pcie_gbps = 24.0
    copy_ms = (bytes_per_expert / (pcie_gbps * 1024**3)) * 1000.0

    for t, step in enumerate(trace):
        if name == "spice":
            draft_overhead_ms += cfg.layers * cfg.draft_ms
            online_overhead_ms += cfg.layers * cfg.online_ms
            depth = cfg.lookahead_min
            confidence = cfg.predictor_acc
            while depth < cfg.lookahead_max and confidence >= cfg.confidence_threshold:
                depth += 1
                confidence *= 0.93
            # Hardware-aware prefetch throttling: SPICE should not enqueue more
            # speculative transfers than the current compute window can overlap.
            max_depth_by_bandwidth = max(
                1,
                int((cfg.compute_ms * cfg.prefetch_util_target) / (cfg.top_k * copy_ms)),
            )
            depth = min(depth, max_depth_by_bandwidth)
            step_prefetch_budget = max(
                cfg.top_k,
                int((cfg.layers * cfg.compute_ms * cfg.prefetch_util_target) / copy_ms),
            )
            issued_this_step = 0
            for delta in range(1, min(depth, len(trace) - t - 1) + 1):
                future = trace[t + delta]
                for l, actual in enumerate(future):
                    pred = predict_set(actual, cfg, rng)
                    for e in pred:
                        if issued_this_step >= step_prefetch_budget:
                            break
                        if cache.add((l, e)):
                            h2d_bytes += bytes_per_expert
                            overlapped_ms += copy_ms
                            issued_this_step += 1
                    wrong_prefetches += len(set(pred).difference(actual))
                    if issued_this_step >= step_prefetch_budget:
                        break
                if issued_this_step >= step_prefetch_budget:
                    break
        elif name == "pregated":
            if t + 1 < len(trace):
                future = trace[t + 1]
                for l, actual in enumerate(future):
                    static_cfg = SimConfig(**{**cfg.__dict__, "predictor_acc": cfg.static_predictor_acc})
                    pred = predict_set(actual, static_cfg, rng)
                    for e in pred:
                        if cache.add((l, e)):
                            h2d_bytes += bytes_per_expert
                            overlapped_ms += copy_ms
                    wrong_prefetches += len(set(pred).difference(actual))

        for l, selected in enumerate(step):
            for e in selected:
                target_slots += 1
                key = (l, e)
                if name in {"lru", "moe_offloading", "pregated", "spice"} and cache.contains(key):
                    hits += 1
                else:
                    if name != "naive":
                        cache.add(key)
                    fallbacks += 1
                    h2d_bytes += bytes_per_expert
                    stall_ms += copy_ms
        if name == "moe_offloading" and t + 1 < len(trace):
            # Conservative history-based prefetch: next-step same layer previous experts.
            for l, selected in enumerate(step):
                for e in selected:
                    if cache.add((l, e)):
                        h2d_bytes += bytes_per_expert
                        overlapped_ms += copy_ms

    compute_ms = cfg.steps * cfg.layers * cfg.compute_ms
    total_ms = compute_ms + stall_ms + draft_overhead_ms + online_overhead_ms
    useful_transfer_ms = stall_ms + overlapped_ms
    return {
        "policy": name,
        "top_k": cfg.top_k,
        "expert_mb": cfg.expert_mb,
        "cache_capacity": cfg.cache_capacity,
        "target_slots": target_slots,
        "cache_hit_rate": hits / max(1, target_slots),
        "fallback_rate": fallbacks / max(1, target_slots),
        "wrong_prefetches": wrong_prefetches,
        "h2d_gb": bytes_to_gb(h2d_bytes),
        "compute_ms": compute_ms,
        "stall_ms": stall_ms,
        "draft_overhead_ms": draft_overhead_ms,
        "online_overhead_ms": online_overhead_ms,
        "sim_total_ms": total_ms,
        "sim_tpot_ms": total_ms / cfg.steps,
        "pcie_active_fraction": min(1.0, useful_transfer_ms / max(1e-9, total_ms)),
    }


def measure_copy_bandwidth(device: torch.device, expert_mb: int, copies: int = 256) -> dict:
    if device.type != "cuda":
        return {"measured_copy_gbps": None, "measured_copy_ms": None}
    elems = expert_mb * 1024 * 1024 // 2
    src = torch.empty(elems, dtype=torch.float16, pin_memory=True)
    dst = torch.empty(elems, dtype=torch.float16, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(copies):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    total_bytes = copies * expert_mb * 1024 * 1024
    return {
        "measured_copy_gbps": total_bytes / dt / 1024**3,
        "measured_copy_ms": dt * 1000.0 / copies,
    }


def main() -> None:
    parser = build_arg_parser("SPICE prefetch system simulator")
    parser.add_argument(
        "--mode",
        choices=["main", "overhead", "topk", "topk_baselines", "cache_sweep"],
        default="main",
    )
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--expert_mb", type=int, default=8)
    parser.add_argument("--copies", type=int, default=192)
    parser.add_argument("--topk_values", type=str, default="2,4,6,8,10,12")
    parser.add_argument("--cache_values", type=str, default="128,256,512,1024,2048")
    parser.add_argument("--policies", type=str, default="naive,lru,moe_offloading,pregated,spice")
    parser.add_argument("--stress_cache_factor", type=int, default=24)
    parser.add_argument("--stress_cache_min", type=int, default=256)
    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = device_from_arg(args.gpu)

    base = SimConfig(steps=args.steps, expert_mb=args.expert_mb)
    copy = measure_copy_bandwidth(device, args.expert_mb, copies=args.copies)
    rows = []
    if args.mode == "main":
        trace = make_trace(base, args.seed)
        for p in ["naive", "lru", "moe_offloading", "pregated", "spice"]:
            rows.append(simulate_policy(p, trace, base, args.seed))
    elif args.mode == "overhead":
        trace = make_trace(base, args.seed)
        variants = [
            ("spice_offline", SimConfig(**{**base.__dict__, "online_ms": 0.0})),
            ("spice_online", SimConfig(**{**base.__dict__, "online_ms": 0.12})),
            ("spice_no_lore", SimConfig(**{**base.__dict__, "predictor_acc": 0.62})),
        ]
        for name, cfg in variants:
            row = simulate_policy("spice", trace, cfg, args.seed)
            row["variant"] = name
            rows.append(row)
    elif args.mode == "topk":
        for k in parse_int_list(args.topk_values):
            cfg = SimConfig(
                **{
                    **base.__dict__,
                    "top_k": k,
                    "cache_capacity": max(args.stress_cache_min, args.stress_cache_factor * k),
                }
            )
            trace = make_trace(cfg, args.seed + k)
            row = simulate_policy("spice", trace, cfg, args.seed + k)
            row["variant"] = f"topk_{k}"
            row["stress_regime"] = "spice_only"
            rows.append(row)
    elif args.mode == "topk_baselines":
        for k in parse_int_list(args.topk_values):
            cfg = SimConfig(
                **{
                    **base.__dict__,
                    "top_k": k,
                    "cache_capacity": max(args.stress_cache_min, args.stress_cache_factor * k),
                }
            )
            trace = make_trace(cfg, args.seed + k)
            for p in parse_policy_list(args.policies):
                row = simulate_policy(p, trace, cfg, args.seed + k)
                row["variant"] = f"topk_{k}"
                row["stress_regime"] = "baseline_vs_spice"
                rows.append(row)
    else:
        policies = parse_policy_list(args.policies)
        for cap in parse_int_list(args.cache_values):
            cfg = SimConfig(**{**base.__dict__, "cache_capacity": cap})
            trace = make_trace(cfg, args.seed + cap)
            for p in policies:
                row = simulate_policy(p, trace, cfg, args.seed + cap)
                row["variant"] = f"cache_{cap}"
                row["cache_budget_gb"] = bytes_to_gb(cap * cfg.expert_mb * 1024 * 1024)
                row["sweep_axis"] = "cache_capacity"
                rows.append(row)

    power = measure_gpu_power_watts(args.gpu)
    result = {
        "experiment": f"prefetch_system_{args.mode}",
        "device": str(device),
        "gpu_power_watts_sample": power,
        "copy_microbench": copy,
        "config": base.__dict__,
        "rows": rows,
    }
    write_json(out_dir / f"prefetch_system_{args.mode}.json", result)
    print(result)


if __name__ == "__main__":
    main()
