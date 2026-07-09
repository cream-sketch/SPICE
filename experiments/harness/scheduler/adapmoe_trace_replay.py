"""AdapMoE-style baseline replay on real MoE routes.

This is a method-level baseline for models unsupported by the original
Mixtral/HQQ AdapMoE runtime.  It uses the same Qwen2-MoE route trace and A800
resource measurements as SPICE, but only applies AdapMoE's baseline mechanisms:

  * adaptive active expert gating, approximated as retaining the top-m routed
    experts by gate rank;
  * a finite GPU expert cache;
  * demand H2D fetch for active expert cache misses.

It intentionally does not use SPICE forecast, CPU residual execution, or LoRE
substitution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdapMoE-style trace replay baseline")
    p.add_argument("--forecast_dir", required=True)
    p.add_argument("--cost_json", required=True)
    p.add_argument("--resource_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--residency", type=float, required=True)
    p.add_argument("--max_test_tokens", type=int, required=True)
    p.add_argument("--active_m", default="8,6,4", help="comma-separated active top-m experts")
    p.add_argument("--dense_ms", type=float, default=0.0)
    p.add_argument("--ttft_prompt_layers", type=int, default=0,
                   help="optional prompt-length multiplier for TTFT estimate; 0 disables TTFT")
    return p.parse_args()


def load_forecast_sequences(forecast_dir: str):
    root = Path(forecast_dir)
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files = man.get("files") or sorted(p.name for p in root.glob("fc_*.pt"))
    seqs = []
    for name in files:
        d = torch.load(root / name, map_location="cpu", weights_only=False)
        seqs.append({"name": name, "true_top": d["true_top"].long()})
    if not seqs:
        raise ValueError(f"no forecast files in {forecast_dir}")
    first = seqs[0]["true_top"]
    return seqs, int(first.shape[0]), int(man["experts"]), int(man["top_k"]), man


def load_fetch_cost(cost_json: str, top_k: int) -> dict[int, float]:
    data = json.loads(Path(cost_json).read_text())
    table = {}
    for row in data["rows"]:
        n = int(row["n_miss"])
        f = int(row["n_fetch"])
        if n == f:
            table[n] = float(row["ms"])
    for n in range(1, top_k + 1):
        if n not in table:
            raise KeyError(f"missing fetch-all cost for n_miss={n}")
    table[0] = 0.0
    return table


def build_layer_caches(train_seqs, n_layers: int, n_experts: int, cap: int):
    """Allocate cache slots across layers by observed expert popularity."""
    counts = [Counter() for _ in range(n_layers)]
    for item in train_seqs:
        true_top = item["true_top"]
        for l in range(n_layers):
            for t in range(true_top.shape[1]):
                counts[l].update(int(x) for x in true_top[l, t].tolist())

    # Greedy global allocation: each slot goes to the next most popular
    # (layer, expert). This is a trace-level proxy for AdapMoE's dynamic cache
    # allocation under a fixed total residency budget.
    candidates = []
    for l in range(n_layers):
        for e, c in counts[l].items():
            candidates.append((c, l, e))
    candidates.sort(reverse=True)
    selected = candidates[:cap]
    caches = [OrderedDict() for _ in range(n_layers)]
    caps = [0 for _ in range(n_layers)]
    for _c, l, e in selected:
        caps[l] += 1
        caches[l][e] = None
    return caches, caps


def touch(cache: OrderedDict, expert: int):
    if expert in cache:
        cache.move_to_end(expert)


def admit(cache: OrderedDict, cap: int, expert: int):
    if cap <= 0:
        return
    if expert in cache:
        cache.move_to_end(expert)
        return
    while len(cache) >= cap and cache:
        cache.popitem(last=False)
    cache[expert] = None


def run_policy(seqs, train_seqs, n_layers: int, n_experts: int, top_k: int,
               cap: int, active_m: int, fetch_cost: dict[int, float],
               t_gpu_ms: float, dense_ms: float, max_test_tokens: int):
    caches, layer_caps = build_layer_caches(train_seqs, n_layers, n_experts, cap)
    stats = defaultdict(float)
    clock = 0.0
    tokens = 0
    active_m = max(1, min(top_k, active_m))
    for item in seqs:
        true_top = item["true_top"]
        T = min(true_top.shape[1], max_test_tokens - tokens)
        for t in range(T):
            for l in range(n_layers):
                active = [int(x) for x in true_top[l, t, :active_m].tolist()]
                dropped = top_k - active_m
                hits = []
                misses = []
                for e in active:
                    stats["active_slots"] += 1
                    if e in caches[l]:
                        hits.append(e)
                        touch(caches[l], e)
                        stats["hits"] += 1
                    else:
                        misses.append(e)
                        stats["misses"] += 1
                stats["dropped_slots"] += dropped
                nmiss = len(misses)
                layer_ms = dense_ms + len(hits) * t_gpu_ms + fetch_cost[nmiss]
                clock += layer_ms
                for e in misses:
                    admit(caches[l], layer_caps[l], e)
                stats["fetches"] += nmiss
                stats["layer_ms"] += layer_ms
            tokens += 1
            if tokens >= max_test_tokens:
                break
        if tokens >= max_test_tokens:
            break
    stats["tokens"] = tokens
    stats["tpot_ms"] = clock / max(1, tokens)
    stats["avg_active_experts_per_token_layer"] = stats["active_slots"] / max(1, tokens * n_layers)
    stats["slot_reduction_frac"] = stats["dropped_slots"] / max(1, tokens * n_layers * top_k)
    stats["hit_rate"] = stats["hits"] / max(1, stats["active_slots"])
    stats["miss_rate"] = stats["misses"] / max(1, stats["active_slots"])
    stats["fetches_per_token"] = stats["fetches"] / max(1, tokens)
    return dict(stats)


def main() -> None:
    args = parse_args()
    seqs, n_layers, n_experts, top_k, manifest = load_forecast_sequences(args.forecast_dir)
    cost = load_fetch_cost(args.cost_json, top_k)
    resource = json.loads(Path(args.resource_json).read_text())
    t_gpu = float(resource["t_gpu_expert_ms"])
    cap = max(1, int(round(args.residency * n_layers * n_experts)))
    train = seqs
    test = seqs
    rows = []
    for m in [int(x) for x in args.active_m.split(",") if x.strip()]:
        row = run_policy(test, train, n_layers, n_experts, top_k, cap, m, cost,
                         t_gpu, args.dense_ms, args.max_test_tokens)
        row.update({
            "method": "AdapMoE-style",
            "active_m": m,
            "residency": args.residency,
            "capacity": cap,
            "t_gpu_ms": t_gpu,
            "dense_ms": args.dense_ms,
        })
        if args.ttft_prompt_layers > 0:
            row["ttft_ms_est"] = row["tpot_ms"] * args.ttft_prompt_layers
        rows.append(row)
        print(
            f"AdapMoE active_m={m} TPOT={row['tpot_ms']:.3f}ms "
            f"active/layer={row['avg_active_experts_per_token_layer']:.2f} "
            f"drop={100*row['slot_reduction_frac']:.1f}% hit={row['hit_rate']:.3f} "
            f"fetch/tok={row['fetches_per_token']:.2f}",
            flush=True,
        )
    out = {
        "config": vars(args),
        "forecast_manifest": manifest,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
