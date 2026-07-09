"""HybriMoE-style baseline replay on real MoE routes.

The released HybriMoE artifact is a KTransformers/GGUF runtime. Running it
directly would mix several variables into a Qwen2-BF16 comparison: quantized
expert format, CPUInfer kernels, KTransformers model injection, and the
HybriMoE policy itself. This replay isolates the policy-level comparison on the
same Qwen2 trace and hardware cost tables used by SPICE.

Modeled HybriMoE mechanisms:
  * per-layer GPU expert cache under the same total residency budget;
  * score-aware cache eviction / next-layer prefetch from recent routing;
  * hybrid CPU-GPU scheduling: resident experts run on GPU, residual misses are
    assigned to exact CPU execution or demand fetch + GPU by measured cost.

No SPICE forecast, CPU residual deadline model, or LoRE substitution is used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HybriMoE-style trace replay baseline")
    p.add_argument("--forecast_dir", required=True)
    p.add_argument("--cost_json", required=True)
    p.add_argument("--resource_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--residency", type=float, required=True)
    p.add_argument("--max_test_tokens", type=int, required=True)
    p.add_argument("--prefetch_size", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--cache_size", type=int, default=0, help="optional fixed per-layer cache size")
    p.add_argument("--dense_ms", type=float, default=0.0)
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


def load_costs(cost_json: str, cost_metric: str = "ms") -> dict[tuple[int, int], float]:
    data = json.loads(Path(cost_json).read_text())
    costs: dict[tuple[int, int], float] = {(0, 0): 0.0}
    for row in data["rows"]:
        costs[(int(row["n_miss"]), int(row["n_fetch"]))] = float(row[cost_metric])
    return costs


def layer_counts(seqs, n_layers: int) -> list[Counter]:
    counts = [Counter() for _ in range(n_layers)]
    for item in seqs:
        true_top = item["true_top"]
        for layer in range(n_layers):
            for tok in range(true_top.shape[1]):
                counts[layer].update(int(x) for x in true_top[layer, tok].tolist())
    return counts


def allocate_layer_caps(counts: list[Counter], n_experts: int, cap: int, fixed: int = 0) -> list[int]:
    n_layers = len(counts)
    if fixed > 0:
        return [min(n_experts, fixed) for _ in range(n_layers)]
    cap = max(1, min(n_layers * n_experts, cap))
    base = cap // n_layers
    rem = cap % n_layers
    demand = [(sum(c.values()), i) for i, c in enumerate(counts)]
    demand.sort(reverse=True)
    caps = [min(n_experts, base) for _ in range(n_layers)]
    for _, layer in demand[:rem]:
        caps[layer] = min(n_experts, caps[layer] + 1)
    return caps


def init_caches(counts: list[Counter], caps: list[int]) -> list[set[int]]:
    caches: list[set[int]] = []
    for count, cap in zip(counts, caps):
        caches.append(set(e for e, _ in count.most_common(cap)))
    return caches


def rank_scores(experts: list[int]) -> dict[int, float]:
    # Approximate router scores when only top-k ids are available. The exact
    # gate probabilities are not needed for correctness; this only drives
    # HybriMoE's score-aware cache priority.
    vals = [1.0 / (i + 1) for i in range(len(experts))]
    z = sum(vals) or 1.0
    return {e: vals[i] / z for i, e in enumerate(experts)}


def update_priority(priority: torch.Tensor, layer: int, experts: list[int], alpha: float) -> None:
    priority[layer] *= 1.0 - alpha
    for expert, score in rank_scores(experts).items():
        priority[layer, expert] += alpha * float(score)


def admit(cache: set[int], cap: int, priority: torch.Tensor, layer: int, expert: int) -> int | None:
    if cap <= 0 or expert in cache:
        return None
    evicted = None
    if len(cache) >= cap:
        victim = min(cache, key=lambda e: float(priority[layer, e]))
        cache.remove(victim)
        evicted = victim
    cache.add(expert)
    return evicted


def choose_split(n_hits: int, n_miss: int, costs: dict[tuple[int, int], float], t_gpu_ms: float):
    best = None
    hit_gpu_ms = n_hits * t_gpu_ms
    for n_fetch in range(n_miss + 1):
        n_cpu = n_miss - n_fetch
        fetch_ms = costs.get((n_fetch, n_fetch), 0.0) if n_fetch else 0.0
        cpu_ms = costs.get((n_cpu, 0), 0.0) if n_cpu else 0.0
        layer_ms = max(hit_gpu_ms + fetch_ms, cpu_ms)
        cand = (layer_ms, n_fetch, n_cpu, fetch_ms, cpu_ms)
        if best is None or cand < best:
            best = cand
    assert best is not None
    return best


def run_policy(seqs, n_layers: int, n_experts: int, top_k: int, caps: list[int],
               costs: dict[tuple[int, int], float], t_gpu_ms: float, max_test_tokens: int,
               prefetch_size: int, alpha: float, dense_ms: float):
    counts = layer_counts(seqs, n_layers)
    caches = init_caches(counts, caps)
    priority = torch.zeros((n_layers, n_experts), dtype=torch.float32)
    for layer, count in enumerate(counts):
        total = float(sum(count.values())) or 1.0
        for expert, value in count.items():
            priority[layer, expert] = float(value) / total

    stats = defaultdict(float)
    clock = 0.0
    tokens = 0
    for item in seqs:
        true_top = item["true_top"]
        T = min(true_top.shape[1], max_test_tokens - tokens)
        for tok in range(T):
            for layer in range(n_layers):
                routed = [int(x) for x in true_top[layer, tok].tolist()]
                update_priority(priority, layer, routed, alpha)
                hits = [e for e in routed if e in caches[layer]]
                misses = [e for e in routed if e not in caches[layer]]
                split = choose_split(len(hits), len(misses), costs, t_gpu_ms)
                layer_ms, n_fetch, n_cpu, fetch_ms, cpu_ms = split
                clock += dense_ms + layer_ms

                # Fetch the highest-priority misses if HSS chooses GPU service.
                fetch_candidates = sorted(misses, key=lambda e: float(priority[layer, e]), reverse=True)[:n_fetch]
                cpu_candidates = [e for e in misses if e not in set(fetch_candidates)]
                for e in fetch_candidates:
                    evicted = admit(caches[layer], caps[layer], priority, layer, e)
                    stats["cache_evictions"] += int(evicted is not None)

                # HybriMoE score-aware next-layer prefetch. Treat transfer as
                # overlapped to keep this baseline optimistic.
                if prefetch_size > 0 and layer + 1 < n_layers and caps[layer + 1] > 0:
                    k = min(prefetch_size, caps[layer + 1], n_experts)
                    top = torch.topk(priority[layer + 1], k=k).indices.tolist()
                    for e in top:
                        e = int(e)
                        if e not in caches[layer + 1]:
                            evicted = admit(caches[layer + 1], caps[layer + 1], priority, layer + 1, e)
                            stats["prefetch_admitted"] += 1
                            stats["cache_evictions"] += int(evicted is not None)

                stats["routed"] += top_k
                stats["hits"] += len(hits)
                stats["misses"] += len(misses)
                stats["gpu_served"] += len(hits) + len(fetch_candidates)
                stats["cpu_served"] += len(cpu_candidates)
                stats["demand_fetches"] += len(fetch_candidates)
                stats["fetch_ms"] += fetch_ms
                stats["cpu_ms"] += cpu_ms
                stats["layer_ms"] += dense_ms + layer_ms
            tokens += 1
            if tokens >= max_test_tokens:
                break
        if tokens >= max_test_tokens:
            break

    stats["tokens"] = tokens
    stats["tpot_ms"] = clock / max(1, tokens)
    stats["hit_rate"] = stats["hits"] / max(1, stats["routed"])
    stats["miss_rate"] = stats["misses"] / max(1, stats["routed"])
    stats["hits_per_tok"] = stats["hits"] / max(1, tokens)
    stats["misses_per_tok"] = stats["misses"] / max(1, tokens)
    stats["gpu_served_per_tok"] = stats["gpu_served"] / max(1, tokens)
    stats["cpu_served_per_tok"] = stats["cpu_served"] / max(1, tokens)
    stats["demand_fetches_per_tok"] = stats["demand_fetches"] / max(1, tokens)
    stats["prefetch_admitted_per_tok"] = stats["prefetch_admitted"] / max(1, tokens)
    return dict(stats)


def main() -> None:
    args = parse_args()
    seqs, n_layers, n_experts, top_k, manifest = load_forecast_sequences(args.forecast_dir)
    costs = load_costs(args.cost_json)
    resource = json.loads(Path(args.resource_json).read_text())
    t_gpu = float(resource["t_gpu_expert_ms"])
    cap = max(1, int(round(args.residency * n_layers * n_experts)))
    counts = layer_counts(seqs, n_layers)
    caps = allocate_layer_caps(counts, n_experts, cap, fixed=args.cache_size)
    row = run_policy(
        seqs, n_layers, n_experts, top_k, caps, costs, t_gpu, args.max_test_tokens,
        args.prefetch_size, args.alpha, args.dense_ms,
    )
    row.update({
        "method": "HybriMoE-style",
        "residency": args.residency,
        "capacity": sum(caps),
        "layer_cache_min": min(caps),
        "layer_cache_max": max(caps),
        "prefetch_size": args.prefetch_size,
        "alpha": args.alpha,
        "t_gpu_ms": t_gpu,
    })
    print(
        f"HybriMoE-style TPOT={row['tpot_ms']:.3f}ms "
        f"hit/tok={row['hits_per_tok']:.2f} miss/tok={row['misses_per_tok']:.2f} "
        f"cpu/tok={row['cpu_served_per_tok']:.2f} fetch/tok={row['demand_fetches_per_tok']:.2f} "
        f"prefetch/tok={row['prefetch_admitted_per_tok']:.2f}",
        flush=True,
    )
    out = {
        "config": vars(args),
        "forecast_manifest": manifest,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "rows": [row],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
