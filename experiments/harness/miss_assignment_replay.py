"""Replay real MoE routes with measured top-k miss-assignment costs.

This is a diagnostic harness for the question:
  when n routed experts miss in a layer, should we CPU-serve all of them, fetch all,
  or split n_fetch/n_cpu using a measured capacity-aware table?

It intentionally separates:
  SERVE: how the current token is computed (CPU vs fetch/GPU)
  ADMIT: whether fetched experts are granted HBM residency for future reuse

No quality changes; all routed experts are computed. This is not an upstream baseline reproduction.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay measured miss assignment on route traces")
    p.add_argument("--trace_dir", required=True)
    p.add_argument("--cost_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train_frac", type=float, required=True)
    p.add_argument("--residency", required=True, help="comma list of HBM routed-expert residency fractions")
    p.add_argument("--max_test_tokens", type=int, required=True)
    p.add_argument("--t_attn", type=float, required=True)
    p.add_argument("--t_gate", type=float, required=True)
    p.add_argument("--t_shared", type=float, required=True)
    p.add_argument("--t_gpu", type=float, required=True)
    p.add_argument("--layer_mode", choices=["additive", "overlap"], required=True)
    p.add_argument("--alpha", type=float, default=32.0, help="smoothing for token-conditioned future-demand table")
    p.add_argument("--policies", default="fetch_all,cpu_all,capacity_noadmit,capacity_popularity,capacity_rank,capacity_spice,oracle_admit")
    return p.parse_args()


def load_sequences(trace_dir: str):
    files = sorted(glob.glob(str(Path(trace_dir) / "dec_*.pt")))
    man = json.loads((Path(trace_dir) / "manifest.json").read_text())
    if not files:
        raise ValueError(
            f"{trace_dir} does not contain dec_*.pt decode traces. "
            "miss_assignment_replay.py expects gen_decode_traces*.py output, not manifest-only "
            "HF router-logit traces."
        )
    seqs = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        steps = d["steps"]
        prompt_ids = d["prompt_ids"]
        prev = int(prompt_ids[0][-1])
        seq = []
        for gen_tid, per_layer in steps:
            if any(x is None for x in per_layer):
                prev = int(gen_tid)
                continue
            seq.append((prev, [[int(e) for e in topk] for topk in per_layer]))
            prev = int(gen_tid)
        if seq:
            seqs.append(seq)
    cfg = man.get("model_config", {})
    n_layers = man.get("num_layers", cfg.get("num_hidden_layers"))
    n_experts = man.get("experts", cfg.get("num_experts", cfg.get("n_routed_experts")))
    top_k = man.get("top_k", cfg.get("num_experts_per_tok"))
    if n_layers is None or n_experts is None or top_k is None:
        raise KeyError(f"manifest missing layer/expert/top_k metadata: {Path(trace_dir) / 'manifest.json'}")
    return seqs, int(n_layers), int(n_experts), int(top_k)


def load_costs(path: str, metric: str = "ms"):
    data = json.loads(Path(path).read_text())
    rows = data["rows"]
    table = {}
    best = {}
    for r in rows:
        n = int(r["n_miss"])
        f = int(r["n_fetch"])
        if metric not in r:
            raise KeyError(f"{path} row is missing requested cost metric {metric!r}")
        table[(n, f)] = float(r[metric])
    for n in sorted({int(r["n_miss"]) for r in rows}):
        choices = [(f, table[(n, f)]) for f in range(n + 1)]
        best[n] = min(choices, key=lambda x: x[1])[0]
    cfg = data.get("config", {})
    expert_mb = float(data.get("expert_mb_bf16", 0.0))
    d_model = int(cfg.get("d_model", 0))
    # D2H activation + H2D output, once per layer that has at least one CPU-served expert.
    act_roundtrip_mb = (2 * d_model * 2) / 1e6 if d_model else 0.0
    return table, best, data, expert_mb, act_roundtrip_mb


def popularity(train, n_layers: int, n_experts: int):
    pop = [[0 for _ in range(n_experts)] for _ in range(n_layers)]
    for seq in train:
        for _tid, per_layer in seq:
            for l, topk in enumerate(per_layer):
                for e in topk:
                    pop[l][e] += 1
    return pop


def future_demand(train, n_layers: int, n_experts: int, alpha: float):
    """Token-conditioned next-token same-layer reuse table.

    This is the replay proxy for a SPICE-style future-demand signal: after seeing current input token
    id, estimate whether (layer, expert) will be demanded again soon. It is train-only and realizable.
    """
    cnt = defaultdict(lambda: [0.0 for _ in range(n_experts)])
    tot = defaultdict(float)
    layer_cnt = [[0.0 for _ in range(n_experts)] for _ in range(n_layers)]
    layer_tot = [0.0 for _ in range(n_layers)]
    for seq in train:
        for _tid, per_layer in seq:
            for l, topk in enumerate(per_layer):
                for e in topk:
                    layer_cnt[l][e] += 1.0
                    layer_tot[l] += 1.0
        for i in range(len(seq) - 1):
            tid = seq[i][0]
            next_layers = seq[i + 1][1]
            for l, topk in enumerate(next_layers):
                for e in topk:
                    cnt[(l, tid)][e] += 1.0
                    tot[(l, tid)] += 1.0

    marg = []
    for l in range(n_layers):
        denom = max(1.0, layer_tot[l])
        marg.append([x / denom for x in layer_cnt[l]])

    table = {}
    for key, row in cnt.items():
        l, _tid = key
        denom = tot[key] + alpha
        table[key] = [(row[e] + alpha * marg[l][e]) / denom for e in range(n_experts)]
    return table, marg


def warm_cache(pop, capacity: int):
    items = []
    for l, row in enumerate(pop):
        for e, c in enumerate(row):
            items.append(((l, e), c))
    items.sort(key=lambda x: -x[1])
    return set(k for k, _ in items[:capacity])


def ls_distance(key, cur_layer: int, n_layers: int):
    d = (key[0] - cur_layer) % n_layers
    return n_layers if d == 0 else d


def evict_ls(cache, last_used, cur_layer: int, n_layers: int):
    victim = max(cache, key=lambda k: (ls_distance(k, cur_layer, n_layers), -last_used.get(k, -1)))
    cache.remove(victim)
    last_used.pop(victim, None)
    return victim


def build_future_positions(seq):
    fut = defaultdict(list)
    for ti, (_tid, per_layer) in enumerate(seq):
        for l, topk in enumerate(per_layer):
            for e in topk:
                fut[(l, e)].append(ti)
    return fut


def oracle_next_use(fut, key, ti: int):
    arr = fut.get(key, [])
    idx = bisect.bisect_right(arr, ti)
    return arr[idx] if idx < len(arr) else 10**12


def select_fetch(policy: str, misses: list[int], l: int, tid: int, ti: int, pop, future_tbl, future_marg,
                 fut, best_fetch: int, n_experts: int):
    if policy == "fetch_all":
        n_fetch = len(misses)
    elif policy in ("cpu_all", "cpu_all_noadmit"):
        n_fetch = 0
    elif policy.startswith("capacity_") or policy == "oracle_admit":
        n_fetch = best_fetch
    else:
        raise ValueError(policy)

    if n_fetch <= 0:
        return []

    if policy in ("fetch_all", "capacity_rank"):
        return misses[:n_fetch]

    if policy == "capacity_spice":
        row = future_tbl.get((l, tid))
        ranked = sorted(misses, key=lambda e: (-(row[e] if row is not None else future_marg[l][e]), -pop[l][e]))
        return ranked[:n_fetch]

    if policy == "oracle_admit":
        ranked = sorted(misses, key=lambda e: (oracle_next_use(fut, (l, e), ti), -pop[l][e]))
        return ranked[:n_fetch]

    ranked = sorted(misses, key=lambda e: -pop[l][e])
    return ranked[:n_fetch]


def layer_time_ms(dense_ms: float, hit_count: int, miss_cost_ms: float, t_gpu: float, mode: str) -> float:
    hit_gpu = hit_count * t_gpu
    if mode == "additive":
        return dense_ms + hit_gpu + miss_cost_ms
    return dense_ms + max(hit_gpu, miss_cost_ms)


def simulate(seq, fut, n_layers, n_experts, capacity, policy, cost_table, best_table, pop,
             future_tbl, future_marg, expert_mb, act_roundtrip_mb, args):
    cache = warm_cache(pop, capacity)
    last_used = {k: 0 for k in cache}
    dense_ms = args.t_attn + args.t_gate + args.t_shared
    total_ms = 0.0
    hits = misses_total = fetches = cpu_served = cpu_layers = 0
    admitted = 0
    admitted_used = 0
    admitted_never_used = 0
    admitted_live = {}  # key -> used?
    pos = 0
    for ti, (tid, per_layer) in enumerate(seq):
        for l, topk in enumerate(per_layer):
            layer_hits = 0
            layer_misses = []
            for e in topk:
                k = (l, e)
                if k in cache:
                    layer_hits += 1
                    hits += 1
                    if k in admitted_live and not admitted_live[k]:
                        admitted_live[k] = True
                        admitted_used += 1
                    last_used[k] = pos
                else:
                    layer_misses.append(e)
                    misses_total += 1
                pos += 1

            nmiss = len(layer_misses)
            if nmiss == 0:
                miss_cost = 0.0
                fetched = []
            else:
                fetched = select_fetch(policy, layer_misses, l, tid, ti, pop, future_tbl, future_marg,
                                       fut, best_table[nmiss], n_experts)
                nfetch = len(fetched)
                miss_cost = cost_table[(nmiss, nfetch)]
                fetches += nfetch
                cpu_served += nmiss - nfetch
                if nmiss - nfetch > 0:
                    cpu_layers += 1

            total_ms += layer_time_ms(dense_ms, layer_hits, miss_cost, args.t_gpu, args.layer_mode)

            if policy in ("fetch_all", "capacity_popularity", "capacity_rank", "capacity_spice", "oracle_admit"):
                for e in fetched:
                    k = (l, e)
                    if k not in cache:
                        admitted += 1
                        admitted_live[k] = False
                    cache.add(k)
                    last_used[k] = pos
                    while len(cache) > capacity:
                        victim = evict_ls(cache, last_used, l, n_layers)
                        used = admitted_live.pop(victim, None)
                        if used is False:
                            admitted_never_used += 1

    for used in admitted_live.values():
        if used is False:
            admitted_never_used += 1

    return {
        "total_ms": total_ms,
        "tokens": len(seq),
        "hits": hits,
        "misses": misses_total,
        "fetches": fetches,
        "cpu_served": cpu_served,
        "cpu_layers": cpu_layers,
        "expert_h2d_mb": fetches * expert_mb,
        "cpu_act_mb": cpu_layers * act_roundtrip_mb,
        "admissions": admitted,
        "admitted_never_used": admitted_never_used,
        "admitted_used": admitted_used,
    }


def main() -> None:
    args = parse_args()
    seqs, n_layers, n_experts, top_k = load_sequences(args.trace_dir)
    n_train = max(1, int(round(len(seqs) * args.train_frac)))
    train, test = seqs[:n_train], seqs[n_train:]
    selected = []
    ntok = 0
    for s in test:
        selected.append(s)
        ntok += len(s)
        if ntok >= args.max_test_tokens:
            break
    test = selected

    cost_table, best_table, cost_meta, expert_mb, act_roundtrip_mb = load_costs(args.cost_json)
    pop = popularity(train, n_layers, n_experts)
    future_tbl, future_marg = future_demand(train, n_layers, n_experts, args.alpha)
    test_futures = [build_future_positions(s) for s in test]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    residencies = [float(x) for x in args.residency.split(",")]
    rows = []
    print(f"[data] traces={len(seqs)} train={len(train)} test={len(test)} tokens={sum(len(s) for s in test)} "
          f"layers={n_layers} experts={n_experts} top_k={top_k} mode={args.layer_mode}", flush=True)
    print(f"[cost] {args.cost_json} best_fetch={best_table} expert_mb={expert_mb:.3f} "
          f"act_roundtrip_mb={act_roundtrip_mb:.6f}", flush=True)

    for r in residencies:
        cap = max(1, int(round(r * n_layers * n_experts)))
        for pol in policies:
            agg = defaultdict(float)
            for s, fut in zip(test, test_futures):
                res = simulate(s, fut, n_layers, n_experts, cap, pol, cost_table, best_table, pop,
                               future_tbl, future_marg, expert_mb, act_roundtrip_mb, args)
                for k, v in res.items():
                    agg[k] += v
            tokens = max(1, agg["tokens"])
            routed = max(1, agg["hits"] + agg["misses"])
            admissions = max(1, agg["admissions"])
            row = {
                "residency": r,
                "capacity": cap,
                "policy": pol,
                "tpot_ms": agg["total_ms"] / tokens,
                "hit_rate": agg["hits"] / routed,
                "misses_per_tok": agg["misses"] / tokens,
                "fetches_per_tok": agg["fetches"] / tokens,
                "cpu_served_per_tok": agg["cpu_served"] / tokens,
                "expert_h2d_mb_per_tok": agg["expert_h2d_mb"] / tokens,
                "cpu_act_mb_per_tok": agg["cpu_act_mb"] / tokens,
                "admissions_per_tok": agg["admissions"] / tokens,
                "admitted_never_used_frac": agg["admitted_never_used"] / admissions,
                "admitted_used_frac": agg["admitted_used"] / admissions,
            }
            rows.append(row)
            print(f"res={r:.3f} {pol:>14} TPOT={row['tpot_ms']:7.3f} "
                  f"hit={row['hit_rate']:.3f} fetch/tok={row['fetches_per_tok']:.2f} "
                  f"cpu/tok={row['cpu_served_per_tok']:.2f} H2D/tok={row['expert_h2d_mb_per_tok']:.1f}MB "
                  f"never_used={row['admitted_never_used_frac']:.2f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "args": vars(args),
        "trace": {"n_layers": n_layers, "n_experts": n_experts, "top_k": top_k},
        "cost_meta": cost_meta["config"],
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
