"""Replay a PCIe-pressure-aware residual-miss scheduler for SPICE.

SPICE prefetch can already saturate PCIe. In that regime, residual fallback fetches are not
"free" capacity-split work; they extend the PCIe critical path. The default layer-serial model
keeps residual fetch as a per-layer predecessor, then enforces the token-end PCIe floor:

  token_time = max(layer_serial_clock,
                   spice_prefetch_h2d_ms_per_tok + residual_fallback_fetches * t_fetch_h2d_ms)

and compares:
  spice_fetch_all      : original residual miss fallback, fetch every miss over PCIe
  cpu_fallback         : Fiddler-style residual fallback, CPU compute every miss
  naive_capacity_split : per-layer split from unconstrained microbench (wrong when PCIe saturated)
  pressure_aware_dp    : token-level DP that chooses per-layer fetch counts to minimize the resource DAG

All routed experts are computed exactly; no drop, no quantization. Diagnostic replay, not an upstream baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from miss_assignment_replay import (  # noqa: E402
    build_future_positions,
    evict_ls,
    future_demand,
    load_costs,
    load_sequences,
    oracle_next_use,
    popularity,
    warm_cache,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPICE residual miss scheduler under PCIe prefetch pressure")
    p.add_argument("--trace_dir", required=True)
    p.add_argument("--cost_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train_frac", type=float, required=True)
    p.add_argument("--residency", required=True)
    p.add_argument("--max_test_tokens", type=int, required=True)
    p.add_argument("--t_attn", type=float, required=True)
    p.add_argument("--t_gate", type=float, required=True)
    p.add_argument("--t_shared", type=float, required=True)
    p.add_argument("--t_gpu", type=float, required=True)
    p.add_argument("--t_fetch_h2d", type=float, default=0.0, help="ms per bf16 expert H2D; inferred if 0")
    p.add_argument("--spice_prefetch_h2d_ms_per_tok", type=float, required=True)
    p.add_argument("--cost_metric", choices=["ms", "mean_ms", "p90_ms"], default="ms",
                   help="miss-service cost column from cost_json; p90_ms stress-tests CPU tails")
    p.add_argument("--cpu_scale", type=float, default=1.0,
                   help="multiply CPU-only miss-service costs to model CPU load/NUMA sensitivity")
    p.add_argument("--fetch_scale", type=float, default=1.0,
                   help="multiply fallback expert H2D time to model PCIe sensitivity")
    p.add_argument("--dag_mode", choices=["layer_serial", "aggregate_lower_bound"], default="layer_serial",
                   help="layer_serial respects per-layer fetch predecessors; aggregate_lower_bound keeps the older token max model")
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--admit_score", choices=["rank", "spice", "oracle"], default="spice")
    p.add_argument("--policies", default="spice_fetch_all,cpu_fallback,naive_capacity_split,pressure_aware_dp")
    return p.parse_args()


def cpu_cost_ms(cost_table, n_cpu: int, cpu_scale: float = 1.0) -> float:
    if n_cpu <= 0:
        return 0.0
    return cpu_scale * cost_table[(n_cpu, 0)]


def select_by_score(misses, n_fetch, l, tid, ti, pop, future_tbl, future_marg, fut, score):
    if n_fetch <= 0:
        return []
    if score == "rank":
        return misses[:n_fetch]
    if score == "spice":
        row = future_tbl.get((l, tid))
        return sorted(misses, key=lambda e: (-(row[e] if row is not None else future_marg[l][e]), -pop[l][e]))[:n_fetch]
    return sorted(misses, key=lambda e: (oracle_next_use(fut, (l, e), ti), -pop[l][e]))[:n_fetch]


def layer_serial_step(clock, prev_fetches, hit_count, nmiss, n_fetch, dense_ms,
                      cost_table, t_gpu, t_fetch_h2d, prefetch_floor, cpu_scale):
    """Conservative per-layer DAG step.

    Residual fetch is a predecessor of fetched-expert GPU compute and of the next layer.
    `cost_table[(nmiss,n_fetch)]` is the measured no-backlog mixed service wall time; PCIe backlog
    can only increase it. CPU service includes activation/output copies from the microbench.
    """
    n_fetch = max(0, min(nmiss, n_fetch))
    n_cpu = nmiss - n_fetch
    base_done = clock + dense_ms + hit_count * t_gpu
    if nmiss == 0:
        return base_done, prev_fetches, 0.0
    if n_fetch == 0:
        return base_done + cpu_cost_ms(cost_table, n_cpu, cpu_scale), prev_fetches, 0.0

    cpu_ms = cpu_cost_ms(cost_table, n_cpu, cpu_scale)
    no_backlog_service = max(cost_table[(nmiss, n_fetch)], cpu_ms, n_fetch * (t_fetch_h2d + t_gpu))
    copy_ready = prefetch_floor + prev_fetches * t_fetch_h2d
    pcie_wait = max(0.0, copy_ready - base_done)
    blocked_fetch_service = pcie_wait + n_fetch * t_fetch_h2d + n_fetch * t_gpu
    service = max(no_backlog_service, cpu_ms, blocked_fetch_service)
    return base_done + service, prev_fetches + n_fetch, pcie_wait


def evaluate_fetch_counts_layer_serial(layer_states, fetch_counts, cost_table, dense_ms, t_gpu,
                                       t_fetch_h2d, prefetch_floor, cpu_scale):
    clock = 0.0
    fetches = 0
    pcie_wait = 0.0
    for (hit_count, nmiss), n_fetch in zip(layer_states, fetch_counts):
        clock, fetches, wait = layer_serial_step(clock, fetches, hit_count, nmiss, n_fetch, dense_ms,
                                                 cost_table, t_gpu, t_fetch_h2d, prefetch_floor, cpu_scale)
        pcie_wait += wait
    pcie_h2d_ms = prefetch_floor + fetches * t_fetch_h2d
    return {"token_ms": max(clock, pcie_h2d_ms), "layer_clock_ms": clock, "fetches": fetches,
            "pcie_wait_ms": pcie_wait, "pcie_h2d_ms": pcie_h2d_ms}


def best_fetch_counts_layer_serial(layer_states, cost_table, dense_ms, t_gpu, t_fetch_h2d,
                                   prefetch_floor, cpu_scale):
    """Return per-layer fetch counts minimizing a sequential layer resource DAG."""
    states = {0: (0.0, [], 0.0)}  # total_fetches -> (clock_ms, choices, pcie_wait_ms)
    for hit_count, nmiss in layer_states:
        next_states = {}
        for prev_fetches, (clock, choices, wait_sum) in states.items():
            for n_fetch in range(nmiss + 1):
                nclock, nf_total, wait = layer_serial_step(clock, prev_fetches, hit_count, nmiss, n_fetch,
                                                           dense_ms, cost_table, t_gpu, t_fetch_h2d,
                                                           prefetch_floor, cpu_scale)
                old = next_states.get(nf_total)
                if old is None or nclock < old[0]:
                    next_states[nf_total] = (nclock, choices + [n_fetch], wait_sum + wait)
        states = next_states
    best = None
    for total_fetches, item in states.items():
        clock, choices, wait_sum = item
        pcie_h2d_ms = prefetch_floor + total_fetches * t_fetch_h2d
        scored = (max(clock, pcie_h2d_ms), clock, choices, wait_sum)
        if best is None or scored < best:
            best = scored
    assert best is not None
    return best[2]


def best_fetch_counts_aggregate_lower_bound(layer_states, cost_table, t_gpu, t_fetch_h2d, prefetch_floor, cpu_scale):
    """Return per-layer fetch counts minimizing max(compute, prefetch_floor + fetch_h2d).

    layer_states entries: (hit_count, nmiss). Dense/hit compute is included in each option.
    DP state is total residual fallback fetch count -> (min_compute_ms, choices).
    """
    dp = {0: (0.0, [])}
    for hit_count, nmiss in layer_states:
        opts = []
        for f in range(nmiss + 1):
            n_cpu = nmiss - f
            compute = hit_count * t_gpu + cpu_cost_ms(cost_table, n_cpu, cpu_scale) + f * t_gpu
            opts.append((f, compute))
        ndp = {}
        for prev_fetches, (prev_compute, prev_choices) in dp.items():
            for f, compute in opts:
                nf = prev_fetches + f
                nc = prev_compute + compute
                old = ndp.get(nf)
                if old is None or nc < old[0]:
                    ndp[nf] = (nc, prev_choices + [f])
        dp = ndp
    best = None
    for total_fetches, (compute, choices) in dp.items():
        token_time = max(compute, prefetch_floor + total_fetches * t_fetch_h2d)
        item = (token_time, total_fetches, compute, choices)
        if best is None or item < best:
            best = item
    assert best is not None
    return best[3]


def policy_fetch_counts(policy, layer_states, best_table, cost_table, dense_ms, t_gpu, t_fetch_h2d,
                        prefetch_floor, cpu_scale, dag_mode):
    if policy == "pressure_aware_dp":
        if dag_mode == "aggregate_lower_bound":
            return best_fetch_counts_aggregate_lower_bound(layer_states, cost_table, t_gpu, t_fetch_h2d,
                                                           prefetch_floor, cpu_scale)
        return best_fetch_counts_layer_serial(layer_states, cost_table, dense_ms, t_gpu, t_fetch_h2d,
                                              prefetch_floor, cpu_scale)
    out = []
    for _hit_count, nmiss in layer_states:
        if nmiss == 0:
            out.append(0)
            continue
        if policy == "spice_fetch_all":
            out.append(nmiss)
        elif policy == "cpu_fallback":
            out.append(0)
        elif policy == "naive_capacity_split":
            out.append(best_table[nmiss])
        else:
            raise ValueError(policy)
    return out


def simulate(seq, fut, n_layers, capacity, policy, cost_table, best_table, pop,
             future_tbl, future_marg, expert_mb, act_roundtrip_mb, args):
    cache = warm_cache(pop, capacity)
    last_used = {k: 0 for k in cache}
    dense_ms = args.t_attn + args.t_gate + args.t_shared
    total_ms = 0.0
    token_dag_ms_total = 0.0
    layer_clock_ms_total = 0.0
    pcie_h2d_ms_total = 0.0
    pcie_wait_bound_tokens = 0
    pcie_floor_bound_tokens = 0
    layer_clock_bound_tokens = 0
    hits = misses = fetches = cpu_served = admitted = admitted_used = admitted_never_used = 0
    cpu_layers = 0
    admitted_live = {}
    pos = 0

    for ti, (tid, per_layer) in enumerate(seq):
        layer_infos = []
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
                    misses += 1
                    layer_misses.append(e)
                pos += 1
            layer_infos.append((l, layer_hits, layer_misses))

        layer_states = [(h, len(m)) for _l, h, m in layer_infos]
        fetch_counts = policy_fetch_counts(policy, layer_states, best_table, cost_table, dense_ms, args.t_gpu,
                                           args.t_fetch_h2d, args.spice_prefetch_h2d_ms_per_tok,
                                           args.cpu_scale, args.dag_mode)

        token_eval = evaluate_fetch_counts_layer_serial(layer_states, fetch_counts, cost_table, dense_ms,
                                                        args.t_gpu, args.t_fetch_h2d,
                                                        args.spice_prefetch_h2d_ms_per_tok, args.cpu_scale)
        token_fetches = 0
        fetched_by_layer = []
        for (l, layer_hits, layer_misses), n_fetch in zip(layer_infos, fetch_counts):
            nmiss = len(layer_misses)
            n_fetch = min(n_fetch, nmiss)
            n_cpu = nmiss - n_fetch
            token_fetches += n_fetch
            fetches += n_fetch
            cpu_served += n_cpu
            if n_cpu:
                cpu_layers += 1
            fetched = select_by_score(layer_misses, n_fetch, l, tid, ti, pop, future_tbl, future_marg,
                                      fut, args.admit_score)
            fetched_by_layer.append((l, fetched))

        pcie_ms = token_eval["pcie_h2d_ms"]
        token_dag_ms_total += token_eval["token_ms"]
        layer_clock_ms_total += token_eval["layer_clock_ms"]
        pcie_h2d_ms_total += pcie_ms
        if token_eval["pcie_wait_ms"] > 1e-9:
            pcie_wait_bound_tokens += 1
        if token_eval["pcie_h2d_ms"] > token_eval["layer_clock_ms"] + 1e-9:
            pcie_floor_bound_tokens += 1
        else:
            layer_clock_bound_tokens += 1
        total_ms += token_eval["token_ms"]

        if policy in ("spice_fetch_all", "naive_capacity_split", "pressure_aware_dp"):
            # Protected post-token admission: fetched residuals can improve future-token residency,
            # but they do not evict current-token later-layer residents after planning.
            for l, fetched in fetched_by_layer:
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

    tokens = max(1, len(seq))
    return {
        "total_ms": total_ms,
        "tokens": len(seq),
        "hits": hits,
        "misses": misses,
        "fetches": fetches,
        "cpu_served": cpu_served,
        "cpu_layers": cpu_layers,
        "expert_h2d_mb": fetches * expert_mb,
        "cpu_act_roundtrip_mb": cpu_layers * act_roundtrip_mb,
        "admissions": admitted,
        "admitted_used": admitted_used,
        "admitted_never_used": admitted_never_used,
        "token_dag_ms": token_dag_ms_total,
        "layer_clock_ms": layer_clock_ms_total,
        "compute_ms": layer_clock_ms_total,
        "pcie_h2d_ms": pcie_h2d_ms_total,
        "pcie_wait_bound_tokens": pcie_wait_bound_tokens,
        "pcie_floor_bound_tokens": pcie_floor_bound_tokens,
        "layer_clock_bound_tokens": layer_clock_bound_tokens,
        "tpot_ms": total_ms / tokens,
    }


def main() -> None:
    args = parse_args()
    seqs, n_layers, n_experts, top_k = load_sequences(args.trace_dir)
    n_train = max(1, int(round(args.train_frac * len(seqs))))
    train, test = seqs[:n_train], seqs[n_train:]
    if not test:
        raise ValueError(
            f"train_frac={args.train_frac} leaves no test traces from {len(seqs)} sequences; "
            "lower train_frac or provide more dec_*.pt files"
        )
    selected = []
    ntok = 0
    for s in test:
        selected.append(s)
        ntok += len(s)
        if ntok >= args.max_test_tokens:
            break
    test = selected
    cost_table, best_table, cost_meta, expert_mb, act_roundtrip_mb = load_costs(args.cost_json, args.cost_metric)
    if args.t_fetch_h2d <= 0:
        args.t_fetch_h2d = max(0.0, cost_table[(1, 1)] - args.t_gpu)
    args.t_fetch_h2d *= args.fetch_scale
    bw_mb_per_ms = expert_mb / args.t_fetch_h2d if args.t_fetch_h2d > 0 else 0.0
    pop = popularity(train, n_layers, n_experts)
    future_tbl, future_marg = future_demand(train, n_layers, n_experts, args.alpha)
    test_futures = [build_future_positions(s) for s in test]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    residencies = [float(x) for x in args.residency.split(",")]
    rows = []
    print(f"[data] train={len(train)} test={len(test)} tokens={sum(len(s) for s in test)} "
          f"layers={n_layers} experts={n_experts} top_k={top_k}", flush=True)
    print(f"[resource] prefetch_floor={args.spice_prefetch_h2d_ms_per_tok:.3f}ms/tok "
          f"fallback_fetch_h2d={args.t_fetch_h2d:.3f}ms/expert cost_metric={args.cost_metric} "
          f"cpu_scale={args.cpu_scale:.2f} fetch_scale={args.fetch_scale:.2f} "
          f"bw={bw_mb_per_ms * 1000 / 1024:.2f}GB/s best_fetch={best_table}", flush=True)

    for r in residencies:
        cap = max(1, int(round(r * n_layers * n_experts)))
        for pol in policies:
            agg = defaultdict(float)
            for s, fut in zip(test, test_futures):
                res = simulate(s, fut, n_layers, cap, pol, cost_table, best_table, pop,
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
                "cpu_layers_per_tok": agg["cpu_layers"] / tokens,
                "expert_h2d_mb_per_tok": agg["expert_h2d_mb"] / tokens,
                "cpu_act_roundtrip_mb_per_tok": agg["cpu_act_roundtrip_mb"] / tokens,
                "token_dag_ms_per_tok": agg["token_dag_ms"] / tokens,
                "layer_clock_ms_per_tok": agg["layer_clock_ms"] / tokens,
                "compute_ms_per_tok": agg["compute_ms"] / tokens,
                "pcie_h2d_ms_per_tok": agg["pcie_h2d_ms"] / tokens,
                "fallback_h2d_ms_per_tok": agg["fetches"] * args.t_fetch_h2d / tokens,
                "prefetch_h2d_floor_ms_per_tok": args.spice_prefetch_h2d_ms_per_tok,
                "pcie_bound_frac": agg["pcie_floor_bound_tokens"] / tokens,
                "pcie_floor_bound_frac": agg["pcie_floor_bound_tokens"] / tokens,
                "pcie_wait_bound_frac": agg["pcie_wait_bound_tokens"] / tokens,
                "layer_clock_bound_frac": agg["layer_clock_bound_tokens"] / tokens,
                "compute_bound_frac": agg["layer_clock_bound_tokens"] / tokens,
                "admitted_never_used_frac": agg["admitted_never_used"] / admissions,
                "admitted_used_frac": agg["admitted_used"] / admissions,
            }
            rows.append(row)
            print(f"res={r:.3f} {pol:>20} TPOT={row['tpot_ms']:7.3f} hit={row['hit_rate']:.3f} "
                  f"fetch/tok={row['fetches_per_tok']:.2f} cpu/tok={row['cpu_served_per_tok']:.2f} "
                  f"H2D/tok={row['expert_h2d_mb_per_tok']:.1f}MB pcie_bound={row['pcie_bound_frac']:.2f} "
                  f"never_used={row['admitted_never_used_frac']:.2f}",
                  flush=True)

    verdict = {}
    by_res = defaultdict(dict)
    for row in rows:
        by_res[row["residency"]][row["policy"]] = row
    for r, d in by_res.items():
        item = {}
        if "spice_fetch_all" in d and "pressure_aware_dp" in d:
            denom = max(1e-9, d["spice_fetch_all"]["tpot_ms"])
            item["gain_vs_spice_fetch_all_pct"] = 100.0 * (d["spice_fetch_all"]["tpot_ms"] - d["pressure_aware_dp"]["tpot_ms"]) / denom
        if "naive_capacity_split" in d and "pressure_aware_dp" in d:
            denom = max(1e-9, d["naive_capacity_split"]["tpot_ms"])
            item["gain_vs_naive_capacity_split_pct"] = 100.0 * (d["naive_capacity_split"]["tpot_ms"] - d["pressure_aware_dp"]["tpot_ms"]) / denom
        if "cpu_fallback" in d and "pressure_aware_dp" in d:
            item["dp_fetches_per_tok"] = d["pressure_aware_dp"]["fetches_per_tok"]
            item["cpu_fallback_tpot_ms"] = d["cpu_fallback"]["tpot_ms"]
            item["pressure_aware_dp_tpot_ms"] = d["pressure_aware_dp"]["tpot_ms"]
        verdict[str(r)] = item

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "args": vars(args),
        "trace": {"n_layers": n_layers, "n_experts": n_experts, "top_k": top_k},
        "cost_meta": cost_meta.get("config", {}),
        "rows": rows,
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
