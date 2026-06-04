"""Replay SPICE draft-forecast residual misses under PCIe pressure.

This harness consumes `spice_draft.cli --dump_forecast` files:

  true_top: [layers, tokens, top_k]
  fcast   : [anchor_layer, horizon, tokens, top_k]

It first simulates verified SPICE-style expert prefetch into a bounded HBM cache using the
draft forecast. Prefetches are not magical hits: they enter a serial H2D queue and become
available only after their ready time reaches a layer-deadline lower bound. Wrong forecasts are
harmless but can waste PCIe and cache space. The residual misses that remain are then served by
the same exact same-precision resource-DAG policies as `prefetch_pressure_scheduler_replay.py`.

This is a diagnostic bridge between the SPICE draft signal and the residual-miss scheduler, not a
full runtime. Important limitation: prefetch hit/miss uses per-layer ready times, but residual
fallback scheduling treats the issued SPICE prefetch bytes as an upper-pressure scalar H2D floor.
That intentionally stress-tests residual fetches under saturated SPICE traffic; it is not a unified
copy-engine event timeline.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from miss_assignment_replay import evict_ls, load_costs, popularity, warm_cache  # noqa: E402
from prefetch_pressure_scheduler_replay import (  # noqa: E402
    best_fetch_counts_layer_serial,
    evaluate_fetch_counts_layer_serial,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPICE forecast residual scheduler under PCIe pressure")
    p.add_argument("--forecast_dir", required=True)
    p.add_argument("--cost_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train_frac", type=float, required=True)
    p.add_argument("--residency", required=True, help="comma list of HBM routed-expert residency fractions")
    p.add_argument("--max_test_tokens", type=int, required=True)
    p.add_argument("--t_attn", type=float, required=True)
    p.add_argument("--t_gate", type=float, required=True)
    p.add_argument("--t_shared", type=float, required=True)
    p.add_argument("--t_gpu", type=float, required=True)
    p.add_argument("--t_fetch_h2d", type=float, default=0.0, help="ms per bf16 expert H2D; inferred if 0")
    p.add_argument("--min_lead_layers", type=int, default=1,
                   help="minimum layer lead for draft prefetch; 1 means next layer, not current layer")
    p.add_argument("--max_lead_layers", type=int, default=6)
    p.add_argument("--cost_metric", choices=["ms", "mean_ms", "p90_ms"], default="ms")
    p.add_argument("--cpu_scale", type=float, default=1.0)
    p.add_argument("--fetch_scale", type=float, default=1.0)
    p.add_argument("--admit_score", choices=["rank", "oracle"], default="rank")
    p.add_argument("--policies", default="spice_fetch_all,cpu_fallback,naive_capacity_split,pressure_aware_dp")
    return p.parse_args()


def load_forecast_sequences(forecast_dir: str):
    root = Path(forecast_dir)
    man = json.loads((root / "manifest.json").read_text())
    files = man.get("files") or sorted(p.name for p in root.glob("fc_*.pt"))
    seqs = []
    n_layers = top_k = max_horizon = None
    n_experts = 0
    for name in files:
        d = torch.load(root / name, map_location="cpu", weights_only=False)
        true_top = d["true_top"].long()  # [L,S,K]
        fcast = d["fcast"].long()        # [L,H,S,K]
        if n_layers is None:
            n_layers = int(d.get("num_layers", true_top.shape[0]))
            top_k = int(d.get("top_k", true_top.shape[-1]))
            max_horizon = int(d.get("max_horizon", fcast.shape[1]))
        n_experts = max(n_experts, int(true_top.max().item()) + 1)
        valid = fcast[fcast >= 0]
        if valid.numel():
            n_experts = max(n_experts, int(valid.max().item()) + 1)
        seqs.append({"name": name, "true_top": true_top, "fcast": fcast})
    if n_layers is None or top_k is None:
        raise ValueError(f"no forecast files in {forecast_dir}")
    return seqs, int(n_layers), int(n_experts), int(top_k), int(max_horizon), man


def seq_for_popularity(item):
    true_top = item["true_top"]
    layers, tokens, _top_k = true_top.shape
    seq = []
    for t in range(tokens):
        seq.append((t, [[int(x) for x in true_top[l, t].tolist()] for l in range(layers)]))
    return seq


def build_future_positions_from_true(true_top: torch.Tensor):
    fut = defaultdict(list)
    layers, tokens, _top_k = true_top.shape
    for t in range(tokens):
        for l in range(layers):
            for e in true_top[l, t].tolist():
                fut[(l, int(e))].append(t)
    return fut


def oracle_next_use(fut, key, ti: int):
    arr = fut.get(key, [])
    idx = bisect.bisect_right(arr, ti)
    return arr[idx] if idx < len(arr) else 10**12


def select_residual_fetches(misses, n_fetch, l, ti, fut, score):
    if n_fetch <= 0:
        return []
    if score == "oracle":
        return sorted(misses, key=lambda e: oracle_next_use(fut, (l, e), ti))[:n_fetch]
    return misses[:n_fetch]


def cache_add(cache, last_used, admitted_live, key, pos, capacity, cur_layer, n_layers, other_live=None):
    inserted = key not in cache
    if inserted:
        admitted_live[key] = False
    cache.add(key)
    last_used[key] = pos
    evicted_unused = 0
    other_evicted_unused = 0
    while len(cache) > capacity:
        victim = evict_ls(cache, last_used, cur_layer, n_layers)
        used = admitted_live.pop(victim, None)
        if used is False:
            evicted_unused += 1
        if other_live is not None:
            other_used = other_live.pop(victim, None)
            if other_used is False:
                other_evicted_unused += 1
    return inserted, evicted_unused, other_evicted_unused


def policy_fetch_counts(policy, layer_states, best_table, cost_table, dense_ms, t_gpu, t_fetch_h2d,
                        prefetch_floor, cpu_scale):
    if policy == "pressure_aware_dp":
        return best_fetch_counts_layer_serial(layer_states, cost_table, dense_ms, t_gpu, t_fetch_h2d,
                                              prefetch_floor, cpu_scale)
    out = []
    for _hit_count, nmiss in layer_states:
        if policy == "spice_fetch_all":
            out.append(nmiss)
        elif policy == "cpu_fallback":
            out.append(0)
        elif policy == "naive_capacity_split":
            out.append(best_table[nmiss] if nmiss else 0)
        else:
            raise ValueError(policy)
    return out


def simulate_one(item, capacity, policy, cost_table, best_table, pop, expert_mb, act_roundtrip_mb, args):
    true_top = item["true_top"]
    fcast = item["fcast"]
    n_layers, tokens, _top_k = true_top.shape
    max_horizon = fcast.shape[1]
    dense_ms = args.t_attn + args.t_gate + args.t_shared
    cache = warm_cache(pop, capacity)
    last_used = {k: 0 for k in cache}
    prefetched_live = {}  # key -> used?
    residual_live = {}    # key -> used?
    fut = build_future_positions_from_true(true_top)

    total_ms = 0.0
    hits = misses = fetches = cpu_served = cpu_layers = 0
    prefetch_issued = prefetch_wrong = prefetch_useful = prefetch_unused = 0
    residual_admit = residual_admit_used = residual_admit_unused = 0
    pcie_h2d_ms_total = 0.0
    token_count = 0
    pos = 0

    for ti in range(tokens):
        if token_count >= args.max_test_tokens:
            break
        token_prefetches = 0
        layer_infos = []
        copy_cursor = 0.0
        pending = []       # (ready_ms, key)
        pending_keys = set()

        def process_ready(now_ms, cur_layer):
            nonlocal prefetch_unused, residual_admit_unused
            ready = [x for x in pending if x[0] <= now_ms]
            if not ready:
                return
            ready.sort(key=lambda x: x[0])
            remain = []
            for ready_ms, key in pending:
                if ready_ms > now_ms:
                    remain.append((ready_ms, key))
                    continue
                pending_keys.discard(key)
                _inserted, unused, residual_unused = cache_add(cache, last_used, prefetched_live, key, pos,
                                                               capacity, cur_layer, n_layers, residual_live)
                prefetch_unused += unused
                residual_admit_unused += residual_unused
            pending[:] = remain

        for l in range(n_layers):
            layer_deadline = l * dense_ms
            process_ready(layer_deadline, l)

            layer_hits = 0
            layer_misses = []
            for e in [int(x) for x in true_top[l, ti].tolist()]:
                k = (l, e)
                if k in cache:
                    hits += 1
                    layer_hits += 1
                    if k in prefetched_live and not prefetched_live[k]:
                        prefetched_live[k] = True
                        prefetch_useful += 1
                    if k in residual_live and not residual_live[k]:
                        residual_live[k] = True
                        residual_admit_used += 1
                    last_used[k] = pos
                else:
                    misses += 1
                    layer_misses.append(e)
                pos += 1
            layer_infos.append((l, layer_hits, layer_misses))

            # Issue draft prefetches for future layers with enough lead. h=0 is current layer and is
            # deliberately excluded by min_lead_layers>=1 to avoid impossible current-layer "hits".
            # Issue time is a lower-bound layer clock; if a prefetch is not ready by the target layer
            # lower-bound, it is a residual miss for this token and is inserted only later.
            max_lead = min(args.max_lead_layers, max_horizon - 1, n_layers - l - 1)
            for lead in range(args.min_lead_layers, max_lead + 1):
                target_l = l + lead
                pred = [int(x) for x in fcast[l, lead, ti].tolist() if int(x) >= 0]
                true_set = set(int(x) for x in true_top[target_l, ti].tolist())
                seen = set()
                for e in pred:
                    if e in seen:
                        continue
                    seen.add(e)
                    k = (target_l, e)
                    if k in cache or k in pending_keys:
                        continue
                    start = max(copy_cursor, layer_deadline)
                    ready = start + args.t_fetch_h2d
                    copy_cursor = ready
                    pending.append((ready, k))
                    pending_keys.add(k)
                    prefetch_issued += 1
                    token_prefetches += 1
                    prefetch_wrong += int(e not in true_set)

        # Late prefetches did not help this token, but by token end their H2D floor has completed,
        # so they may enter HBM and affect future-token residency.
        process_ready(float("inf"), max(0, n_layers - 1))

        layer_states = [(h, len(m)) for _l, h, m in layer_infos]
        prefetch_floor = token_prefetches * args.t_fetch_h2d
        fetch_counts = policy_fetch_counts(policy, layer_states, best_table, cost_table, dense_ms,
                                           args.t_gpu, args.t_fetch_h2d, prefetch_floor, args.cpu_scale)
        token_eval = evaluate_fetch_counts_layer_serial(layer_states, fetch_counts, cost_table, dense_ms,
                                                        args.t_gpu, args.t_fetch_h2d, prefetch_floor,
                                                        args.cpu_scale)
        total_ms += token_eval["token_ms"]
        pcie_h2d_ms_total += token_eval["pcie_h2d_ms"]

        if policy in ("spice_fetch_all", "naive_capacity_split", "pressure_aware_dp"):
            for (l, _h, layer_misses), n_fetch in zip(layer_infos, fetch_counts):
                n_fetch = min(n_fetch, len(layer_misses))
                n_cpu = len(layer_misses) - n_fetch
                fetches += n_fetch
                cpu_served += n_cpu
                if n_cpu:
                    cpu_layers += 1
                fetched = select_residual_fetches(layer_misses, n_fetch, l, ti, fut, args.admit_score)
                for e in fetched:
                    k = (l, e)
                    inserted, unused, prefetch_evicted_unused = cache_add(cache, last_used, residual_live, k, pos,
                                                                          capacity, l, n_layers, prefetched_live)
                    if inserted:
                        residual_admit += 1
                    residual_admit_unused += unused
                    prefetch_unused += prefetch_evicted_unused
        else:
            for _l, _h, layer_misses in layer_infos:
                cpu_served += len(layer_misses)
                if layer_misses:
                    cpu_layers += 1

        token_count += 1

    for used in prefetched_live.values():
        if used is False:
            prefetch_unused += 1
    for used in residual_live.values():
        if used is False:
            residual_admit_unused += 1

    routed = max(1, hits + misses)
    tokens_done = max(1, token_count)
    return {
        "total_ms": total_ms,
        "tokens": token_count,
        "hits": hits,
        "misses": misses,
        "fetches": fetches,
        "cpu_served": cpu_served,
        "cpu_layers": cpu_layers,
        "spice_prefetch_issued": prefetch_issued,
        "spice_prefetch_wrong": prefetch_wrong,
        "spice_prefetch_useful": prefetch_useful,
        "spice_prefetch_unused": prefetch_unused,
        "residual_admissions": residual_admit,
        "residual_admitted_used": residual_admit_used,
        "residual_admitted_unused": residual_admit_unused,
        "expert_h2d_mb": (prefetch_issued + fetches) * expert_mb,
        "fallback_h2d_mb": fetches * expert_mb,
        "cpu_act_roundtrip_mb": cpu_layers * act_roundtrip_mb,
        "pcie_h2d_ms": pcie_h2d_ms_total,
        "hit_rate": hits / routed,
        "residual_rate": misses / routed,
        "tpot_ms": total_ms / tokens_done,
    }


def main() -> None:
    args = parse_args()
    cost_table, best_table, cost_meta, expert_mb, act_roundtrip_mb = load_costs(args.cost_json, args.cost_metric)
    if args.t_fetch_h2d <= 0:
        bw = float(cost_meta.get("config", {}).get("bw_gbps", 0.0))
        args.t_fetch_h2d = (expert_mb / (bw * 1024.0 / 1000.0)) if bw else 0.769
    args.t_fetch_h2d *= args.fetch_scale

    seqs, n_layers, n_experts, top_k, max_horizon, manifest = load_forecast_sequences(args.forecast_dir)
    split = max(1, int(round(len(seqs) * args.train_frac)))
    train_items = seqs[:split]
    test_items = seqs[split:] or seqs
    train_for_pop = [seq_for_popularity(x) for x in train_items]
    pop = popularity(train_for_pop, n_layers, n_experts)

    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    rows = []
    for r in [float(x) for x in args.residency.split(",")]:
        cap = max(1, int(round(r * n_layers * n_experts)))
        for pol in policies:
            agg = defaultdict(float)
            for item in test_items:
                remaining = args.max_test_tokens - int(agg["tokens"])
                if remaining <= 0:
                    break
                old_max = args.max_test_tokens
                args.max_test_tokens = remaining
                res = simulate_one(item, cap, pol, cost_table, best_table, pop, expert_mb, act_roundtrip_mb, args)
                args.max_test_tokens = old_max
                for k, v in res.items():
                    agg[k] += v
            tokens = max(1, agg["tokens"])
            routed = max(1, agg["hits"] + agg["misses"])
            prefetch_issued = max(1, agg["spice_prefetch_issued"])
            row = {
                "policy": pol,
                "residency": r,
                "capacity": cap,
                "tokens": agg["tokens"],
                "tpot_ms": agg["total_ms"] / tokens,
                "hit_rate": agg["hits"] / routed,
                "residual_rate": agg["misses"] / routed,
                "residual_misses_per_tok": agg["misses"] / tokens,
                "fallback_fetches_per_tok": agg["fetches"] / tokens,
                "cpu_served_per_tok": agg["cpu_served"] / tokens,
                "spice_prefetch_issued_per_tok": agg["spice_prefetch_issued"] / tokens,
                "spice_prefetch_wrong_frac": agg["spice_prefetch_wrong"] / prefetch_issued,
                "spice_prefetch_useful_frac": agg["spice_prefetch_useful"] / prefetch_issued,
                "spice_prefetch_unused_frac": agg["spice_prefetch_unused"] / prefetch_issued,
                "pcie_h2d_ms_per_tok": agg["pcie_h2d_ms"] / tokens,
                "expert_h2d_mb_per_tok": agg["expert_h2d_mb"] / tokens,
                "fallback_h2d_mb_per_tok": agg["fallback_h2d_mb"] / tokens,
                "cpu_act_roundtrip_mb_per_tok": agg["cpu_act_roundtrip_mb"] / tokens,
            }
            rows.append(row)
            print(f"res={r:>5} {pol:>18} TPOT={row['tpot_ms']:7.2f} hit={row['hit_rate']:.3f} "
                  f"resid/tok={row['residual_misses_per_tok']:6.2f} pf/tok={row['spice_prefetch_issued_per_tok']:6.2f} "
                  f"fb_fetch/tok={row['fallback_fetches_per_tok']:6.2f}", flush=True)

    by = defaultdict(dict)
    for row in rows:
        by[row["residency"]][row["policy"]] = row
    verdict = {}
    for r, d in by.items():
        base = d.get("spice_fetch_all", {}).get("tpot_ms")
        dp = d.get("pressure_aware_dp", {}).get("tpot_ms")
        cpu = d.get("cpu_fallback", {}).get("tpot_ms")
        naive = d.get("naive_capacity_split", {}).get("tpot_ms")
        verdict[str(r)] = {
            "spice_fetch_all": base,
            "cpu_fallback": cpu,
            "naive_capacity_split": naive,
            "pressure_aware_dp": dp,
            "dp_gain_vs_fetch_all_pct": 100.0 * (base - dp) / base if base and dp else None,
            "dp_gain_vs_naive_pct": 100.0 * (naive - dp) / naive if naive and dp else None,
            "dp_gain_vs_cpu_pct": 100.0 * (cpu - dp) / cpu if cpu and dp else None,
        }
    out = {
        "config": vars(args),
        "forecast_manifest": manifest,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "max_horizon": max_horizon,
        "rows": rows,
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
