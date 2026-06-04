"""DECISIVE: does the speculative-K window help or HURT offloaded miss cost vs batch=1 AR?

DECISIVE: speculative-K 窗口对 offloaded miss 成本是帮助还是损害 (vs batch=1 AR)?

Hypothesis to test (back-of-envelope said spec HURTS): a verify step computes K candidate
positions' experts but advances only ~accept+1 tokens; rejected positions' expert work is wasted.
Window cross-position reuse (C/U 1.27-1.88) dedups some, but may not compensate.
要检验的假设 (粗算说 spec 损害): verify step 计算 K 个候选位置的专家, 但只推进 ~accept+1 个 token;
被拒位置的专家计算浪费. 窗口内复用去重一部分, 但可能补偿不了.

Method (uses REAL P1 windows + REAL measured cost table; cache = top-popular resident set):
  For residency r, resident set R = top (r*L*E) experts by popularity over the windows.
  SPEC per advanced-token miss cost = mean_step( sum_layers cost_best[ U_miss(layer) ] ) / tokens_per_step,
    where U_miss(layer) = #unique experts in the K positions' top-k routes NOT in R.
  AR  per-token miss cost = mean over single candidate positions( sum_layers cost_best[ n_miss(layer) ] ),
    where n_miss(layer) = #experts in that ONE position's top-k NOT in R  (tokens advanced = 1).
  Also report cpu_all and fetch_all variants. Lower = better.
All printed English. Core params: no defaults.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def parse_args():
    ap = argparse.ArgumentParser(description="Spec-window vs AR per-token offloaded miss cost (decisive)")
    ap.add_argument("--windows", required=True, help="windows .pt from spec_decode_capture.py")
    ap.add_argument("--cost_json", required=True, help="miss_assignment_microbench output (extended n_miss)")
    ap.add_argument("--residency", required=True, help="comma list of resident fractions, e.g. 0.05,0.1,0.2")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load_cost(path: str):
    rows = json.loads(Path(path).read_text())["rows"]
    table = {}
    for r in rows:
        table[(int(r["n_miss"]), int(r["n_fetch"]))] = float(r["ms"])
    ns = sorted({int(r["n_miss"]) for r in rows})
    best = {}; cpu_all = {}; fetch_all = {}
    for n in ns:
        choices = [(f, table[(n, f)]) for f in range(n + 1) if (n, f) in table]
        best[n] = min(c[1] for c in choices)
        cpu_all[n] = table[(n, 0)]
        fetch_all[n] = table[(n, n)]
    nmax = max(ns)
    return best, cpu_all, fetch_all, nmax


def cost_lookup(tbl: dict, n: int, nmax: int) -> float:
    if n <= 0:
        return 0.0
    if n in tbl:
        return tbl[n]
    # extrapolate linearly from the top two measured points for n > nmax
    a = tbl[nmax]; b = tbl[nmax - 1]
    return a + (a - b) * (n - nmax)


def main():
    a = parse_args()
    blob = torch.load(a.windows, map_location="cpu", weights_only=False)
    wins = blob["windows"]
    L = int(blob["n_layers"]); E = int(blob["n_experts"]); top_k = int(blob["top_k"]); K = int(blob["K"])
    tps = float(blob["mean_tokens_per_step"])
    best, cpu_all, fetch_all, nmax = load_cost(a.cost_json)

    # popularity over all windows (per-layer expert counts) -> resident set per layer-global ranking
    pop = Counter()
    for w in wins:
        routes = w["routes"]  # [L, K, top_k]
        for l in range(L):
            for e in routes[l].reshape(-1).tolist():
                pop[(l, int(e))] += 1

    residencies = [float(x) for x in a.residency.split(",")]
    out_rows = []
    for r in residencies:
        cap = max(1, int(round(r * L * E)))
        resident = set(k for k, _ in pop.most_common(cap))

        # SPEC: per step, per layer unique missed experts -> cost_best, summed; advanced tokens = tps
        spec_best = spec_cpu = spec_fetch = 0.0
        # AR: per single candidate position, per layer missed experts -> cost; tokens = 1
        ar_best = ar_cpu = ar_fetch = 0.0
        committed_tokens = 0  # total AR-decoded tokens = sum of accepted positions
        for w in wins:
            routes = w["routes"]  # [L,K,top_k]
            # AR only ever decodes COMMITTED tokens = accepted candidate positions [0..ac-1]
            # (rejected candidates are never decoded in plain AR). Use them as AR's per-token routes.
            # AR 只解码已提交 token = 已接受候选位置 [0..ac-1] (被拒候选 AR 从不解码).
            ac = max(1, int(w["accept_count"]))  # at least the first position is decoded
            committed_tokens += min(ac, routes.shape[1])
            for l in range(L):
                rl = routes[l]  # [K, top_k]
                # spec: unique missed across ALL K candidate positions (the verify forward computes all K)
                uniq = set(int(e) for e in rl.reshape(-1).tolist())
                u_miss = sum(1 for e in uniq if (l, e) not in resident)
                spec_best += cost_lookup(best, u_miss, nmax)
                spec_cpu += cost_lookup(cpu_all, u_miss, nmax)
                spec_fetch += cost_lookup(fetch_all, u_miss, nmax)
                # ar: only accepted (committed) positions, each processed independently (K=1)
                for p in range(min(ac, rl.shape[0])):
                    miss = sum(1 for e in set(int(x) for x in rl[p].tolist()) if (l, e) not in resident)
                    ar_best += cost_lookup(best, miss, nmax)
                    ar_cpu += cost_lookup(cpu_all, miss, nmax)
                    ar_fetch += cost_lookup(fetch_all, miss, nmax)
        n_steps = len(wins)
        # both costs are summed over ALL layers; normalize to PER COMMITTED TOKEN.
        # spec advances tps tokens per step over n_steps; AR decodes committed_tokens one at a time.
        spec_per_tok = spec_best / (n_steps * tps)
        spec_per_tok_cpu = spec_cpu / (n_steps * tps)
        ar_per_tok = ar_best / committed_tokens
        ar_per_tok_cpu = ar_cpu / committed_tokens
        ar_per_tok_fetch = ar_fetch / committed_tokens
        row = {
            "residency": r, "capacity": cap, "K": K, "tokens_per_step": tps,
            "ar_fiddler_cpu_per_tok_ms": ar_per_tok_cpu,
            "ar_split_per_tok_ms": ar_per_tok,
            "ar_fetchall_per_tok_ms": ar_per_tok_fetch,
            "spec_split_per_tok_ms": spec_per_tok,
            "spec_cpu_per_tok_ms": spec_per_tok_cpu,
            "spec_vs_ar_split_ratio": spec_per_tok / ar_per_tok,
            "spec_vs_ar_fiddler_ratio": spec_per_tok / ar_per_tok_cpu,
        }
        out_rows.append(row)
        verdict = "SPEC HELPS" if row["spec_vs_ar_split_ratio"] < 1.0 else "SPEC HURTS"
        print(f"res={r:.3f} K={K} tps={tps:.2f} | AR-Fiddler={ar_per_tok_cpu:.3f} AR-split={ar_per_tok:.3f} "
              f"| SPEC-split={spec_per_tok:.3f} | spec/AR-split={row['spec_vs_ar_split_ratio']:.2f} "
              f"spec/AR-Fiddler={row['spec_vs_ar_fiddler_ratio']:.2f} -> {verdict}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"windows": a.windows, "cost": a.cost_json,
                                       "L": L, "E": E, "top_k": top_k, "rows": out_rows}, indent=2))


if __name__ == "__main__":
    main()
