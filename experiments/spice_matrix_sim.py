"""Matrix-granular exact residency simulator (user step 2; decides if finer-than-expert HBM caching
beats whole-expert caching on real traces). EXACT same-precision. Uses measured A800 matrix edges.

Question: cache at MATRIX granularity (3 objects/expert, 5.77MB each) vs WHOLE-expert (17.3MB) at the
SAME HBM-MB budget -- does finer granularity + lower partial-fetch cost give >=15% TPOT over whole?
Misses served exactly by Fiddler CPU compute (0.167ms) unless cached. Capacity in MB (so finer
granularity's packing benefit is captured). All printed English. Core params: no defaults.

Policies (all EXACT):
  cpu_fiddler   : warm whole-expert cache; every nonresident expert CPU-served (0.167ms). Baseline.
  whole_cache   : whole-expert objects, LRU; hit=resident_hit; miss=CPU-serve(0.167) + admit whole (fetch 0.78 bg).
  matrix_cache  : 3 matrices/expert as independent LRU objects; expert hit iff all 3 resident; else
                  CPU-serve(0.167) + admit the expert's 3 matrices.
  oracle_matrix : matrix objects with Belady eviction (true next use). Upper bound.
"""
import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from spice_x_eviction_value import load_sequences


def parse_args():
    ap = argparse.ArgumentParser(description="Matrix-granular exact residency simulator")
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--train_frac", type=float, required=True)
    ap.add_argument("--hbm_mb", type=str, required=True, help="comma list of HBM expert-residency budgets (MB)")
    ap.add_argument("--max_test_tokens", type=int, required=True)
    ap.add_argument("--matrix_mb", type=float, required=True)
    ap.add_argument("--t_hit", type=float, required=True, help="resident whole-expert compute ms")
    ap.add_argument("--t_cpu", type=float, required=True, help="Fiddler CPU exact serve ms")
    ap.add_argument("--t_attn_layer", type=float, required=True, help="per-layer non-expert const ms (attn+gate+shared)")
    return ap.parse_args()


def pop_experts(train, n_layers, n_experts):
    pop = np.zeros((n_layers, n_experts))
    for seq in train:
        for (_v, pl) in seq:
            for l, topk in enumerate(pl):
                for e in topk:
                    pop[l, e] += 1
    return pop


def simulate(seq, n_layers, n_experts, cap_objs, policy, pop, t_hit, t_cpu, t_attn):
    """Return (total_ms, tokens, hits, cpu_serves). Objects: whole=(l,e); matrix=(l,e,m) m in 0..2.
    Capacity cap_objs in OBJECTS (whole: experts; matrix/oracle: matrices)."""
    matrix = policy in ("matrix_cache", "oracle_matrix")
    # flatten true stream for belady (matrix granularity)
    flat = []
    for ti, (_v, pl) in enumerate(seq):
        for l, topk in enumerate(pl):
            for e in topk:
                if matrix:
                    for m in range(3): flat.append((l, e, m))
                else:
                    flat.append((l, e))
    occ = defaultdict(list)
    if policy == "oracle_matrix":
        for i, k in enumerate(flat): occ[k].append(i)
    occ_ptr = defaultdict(int)

    # warm cache by popularity
    objs = []
    for l in range(n_layers):
        for e in range(n_experts):
            if matrix:
                for m in range(3): objs.append(((l, e, m), pop[l, e]))
            else:
                objs.append(((l, e), pop[l, e]))
    objs.sort(key=lambda x: -x[1])
    cache = set() if policy == "cpu_fiddler" else set(k for k, _ in objs[:cap_objs])  # fiddler = no cache floor
    last_used = {k: 0 for k in cache}

    def nxt(k, pos):
        ps = occ[k]; p = occ_ptr[k]
        while p < len(ps) and ps[p] <= pos: p += 1
        occ_ptr[k] = p
        return ps[p] if p < len(ps) else len(flat) + 10

    def evict(pos):
        if policy == "oracle_matrix":
            v = max(cache, key=lambda k: nxt(k, pos))
        else:
            v = min(cache, key=lambda k: last_used[k])
        cache.discard(v); last_used.pop(v, None)

    total = 0.0; hits = 0; cpu = 0; pos = 0
    for ti, (_v, pl) in enumerate(seq):
        for l in range(n_layers):
            total += t_attn
            for e in pl[l]:
                if matrix:
                    keys = [(l, e, m) for m in range(3)]
                    resident = all(k in cache for k in keys)
                else:
                    keys = [(l, e)]; resident = keys[0] in cache
                if resident:
                    hits += 1; total += t_hit
                    for k in keys: last_used[k] = pos
                else:
                    cpu += 1; total += t_cpu          # exact Fiddler serve
                    if policy == "oracle_matrix":     # only oracle admits dynamically (Belady)
                        for k in keys:
                            if k not in cache:
                                while len(cache) >= cap_objs and cache: evict(pos)
                                if cap_objs >= 1: cache.add(k)
                            last_used[k] = pos
                    # static policies (cpu_fiddler/whole_cache/matrix_cache): keep popularity-warm set, no admit
                pos += (3 if matrix else 1)
    return total, len(seq), hits, cpu


def main():
    a = parse_args()
    seqs, n_layers, n_experts = load_sequences(a.trace_dir)
    n_train = max(1, int(round(a.train_frac * len(seqs))))
    train, test = seqs[:n_train], seqs[n_train:]
    if not test: raise ValueError("empty test; lower train_frac")
    pop = pop_experts(train, n_layers, n_experts)
    tot = 0; t2 = []
    for s in test:
        t2.append(s); tot += len(s)
        if tot >= a.max_test_tokens: break
    test = t2
    expert_mb = 3 * a.matrix_mb
    print(f"[data] train={len(train)} test={len(test)} tokens={sum(len(s) for s in test)} "
          f"layers={n_layers} experts={n_experts} expert={expert_mb:.1f}MB", flush=True)
    rows = []
    for hbm in [float(x) for x in a.hbm_mb.split(",")]:
        for pol in ["cpu_fiddler", "whole_cache", "matrix_cache", "oracle_matrix"]:
            obj_mb = a.matrix_mb if pol in ("matrix_cache", "oracle_matrix") else expert_mb
            cap_objs = max(1, int(hbm / obj_mb))
            T = hits = cpu = toks = 0
            for s in test:
                tt, tk, h, c = simulate(s, n_layers, n_experts, cap_objs, pol, pop,
                                        a.t_hit, a.t_cpu, a.t_attn_layer)
                T += tt; toks += tk; hits += h; cpu += c
            tpot = T / max(1, toks)
            rows.append({"hbm_mb": hbm, "policy": pol, "cap_objs": cap_objs, "tpot_ms": tpot,
                         "hit_rate": hits / max(1, hits + cpu)})
            print(f"hbm={hbm:>6}MB {pol:>14} cap={cap_objs:>6} TPOT={tpot:7.3f} hit={hits/max(1,hits+cpu):.3f}", flush=True)
    # verdict
    print("\n===== MATRIX-GRANULAR VERDICT =====", flush=True)
    by = defaultdict(dict)
    for x in rows: by[x["hbm_mb"]][x["policy"]] = x["tpot_ms"]
    verdict = {}
    for hbm, d in by.items():
        whole = d["whole_cache"]; mat = d["matrix_cache"]; orc = d["oracle_matrix"]; fid = d["cpu_fiddler"]
        verdict[str(hbm)] = {"cpu_fiddler": fid, "whole": whole, "matrix": mat, "oracle_matrix": orc,
                             "matrix_vs_whole_pct": 100 * (whole - mat) / whole,
                             "oracle_vs_whole_pct": 100 * (whole - orc) / whole}
        print(f"hbm={hbm}MB: fiddler={fid:.2f} whole={whole:.2f} matrix={mat:.2f} oracle={orc:.2f} | "
              f"matrix_vs_whole={100*(whole-mat)/whole:+.1f}% oracle_vs_whole={100*(whole-orc)/whole:+.1f}%", flush=True)
    Path(a.out).write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
