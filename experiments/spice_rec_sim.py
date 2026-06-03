"""SPICE-REC faithful timeline simulator (both codex reviews mandate this over scalar max()).

Conservative discrete-event model on real decode traces. Resources: GPU (serial), CPU (burst cost
for N concurrent missed experts), ONE copy engine (PCIe, serial, priority queue). Demand weight-fetch
is a PREDECESSOR of its GPU compute (NO magic overlap of an expert's compute with its own H2D).
REC prefetch runs ONLY in copy-engine idle time, low priority, into a PROTECTED shadow buffer (does
not evict hot main-cache residents -> avoids recreating the eviction-pollution failure).

Policies:
  fetch_fallback : miss -> demand H2D weight (blocks its GPU compute, occupies PCIe). SPICE default.
  cpu_fiddler    : miss -> CPU exact serve (no PCIe weight). PCIe left IDLE (wasted). Fiddler baseline.
  spice_rec      : miss -> CPU serve; during the CPU/idle window, prefetch CURRENT token's downstream
                   experts (ranked by within-token draft proxy A[j,v_cur,.]) on the freed PCIe.
  oracle_shadow  : spice_rec but prefetch the TRUE downstream experts (upper bound).
  random_rec     : spice_rec but prefetch RANDOM downstream experts, same byte budget (control).
  dummy_prefetch : cpu serve + same prefetch bytes but discarded/never demanded (isolate contention).

Key metric (codex): useful_on_time_REC_bytes / theoretical_window_bytes. GO: spice_rec beats
cpu_fiddler by >=15-20% exposed stall AND meaningful wall-clock. All printed English. No core defaults.
"""
import argparse, json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from spice_x_eviction_value import load_sequences, build_A, a_lookup


def parse_args():
    ap = argparse.ArgumentParser(description="SPICE-REC faithful timeline simulator")
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--train_frac", type=float, required=True)
    ap.add_argument("--residency", type=str, required=True)
    ap.add_argument("--max_test_tokens", type=int, required=True)
    ap.add_argument("--prefetch_horizon", type=int, required=True, help="downstream layers ahead to prefetch in REC window")
    ap.add_argument("--shadow_slots", type=int, required=True, help="protected shadow-buffer capacity for REC prefetch (separate from main cache)")
    ap.add_argument("--bw_gbps", type=str, required=True, help="comma list of PCIe bandwidths to sweep (tight regime)")
    ap.add_argument("--t_attn", type=float, required=True)
    ap.add_argument("--t_gate", type=float, required=True)
    ap.add_argument("--t_shared", type=float, required=True)
    ap.add_argument("--t_gpu", type=float, required=True)
    ap.add_argument("--t_cpu1", type=float, required=True, help="CPU exact compute per single expert (ms)")
    ap.add_argument("--expert_mb", type=float, required=True)
    ap.add_argument("--act_kb", type=float, required=True, help="CPU activation roundtrip bytes (KB) over PCIe")
    return ap.parse_args()


CPU_BURST = {0: 0.0, 1: 0.18, 2: 0.56, 3: 1.62, 4: 2.19}


def cpu_burst(n):
    if n <= 4:
        return CPU_BURST[n]
    return CPU_BURST[4] + (n - 4) * (CPU_BURST[4] - CPU_BURST[3])


def popularity(train, n_layers, n_experts):
    pop = np.zeros((n_layers, n_experts))
    for seq in train:
        for (_v, pl) in seq:
            for l, topk in enumerate(pl):
                for e in topk:
                    pop[l, e] += 1
    return pop


def simulate(seq, n_layers, n_experts, capacity, policy, comp, A, layer_marg, pop, horizon, shadow_slots, rng):
    """Faithful timeline. Returns dict with total_ms, exposed components, REC usefulness counters."""
    tok_ids = [v for (v, _) in seq]
    t_fetch = comp["t_fetch"]; t_act = comp["t_act"]  # weight H2D, activation roundtrip (ms)
    warmpop = sorted([((l, e), pop[l, e]) for l in range(n_layers) for e in range(n_experts)], key=lambda x: -x[1])
    cache = set(k for k, _ in warmpop[:capacity])      # main resident cache (static warm; demand-fetch admits+LS)
    last_used = {k: 0 for k in cache}; key_layer = {k: k[0] for k in cache}
    shadow = {}                                         # key -> completion_time (REC prefetched, protected)

    clock = 0.0; copy_free = 0.0
    total = 0.0
    rec_bytes_issued = 0.0; rec_bytes_useful = 0.0; window_bytes_theory = 0.0
    exposed_fetch = 0.0; exposed_cpu = 0.0

    def ls_evict():
        victim = min(cache, key=lambda k: last_used[k])
        cache.discard(victim); last_used.pop(victim, None); key_layer.pop(victim, None)

    pos = 0
    for ti, (_v, pl) in enumerate(seq):
        for l in range(n_layers):
            layer_start = clock
            # promote any shadow prefetches that have completed by now into availability
            routed = pl[l]
            avail = []
            misses = []
            for e in routed:
                k = (l, e)
                if k in cache:
                    avail.append(e); last_used[k] = pos
                elif k in shadow and shadow[k] <= clock:
                    avail.append(e); rec_bytes_useful += comp["expert_mb"]  # on-time useful REC prefetch
                    del shadow[k]
                    cache.add(k); last_used[k] = pos; key_layer[k] = l
                    if len(cache) > capacity: ls_evict()
                else:
                    misses.append(e)
            # GPU base (attn+gate+shared+resident routed compute)
            gpu_base = comp["t_attn"] + comp["t_gate"] + comp["t_shared"] + len(avail) * comp["t_gpu"]

            if policy == "fetch_fallback":
                # each miss demand-fetched on copy engine (serial, predecessor of its gpu compute)
                fetch_end = max(copy_free, layer_start)
                for _e in misses:
                    fetch_end += t_fetch
                copy_free = fetch_end
                # gpu compute of fetched experts after their fetch completes
                gpu_done = max(layer_start + gpu_base, fetch_end) + len(misses) * comp["t_gpu"]
                layer_end = gpu_done
                exposed_fetch += max(0.0, fetch_end - (layer_start + gpu_base))
                # admit fetched into cache (LS)
                for e in misses:
                    k = (l, e)
                    cache.add(k); last_used[k] = pos; key_layer[k] = l
                    if len(cache) > capacity: ls_evict()
            else:
                # CPU serve all misses; activation roundtrip on copy engine (high priority, KB)
                n_cpu = len(misses)
                cpu_end = layer_start + cpu_burst(n_cpu)
                act_end = max(copy_free, layer_start) + (t_act if n_cpu > 0 else 0.0)
                copy_free = max(copy_free, act_end)
                gpu_done = layer_start + gpu_base
                layer_end = max(gpu_done, cpu_end, act_end)
                exposed_cpu += max(0.0, cpu_end - gpu_done)
                # REC prefetch in the freed PCIe window [copy_free, layer_end] (low priority, idle-fill)
                if policy in ("spice_rec", "oracle_shadow", "random_rec", "dummy_prefetch"):
                    window = max(0.0, layer_end - copy_free)
                    window_bytes_theory += comp["bw_mb_per_ms"] * window
                    n_slots = int(window // t_fetch)  # whole experts prefetchable in the idle window
                    if n_slots > 0:
                        cand = pick_prefetch(policy, l, tok_ids[ti], routed_future(pl, l, horizon),
                                             A, layer_marg, n_experts, cache, shadow, n_slots, rng, n_layers)
                        t = copy_free
                        for k in cand:
                            t += t_fetch
                            if policy != "dummy_prefetch":
                                if len(shadow) < shadow_slots:
                                    shadow[k] = t
                            rec_bytes_issued += comp["expert_mb"]
                        copy_free = t
            total += (layer_end - layer_start)
            clock = layer_end
            pos += len(routed)
    return {"total_ms": total, "tokens": len(seq), "exposed_fetch_ms": exposed_fetch,
            "exposed_cpu_ms": exposed_cpu, "rec_bytes_issued": rec_bytes_issued,
            "rec_bytes_useful": rec_bytes_useful, "window_bytes_theory": window_bytes_theory}


def routed_future(pl, l, horizon):
    """True experts at downstream layers l+1..l+horizon (for oracle); list of (layer, expert)."""
    out = []
    for j in range(l + 1, min(l + 1 + horizon, len(pl))):
        for e in pl[j]:
            out.append((j, e))
    return out


def pick_prefetch(policy, l, v_cur, true_future, A, layer_marg, n_experts, cache, shadow, n_slots, rng, n_layers):
    """Choose up to n_slots downstream (layer,expert) to prefetch, not already resident/inflight."""
    horizon_layers = sorted(set(j for (j, _e) in true_future))
    cand = []
    if policy == "oracle_shadow":
        ranked = true_future  # true downstream experts
    elif policy == "spice_rec":
        # within-token draft proxy: top experts by A[j, v_cur, .] for each downstream layer
        ranked = []
        for j in horizon_layers:
            probs = [(A.get((j, v_cur))[e] if A.get((j, v_cur)) is not None else layer_marg[j, e], (j, e))
                     for e in range(n_experts)]
            probs.sort(key=lambda x: -x[0])
            ranked += [k for _p, k in probs[:4]]  # top-4 predicted per layer (~top_k)
    elif policy in ("random_rec", "dummy_prefetch"):
        ranked = [(j, int(rng.integers(0, n_experts))) for j in horizon_layers for _ in range(4)]
    else:
        ranked = []
    for k in ranked:
        if k in cache or k in shadow:
            continue
        cand.append(k)
        if len(cand) >= n_slots:
            break
    return cand


def main():
    a = parse_args()
    seqs, n_layers, n_experts = load_sequences(a.trace_dir)
    n_train = max(1, int(round(a.train_frac * len(seqs))))
    train, test = seqs[:n_train], seqs[n_train:]
    if not test:
        raise ValueError("empty test split; lower train_frac")
    A, layer_marg = build_A(train, n_layers, n_experts, a.alpha)
    pop = popularity(train, n_layers, n_experts)
    tot = 0; test2 = []
    for s in test:
        test2.append(s); tot += len(s)
        if tot >= a.max_test_tokens: break
    test = test2
    print(f"[data] train={len(train)} test={len(test)} tokens={sum(len(s) for s in test)} "
          f"layers={n_layers} experts={n_experts}", flush=True)

    total_slots = n_layers * n_experts
    residencies = [float(x) for x in a.residency.split(",")]
    bws = [float(x) for x in a.bw_gbps.split(",")]
    policies = ["fetch_fallback", "cpu_fiddler", "spice_rec", "oracle_shadow", "random_rec", "dummy_prefetch"]
    rows = []
    for bw in bws:
        bw_mb_per_ms = bw * 1024.0 / 1000.0  # GB/s -> MB/ms
        t_fetch = a.expert_mb / bw_mb_per_ms
        t_act = (a.act_kb / 1024.0) / bw_mb_per_ms
        comp = {"t_attn": a.t_attn, "t_gate": a.t_gate, "t_shared": a.t_shared, "t_gpu": a.t_gpu,
                "t_fetch": t_fetch, "t_act": t_act, "expert_mb": a.expert_mb, "bw_mb_per_ms": bw_mb_per_ms}
        for r in residencies:
            cap = max(1, int(round(r * total_slots)))
            for pol in policies:
                rng = np.random.default_rng(0)
                agg = defaultdict(float)
                for s in test:
                    res = simulate(s, n_layers, n_experts, cap, pol, comp, A, layer_marg, pop,
                                   a.prefetch_horizon, a.shadow_slots, rng)
                    for k, v in res.items(): agg[k] += v
                tpot = agg["total_ms"] / max(1, agg["tokens"])
                useful_frac = agg["rec_bytes_useful"] / max(1e-9, agg["window_bytes_theory"])
                rows.append({"bw": bw, "residency": r, "policy": pol, "tpot_ms": tpot,
                             "exposed_fetch_ms_tok": agg["exposed_fetch_ms"] / max(1, agg["tokens"]),
                             "exposed_cpu_ms_tok": agg["exposed_cpu_ms"] / max(1, agg["tokens"]),
                             "rec_useful_mb_tok": agg["rec_bytes_useful"] / max(1, agg["tokens"]),
                             "rec_issued_mb_tok": agg["rec_bytes_issued"] / max(1, agg["tokens"]),
                             "useful_on_time_frac": useful_frac})
                print(f"bw={bw:>4} res={r:>5} {pol:>15} TPOT={tpot:7.3f} exp_cpu={rows[-1]['exposed_cpu_ms_tok']:.3f} "
                      f"rec_useful={rows[-1]['rec_useful_mb_tok']:6.1f}MB issued={rows[-1]['rec_issued_mb_tok']:6.1f} "
                      f"useful_frac={useful_frac:.2f}", flush=True)
    # verdict
    print("\n===== SPICE-REC VERDICT =====", flush=True)
    by = defaultdict(dict)
    for x in rows: by[(x["bw"], x["residency"])][x["policy"]] = x
    verdict = {}
    for (bw, r), d in by.items():
        fid = d["cpu_fiddler"]["tpot_ms"]; rec = d["spice_rec"]["tpot_ms"]; orc = d["oracle_shadow"]["tpot_ms"]
        ff = d["fetch_fallback"]["tpot_ms"]; rnd = d["random_rec"]["tpot_ms"]
        gain = (fid - rec) / fid if fid > 0 else 0.0
        verdict[f"bw{bw}_r{r}"] = {"fetch_fallback": ff, "cpu_fiddler": fid, "spice_rec": rec,
                                   "oracle_shadow": orc, "random_rec": rnd,
                                   "rec_gain_over_fiddler_pct": 100 * gain,
                                   "GO": gain >= 0.15 and rec < rnd}
        print(f"bw{bw} r{r}: fetch={ff:.2f} fiddler={fid:.2f} spice_rec={rec:.2f} oracle={orc:.2f} "
              f"random={rnd:.2f} | REC_gain_vs_fiddler={100*gain:+.1f}% {'GO' if (gain>=0.15 and rec<rnd) else 'no'}", flush=True)
    Path(a.out).write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
