SCRUTINIZE this faithful-DAG result (positive, so suspect it). spice_rec_sim.py now has faithful CPU path (D2H->CPU->H2D->merge serial) + two DAG-reshaping levers. Result bw12 res10%: cpu_fiddler 44.7, spice_rec 44.3 (+0.8%), compressed_fetch(int4 0.25x H2D) 34.5 (-23%), slo_drop(drop 25% lowest-rank) 29.9 (-33%).
Questions:
1. Is compressed_fetch's -23% real under this model or an artifact? It demand-fetches every miss at 0.25x bytes and caches; exp_cpu=0. Is "compression 4x-shortens H2D -> fetch fits slack -> high residency" a faithful conclusion, or does the model under-charge something (e.g. int4 still needs dequant compute on GPU; quality loss not modeled)?
2. slo_drop -33%: dropping 25% lowest-gate-rank missed experts. Faithful that this just removes their CPU-serve cost? Quality cost is OUT of this latency model (must be added separately) - correct to report latency-only here with that caveat?
3. The headline claim "at batch=1 exact, pure scheduling (spice_rec/admission) ~1% but DAG-reshaping (compression/drop) 23-33%" - is this a sound, defensible characterization from this faithful sim, or are there remaining fidelity bugs that inflate compression/drop or deflate spice_rec?
4. CPU path now: cpu_start=layer_start+t_act_in, cpu_done=+cpu_burst, out_done=+t_act_out, layer_end=max(gpu,out_done). Faithful enough? host DRAM contention still unmodeled - does that change the RELATIVE ranking (compression uses PCIe+GPU-dequant; cpu_fiddler uses CPU+DRAM; drop uses less CPU)?
5. Verdict: trust "compression & drop are the only effective batch=1 levers" as a paper characterization? What single fidelity fix would most change it?
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
    ap.add_argument("--expert_mb", type=float, required=True)
    ap.add_argument("--act_kb", type=float, required=True, help="CPU activation roundtrip bytes (KB) over PCIe")
    ap.add_argument("--compress_ratio", type=float, required=True, help="compressed_fetch: expert H2D bytes fraction (e.g. 0.25 int4)")
    ap.add_argument("--drop_ratio", type=float, required=True, help="slo_drop: fraction of routed experts to drop (lowest gate rank)")
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


PREFETCH_POLICIES = ("spice_rec", "oracle_shadow", "random_rec", "dummy_prefetch", "fetch_idle_prefetch")
FETCH_POLICIES = ("fetch_fallback", "fetch_idle_prefetch", "compressed_fetch")  # demand-fetch misses to GPU


def simulate(seq, n_layers, n_experts, capacity, policy, comp, A, layer_marg, pop, horizon, shadow_slots, top_k, rng):
    """Faithful timeline with a PREEMPTIBLE prefetch queue (codex fixes 1-8).
    Copy engine: demand weight-fetch / activation copy-out are high priority (counted in layer time);
    REC prefetch is interruptible and consumes ONLY genuine PCIe idle accumulated across layers, so it
    NEVER delays demand (dummy_prefetch is correctly a timing no-op -> isolates prediction value, not
    contention). idle is large for cpu policies (no weight fetch) and small for fetch policies -> this
    isolates the bandwidth FREED by CPU-serving. shadow entries carry a within-token deadline."""
    tok_ids = [v for (v, _) in seq]
    t_fetch = comp["t_fetch"]; t_act = comp["t_act"]
    warmpop = sorted([((l, e), pop[l, e]) for l in range(n_layers) for e in range(n_experts)], key=lambda x: -x[1])
    cache = set(k for k, _ in warmpop[:capacity])
    last_used = {k: 0 for k in cache}; key_layer = {k: k[0] for k in cache}
    shadow = {}                       # key -> (target_tok, target_layer): completed prefetch, available
    pf_queue = []                     # in-flight preemptible prefetch: [key, target_tok, target_layer, remaining_ms]

    clock = 0.0; total = 0.0
    rec_bytes_issued = 0.0; rec_bytes_useful = 0.0; idle_ms_total = 0.0
    exposed_fetch = 0.0; exposed_cpu = 0.0

    def ls_dist(k, cur_layer):
        d = (key_layer[k] - cur_layer) % n_layers
        return n_layers if d == 0 else d

    def ls_evict(cur_layer):
        victim = max(cache, key=lambda k: (ls_dist(k, cur_layer), -last_used[k]))  # cyclic Least-Stale (codex #8)
        cache.discard(victim); last_used.pop(victim, None); key_layer.pop(victim, None)

    def expired(target_tok, target_layer, ti, l):
        return target_tok < ti or (target_tok == ti and target_layer < l)  # within-token deadline (codex #2)

    pos = 0
    for ti, (_v, pl) in enumerate(seq):
        for l in range(n_layers):
            layer_start = clock
            routed = pl[l]
            # drop expired shadow / in-flight prefetches (missed their within-token demand)
            shadow = {k: v for k, v in shadow.items() if not expired(v[0], v[1], ti, l)}
            pf_queue = [q for q in pf_queue if not expired(q[1], q[2], ti, l)]
            avail = []; misses = []
            for e in routed:
                k = (l, e)
                if k in cache:
                    avail.append(e); last_used[k] = pos
                elif k in shadow:                       # prefetch completed on time
                    avail.append(e); rec_bytes_useful += comp["expert_mb"]; del shadow[k]
                    cache.add(k); last_used[k] = pos; key_layer[k] = l
                    if len(cache) > capacity: ls_evict(l)
                else:
                    misses.append(e)
            gpu_base = comp["t_attn"] + comp["t_gate"] + comp["t_shared"] + len(avail) * comp["t_gpu"]

            if policy in FETCH_POLICIES:
                tf = t_fetch * (comp["compress_ratio"] if policy == "compressed_fetch" else 1.0)  # low-bit shortens H2D
                demand_copy = len(misses) * tf
                fetch_end = layer_start + demand_copy
                gpu_done = max(layer_start + gpu_base, fetch_end) + len(misses) * comp["t_gpu"]
                layer_end = gpu_done
                exposed_fetch += max(0.0, fetch_end - (layer_start + gpu_base))
                for e in misses:
                    k = (l, e); cache.add(k); last_used[k] = pos; key_layer[k] = l
                    if len(cache) > capacity: ls_evict(l)
            else:
                served = misses
                if policy == "slo_drop":            # break exact dependency: drop lowest-rank missed experts
                    keep = max(0, len(misses) - int(round(comp["drop_ratio"] * len(routed))))
                    served = misses[:keep]          # misses keep routed (gate-rank) order; drop tail
                n_cpu = len(served)
                if n_cpu > 0:
                    # FAITHFUL CPU path (codex/user): D2H(h) -> CPU compute -> H2D(out) -> merge, SERIAL
                    t_act_in = t_act; t_act_out = n_cpu * t_act
                    cpu_start = layer_start + t_act_in
                    cpu_done = cpu_start + cpu_burst(n_cpu)
                    out_done = cpu_done + t_act_out
                    demand_copy = t_act_in + t_act_out
                    layer_end = max(layer_start + gpu_base, out_done)
                    exposed_cpu += max(0.0, out_done - (layer_start + gpu_base))
                else:
                    demand_copy = 0.0
                    layer_end = layer_start + gpu_base

            layer_dur = layer_end - layer_start
            # genuine PCIe idle this layer = layer time minus high-priority demand copy
            idle = max(0.0, layer_dur - demand_copy)
            idle_ms_total += idle

            if policy in PREFETCH_POLICIES:
                # enqueue new candidates (bounded by shadow_slots + queue length)
                room = shadow_slots - len(shadow) - len(pf_queue)
                if room > 0:
                    cand = pick_prefetch(policy, tok_ids[ti], pl, l, ti, A, layer_marg, n_experts,
                                         cache, shadow, pf_queue, room, top_k, horizon, rng)
                    for k, tt, tl in cand:
                        pf_queue.append([k, tt, tl, t_fetch]); rec_bytes_issued += comp["expert_mb"]
                # drain idle PCIe into queue by EARLIEST-DEADLINE-FIRST (codex: near-demand prefetch must
                # not starve behind far-future ones). Sort by (target_tok, target_layer).
                pf_queue.sort(key=lambda q: (q[1], q[2]))
                budget = idle
                while budget > 1e-9 and pf_queue:
                    q = pf_queue[0]
                    step = min(budget, q[3]); q[3] -= step; budget -= step
                    if q[3] <= 1e-9:
                        shadow[q[0]] = (q[1], q[2]); pf_queue.pop(0)
            total += layer_dur; clock = layer_end; pos += len(routed)
    return {"total_ms": total, "tokens": len(seq), "exposed_fetch_ms": exposed_fetch,
            "exposed_cpu_ms": exposed_cpu, "rec_bytes_issued": rec_bytes_issued,
            "rec_bytes_useful": rec_bytes_useful, "idle_ms_total": idle_ms_total}


def pick_prefetch(policy, v_cur, pl, l, ti, A, layer_marg, n_experts, cache, shadow, pf_queue, room, top_k, horizon, rng):
    """Return up to `room` (key, target_tok, target_layer) downstream prefetch candidates for the CURRENT
    token's layers l+1..l+horizon, not already resident/in-flight. dedup within candidates (codex #5)."""
    inflight = set(q[0] for q in pf_queue)
    layers = list(range(l + 1, min(l + 1 + horizon, len(pl))))
    ranked = []
    if policy in ("oracle_shadow",):
        for j in layers:
            for e in pl[j]:
                ranked.append((j, e))
    elif policy in ("spice_rec", "fetch_idle_prefetch"):
        for j in layers:
            row = A.get((j, v_cur))
            probs = [((row[e] if row is not None else layer_marg[j, e]), e) for e in range(n_experts)]
            probs.sort(key=lambda x: -x[0])
            for _p, e in probs[:top_k]:
                ranked.append((j, e))
    elif policy in ("random_rec", "dummy_prefetch"):
        for j in layers:
            for _ in range(top_k):
                ranked.append((j, int(rng.integers(0, n_experts))))
    cand = []; seen = set()
    for (j, e) in ranked:
        k = (j, e)
        if k in cache or k in shadow or k in inflight or k in seen:
            continue
        seen.add(k); cand.append((k, ti, j))   # target = current token, layer j (within-token deadline)
        if len(cand) >= room:
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
    policies = ["fetch_fallback", "fetch_idle_prefetch", "compressed_fetch", "cpu_fiddler",
                "spice_rec", "oracle_shadow", "random_rec", "dummy_prefetch", "slo_drop"]
    rows = []
    for bw in bws:
        bw_mb_per_ms = bw * 1024.0 / 1000.0  # GB/s -> MB/ms
        t_fetch = a.expert_mb / bw_mb_per_ms
        t_act = (a.act_kb / 1024.0) / bw_mb_per_ms
        comp = {"t_attn": a.t_attn, "t_gate": a.t_gate, "t_shared": a.t_shared, "t_gpu": a.t_gpu,
                "t_fetch": t_fetch, "t_act": t_act, "expert_mb": a.expert_mb, "bw_mb_per_ms": bw_mb_per_ms,
                "compress_ratio": a.compress_ratio, "drop_ratio": a.drop_ratio}
        for r in residencies:
            cap = max(1, int(round(r * total_slots)))
            for pol in policies:
                rng = np.random.default_rng(0)
                agg = defaultdict(float)
                for s in test:
                    res = simulate(s, n_layers, n_experts, cap, pol, comp, A, layer_marg, pop,
                                   a.prefetch_horizon, a.shadow_slots, a.top_k, rng)
                    for k, v in res.items(): agg[k] += v
                tpot = agg["total_ms"] / max(1, agg["tokens"])
                idle_bytes = agg["idle_ms_total"] * bw_mb_per_ms
                useful_frac = agg["rec_bytes_useful"] / max(1e-9, idle_bytes)
                rows.append({"bw": bw, "residency": r, "policy": pol, "tpot_ms": tpot,
                             "exposed_cpu_ms_tok": agg["exposed_cpu_ms"] / max(1, agg["tokens"]),
                             "rec_useful_mb_tok": agg["rec_bytes_useful"] / max(1, agg["tokens"]),
                             "rec_issued_mb_tok": agg["rec_bytes_issued"] / max(1, agg["tokens"]),
                             "useful_on_time_frac": useful_frac})
                print(f"bw={bw:>4} res={r:>5} {pol:>18} TPOT={tpot:7.3f} exp_cpu={rows[-1]['exposed_cpu_ms_tok']:6.3f} "
                      f"rec_useful={rows[-1]['rec_useful_mb_tok']:6.1f} issued={rows[-1]['rec_issued_mb_tok']:6.1f} "
                      f"uf={useful_frac:.2f}", flush=True)
    # verdict: REC-specific gain = spice_rec vs fetch_idle_prefetch (same predictor, ONLY diff = CPU-serve
    # frees bandwidth) AND vs cpu_fiddler (adds prefetch). random/dummy isolate prediction value.
    print("\n===== SPICE-REC VERDICT =====", flush=True)
    by = defaultdict(dict)
    for x in rows: by[(x["bw"], x["residency"])][x["policy"]] = x
    verdict = {}
    for (bw, r), d in by.items():
        fid = d["cpu_fiddler"]["tpot_ms"]; rec = d["spice_rec"]["tpot_ms"]; orc = d["oracle_shadow"]["tpot_ms"]
        ff = d["fetch_fallback"]["tpot_ms"]; fip = d["fetch_idle_prefetch"]["tpot_ms"]
        rnd = d["random_rec"]["tpot_ms"]; dum = d["dummy_prefetch"]["tpot_ms"]
        g_fid = (fid - rec) / fid if fid > 0 else 0.0          # adding prefetch to cpu-serve
        g_fip = (fip - rec) / fip if fip > 0 else 0.0          # REC-specific: CPU-serve freed bandwidth
        verdict[f"bw{bw}_r{r}"] = {"fetch_fallback": ff, "fetch_idle_prefetch": fip, "cpu_fiddler": fid,
                                   "spice_rec": rec, "oracle_shadow": orc, "random_rec": rnd, "dummy": dum,
                                   "gain_vs_fiddler_pct": 100 * g_fid, "gain_vs_fetch_idle_pct": 100 * g_fip,
                                   "GO": g_fid >= 0.15 and rec < rnd}
        print(f"bw{bw} r{r}: ff={ff:.2f} fetch+pf={fip:.2f} fiddler={fid:.2f} spice_rec={rec:.2f} "
              f"oracle={orc:.2f} rand={rnd:.2f} dummy={dum:.2f} | vs_fiddler={100*g_fid:+.1f}% "
              f"vs_fetch+pf={100*g_fip:+.1f}% rec<rand={rec<rnd}", flush=True)
    Path(a.out).write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
