"""SPICE-X decisive pre-check: does target-conditioned cache VALUE beat Least-Stale on hit-rate?

先跑出效果 (user). Pure cache hit-rate (no DMA sim) at tight cache, on real autoregressive decode
traces (dec_*.pt with token ids). The ONE question: a value policy that ranks residents by a
survival next-use probability fusing within-token prior A[j,v_cur,e] (current token future layers)
and cross-token prior A[j,v_next,e] (next token same layer) -- does it evict better than Least-Stale,
and how close to oracle Belady? If value ~= LS -> dead (matches prior token_table result -> paper I).

A[j,v,e] = P(expert e at layer j | token id v), estimated from TRAIN sequences only (leakage guard),
Laplace-smoothed toward layer marginal. Greedy decode => v_cur, v_next are known exactly from trace.

Baselines reuse the existing harness definitions: specmd_ls (cyclic least-stale), oracle_belady
(true future of THIS sequence), lru. All printed content English. Core params: no defaults.
"""
import argparse, json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser(description="SPICE-X eviction value pre-check (hit-rate vs LS vs Belady)")
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True, help="Laplace smoothing for A[j,v,e] toward layer marginal")
    ap.add_argument("--rho", type=float, required=True, help="discount on cross-token (next token) term")
    ap.add_argument("--w_rec", type=float, required=True, help="weight of recency (cyclic-proximity) survival term; recovers LS at larger cache")
    ap.add_argument("--train_frac", type=float, required=True)
    ap.add_argument("--residency", type=str, required=True, help="comma list of cache residency fractions of (layers*experts)")
    ap.add_argument("--max_test_tokens", type=int, required=True)
    return ap.parse_args()


def load_sequences(trace_dir):
    """Return sequences as list of (INPUT_token_id, [layer][top_k]). codex alignment fix:
    gen_decode_traces saves (generated_tid_g, per_layer_g) where per_layer_g is the routing of the
    INPUT token x_g, and x_g = generated_tid_{g-1} (x_0 = last prompt token). So we shift: the token
    that PRODUCED per_layer_g (and is known BEFORE that forward) is generated_tid_{g-1}.
    This makes A/B/simulate use only the realizable, correctly-aligned input token."""
    files = sorted(glob.glob(str(Path(trace_dir) / "dec_*.pt")))
    man = json.loads((Path(trace_dir) / "manifest.json").read_text())
    n_layers = man["num_layers"]; n_experts = man["experts"]
    seqs = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        steps = d["steps"]  # list of (generated_tid, [L][top_k])
        prompt_ids = d["prompt_ids"]
        prev_out = int(prompt_ids[0][-1])  # x_0 = last prompt token
        seq = []
        for (gen_tid, per_layer) in steps:
            if any(sl is None for sl in per_layer):  # guard: skip incomplete capture
                prev_out = int(gen_tid); continue
            plist = [[int(e) for e in topk] for topk in per_layer]
            seq.append((prev_out, plist))   # (input_tid_g = produced per_layer_g, routing_g)
            prev_out = int(gen_tid)          # next step's input = this step's output
        if seq: seqs.append(seq)
    return seqs, n_layers, n_experts


def build_A(train_seqs, n_layers, n_experts, alpha):
    """A[layer][token_id] -> np.array(n_experts) of P(expert | layer, token). Laplace toward layer marginal."""
    # counts
    tok_cnt = defaultdict(lambda: np.zeros(n_experts, dtype=np.float64))  # key (layer, token)
    tok_tot = defaultdict(float)
    layer_cnt = np.zeros((n_layers, n_experts), dtype=np.float64)
    for seq in train_seqs:
        for (tid, per_layer) in seq:
            for l, topk in enumerate(per_layer):
                for e in topk:
                    tok_cnt[(l, tid)][e] += 1.0; layer_cnt[l, e] += 1.0
                tok_tot[(l, tid)] += len(topk)
    layer_marg = layer_cnt / np.clip(layer_cnt.sum(axis=1, keepdims=True), 1, None)
    A = {}
    for (l, tid), c in tok_cnt.items():
        A[(l, tid)] = (c + alpha * layer_marg[l]) / (tok_tot[(l, tid)] + alpha)
    return A, layer_marg


def build_B(train_seqs, n_layers, n_experts, alpha, layer_marg):
    """REALIZABLE cross-token transition table (codex leakage fix): B[layer][token_id] ->
    P(next token uses expert e at layer j | CURRENT token id = v). Built from consecutive token
    pairs in train. Usable during current token's forward (v_cur known; v_next NOT known)."""
    btok_cnt = defaultdict(lambda: np.zeros(n_experts, dtype=np.float64))
    btok_tot = defaultdict(float)
    for seq in train_seqs:
        for ti in range(len(seq) - 1):
            v_cur = seq[ti][0]; nxt_per_layer = seq[ti + 1][1]
            for l, topk in enumerate(nxt_per_layer):
                for e in topk:
                    btok_cnt[(l, v_cur)][e] += 1.0
                btok_tot[(l, v_cur)] += len(topk)
    B = {}
    for (l, v), c in btok_cnt.items():
        B[(l, v)] = (c + alpha * layer_marg[l]) / (btok_tot[(l, v)] + alpha)
    return B


def a_lookup(tbl, layer_marg, l, tid, e):
    v = tbl.get((l, tid))
    return float(v[e]) if v is not None else float(layer_marg[l, e])


def simulate(seq, n_layers, n_experts, capacity, policy, A, B, layer_marg, rho, w_rec, within_mode):
    """Replay one sequence's (layer,expert) stream under a policy. Return (hits, misses).
    Stream order = token by token, layer 0..L-1, experts within a layer. token context available."""
    # flatten with token index + per-position token ids
    flat = []  # (layer, expert, tok_idx)
    tok_ids = [tid for (tid, _) in seq]
    for ti, (_tid, per_layer) in enumerate(seq):
        for l, topk in enumerate(per_layer):
            for e in topk:
                flat.append((l, e, ti))
    # belady precompute
    occ = defaultdict(list)
    if policy == "oracle_belady":
        for i, (l, e, _t) in enumerate(flat):
            occ[(l, e)].append(i)
    occ_ptr = defaultdict(int)
    INF = len(flat) + 10

    cache = set(); last_used = {}; key_layer = {}
    hits = misses = 0

    def next_use(key, pos):
        ps = occ.get(key, ()); p = occ_ptr[key]
        while p < len(ps) and ps[p] <= pos: p += 1
        occ_ptr[key] = p
        return ps[p] if p < len(ps) else INF

    def ls_dist(k, cur_layer):
        d = (key_layer[k] - cur_layer) % n_layers
        return n_layers if d == 0 else d

    true_routing = [set().union(*[set(t) for t in pl]) if pl else set() for (_v, pl) in seq]
    true_by_layer = [[set(t) for t in pl] for (_v, pl) in seq]  # true_by_layer[tok][layer] = set of experts

    def value(k, cur_layer, cur_tok):
        """REALIZABLE survival next-use prob: within-token + cross-token transition B + recency.
        within_mode 'table' -> A[j,v_cur,e] (realizable floor, token-id predictor);
        within_mode 'oracle' -> 1 if e truly used at layer j>cur_layer of current token (CEILING:
        upper bound on what a perfect within-token draft like SPICE could give)."""
        j, e = k
        v_cur = tok_ids[cur_tok]
        if j > cur_layer:
            within = (1.0 if e in true_by_layer[cur_tok][j] else 0.0) if within_mode == "oracle" \
                     else a_lookup(A, layer_marg, j, v_cur, e)
        else:
            within = 0.0
        cross = rho * a_lookup(B, layer_marg, j, v_cur, e)
        recency = w_rec * (1.0 - (ls_dist(k, cur_layer) - 1) / n_layers)
        return 1.0 - (1.0 - within) * (1.0 - min(cross, 1.0)) * (1.0 - min(recency, 1.0))

    def evict(cur_layer, pos, cur_tok):
        if policy == "lru":
            victim = min(cache, key=lambda k: last_used[k])
        elif policy == "specmd_ls":
            victim = max(cache, key=lambda k: (ls_dist(k, cur_layer), -last_used[k]))
        elif policy == "oracle_belady":
            victim = max(cache, key=lambda k: next_use(k, pos))
        elif policy == "value":
            victim = min(cache, key=lambda k: (value(k, cur_layer, cur_tok), last_used[k]))
        else:
            raise ValueError(policy)
        cache.discard(victim); last_used.pop(victim, None); key_layer.pop(victim, None)

    for pos, (l, e, ti) in enumerate(flat):
        key = (l, e)
        if key in cache:
            hits += 1; last_used[key] = pos
        else:
            misses += 1
            while len(cache) >= capacity and cache:
                evict(l, pos, ti)
            if capacity >= 1:
                cache.add(key); last_used[key] = pos; key_layer[key] = l
    return hits, misses


def main():
    a = parse_args()
    seqs, n_layers, n_experts = load_sequences(a.trace_dir)
    n_train = max(1, int(round(a.train_frac * len(seqs))))
    train, test = seqs[:n_train], seqs[n_train:]
    if not test: test = train[-1:]
    A, layer_marg = build_A(train, n_layers, n_experts, a.alpha)
    B = build_B(train, n_layers, n_experts, a.alpha, layer_marg)
    # cap test tokens
    tot = 0; test2 = []
    for s in test:
        test2.append(s); tot += len(s)
        if tot >= a.max_test_tokens: break
    test = test2
    print(f"[data] seqs train={len(train)} test={len(test)} layers={n_layers} experts={n_experts} "
          f"A_entries={len(A)} test_tokens={sum(len(s) for s in test)}", flush=True)

    total = n_layers * n_experts
    residencies = [float(x) for x in a.residency.split(",")]
    policies = ["lru", "specmd_ls", "value_within", "value", "value_oracle_within", "oracle_belady"]
    rows = []
    for r in residencies:
        cap = max(1, int(round(r * total)))
        for pol in policies:
            h = m = 0
            # value_within = within-token only (rho=0): isolates within-token A contribution
            rho_use = 0.0 if pol in ("value_within", "value_oracle_within") else a.rho
            wrec_use = 0.0 if pol in ("value_within", "value_oracle_within") else a.w_rec
            wmode = "oracle" if pol == "value_oracle_within" else "table"
            pol_run = "value" if pol in ("value_within", "value_oracle_within") else pol
            for s in test:
                hi, mi = simulate(s, n_layers, n_experts, cap, pol_run, A, B, layer_marg, rho_use, wrec_use, wmode)
                h += hi; m += mi
            hr = h / max(1, h + m)
            rows.append({"residency": r, "capacity": cap, "policy": pol, "hit_rate": hr,
                         "slots": h + m})
            print(f"res={r:>5} cap={cap:>4} {pol:>14} hit={hr:.4f}", flush=True)

    # per-sequence paired bootstrap CI on (value - LS) hit-rate gap (codex: slots highly correlated)
    print("\n===== paired bootstrap CI (value - LS), per-sequence =====", flush=True)
    boot = {}
    rng = np.random.default_rng(0)
    for r in residencies:
        cap = max(1, int(round(r * total)))
        per_seq = []  # (value_hits, value_slots, ls_hits, ls_slots) per sequence
        for s in test:
            vh, vm = simulate(s, n_layers, n_experts, cap, "value", A, B, layer_marg, a.rho, a.w_rec, "table")
            lh, lm = simulate(s, n_layers, n_experts, cap, "specmd_ls", A, B, layer_marg, a.rho, a.w_rec, "table")
            per_seq.append((vh, vh + vm, lh, lh + lm))
        arr = np.array(per_seq, dtype=float)
        n = len(arr)
        diffs = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            b = arr[idx]
            vhr = b[:, 0].sum() / max(1, b[:, 1].sum()); lhr = b[:, 2].sum() / max(1, b[:, 3].sum())
            diffs.append(vhr - lhr)
        lo, hi = np.percentile(diffs, [2.5, 97.5]); mean = float(np.mean(diffs))
        boot[str(r)] = {"mean_gap": mean, "ci95_lo": float(lo), "ci95_hi": float(hi), "n_seq": n,
                        "significant": bool(lo > 0)}
        print(f"res={r}: value-LS mean={mean:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] "
              f"{'SIGNIFICANT(>0)' if lo>0 else 'not sig' if hi>0 else 'SIG NEGATIVE'}", flush=True)

    # verdict: value vs LS gap, and value's share of (Belady - LS) headroom
    print("\n===== VALUE vs LS verdict =====", flush=True)
    by = defaultdict(dict)
    for x in rows: by[x["residency"]][x["policy"]] = x["hit_rate"]
    verdict = {}
    for r, d in by.items():
        ls, val, bel = d["specmd_ls"], d["value"], d["oracle_belady"]
        head = bel - ls
        share = (val - ls) / head if head > 1e-9 else 0.0
        verdict[str(r)] = {"ls": ls, "value": val, "belady": bel, "value_minus_ls": val - ls,
                           "belady_minus_ls": head, "value_share_of_headroom": share,
                           "GO_value_beats_ls": (val - ls) > 0.005}
        print(f"res={r}: LS={ls:.4f} VALUE={val:.4f} Belady={bel:.4f} | "
              f"value-LS={val-ls:+.4f} share_of_headroom={share:+.2%} "
              f"{'GO' if (val-ls)>0.005 else 'NO-GO(~=LS)'}", flush=True)
    Path(a.out).write_text(json.dumps({"rows": rows, "verdict": verdict, "bootstrap": boot}, indent=2))


if __name__ == "__main__":
    main()
