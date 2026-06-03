AUDIT this corrected experiment for CORRECTNESS and LEAKAGE before I trust the result. You previously caught a leakage (cross-token used actual v_{t+1}). I replaced it with a realizable transition table B[j,v_cur,e]=P(next token uses expert e at layer j | current token id). 

SURPRISING RESULT I DO NOT TRUST: the corrected (no-leak) version is BETTER than the leaked version at 5% residency (0.1193 vs 0.1055). Removing oracle next-token info IMPROVED hit-rate. Explain whether this signals residual leakage in B, or is legitimately because B (aggregated transition) is a denser/better eviction signal than a single realized next token. 

Corrected results (Qwen, 16 test seqs, 2048 tokens, train_frac 0.6, alpha 5, rho 0.5):
  res=0.05: LS=0.0666 value_within=0.0801 value=0.1193 Belady=0.3148
  res=0.10: LS=0.1573 value_within=0.1638 value=0.1896 Belady=0.4537
  res=0.20: LS=0.3348 value_within=0.3232 value=0.3138 Belady=0.6204

Audit these specifically:
1. LEAKAGE: Is build_B (transitions from train only) + a_lookup during simulate truly realizable at decode time? Does using tok_ids[cur_tok] (current token) for both within (A) and cross (B) leak anything? Is there any way test info enters A/B?
2. Why corrected > leaked at 5%? Residual leak in B, or legit? Be concrete.
3. value_within (rho=0) beats LS at 5%/10% but loses at 20%. value(+transition) bigger win at 5%/10%, still loses at 20%. Is the 20% regression a real property or a bug (e.g. staleness tiebreak, or A overvaluing j>cur_layer experts that are imminent anyway and that LS already keeps)?
4. Is the train/test split by SEQUENCE (not random events)? Confirm from code. Is 16 test seqs / 2048 tokens enough to trust +0.05?
5. Belady here is per-(layer,expert) hit-rate oracle on the same global capacity as LS/value. Is the comparison fair?
6. VERDICT: is this corrected result trustworthy enough to (a) proceed to deadline-replay TPOT + DeepSeek replication, or (b) is there a flaw making it invalid? And does it change your earlier "appendix" verdict?

Full corrected script below.
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
    ap.add_argument("--train_frac", type=float, required=True)
    ap.add_argument("--residency", type=str, required=True, help="comma list of cache residency fractions of (layers*experts)")
    ap.add_argument("--max_test_tokens", type=int, required=True)
    return ap.parse_args()


def load_sequences(trace_dir):
    """Return list of sequences; each = list of (token_id, [layer][top_k experts]). num_layers, num_experts."""
    files = sorted(glob.glob(str(Path(trace_dir) / "dec_*.pt")))
    man = json.loads((Path(trace_dir) / "manifest.json").read_text())
    n_layers = man["num_layers"]; n_experts = man["experts"]
    seqs = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        steps = d["steps"]  # list of (token_id, [L][top_k])
        seq = [(int(tid), [[int(e) for e in topk] for topk in per_layer]) for (tid, per_layer) in steps]
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


def simulate(seq, n_layers, n_experts, capacity, policy, A, B, layer_marg, rho):
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

    def value(k, cur_layer, cur_tok):
        """REALIZABLE survival next-use prob (codex leakage fix): within-token uses current token id
        (known) for its future layers j>cur_layer; cross-token uses transition table B[j,v_cur,e]
        (P next token uses (j,e) | current token) -- does NOT peek at v_{t+1}. Higher = keep."""
        j, e = k
        v_cur = tok_ids[cur_tok]
        within = a_lookup(A, layer_marg, j, v_cur, e) if j > cur_layer else 0.0
        cross = rho * a_lookup(B, layer_marg, j, v_cur, e)  # transition, realizable for all j
        return 1.0 - (1.0 - within) * (1.0 - min(cross, 1.0))

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
    policies = ["lru", "specmd_ls", "value_within", "value", "oracle_belady"]
    rows = []
    for r in residencies:
        cap = max(1, int(round(r * total)))
        for pol in policies:
            h = m = 0
            # value_within = within-token only (rho=0): isolates within-token A contribution
            rho_use = 0.0 if pol == "value_within" else a.rho
            pol_run = "value" if pol == "value_within" else pol
            for s in test:
                hi, mi = simulate(s, n_layers, n_experts, cap, pol_run, A, B, layer_marg, rho_use)
                h += hi; m += mi
            hr = h / max(1, h + m)
            rows.append({"residency": r, "capacity": cap, "policy": pol, "hit_rate": hr,
                         "slots": h + m})
            print(f"res={r:>5} cap={cap:>4} {pol:>14} hit={hr:.4f}", flush=True)

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
    Path(a.out).write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
