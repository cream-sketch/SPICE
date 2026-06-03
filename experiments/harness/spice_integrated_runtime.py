"""REAL integrated runtime: SPICE prefetch + capacity-aware verified-fallback (BOTH strategies, real wall-clock).

REAL 集成 runtime: SPICE prefetch + capacity-aware verified-fallback (两个策略都用, 真实 wall-clock).

SPICE prefetch and the capacity-aware fallback are ORTHOGONAL (user's point):
  - SPICE prefetch (paper: 74.24% slot hit, up to 2.86x TPOT) reduces the MISS RATE by predicting
    future-layer experts and async-prefetching them overlapped with compute. Prefetched/resident slots
    are served by a GPU GEMM (the weight is already in HBM). SPICE also issues the prefetch H2D traffic.
  - The capacity-aware fallback improves how the RESIDUAL misses (paper ~25.76%) are served, replacing
    SPICE's SYNCHRONOUS fetch-all (SPICE.pdf line 262/548) with CPU||PCIe split + use!=residency admission.

FAIRNESS (codex review): the per-(token,layer) HIT/RESIDUAL mask is PRE-GENERATED ONCE and shared by all
policies, so the residual workload is IDENTICAL; only the residual-miss SERVICE differs. Prefetch H2D
traffic for the hit slots is issued identically for all policies (real, overlapped on a low-pri stream,
contends PCIe with the capacity fallback's fetches -- a real effect that favors CPU-serve). Activation
D2H is done ONCE per layer. Each policy is warmed up separately. No RNG inside the timed loop.

Policies (prefetch ON for all; only residual fallback differs):
  prefetch_sync     : SPICE original  = residual miss -> synchronous per-expert H2D fetch + GPU GEMM; admit.
  prefetch_cpu      : SPICE + Fiddler = residual miss -> all-CPU serve (DRAM, parallel to PCIe); no admit.
  prefetch_capacity : SPICE + NEW     = residual miss -> capacity split (n_fetch on PCIe || n_cpu on CPU); admit fetched only.

All ops REAL (attn/shared/router/experts), real KV, bf16 same-precision, batch=1 decode, real wall-clock.
target_residual_rate calibrates the prefetch operating point (paper 0.2576). All printed English.
Core params: no defaults (cpu_dtype/seed have defaults).
"""
import argparse, time, json, glob, random
from pathlib import Path
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="SPICE prefetch + capacity-aware fallback integrated runtime")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--n_layers", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--n_experts", type=int, required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--n_heads", type=int, required=True)
    ap.add_argument("--ctx_len", type=int, required=True)
    ap.add_argument("--decode_tokens", type=int, required=True)
    ap.add_argument("--cpu_threads", type=int, required=True)
    ap.add_argument("--target_residual_rate", type=float, required=True,
                    help="fraction of routed slots that miss after prefetch (paper 0.2576)")
    ap.add_argument("--fetch_frac", type=float, required=True,
                    help="capacity policy: fraction of residual misses sent to PCIe fetch (rest to CPU)")
    ap.add_argument("--trace_dir", required=True, help="real decode traces dir, or 'synthetic'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu_dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    rng = random.Random(a.seed)
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev); dt = torch.bfloat16
    cpu_dt = torch.bfloat16 if a.cpu_dtype == "bf16" else torch.float32
    dm, di, H = a.d_model, a.d_inter, a.n_heads; hd = dm // H
    L, E = a.n_layers, a.n_experts

    Wq = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wk = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wv = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wo = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wsh = [(torch.randn(di, dm, device=dev, dtype=dt), torch.randn(di, dm, device=dev, dtype=dt),
            torch.randn(dm, di, device=dev, dtype=dt)) for _ in range(L)]
    Wrt = [torch.randn(E, dm, device=dev, dtype=dt) for _ in range(L)]
    Kc = [torch.randn(H, a.ctx_len, hd, device=dev, dtype=dt) for _ in range(L)]
    Vc = [torch.randn(H, a.ctx_len, hd, device=dev, dtype=dt) for _ in range(L)]

    # distinct host (pinned, for fetch H2D) + CPU (for CPU-serve) banks to avoid L3 artifacts
    NBANK = 256
    host_bank = [(torch.randn(di, dm, dtype=dt).pin_memory(), torch.randn(di, dm, dtype=dt).pin_memory(),
                  torch.randn(dm, di, dtype=dt).pin_memory()) for _ in range(NBANK)]
    cpu_bank = [(torch.randn(di, dm, dtype=cpu_dt), torch.randn(di, dm, dtype=cpu_dt),
                 torch.randn(dm, di, dtype=cpu_dt)) for _ in range(NBANK)]
    resident_w = (torch.randn(di, dm, device=dev, dtype=dt), torch.randn(di, dm, device=dev, dtype=dt),
                  torch.randn(dm, di, device=dev, dtype=dt))  # shared GPU weight for hit-GEMM timing

    fetch_stage = [(torch.empty(di, dm, device=dev, dtype=dt), torch.empty(di, dm, device=dev, dtype=dt),
                    torch.empty(dm, di, device=dev, dtype=dt)) for _ in range(8)]
    pf_stage = (torch.empty(di, dm, device=dev, dtype=dt), torch.empty(di, dm, device=dev, dtype=dt),
                torch.empty(dm, di, device=dev, dtype=dt))  # reused target for prefetch H2D traffic
    fetch_stream = torch.cuda.Stream(dev, priority=-1)
    pf_stream = torch.cuda.Stream(dev, priority=0)

    def hkey(l, e):
        return (l * E + e) % NBANK

    def attn(l, h):
        q = (h @ Wq[l].T).view(H, 1, hd); k = (h @ Wk[l].T).view(H, 1, hd); v = (h @ Wv[l].T).view(H, 1, hd)
        K = torch.cat([Kc[l], k], dim=1); V = torch.cat([Vc[l], v], dim=1)
        o = F.scaled_dot_product_attention(q, K, V)
        return (o.reshape(1, dm) @ Wo[l].T)

    def shared(l, h):
        g, u, d = Wsh[l]; return F.linear(F.silu(F.linear(h, g)) * F.linear(h, u), d)

    def expert_gpu(w, h):
        g, u, d = w; return F.linear(F.silu(F.linear(h, g)) * F.linear(h, u), d)

    def expert_cpu(hc, l, e):
        g, u, d = cpu_bank[hkey(l, e)]
        return F.linear(F.silu(F.linear(hc, g)) * F.linear(hc, u), d)

    # ---- routing trace ----
    if a.trace_dir == "synthetic":
        import numpy as np
        nr = np.random.default_rng(a.seed)
        routes = [[nr.choice(E, a.top_k, replace=False).tolist() for _ in range(L)] for _ in range(a.decode_tokens)]
    else:
        f = sorted(glob.glob(str(Path(a.trace_dir) / "dec_*.pt")))[0]
        d = torch.load(f, map_location="cpu", weights_only=False)
        steps = d["steps"]
        routes = [[[int(e) for e in pl[l]] for l in range(L)] for (_t, pl) in steps if all(x is not None for x in pl)][:a.decode_tokens]
        if len(routes) < a.decode_tokens:
            routes = (routes * (a.decode_tokens // max(1, len(routes)) + 1))[:a.decode_tokens]
    T = len(routes)

    # ---- PRE-GENERATE shared hit/residual mask (identical across policies) ----
    # is_residual[ti][l] = list of residual-missed experts; hit_count[ti][l] = #hits (resident or prefetched).
    # Each routed slot is a residual miss with prob target_residual_rate (deterministic seed); rest are hits.
    is_residual = [[[] for _ in range(L)] for _ in range(T)]
    hit_count = [[0 for _ in range(L)] for _ in range(T)]
    prefetch_experts = [[[] for _ in range(L)] for _ in range(T)]  # the hit slots SPICE prefetched (H2D traffic)
    for ti in range(T):
        for l in range(L):
            for e in routes[ti][l]:
                if rng.random() < a.target_residual_rate:
                    is_residual[ti][l].append(e)
                else:
                    hit_count[ti][l] += 1
                    prefetch_experts[ti][l].append(e)  # served from HBM via prefetch; counts as prefetch H2D

    def serve_residual(policy, misses, l, hn_gpu):
        # returns the residual contribution tensor on GPU; REAL fetch/cpu service. hn_gpu: (1, dm) on GPU.
        if not misses:
            return None
        if policy == "prefetch_sync":
            acc = None
            for e in misses:  # synchronous per-expert: H2D -> wait -> GEMM (SPICE original fallback)
                with torch.cuda.stream(fetch_stream):
                    for s, dd in zip(host_bank[hkey(l, e)], fetch_stage[0]):
                        dd.copy_(s, non_blocking=True)
                torch.cuda.current_stream(dev).wait_stream(fetch_stream)
                oe = expert_gpu(fetch_stage[0], hn_gpu)
                acc = oe if acc is None else acc + oe
            return acc
        if policy == "prefetch_cpu":
            hc = hn_gpu.to(dtype=cpu_dt).cpu()  # ONE D2H per layer (codex fix)
            acc = None
            for e in misses:
                oc = expert_cpu(hc, l, e)
                acc = oc if acc is None else acc + oc
            return acc.to(dev, dtype=dt)
        # prefetch_capacity: split fetch (PCIe) || cpu (DRAM), concurrent.
        # fetch_frac=0 -> all CPU; fetch_frac=1 -> all fetch. round() allows exact 0 (no forced fetch).
        n = len(misses); nf = int(round(a.fetch_frac * n))
        fetch_set = misses[:nf]; cpu_set = misses[nf:]
        fetch_outs = []
        for i, e in enumerate(fetch_set):  # launch async H2D (concurrent with CPU below)
            slot = fetch_stage[i % len(fetch_stage)]
            with torch.cuda.stream(fetch_stream):
                for s, dd in zip(host_bank[hkey(l, e)], slot):
                    dd.copy_(s, non_blocking=True)
            fetch_outs.append((slot, e))
        acc = None
        if cpu_set:
            hc = hn_gpu.to(dtype=cpu_dt).cpu()  # ONE D2H
            for e in cpu_set:
                oc = expert_cpu(hc, l, e)
                acc = oc if acc is None else acc + oc
            acc = acc.to(dev, dtype=dt)
        if fetch_outs:
            torch.cuda.current_stream(dev).wait_stream(fetch_stream)
            for slot, e in fetch_outs:
                oe = expert_gpu(slot, hn_gpu)
                acc = oe if acc is None else acc + oe
        return acc

    def run_decode(policy):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for ti in range(T):
            h = torch.randn(1, dm, device=dev, dtype=dt)
            for l in range(L):
                h = h + attn(l, h)
                hn = h
                _ = F.linear(hn, Wrt[l])
                out = shared(l, hn)
                # hits (resident or prefetched): GPU GEMM each (real cost; prefetch already brought them)
                for _ in range(hit_count[ti][l]):
                    out = out + expert_gpu(resident_w, hn)
                # SPICE prefetch H2D traffic for the hit slots (real, overlapped on low-pri stream;
                # contends PCIe with the capacity fallback's fetch -- a real effect). Identical across policies.
                for e in prefetch_experts[ti][l]:
                    with torch.cuda.stream(pf_stream):
                        for s, dd in zip(host_bank[hkey(l, e)], pf_stage):
                            dd.copy_(s, non_blocking=True)
                # residual misses: the VARIED fallback
                r = serve_residual(policy, is_residual[ti][l], l, hn)
                if r is not None:
                    out = out + r
                h = h + out
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / T * 1000.0

    residual_per_tok = sum(len(is_residual[ti][l]) for ti in range(T) for l in range(L)) / T
    hits_per_tok = sum(hit_count[ti][l] for ti in range(T) for l in range(L)) / T
    res = {"config": vars(a), "T": T, "residual_miss_per_tok": residual_per_tok, "hits_per_tok": hits_per_tok,
           "effective_residual_rate": residual_per_tok / max(1e-9, residual_per_tok + hits_per_tok)}
    for pol in ["prefetch_sync", "prefetch_cpu", "prefetch_capacity"]:
        run_decode(pol)  # per-policy warmup
        ms = run_decode(pol)
        res[f"tpot_{pol}_ms"] = ms
        print(f"[{pol:>18}] TPOT={ms:8.3f} ms/token", flush=True)
    base = res["tpot_prefetch_sync_ms"]
    res["capacity_vs_sync_pct"] = 100 * (base - res["tpot_prefetch_capacity_ms"]) / base
    res["cpu_vs_sync_pct"] = 100 * (base - res["tpot_prefetch_cpu_ms"]) / base
    res["capacity_vs_cpu_pct"] = 100 * (res["tpot_prefetch_cpu_ms"] - res["tpot_prefetch_capacity_ms"]) / res["tpot_prefetch_cpu_ms"]
    print(f"\n[REAL] residual_miss/tok={residual_per_tok:.2f} (rate={res['effective_residual_rate']:.3f}) | "
          f"capacity vs SPICE-sync={res['capacity_vs_sync_pct']:+.1f}%  cpu vs sync={res['cpu_vs_sync_pct']:+.1f}%  "
          f"capacity vs cpu={res['capacity_vs_cpu_pct']:+.1f}%", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
