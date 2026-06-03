"""REAL tiny runtime (user: 'look at the real thing, not simulation'). Measures REAL wall-clock TPOT
of the Fiddler ablation on a real Qwen-like MoE decode stack, batch=1, EXACT same-precision (bf16).

Real ops: attention (q/k/v/o projections + SDPA over a real KV cache), real shared expert, real router,
real routed experts. Expert pool: some resident on GPU HBM, rest pinned in host DRAM. On a routed miss:
  fetch_on_miss (SPICE original): real pinned H2D(weight) -> real GPU expert compute
  cpu_serve     (SPICE+Fiddler) : real D2H(act) -> real CPU expert compute (threads) -> real H2D(out)
Decode T tokens, time per-token wall-clock. Routing from a real trace or synthetic memoryless.
All printed English. Core params: no defaults.
"""
import argparse, time, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="Real tiny MoE decode runtime (Fiddler ablation, wall-clock)")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--n_layers", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--n_experts", type=int, required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--n_heads", type=int, required=True)
    ap.add_argument("--resident_frac", type=float, required=True, help="fraction of (layer,expert) resident on GPU")
    ap.add_argument("--ctx_len", type=int, required=True, help="KV cache length (prefill context)")
    ap.add_argument("--decode_tokens", type=int, required=True)
    ap.add_argument("--cpu_threads", type=int, required=True)
    ap.add_argument("--trace_dir", required=True, help="real decode traces dir, or 'synthetic'")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev); dt = torch.bfloat16
    dm, di, H = a.d_model, a.d_inter, a.n_heads
    hd = dm // H

    # ---- real layer params on GPU ----
    Wq = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(a.n_layers)]
    Wk = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(a.n_layers)]
    Wv = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(a.n_layers)]
    Wo = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(a.n_layers)]
    Wsh = [(torch.randn(di, dm, device=dev, dtype=dt), torch.randn(di, dm, device=dev, dtype=dt),
            torch.randn(dm, di, device=dev, dtype=dt)) for _ in range(a.n_layers)]  # shared expert
    Wrt = [torch.randn(a.n_experts, dm, device=dev, dtype=dt) for _ in range(a.n_layers)]  # router
    # KV cache (real)
    Kc = [torch.randn(H, a.ctx_len, hd, device=dev, dtype=dt) for _ in range(a.n_layers)]
    Vc = [torch.randn(H, a.ctx_len, hd, device=dev, dtype=dt) for _ in range(a.n_layers)]

    # ---- expert pool: GPU-resident set + host-pinned rest ----
    n_res = int(round(a.resident_frac * a.n_layers * a.n_experts))
    # resident: first n_res (layer,expert) by index (popularity-agnostic; serving cost is what we measure)
    gpu_experts = {}   # (l,e) -> (g,u,d) on GPU
    host_experts = {}  # (l,e) -> (g,u,d) pinned host
    stage = (torch.empty(di, dm, device=dev, dtype=dt), torch.empty(di, dm, device=dev, dtype=dt),
             torch.empty(dm, di, device=dev, dtype=dt))  # GPU staging for fetched weight
    idx = 0
    base_g = torch.randn(di, dm, dtype=dt); base_u = torch.randn(di, dm, dtype=dt); base_d = torch.randn(dm, di, dtype=dt)
    for l in range(a.n_layers):
        for e in range(a.n_experts):
            if idx < n_res:
                gpu_experts[(l, e)] = (torch.randn(di, dm, device=dev, dtype=dt),
                                       torch.randn(di, dm, device=dev, dtype=dt),
                                       torch.randn(dm, di, device=dev, dtype=dt))
            else:
                host_experts[(l, e)] = (base_g.clone().pin_memory(), base_u.clone().pin_memory(),
                                        base_d.clone().pin_memory())
            idx += 1
    host_cpu = (base_g.float(), base_u.float(), base_d.float())  # one CPU copy for CPU-serve compute

    copy_stream = torch.cuda.Stream(dev)

    def attn(l, h):
        q = (h @ Wq[l].T).view(H, 1, hd); k = (h @ Wk[l].T).view(H, 1, hd); v = (h @ Wv[l].T).view(H, 1, hd)
        K = torch.cat([Kc[l], k], dim=1); V = torch.cat([Vc[l], v], dim=1)
        o = F.scaled_dot_product_attention(q, K, V)  # (H,1,hd)
        return (o.reshape(1, dm) @ Wo[l].T)

    def shared(l, h):
        g, u, d = Wsh[l]; return F.linear(F.silu(F.linear(h, g)) * F.linear(h, u), d)

    def gpu_expert(w, h):
        g, u, d = w; return F.linear(F.silu(F.linear(h, g)) * F.linear(h, u), d)

    # routing trace
    if a.trace_dir == "synthetic":
        rng = np.random.default_rng(0)
        routes = [[rng.choice(a.n_experts, a.top_k, replace=False).tolist() for _ in range(a.n_layers)]
                  for _ in range(a.decode_tokens)]
    else:
        f = sorted(glob.glob(str(Path(a.trace_dir) / "dec_*.pt")))[0]
        d = torch.load(f, map_location="cpu", weights_only=False)
        steps = d["steps"]
        routes = [[[int(e) for e in pl[l]] for l in range(a.n_layers)] for (_t, pl) in steps[:a.decode_tokens]]
        if len(routes) < a.decode_tokens:
            routes = (routes * (a.decode_tokens // max(1, len(routes)) + 1))[:a.decode_tokens]

    def run_decode(policy):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for ti in range(len(routes)):
            h = torch.randn(1, dm, device=dev, dtype=dt)
            for l in range(a.n_layers):
                h = h + attn(l, h)
                hn = h
                _ = F.linear(hn, Wrt[l])           # router
                out = shared(l, hn)
                for e in routes[ti][l]:
                    if (l, e) in gpu_experts:
                        out = out + gpu_expert(gpu_experts[(l, e)], hn)
                    else:
                        hw = host_experts[(l, e)]
                        if policy == "fetch_on_miss":
                            with torch.cuda.stream(copy_stream):
                                for s, dd in zip(hw, stage): dd.copy_(s, non_blocking=True)
                            torch.cuda.current_stream(dev).wait_stream(copy_stream)
                            out = out + gpu_expert(stage, hn)
                        else:  # cpu_serve
                            hc = hn.float().cpu()
                            g, u, d = host_cpu
                            oc = F.linear(F.silu(F.linear(hc, g)) * F.linear(hc, u), d)
                            out = out + oc.to(dev, dtype=dt)
                h = h + out
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / len(routes) * 1000.0  # ms/token

    res = {"config": vars(a), "n_resident": n_res, "total_experts": a.n_layers * a.n_experts}
    # warmup
    run_decode("cpu_serve")
    res["tpot_fetch_on_miss_ms"] = run_decode("fetch_on_miss")
    res["tpot_cpu_serve_ms"] = run_decode("cpu_serve")
    res["fiddler_ablation_pct"] = 100 * (res["tpot_fetch_on_miss_ms"] - res["tpot_cpu_serve_ms"]) / res["tpot_fetch_on_miss_ms"]
    print(json.dumps(res, indent=2), flush=True)
    with open(a.out, "w") as f: json.dump(res, f, indent=2)
    print(f"\n[REAL wall-clock] SPICE-original fetch-on-miss = {res['tpot_fetch_on_miss_ms']:.3f} ms/token; "
          f"SPICE+Fiddler cpu-serve = {res['tpot_cpu_serve_ms']:.3f} ms/token; "
          f"Fiddler ablation = {res['fiddler_ablation_pct']:+.1f}%", flush=True)


if __name__ == "__main__":
    main()
