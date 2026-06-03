"""REAL speculative-microbatch + CPU-grouped-Fiddler runtime (codex decisive positive experiment).
request batch=1, speculative verify length K (target forward processes K positions), EXACT same-precision.

Mechanism (the positive lever): K consecutive positions share experts (measured reuse C/U 1.9-2.5 at K=16).
On a target verify forward, per layer: gather the K positions' routed experts; resident experts -> GPU
batched over their positions; each UNIQUE missed expert -> CPU computed ONCE over the positions routing
to it (grouped), scatter outputs. vs K=1 AR-Fiddler (serve top_k experts per single token).
Real ops, real wall-clock. TPOT = step_time / (K * accept_rate). Sweep K, compare to K=1.

All printed English. Core params: no defaults. Random weights (timing real, outputs garbage).
"""
import argparse, time, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="Real speculative + CPU-grouped-Fiddler runtime")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--n_layers", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--n_experts", type=int, required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--n_heads", type=int, required=True)
    ap.add_argument("--resident_frac", type=float, required=True)
    ap.add_argument("--ctx_len", type=int, required=True)
    ap.add_argument("--ks", type=str, required=True, help="comma list of speculative K (1=AR baseline)")
    ap.add_argument("--steps", type=int, required=True, help="number of target forwards to time per K")
    ap.add_argument("--accept_rate", type=float, required=True, help="avg fraction of K drafted tokens accepted")
    ap.add_argument("--cpu_threads", type=int, required=True)
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev); dt = torch.bfloat16
    dm, di, H = a.d_model, a.d_inter, a.n_heads; hd = dm // H
    L = a.n_layers

    Wq = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wk = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wv = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wo = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wsh = [(torch.randn(di, dm, device=dev, dtype=dt), torch.randn(di, dm, device=dev, dtype=dt),
            torch.randn(dm, di, device=dev, dtype=dt)) for _ in range(L)]
    Wrt = [torch.randn(a.n_experts, dm, device=dev, dtype=dt) for _ in range(L)]
    Kc = [torch.randn(H, a.ctx_len, hd, device=dev, dtype=dt) for _ in range(L)]
    Vc = [torch.randn(H, a.ctx_len, hd, device=dev, dtype=dt) for _ in range(L)]

    n_res = int(round(a.resident_frac * L * a.n_experts))
    gpu_experts = {}; idx = 0
    for l in range(L):
        for e in range(a.n_experts):
            if idx < n_res:
                gpu_experts[(l, e)] = (torch.randn(di, dm, device=dev, dtype=dt),
                                       torch.randn(di, dm, device=dev, dtype=dt),
                                       torch.randn(dm, di, device=dev, dtype=dt))
            idx += 1
    # CPU host weights for missed experts (one shared bank; timing-real)
    cpu_g = torch.randn(di, dm); cpu_u = torch.randn(di, dm); cpu_d = torch.randn(dm, di)

    f = sorted(glob.glob(str(Path(a.trace_dir) / "dec_*.pt")))[0]
    d = torch.load(f, map_location="cpu", weights_only=False)
    trace = [[[int(e) for e in pl[l]] for l in range(L)] for (_t, pl) in d["steps"]]

    def attn_batched(l, hb):  # hb: (P, dm)
        P = hb.shape[0]
        q = (hb @ Wq[l].T).view(P, H, hd).transpose(0, 1)   # (H,P,hd)
        k = (hb @ Wk[l].T).view(P, H, hd).transpose(0, 1); v = (hb @ Wv[l].T).view(P, H, hd).transpose(0, 1)
        K = torch.cat([Kc[l].unsqueeze(0).expand(1, H, a.ctx_len, hd).reshape(H, a.ctx_len, hd), k], dim=1)
        V = torch.cat([Vc[l], v], dim=1)
        o = F.scaled_dot_product_attention(q, K, V)         # (H,P,hd)
        return (o.transpose(0, 1).reshape(P, dm) @ Wo[l].T)

    def shared_b(l, hb):
        g, u, dd = Wsh[l]; return F.linear(F.silu(F.linear(hb, g)) * F.linear(hb, u), dd)

    def gpu_expert_b(w, hb):
        g, u, dd = w; return F.linear(F.silu(F.linear(hb, g)) * F.linear(hb, u), dd)

    def cpu_serve_grouped(hc):  # hc: (m, dm) positions routing to one missed expert -> (m, dm)
        return F.linear(F.silu(F.linear(hc, cpu_g)) * F.linear(hc, cpu_u), cpu_d)

    def run_K(K):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for s in range(a.steps):
            base = (s * K) % (len(trace) - K)
            hb = torch.randn(K, dm, device=dev, dtype=dt)        # K positions
            for l in range(L):
                hb = hb + attn_batched(l, hb)
                hn = hb
                _ = F.linear(hn, Wrt[l])
                out = shared_b(l, hn)
                # gather positions per expert
                pos_by_expert = {}
                for p in range(K):
                    for e in trace[base + p][l]:
                        pos_by_expert.setdefault(e, []).append(p)
                for e, plist in pos_by_expert.items():
                    sub = hn[plist]                              # (m, dm)
                    if (l, e) in gpu_experts:
                        oe = gpu_expert_b(gpu_experts[(l, e)], sub)
                        out.index_add_(0, torch.tensor(plist, device=dev), oe.to(dt))
                    else:
                        oc = cpu_serve_grouped(sub.float().cpu())  # CPU grouped over positions (real)
                        out.index_add_(0, torch.tensor(plist, device=dev), oc.to(dev, dtype=dt))
                hb = hb + out
        torch.cuda.synchronize(dev)
        step_ms = (time.perf_counter() - t0) / a.steps * 1000.0
        accepted = max(1.0, K * a.accept_rate) if K > 1 else 1.0
        return step_ms, step_ms / accepted

    res = {"config": vars(a), "n_resident": n_res, "rows": []}
    run_K(2)  # warmup
    base_tpot = None
    for K in [int(x) for x in a.ks.split(",")]:
        step_ms, tpot = run_K(K)
        if K == 1: base_tpot = tpot
        res["rows"].append({"K": K, "step_ms": step_ms, "tpot_ms": tpot})
        print(f"K={K:>3} step={step_ms:7.3f}ms TPOT={tpot:7.3f}ms/token "
              f"(accept={a.accept_rate if K>1 else 1.0})", flush=True)
    if base_tpot:
        for r in res["rows"]:
            r["speedup_vs_K1"] = base_tpot / r["tpot_ms"]
        best = max(res["rows"], key=lambda r: r["speedup_vs_K1"])
        print(f"\n[verdict] best K={best['K']} TPOT={best['tpot_ms']:.3f} -> {best['speedup_vs_K1']:.2f}x vs AR-Fiddler-K1 "
              f"(accept_rate={a.accept_rate})", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
