"""P2: REAL-window grouped-CPU-serve runtime, head-to-head vs SpecMoEOff-style GPU-fetch.

P2: 基于真实 speculative 窗口的 grouped-CPU-serve runtime, 与 SpecMoEOff 式 GPU-fetch 真机对比.

Consumes the REAL windows captured by spec_decode_capture.py (each window = one verify step's
K candidate positions' per-layer routes + accept_count) and the REAL mean_tokens_per_step.
Replaces v2's oracle consecutive-AR trace and scalar accept_rate.
消费 spec_decode_capture.py 采集的真实窗口 (每窗口 = 一次 verify step 的 K 个候选位置每层路由 + 接受数)
与真实 mean_tokens_per_step. 取代 v2 的 oracle 连续 AR trace 与 scalar accept_rate.

Policies (same resident HBM cache; ONLY the MISS path differs):
  ar_k1            : K=1 AR baseline (Fiddler), serve each missed top-k expert on CPU per single token.
  spice_w          : spec window, each UNIQUE missed expert served ONCE via grouped CPU GEMM (the lever).
  specmoeoff_fetch : spec window, each UNIQUE missed expert FETCHED to GPU (real H2D copy) then GPU GEMM.
  cpu_no_group     : spec window, missed experts served on CPU per-position GEMV (no grouping) -> isolates grouping gain.
策略 (相同 resident HBM 缓存; 仅 MISS 路径不同).

TPOT:
  spec policies : (window_step_ms + K*draft_ms_per_tok) / mean_tokens_per_step   (real amortization)
  ar_k1         : per-token step_ms   (tokens_per_step=1, no draft cost)

All ops REAL (attention, shared, experts), REAL wall-clock, REAL H2D copy for fetch. Same-precision bf16.
All printed strings English. Core params: no defaults (cpu_dtype has one).
"""
import argparse, time, json
from pathlib import Path
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="P2 real-window grouped-CPU vs GPU-fetch runtime")
    ap.add_argument("--windows", required=True, help="windows .pt from spec_decode_capture.py")
    ap.add_argument("--policy", required=True, choices=["ar_k1", "spice_w", "specmoeoff_fetch", "cpu_no_group"])
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--n_heads", type=int, required=True)
    ap.add_argument("--ctx_len", type=int, required=True)
    ap.add_argument("--resident_frac", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True, help="verify windows to time (cycles real windows)")
    ap.add_argument("--draft_ms_per_tok", type=float, required=True)
    ap.add_argument("--cpu_threads", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu_dtype", choices=["bf16", "fp32"], default="bf16")
    return ap.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev); dt = torch.bfloat16
    cpu_dt = torch.bfloat16 if a.cpu_dtype == "bf16" else torch.float32

    blob = torch.load(a.windows, map_location="cpu", weights_only=False)
    wins = blob["windows"]
    K_real = int(blob["K"]); L = int(blob["n_layers"]); n_experts = int(blob["n_experts"])
    top_k = int(blob["top_k"]); tps = float(blob["mean_tokens_per_step"])
    assert len(wins) > 0, "no windows in file"
    dm, di, H = a.d_model, a.d_inter, a.n_heads; hd = dm // H
    # ar_k1 collapses the window to a single token (K=1, no batching, no draft amortization)
    # ar_k1 把窗口塌缩为单 token (K=1, 无批处理, 无 draft 摊销)
    K = 1 if a.policy == "ar_k1" else K_real

    # weights (random; timing real, outputs garbage but shapes/dtype real)
    Wq = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wk = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wv = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wo = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wsh = [(torch.randn(di, dm, device=dev, dtype=dt), torch.randn(di, dm, device=dev, dtype=dt),
            torch.randn(dm, di, device=dev, dtype=dt)) for _ in range(L)]
    Wrt = [torch.randn(n_experts, dm, device=dev, dtype=dt) for _ in range(L)]
    Kc = [torch.randn(H, a.ctx_len, hd, device=dev, dtype=dt) for _ in range(L)]
    Vc = [torch.randn(H, a.ctx_len, hd, device=dev, dtype=dt) for _ in range(L)]

    n_res = int(round(a.resident_frac * L * n_experts))
    gpu_experts = {}; idx = 0
    for l in range(L):
        for e in range(n_experts):
            if idx < n_res:
                gpu_experts[(l, e)] = (torch.randn(di, dm, device=dev, dtype=dt),
                                       torch.randn(di, dm, device=dev, dtype=dt),
                                       torch.randn(dm, di, device=dev, dtype=dt))
            idx += 1
    # DISTINCT CPU host weight bank (NBANK >> LLC) so missed-expert reads hit distinct DRAM (codex fix).
    # DISTINCT CPU 端权重 bank (NBANK >> LLC) 使缺失专家读取命中不同 DRAM 区 (codex 修复).
    NBANK = 256
    cpu_bank = [(torch.randn(di, dm, dtype=cpu_dt), torch.randn(di, dm, dtype=cpu_dt),
                 torch.randn(dm, di, dtype=cpu_dt)) for _ in range(NBANK)]
    # pinned host weights for the FETCH path (real H2D copy cost); distinct bank too.
    # FETCH 路径的 pinned host 权重 (真实 H2D 拷贝成本); 同样 distinct bank.
    fetch_bank = [(torch.randn(di, dm, dtype=dt).pin_memory(), torch.randn(di, dm, dtype=dt).pin_memory(),
                   torch.randn(dm, di, dtype=dt).pin_memory()) for _ in range(NBANK)]
    bytes_per_expert = (di * dm * 2 + dm * di) * torch.empty(0, dtype=dt).element_size()

    def build_mask(P):
        mask = torch.zeros(P, a.ctx_len + P, device=dev, dtype=dt)
        block = torch.triu(torch.full((P, P), float("-inf"), device=dev, dtype=dt), diagonal=1)
        mask[:, a.ctx_len:] = block
        return mask.unsqueeze(0)

    def attn_batched(l, hb, mask):
        P = hb.shape[0]
        q = (hb @ Wq[l].T).view(P, H, hd).transpose(0, 1)
        k = (hb @ Wk[l].T).view(P, H, hd).transpose(0, 1); v = (hb @ Wv[l].T).view(P, H, hd).transpose(0, 1)
        Kf = torch.cat([Kc[l], k], dim=1); Vf = torch.cat([Vc[l], v], dim=1)
        o = F.scaled_dot_product_attention(q, Kf, Vf, attn_mask=mask)
        return (o.transpose(0, 1).reshape(P, dm) @ Wo[l].T)

    def shared_b(l, hb):
        g, u, dd = Wsh[l]; return F.linear(F.silu(F.linear(hb, g)) * F.linear(hb, u), dd)

    def expert_gpu(w, hb):
        g, u, dd = w; return F.linear(F.silu(F.linear(hb, g)) * F.linear(hb, u), dd)

    def expert_cpu(hc, l, e):
        g, u, dd = cpu_bank[(l * n_experts + e) % NBANK]
        return F.linear(F.silu(F.linear(hc, g)) * F.linear(hc, u), dd)

    def expert_fetch_gpu(hb, l, e):
        # real H2D copy of the 3 weight matrices, then GPU compute (SpecMoEOff-style)
        # 真实 H2D 拷贝 3 个权重矩阵, 再 GPU 计算 (SpecMoEOff 式)
        g0, u0, d0 = fetch_bank[(l * n_experts + e) % NBANK]
        g = g0.to(dev, non_blocking=True); u = u0.to(dev, non_blocking=True); dd = d0.to(dev, non_blocking=True)
        return F.linear(F.silu(F.linear(hb, g)) * F.linear(hb, u), dd)

    mask = build_mask(K)

    def routes_for_window(w, l):
        # w["routes"]: [L, K_real, top_k]; for ar_k1 use only position 0
        r = w["routes"][l]
        if a.policy == "ar_k1":
            return r[:1]
        return r

    def run_once(timed):
        diag_C = diag_U = diag_cpu = diag_fetch = diag_bytes = 0
        for s in range(a.steps):
            w = wins[s % len(wins)]
            hb = torch.randn(K, dm, device=dev, dtype=dt)
            sC = sU = scpu = sfetch = sbytes = 0
            for l in range(L):
                hb = hb + attn_batched(l, hb, mask)
                hn = hb
                _ = F.linear(hn, Wrt[l])           # router (real)
                out = shared_b(l, hn)
                routes = routes_for_window(w, l)    # [K, top_k]
                pos_by_expert = {}
                for p in range(routes.shape[0]):
                    for e in routes[p].tolist():
                        pos_by_expert.setdefault(int(e), []).append(p)
                sC += sum(len(pl) for pl in pos_by_expert.values()); sU += len(pos_by_expert)
                for e, plist in pos_by_expert.items():
                    sub = hn[plist]
                    if (l, e) in gpu_experts:
                        oe = expert_gpu(gpu_experts[(l, e)], sub)
                        out.index_add_(0, torch.tensor(plist, device=dev), oe.to(dt))
                    elif a.policy == "specmoeoff_fetch":
                        oe = expert_fetch_gpu(sub, l, e)
                        out.index_add_(0, torch.tensor(plist, device=dev), oe.to(dt))
                        sfetch += 1; sbytes += bytes_per_expert
                    elif a.policy == "cpu_no_group":
                        # per-position GEMV (no grouping): serve each position separately
                        # 逐位置 GEMV (无分组): 每个位置单独服务
                        for p in plist:
                            oc = expert_cpu(hn[p:p+1].to(dtype=cpu_dt).cpu(), l, e)
                            out.index_add_(0, torch.tensor([p], device=dev), oc.to(dev, dtype=dt))
                        scpu += 1; sbytes += bytes_per_expert
                    else:  # spice_w or ar_k1: grouped CPU serve (one GEMM over the expert's positions)
                        oc = expert_cpu(sub.to(dtype=cpu_dt).cpu(), l, e)
                        out.index_add_(0, torch.tensor(plist, device=dev), oc.to(dev, dtype=dt))
                        scpu += 1; sbytes += bytes_per_expert
            diag_C, diag_U, diag_cpu, diag_fetch, diag_bytes = sC, sU, scpu, sfetch, sbytes
        return diag_C, diag_U, diag_cpu, diag_fetch, diag_bytes

    # warmup
    run_once(False)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    C, U, cpu_n, fetch_n, bytes_n = run_once(True)
    torch.cuda.synchronize(dev)
    step_ms = (time.perf_counter() - t0) / a.steps * 1000.0

    if a.policy == "ar_k1":
        tpot = step_ms                                   # 1 token per step, no draft
    else:
        tpot = (step_ms + K * a.draft_ms_per_tok) / tps  # real amortization
    reuse = (C / U) if U > 0 else 0.0
    res = {"policy": a.policy, "K": K, "K_real": K_real, "resident_frac": a.resident_frac,
           "mean_tokens_per_step": tps, "step_ms": step_ms, "tpot_ms": tpot,
           "C": C, "U": U, "reuse_C_over_U": reuse, "cpu_served": cpu_n, "fetched": fetch_n,
           "bytes_moved": int(bytes_n), "cpu_dtype": a.cpu_dtype, "draft_ms_per_tok": a.draft_ms_per_tok}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
