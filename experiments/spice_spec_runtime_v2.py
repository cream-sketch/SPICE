"""REAL speculative-microbatch + CPU-grouped-Fiddler runtime v2 (codex 4-fix revision).
request batch=1, speculative verify length K (target forward processes K positions), EXACT same-precision.

Mechanism (the positive lever): K consecutive positions share experts (measured reuse C/U 1.9-2.5 at K=16).
On a target verify forward, per layer: gather the K positions' routed experts; resident experts -> GPU
batched over their positions; each UNIQUE missed expert -> CPU computed ONCE over the positions routing
to it (grouped), scatter outputs. vs K=1 AR-Fiddler (serve top_k experts per single token).
Real ops, real wall-clock. TPOT = (step_ms + K*draft_ms_per_tok) / (K * accept_rate). Sweep K, compare to K=1.

v2 changes vs spice_spec_runtime.py (codex flagged):
  1. CAUSAL MASK inside the K new-token block (prefix KV fully visible, K-block lower-triangular).
  2. SAME-PRECISION CPU path: CPU bank + activations in bf16 by default (--cpu_dtype {bf16,fp32}).
  3. DISTINCT CPU weight bank kept (NBANK=256) -- no regression to a shared bank.
  4. Per-K diagnostics: total expert calls C, unique experts U, reuse C/U, CPU-served count, bytes moved.

All printed English. Core params: no defaults (--cpu_dtype has a default). Random weights (timing real, outputs garbage).
"""
import argparse, time, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="Real speculative + CPU-grouped-Fiddler runtime v2")
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
    ap.add_argument("--draft_ms_per_tok", type=float, required=True, help="SPICE draft cost per drafted token (ms); added to TPOT for K>1")
    ap.add_argument("--cpu_threads", type=int, required=True)
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--out", required=True)
    # v2: CPU compute precision; default bf16 to match GPU same-precision intent.
    # v2: CPU 计算精度；默认 bf16 以匹配 GPU 的同精度意图。
    ap.add_argument("--cpu_dtype", choices=["bf16", "fp32"], default="bf16",
                    help="CPU expert bank/compute dtype; bf16 matches GPU same-precision intent (default)")
    return ap.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev); dt = torch.bfloat16
    # v2: resolve CPU dtype from flag; bf16 keeps same-precision with GPU.
    # v2: 由参数解析 CPU dtype；bf16 保持与 GPU 同精度。
    cpu_dt = torch.bfloat16 if a.cpu_dtype == "bf16" else torch.float32
    print(f"[cpu-precision] cpu_dtype={a.cpu_dtype} (bf16 matches GPU same-precision intent; "
          f"fp32 is fallback if bf16 CPU matmul is slow/unsupported in this torch)", flush=True)
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
    # CPU host weights for missed experts: DISTINCT bank (codex fix: shared bank -> L3 cache artifact).
    # CPU 端缺失专家权重：DISTINCT bank（codex 修复：共享 bank 会变成 L3 缓存伪命中）。
    # NBANK distinct expert weight sets (>> LLC) so missed-expert reads hit distinct DRAM regions.
    # v2: bank stored in cpu_dt (bf16 by default) for same-precision with GPU.
    # v2: bank 以 cpu_dt 存储（默认 bf16），与 GPU 同精度。
    NBANK = 256
    cpu_bank = [(torch.randn(di, dm, dtype=cpu_dt), torch.randn(di, dm, dtype=cpu_dt),
                 torch.randn(dm, di, dtype=cpu_dt)) for _ in range(NBANK)]
    # v2: per-expert bytes moved when CPU-served (3 weight matrices), in cpu_dt element size.
    # v2: 单专家 CPU 服务时搬运的字节数（3 个权重矩阵），按 cpu_dt 元素大小计。
    bytes_per_expert = (di * dm + di * dm + dm * di) * torch.empty(0, dtype=cpu_dt).element_size()

    f = sorted(glob.glob(str(Path(a.trace_dir) / "dec_*.pt")))[0]
    d = torch.load(f, map_location="cpu", weights_only=False)
    trace = [[[int(e) for e in pl[l]] for l in range(L)] for (_t, pl) in d["steps"]]

    def build_kblock_mask(P):
        # v2 (FIX 1): explicit additive attn mask of shape (P, ctx_len+P).
        #   columns [0:ctx_len]      = prefix KV, fully visible to all K queries (0.0).
        #   columns [ctx_len:ctx_len+P] = K new-token block, lower-triangular causal:
        #       query p may attend to new key j only if j <= p; future candidates masked (-inf).
        # v2 (FIX 1): 形状 (P, ctx_len+P) 的显式加性注意力掩码。
        #   列 [0:ctx_len]            = 前缀 KV，对所有 K 个 query 全可见 (0.0)。
        #   列 [ctx_len:ctx_len+P]    = K 个新 token 块，下三角因果：
        #       query p 只能 attend 新 key j（j <= p）；未来候选被屏蔽 (-inf)。
        mask = torch.zeros(P, a.ctx_len + P, device=dev, dtype=dt)
        block = torch.full((P, P), float("-inf"), device=dev, dtype=dt)
        block = torch.triu(block, diagonal=1)               # upper part (future) = -inf, diag+lower = 0
        mask[:, a.ctx_len:] = block
        # broadcast over heads inside SDPA: (P, ctx_len+P) -> (1, P, ctx_len+P) matches (H,P,hd) q.
        # 在 SDPA 内对 head 广播：(P, ctx_len+P) -> (1, P, ctx_len+P) 与 (H,P,hd) 的 q 匹配。
        return mask.unsqueeze(0)

    def attn_batched(l, hb, attn_mask):  # hb: (P, dm)
        P = hb.shape[0]
        q = (hb @ Wq[l].T).view(P, H, hd).transpose(0, 1)   # (H,P,hd)
        k = (hb @ Wk[l].T).view(P, H, hd).transpose(0, 1); v = (hb @ Wv[l].T).view(P, H, hd).transpose(0, 1)
        K = torch.cat([Kc[l].unsqueeze(0).expand(1, H, a.ctx_len, hd).reshape(H, a.ctx_len, hd), k], dim=1)
        V = torch.cat([Vc[l], v], dim=1)
        # v2 (FIX 1): pass the (1,P,ctx_len+P) additive mask so K-block is causal, prefix fully visible.
        # v2 (FIX 1): 传入 (1,P,ctx_len+P) 加性掩码，使 K 块因果、前缀全可见。
        o = F.scaled_dot_product_attention(q, K, V, attn_mask=attn_mask)   # (H,P,hd)
        return (o.transpose(0, 1).reshape(P, dm) @ Wo[l].T)

    def shared_b(l, hb):
        g, u, dd = Wsh[l]; return F.linear(F.silu(F.linear(hb, g)) * F.linear(hb, u), dd)

    def gpu_expert_b(w, hb):
        g, u, dd = w; return F.linear(F.silu(F.linear(hb, g)) * F.linear(hb, u), dd)

    def cpu_serve_grouped(hc, l, e):  # hc: (m, dm) in cpu_dt; expert's OWN distinct bank weights (real DRAM)
        g, u, dd = cpu_bank[(l * a.n_experts + e) % NBANK]
        return F.linear(F.silu(F.linear(hc, g)) * F.linear(hc, u), dd)

    def run_K(K):
        # v2 (FIX 4): per-K diagnostics accumulators (per single step, not summed across steps).
        # v2 (FIX 4): 单步诊断累加器（按单步统计，不跨步累加）。
        diag_C = 0; diag_U = 0; diag_cpu = 0; diag_bytes = 0
        attn_mask = build_kblock_mask(K)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for s in range(a.steps):
            base = (s * K) % (len(trace) - K)
            hb = torch.randn(K, dm, device=dev, dtype=dt)        # K positions
            step_C = 0; step_U = 0; step_cpu = 0; step_bytes = 0
            for l in range(L):
                hb = hb + attn_batched(l, hb, attn_mask)
                hn = hb
                _ = F.linear(hn, Wrt[l])
                out = shared_b(l, hn)
                # gather positions per expert
                pos_by_expert = {}
                for p in range(K):
                    for e in trace[base + p][l]:
                        pos_by_expert.setdefault(e, []).append(p)
                # v2 (FIX 4): C = total routed expert calls (top_k*K), U = unique experts this layer.
                # v2 (FIX 4): C = 路由专家调用总数 (top_k*K)，U = 本层唯一专家数。
                step_C += sum(len(pl) for pl in pos_by_expert.values())
                step_U += len(pos_by_expert)
                for e, plist in pos_by_expert.items():
                    sub = hn[plist]                              # (m, dm)
                    if (l, e) in gpu_experts:
                        oe = gpu_expert_b(gpu_experts[(l, e)], sub)
                        out.index_add_(0, torch.tensor(plist, device=dev), oe.to(dt))
                    else:
                        # v2 (FIX 2): CPU path in cpu_dt (bf16 by default) for same-precision with GPU.
                        # v2 (FIX 2): CPU 路径使用 cpu_dt（默认 bf16），与 GPU 同精度。
                        oc = cpu_serve_grouped(sub.to(dtype=cpu_dt).cpu(), l, e)
                        out.index_add_(0, torch.tensor(plist, device=dev), oc.to(dev, dtype=dt))
                        step_cpu += 1
                        step_bytes += bytes_per_expert        # v2 (FIX 4): weight bytes moved CPU->GPU served
            # keep last step's per-step diagnostics (steady-state representative)
            diag_C = step_C; diag_U = step_U; diag_cpu = step_cpu; diag_bytes = step_bytes
        torch.cuda.synchronize(dev)
        step_ms = (time.perf_counter() - t0) / a.steps * 1000.0
        if K > 1:
            accepted = max(1.0, K * a.accept_rate)
            tpot = (step_ms + K * a.draft_ms_per_tok) / accepted   # include serial draft cost (codex fix)
        else:
            tpot = step_ms
        reuse = (diag_C / diag_U) if diag_U > 0 else 0.0
        diag = {"C": diag_C, "U": diag_U, "reuse_C_over_U": reuse,
                "cpu_served": diag_cpu, "bytes_moved": int(diag_bytes)}
        return step_ms, tpot, diag

    res = {"config": vars(a), "n_resident": n_res, "rows": []}
    run_K(2)  # warmup
    base_tpot = None
    for K in [int(x) for x in a.ks.split(",")]:
        step_ms, tpot, diag = run_K(K)
        if K == 1: base_tpot = tpot
        row = {"K": K, "step_ms": step_ms, "tpot_ms": tpot}
        row.update(diag)
        res["rows"].append(row)
        # v2 (FIX 4): print per-K diagnostics so the result is interpretable.
        # v2 (FIX 4): 打印每个 K 的诊断信息，使结果可解释。
        print(f"K={K:>3} step={step_ms:7.3f}ms TPOT={tpot:7.3f}ms/token "
              f"(accept={a.accept_rate if K>1 else 1.0}) | "
              f"C={diag['C']:>4} U={diag['U']:>4} reuse={diag['reuse_C_over_U']:5.2f} "
              f"cpu_served={diag['cpu_served']:>4} bytes_moved={diag['bytes_moved']:>10}", flush=True)
    if base_tpot:
        for r in res["rows"]:
            r["speedup_vs_K1"] = base_tpot / r["tpot_ms"]
        best = max(res["rows"], key=lambda r: r["speedup_vs_K1"])
        print(f"\n[verdict] best K={best['K']} TPOT={best['tpot_ms']:.3f} -> {best['speedup_vs_K1']:.2f}x vs AR-Fiddler-K1 "
              f"(accept_rate={a.accept_rate}, cpu_dtype={a.cpu_dtype})", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
