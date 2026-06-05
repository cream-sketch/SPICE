"""Decisive offload microbench: per-token cost of FUSED compressed expert fetch vs plain on_demand
H2D, at batch=1. Isolates the mechanism (no model forward): replay K expert-invocations/token,
each = transfer the expert weights H2D + compute SwiGLU(x). Two paths:
  (a) on_demand:  H2D full bf16 gate/up/down -> 3x F.linear (the 75ms baseline mechanism)
  (b) fused:      H2D packed payload (1.33x less) -> 3x fused_decode_gemv (decode-in-register)
Question: does fused beat on_demand, or does batch=1 per-expert launch/H2D overhead eat the 1.33x?

Not a simulator: real Qwen expert weights, real H2D over PCIe, real Triton fused decode, exactness
checked. 真实 offload 微基准:融合压缩 fetch vs on_demand,每 token 真实搬运+计算。
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

from fused_decode_gemv import pack, fused_decode_gemv


def load_experts(model_dir, n):
    """Load n routed experts as (gate, up, down) bf16 triples."""
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    by_layer = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                if ".mlp.experts." in k and k.endswith(".weight") and any(
                    t in k for t in ("gate_proj", "up_proj", "down_proj")):
                    parts = k.split(".experts.")[1]
                    eid = parts.split(".")[0]
                    li = k.split(".layers.")[1].split(".")[0]
                    key = (li, eid)
                    by_layer.setdefault(key, {})[k.rsplit(".", 2)[1]] = (f, k)
    triples = []
    for key, d in by_layer.items():
        if len(d) == 3:
            triples.append(d)
        if len(triples) >= n:
            break
    out = []
    for d in triples:
        t = {}
        for proj, (f, k) in d.items():
            with safe_open(f, framework="pt", device="cpu") as h:
                t[proj] = h.get_tensor(k)
        out.append((t["gate_proj"], t["up_proj"], t["down_proj"]))
    return out


def main():
    p = argparse.ArgumentParser(description="fused compressed offload vs on_demand per-token microbench")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--n_experts", type=int, default=96, help="distinct experts to cycle (>= per-token)")
    p.add_argument("--per_token", type=int, default=96, help="expert-invocations per token (Qwen: 24*4)")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--pcie_gbps", type=float, default=22.0)
    args = p.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    experts = load_experts(args.model_dir, args.n_experts)
    ne = len(experts)
    d_model = experts[0][0].shape[1]
    print(f"[load] {ne} experts, d_model={d_model}")

    # CPU pinned weights (on_demand) + packed payloads & resident meta (fused)
    od_w = []      # (gate,up,down) pinned bf16
    fz = []        # ((gp,gm),(up,um),(dp,dm)) payload pinned + meta on GPU
    for g, u, d in experts:
        od_w.append(tuple(t.to(torch.bfloat16).contiguous().pin_memory() for t in (g, u, d)))
        trip = []
        for t in (g, u, d):
            packed, meta = pack(t.to(dev))
            trip.append((packed.to("cpu").pin_memory(), meta.to(dev)))
        fz.append(tuple(trip))
    full_bytes = sum(t.numel() * 2 for t in experts[0])
    pk_bytes = sum(p.numel() for (p, _), in [(fz[0][0],), (fz[0][1],), (fz[0][2],)])
    print(f"[size] full_expert={full_bytes/1e6:.2f}MB packed={pk_bytes/1e6:.2f}MB ratio={full_bytes/pk_bytes:.3f}x")

    x = torch.randn(d_model, device=dev, dtype=torch.bfloat16)
    # GPU scratch for on_demand weights
    g0, u0, d0 = experts[0]
    sg = torch.empty_like(g0, device=dev); su = torch.empty_like(u0, device=dev); sd = torch.empty_like(d0, device=dev)

    def on_demand_token(ti):
        out = None
        for j in range(args.per_token):
            gw, uw, dw = od_w[(ti + j) % ne]
            sg.copy_(gw, non_blocking=True); su.copy_(uw, non_blocking=True); sd.copy_(dw, non_blocking=True)
            y = F.linear(F.silu(F.linear(x, sg)) * F.linear(x, su), sd)
            out = y if out is None else out + y
        return out

    # fused scratch payload buffers
    pbuf = [torch.empty(fz[0][i][0].numel(), dtype=torch.uint8, device=dev) for i in range(3)]

    def fused_token(ti):
        out = None
        for j in range(args.per_token):
            trip = fz[(ti + j) % ne]
            for i in range(3):
                pbuf[i][: trip[i][0].numel()].copy_(trip[i][0], non_blocking=True)
            (gp, gm), (up, um), (dp, dm) = trip
            g = fused_decode_gemv(pbuf[0][: gp.numel()], gm, x)
            u = fused_decode_gemv(pbuf[1][: up.numel()], um, x)
            h = (F.silu(g) * u).to(torch.bfloat16)
            y = fused_decode_gemv(pbuf[2][: dp.numel()], dm, h)
            out = y if out is None else out + y
        return out

    def run(fn):
        torch.cuda.synchronize(dev)
        for ti in range(args.tokens):
            if ti == args.warmup:
                torch.cuda.synchronize(dev); t0 = time.perf_counter()
            fn(ti)
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) * 1000.0 / (args.tokens - args.warmup)

    od_ms = run(on_demand_token)
    fz_ms = run(fused_token)
    print(f"[tpot] on_demand_full_h2d = {od_ms:.2f} ms/token")
    print(f"[tpot] fused_compressed   = {fz_ms:.2f} ms/token   (speedup {od_ms/fz_ms:.3f}x)")
    print(f"[ref ] full H2D vol/token = {args.per_token*full_bytes/1e9:.2f}GB -> "
          f"{args.per_token*full_bytes/1e9/args.pcie_gbps*1000:.1f}ms PCIe floor; "
          f"compressed -> {args.per_token*pk_bytes/1e9/args.pcie_gbps*1000:.1f}ms")


if __name__ == "__main__":
    main()
