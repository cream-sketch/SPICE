"""Feasibility probe: can nvCOMP GPU decompression OVERLAP a torch H2D copy, so that
compressed expert fetch reaches the pipelined floor max(transfer, decompress) instead of the
serial sum? This is the make-or-break for lossless-compressed offload at batch=1 (serial loses
at ~102 ms; pipelined ~54 ms beats cpu_serve 77 ms).

Key mechanism (per nvCOMP 5.2): the decode stream is bound at CODEC CONSTRUCTION via
`cuda_stream=`, NOT at decode-call time -- so `with torch.cuda.stream(s): decode()` does NOT
work. Use nvcomp.CudaStream.make_new(dev) -> wrap as torch.cuda.ExternalStream, build the codec
on that stream, run H2D on a separate torch stream, and gate decode with an event. Needs >=4
staging slots so an in-flight decode's input is not overwritten by a later copy.

Not a simulator: real Qwen expert weights, real nvCOMP encode/decode, real H2D, real streams.
真实重叠探针:验证 GPU 解压能否与 H2D 并发重叠,使压缩搬运达到 max(传输,解压) 流水线下界。
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
from nvidia import nvcomp
from safetensors import safe_open


def load_gate_tensors(model_dir, n):
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    ws = []
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                if ".mlp.experts." in k and "gate_proj" in k and k.endswith(".weight"):
                    ws.append(h.get_tensor(k))
        if len(ws) >= n:
            break
    return ws[:n]


def byteplane_split(u8):
    p = u8.reshape(-1, 2)
    return torch.cat([p[:, 1].contiguous(), p[:, 0].contiguous()])


def main():
    p = argparse.ArgumentParser(description="nvCOMP decode / H2D overlap feasibility probe")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--n", type=int, default=16, help="expert tensors to stream")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--slots", type=int, default=4, help=">=4 to avoid in-flight-input overwrite")
    args = p.parse_args()

    dev = args.gpu
    torch.cuda.set_device(dev)
    nv_s = nvcomp.CudaStream.make_new(dev)                       # nvCOMP-owned stream (keep alive)
    codec = nvcomp.Codec(algorithm="ANS", device_id=dev, cuda_stream=nv_s.ptr, data_type="|u1")
    ext_dec = torch.cuda.ExternalStream(nv_s.ptr, device=dev)   # torch view of the decode stream

    ws = load_gate_tensors(args.model_dir, args.n)
    comp_host = []
    for w in ws:
        u8 = w.contiguous().view(torch.uint8).reshape(-1).cuda()
        comp = codec.encode(nvcomp.as_array(byteplane_split(u8)))
        host = comp.cpu()                                        # keep alive across from_dlpack
        comp_host.append(torch.from_numpy(np.from_dlpack(host).view(np.uint8).copy()).pin_memory())
    maxsz = max(t.numel() for t in comp_host)
    usz = ws[0].numel() * 2
    n = len(comp_host)
    print(f"[probe] n={n} max_comp={maxsz/1e6:.2f}MB uncomp={usz/1e6:.2f}MB slots={args.slots}")

    def h2d_only(reps):
        g = torch.empty(maxsz, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            for t in comp_host:
                g[: t.numel()].copy_(t, non_blocking=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps / n * 1000

    def decode_only(reps):
        gc = [nvcomp.as_array(t.cuda(), cuda_stream=nv_s.ptr) for t in comp_host]
        outs = [torch.empty(usz, dtype=torch.uint8, device="cuda") for _ in comp_host]
        oa = [nvcomp.as_array(o, cuda_stream=nv_s.ptr) for o in outs]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            for g, o in zip(gc, oa):
                codec.decode(g, data_type="|u1", out=o)
        ext_dec.synchronize(); torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps / n * 1000

    def pipelined(reps):
        h2d_s = torch.cuda.Stream(device=dev)
        ns = args.slots
        comp_dev = [torch.empty(maxsz, dtype=torch.uint8, device="cuda") for _ in range(ns)]
        split_out = [torch.empty(usz, dtype=torch.uint8, device="cuda") for _ in range(ns)]
        split_arr = [nvcomp.as_array(o, cuda_stream=nv_s.ptr) for o in split_out]
        cdone = [torch.cuda.Event() for _ in range(ns)]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            for i in range(n + 1):
                if i < n:                                        # copy expert i on h2d stream
                    s = i % ns; t = comp_host[i]
                    with torch.cuda.stream(h2d_s):
                        comp_dev[s][: t.numel()].copy_(t, non_blocking=True)
                        cdone[s].record(h2d_s)
                if i >= 1:                                       # decode expert i-1 overlaps copy i
                    j = i - 1; s = j % ns
                    ext_dec.wait_event(cdone[s])
                    src = nvcomp.as_array(comp_dev[s][: comp_host[j].numel()], cuda_stream=nv_s.ptr)
                    codec.decode(src, data_type="|u1", out=split_arr[s])
        ext_dec.synchronize(); torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps / n * 1000

    h = h2d_only(args.reps); de = decode_only(args.reps); pi = pipelined(args.reps)
    print(f"[probe] h2d_only={h:.3f}  decode_only={de:.3f}  serial_sum={h+de:.3f}  "
          f"ideal_max={max(h, de):.3f}  PIPELINED={pi:.3f} ms/expert")
    print(f"[probe] verdict: {'OVERLAP WORKS' if pi < 0.85 * (h + de) else 'NO OVERLAP'} "
          f"(pipelined {'~=' if abs(pi-max(h,de)) < 0.3*max(h,de) else '!='} max)")


if __name__ == "__main__":
    main()
