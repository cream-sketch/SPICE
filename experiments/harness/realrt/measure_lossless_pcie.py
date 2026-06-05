"""Measure LOSSLESS expert-weight compression for offloaded-MoE PCIe reduction.

Premise (batch=1 offload): GPU compute is ~0.8 ms/token while H2D PCIe is ~75 ms/token, so we
have huge compute headroom. We can SPEND GPU compute to TRANSFER FEWER BYTES -- but it must be
EXACT (lossless): decompress(compress(W)) == W bit-for-bit. This script measures, on REAL Qwen
expert weights, the achievable lossless ratio and the GPU decompression throughput for several
nvCOMP algorithms, raw vs byte-plane-separated (bf16 high/low byte split, which exposes the
low-entropy exponent plane). Net effective PCIe and projected per-token transfer follow.

Not a simulator: real safetensors weights, real GPU compress/decompress, bit-exact verification.
真实无损压缩测量:省 PCIe 字节、花 GPU 算力解压、保证逐位无损。
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from collections import defaultdict

import numpy as np
import torch
from nvidia import nvcomp
from safetensors import safe_open

EXPERT_TAGS = ("gate_proj", "up_proj", "down_proj")


def load_expert_tensors(model_dir, n_sample):
    """Sample n_sample routed-expert weight tensors (bf16) spread across layers/files."""
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {model_dir}")
    key2file = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                if ".mlp.experts." in k and k.endswith(".weight") and any(t in k for t in EXPERT_TAGS):
                    key2file[k] = f
    keys = sorted(key2file)
    if not keys:
        raise RuntimeError("no expert weight keys found (check model architecture)")
    step = max(1, len(keys) // n_sample)
    sel = keys[::step][:n_sample]
    by_file = defaultdict(list)
    for k in sel:
        by_file[key2file[k]].append(k)
    tensors = []
    for f, ks in by_file.items():
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in ks:
                tensors.append(h.get_tensor(k))
    return tensors


def bytes_u8(t):
    """Flat little-endian byte view of a tensor on its current device."""
    return t.contiguous().view(torch.uint8).reshape(-1)


def byteplane_split(u8):
    """bf16/fp16 byte-plane separation: [..,low,high] -> concat(all_high, all_low).
    High byte (sign+exponent) is low-entropy -> compresses; low byte (mantissa) is near-random."""
    pairs = u8.reshape(-1, 2)
    return torch.cat([pairs[:, 1].contiguous(), pairs[:, 0].contiguous()])


def to_cpu_u8(arr):
    """nvcomp GPU Array -> host numpy uint8. Must KEEP the host Array alive across from_dlpack
    and copy out (dlpack borrows; a temporary .cpu() would be freed mid-read -> segfault)."""
    host = arr.cpu()
    return np.from_dlpack(host).view(np.uint8).copy()


def try_codec(name, **kw):
    try:
        return nvcomp.Codec(algorithm=name, **kw)
    except Exception:
        try:
            return nvcomp.Codec(algorithm=name)
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__} {e}")
            return None


def measure(codec, tensors, plane, reps, dev):
    """Return (ratio, exact, decompress_GBps). ratio = orig/compressed over all sampled experts;
    exact = bit-identical roundtrip on every expert; throughput timed on the concatenated buffer."""
    orig_total = 0
    comp_total = 0
    exact = True
    blobs = []
    for t in tensors:
        u8 = bytes_u8(t.to(dev))
        src = byteplane_split(u8) if plane else u8       # plane transform is deterministic + invertible
        arr = nvcomp.as_array(src)
        comp = codec.encode(arr)
        orig_total += int(src.numel())
        comp_total += int(comp.buffer_size)
        dec = codec.decode(comp)
        # lossless iff decoded bytes == the exact bytes fed to the codec
        if not np.array_equal(to_cpu_u8(dec)[: src.numel()], src.cpu().numpy().view(np.uint8)):
            exact = False
        blobs.append(src)
    # throughput on the concatenated working set
    big = torch.cat(blobs)
    arr = nvcomp.as_array(big)
    comp = codec.encode(arr)
    torch.cuda.synchronize(dev)
    for _ in range(3):
        codec.decode(comp)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        codec.decode(comp)
    torch.cuda.synchronize(dev)
    dt = (time.perf_counter() - t0) / reps
    gbps = big.numel() / dt / 1e9
    return orig_total / comp_total, exact, gbps


def main():
    p = argparse.ArgumentParser(description="Lossless expert-weight compression for PCIe reduction")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--n_sample", type=int, default=24, help="expert weight tensors to sample")
    p.add_argument("--reps", type=int, default=10, help="decompress timing repeats")
    p.add_argument("--pcie_gbps", type=float, default=22.0, help="measured H2D bandwidth")
    p.add_argument("--bytes_per_token_gb", type=float, default=1.66, help="full-bf16 H2D per token (all-miss)")
    args = p.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    tensors = load_expert_tensors(args.model_dir, args.n_sample)
    mb = sum(bytes_u8(t).numel() for t in tensors) / 1e6
    print(f"[load] {len(tensors)} expert weight tensors, {mb:.1f} MB bf16, dtype={tensors[0].dtype}")

    full_h2d_ms = args.bytes_per_token_gb / args.pcie_gbps * 1000.0
    print(f"[ref ] full-bf16 H2D per token = {args.bytes_per_token_gb} GB / {args.pcie_gbps} GB/s "
          f"= {full_h2d_ms:.1f} ms (the wall to beat)")
    cpu_serve_ms = 77.0  # measured CPU-serve baseline (no weight H2D) -- the real bar to beat
    print(f"[bar ] cpu_serve baseline = {cpu_serve_ms} ms (compute experts on CPU, no weight PCIe)")
    print(f"{'method':28s} {'ratio':>6s} {'exact':>6s} {'decompGBps':>11s} {'PCIe_ms':>8s} "
          f"{'decomp_ms':>10s} {'serial':>7s} {'pipelined':>10s}")

    configs = [
        ("LZ4", False), ("LZ4", True),
        ("Deflate", False), ("Deflate", True),
        ("GDeflate", False), ("GDeflate", True),
        ("Zstd", False), ("Zstd", True),
        ("ANS", False), ("ANS", True),
        ("Bitcomp", False), ("Bitcomp", True),
    ]
    for name, plane in configs:
        codec = try_codec(name)
        if codec is None:
            continue
        try:
            ratio, exact, gbps = measure(codec, tensors, plane, args.reps, dev)
        except Exception as e:
            print(f"  [fail] {name}{'+plane' if plane else '':6s}: {type(e).__name__} {e}")
            continue
        comp_gb = args.bytes_per_token_gb / ratio
        pcie_ms = comp_gb / args.pcie_gbps * 1000.0          # transfer the COMPRESSED bytes
        decomp_ms = args.bytes_per_token_gb / gbps * 1000.0  # GPU decompress to full bytes
        serial = pcie_ms + decomp_ms                          # naive (no overlap)
        pipelined = max(pcie_ms, decomp_ms)                   # double-buffered steady state
        tag = f"{name}{'+byteplane' if plane else ''}"
        win = " <-- beats cpu_serve" if pipelined < cpu_serve_ms else ""
        print(f"{tag:28s} {ratio:6.3f} {str(exact):>6s} {gbps:11.1f} {pcie_ms:8.1f} "
              f"{decomp_ms:10.1f} {serial:7.1f} {pipelined:10.1f}{win}")


if __name__ == "__main__":
    main()
