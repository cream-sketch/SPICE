"""Diagnostic microbench: capacity-aware miss assignment across CPU and fetch/GPU.

For a layer with n missed routed experts (n <= model top_k), enumerate assignments:
  f experts: demand-fetch weights to GPU, then GPU compute
  n-f experts: D2H activation once, CPU exact expert compute, H2D summed output once

The measured wall time answers whether "all CPU" or "split CPU+fetch" wins as n grows.
This is a diagnostic microbench, not an upstream baseline reproduction.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Miss assignment microbench")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--d_inter", type=int, required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--cpu_threads", type=int, required=True)
    p.add_argument("--cpu_dtype", choices=["bf16", "fp32"], required=True)
    p.add_argument("--bank", type=int, default=32, help="distinct expert bank to avoid tiny-cache artifacts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def expert(x: torch.Tensor, g: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


def make_cpu_bank(bank: int, dm: int, di: int, dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    scale_g = dm ** -0.5
    scale_d = di ** -0.5
    return [
        (
            (torch.randn(di, dm) * scale_g).to(dtype=dtype).contiguous(),
            (torch.randn(di, dm) * scale_g).to(dtype=dtype).contiguous(),
            (torch.randn(dm, di) * scale_d).to(dtype=dtype).contiguous(),
        )
        for _ in range(bank)
    ]


def make_fetch_bank(bank: int, dm: int, di: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Pinned bf16 host weights for demand H2D fetch.
    scale_g = dm ** -0.5
    scale_d = di ** -0.5
    host = []
    for _ in range(bank):
        host.append(
            (
                (torch.randn(di, dm) * scale_g).to(dtype=torch.bfloat16).pin_memory(),
                (torch.randn(di, dm) * scale_g).to(dtype=torch.bfloat16).pin_memory(),
                (torch.randn(dm, di) * scale_d).to(dtype=torch.bfloat16).pin_memory(),
            )
        )
    dst = [
        (
            torch.empty(di, dm, device=device, dtype=torch.bfloat16),
            torch.empty(di, dm, device=device, dtype=torch.bfloat16),
            torch.empty(dm, di, device=device, dtype=torch.bfloat16),
        )
        for _ in range(max(1, 8))
    ]
    return host, dst


def main() -> None:
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)
    cpu_dt = torch.bfloat16 if a.cpu_dtype == "bf16" else torch.float32
    dm, di = a.d_model, a.d_inter
    bank = max(a.bank, a.top_k * 4)

    cpu_bank = make_cpu_bank(bank, dm, di, cpu_dt)
    fetch_bank, dst_bank = make_fetch_bank(bank, dm, di, dev)
    x_gpu = torch.randn(1, dm, device=dev, dtype=torch.bfloat16)
    x_cpu_pin = torch.empty(1, dm, dtype=cpu_dt, pin_memory=True)
    y_cpu_pin = torch.empty(1, dm, dtype=cpu_dt, pin_memory=True)
    y_gpu = torch.empty(1, dm, device=dev, dtype=torch.bfloat16)
    stream = torch.cuda.Stream(device=dev)
    cpu_stream = torch.cuda.Stream(device=dev)

    expert_bytes = (2 * di * dm + dm * di) * 2
    results = {
        "config": vars(a),
        "bank": bank,
        "expert_mb_bf16": expert_bytes / 1e6,
        "rows": [],
    }

    def run_once(n_miss: int, n_fetch: int, offset: int) -> None:
        n_cpu = n_miss - n_fetch
        # Activation D2H once if any CPU-served expert. This is tiny but on the critical path.
        if n_cpu:
            x_cpu_pin.copy_(x_gpu.to(dtype=cpu_dt), non_blocking=True)
            torch.cuda.synchronize(dev)

        # Launch fetch/GPU side asynchronously. H2D(weight) is a predecessor of its GPU compute.
        if n_fetch:
            with torch.cuda.stream(stream):
                acc = None
                for j in range(n_fetch):
                    src = fetch_bank[(offset + j) % bank]
                    dst = dst_bank[j % len(dst_bank)]
                    for s, d in zip(src, dst):
                        d.copy_(s, non_blocking=True)
                    y = expert(x_gpu, dst[0], dst[1], dst[2])
                    acc = y if acc is None else acc + y

        # CPU side uses distinct experts and sums outputs before one H2D merge copy.
        if n_cpu:
            out = None
            for j in range(n_cpu):
                g, u, d = cpu_bank[(offset + n_fetch + j) % bank]
                y = expert(x_cpu_pin, g, u, d)
                out = y if out is None else out + y
            y_cpu_pin.copy_(out)
            with torch.cuda.stream(cpu_stream):
                y_gpu.copy_(y_cpu_pin.to(dtype=torch.bfloat16), non_blocking=True)

        torch.cuda.synchronize(dev)

    def bench(n_miss: int, n_fetch: int) -> dict[str, float]:
        for w in range(a.warmup):
            run_once(n_miss, n_fetch, w * n_miss)
        torch.cuda.synchronize(dev)
        samples = []
        for it in range(a.iters):
            t0 = time.perf_counter()
            run_once(n_miss, n_fetch, it * n_miss)
            samples.append((time.perf_counter() - t0) * 1000.0)
        torch.cuda.synchronize(dev)
        samples.sort()
        p90 = samples[min(len(samples) - 1, int(0.9 * len(samples)))]
        return {
            "ms": float(median(samples)),
            "mean_ms": float(sum(samples) / len(samples)),
            "p90_ms": float(p90),
            "min_ms": float(samples[0]),
            "max_ms": float(samples[-1]),
        }

    print(f"[setup] top_k={a.top_k} cpu_dtype={a.cpu_dtype} expert={expert_bytes/1e6:.2f}MB "
          f"threads={torch.get_num_threads()} bank={bank}", flush=True)
    combos = [(n, f) for n in range(1, a.top_k + 1) for f in range(0, n + 1)]
    # Touch every combination before timed runs so oneDNN/CUDA lazy setup cannot bias the first row.
    for n, f in combos:
        run_once(n, f, (n * 100 + f) % bank)
    torch.cuda.synchronize(dev)

    rng = random.Random(a.seed)
    shuffled = combos[:]
    rng.shuffle(shuffled)
    measured = {}
    for n, f in shuffled:
        stats = bench(n, f)
        measured[(n, f)] = stats

    for n in range(1, a.top_k + 1):
        best = None
        for f in range(0, n + 1):
            stats = measured[(n, f)]
            row = {"n_miss": n, "n_fetch": f, "n_cpu": n - f, **stats}
            results["rows"].append(row)
            if best is None or row["ms"] < best["ms"]:
                best = row
            print(f"n_miss={n:>2} fetch={f:>2} cpu={n-f:>2} "
                  f"median={row['ms']:8.4f} mean={row['mean_ms']:8.4f} "
                  f"p90={row['p90_ms']:8.4f} max={row['max_ms']:8.4f} ms", flush=True)
        assert best is not None
        print(f"[best] n_miss={n}: fetch={best['n_fetch']} cpu={best['n_cpu']} "
              f"median={best['ms']:.4f} ms", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
