"""PCIe topology microbench for SPICE scheduler decisions.

Measures whether A800 PCIe benefits from:
  1. H2D/D2H full-duplex overlap.
  2. Multi-stream tiled H2D prefetch for one bf16 expert.
  3. Small D2H activation latency while large H2D prefetch is in flight.
  4. Small H2D CPU-result latency while large H2D prefetch is in flight.
  5. Whether chunked/tiled prefetch reduces small H2D head-of-line blocking.

This is a diagnostic hardware probe. It does not implement an upstream baseline.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCIe topology microbench")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--d_inter", type=int, required=True)
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--streams", default="1,2,4,8")
    p.add_argument("--tile_mb", default="1,2,4,8")
    p.add_argument("--out", required=True)
    return p.parse_args()


def make_expert(dm: int, di: int, device: torch.device):
    host = [
        torch.empty(di, dm, dtype=torch.bfloat16, pin_memory=True).normal_(),
        torch.empty(di, dm, dtype=torch.bfloat16, pin_memory=True).normal_(),
        torch.empty(dm, di, dtype=torch.bfloat16, pin_memory=True).normal_(),
    ]
    dev = [torch.empty_like(x, device=device) for x in host]
    d2h = [torch.empty_like(x, pin_memory=True) for x in host]
    return host, dev, d2h


def sync_ms(fn, iters: int, device: torch.device) -> dict[str, float]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "median_ms": float(statistics.median(samples)),
        "mean_ms": float(sum(samples) / len(samples)),
        "p90_ms": float(samples[min(len(samples) - 1, int(0.9 * len(samples)))]),
        "min_ms": float(samples[0]),
        "max_ms": float(samples[-1]),
    }


def h2d_whole(host, dev):
    for src, dst in zip(host, dev):
        dst.copy_(src, non_blocking=True)


def d2h_whole(dev, d2h):
    for src, dst in zip(dev, d2h):
        dst.copy_(src, non_blocking=True)


def h2d_tiled(host, dev, streams, tile_elems: int):
    idx = 0
    for src, dst in zip(host, dev):
        sf = src.flatten()
        df = dst.flatten()
        n = sf.numel()
        for off in range(0, n, tile_elems):
            st = streams[idx % len(streams)]
            with torch.cuda.stream(st):
                df[off: min(off + tile_elems, n)].copy_(sf[off: min(off + tile_elems, n)], non_blocking=True)
            idx += 1


def h2d_first_tile(host, dev, tile_elems: int):
    src = host[0].flatten()
    dst = dev[0].flatten()
    n = min(tile_elems, src.numel())
    dst[:n].copy_(src[:n], non_blocking=True)


def main() -> None:
    a = parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)
    host, device_tensors, d2h_host = make_expert(a.d_model, a.d_inter, dev)
    expert_bytes = sum(x.numel() * x.element_size() for x in host)
    expert_mb = expert_bytes / 1e6
    results = {
        "config": vars(a),
        "expert_mb": expert_mb,
        "rows": [],
    }

    whole = sync_ms(lambda: h2d_whole(host, device_tensors), a.iters, dev)
    whole["case"] = "h2d_whole_one_stream"
    whole["gbps"] = expert_mb / whole["median_ms"] / 1.024
    results["rows"].append(whole)
    print(f"h2d_whole median={whole['median_ms']:.4f}ms gbps={whole['gbps']:.2f}", flush=True)

    d2h = sync_ms(lambda: d2h_whole(device_tensors, d2h_host), a.iters, dev)
    d2h["case"] = "d2h_whole_one_stream"
    d2h["gbps"] = expert_mb / d2h["median_ms"] / 1.024
    results["rows"].append(d2h)
    print(f"d2h_whole median={d2h['median_ms']:.4f}ms gbps={d2h['gbps']:.2f}", flush=True)

    s_h2d = torch.cuda.Stream(device=dev)
    s_d2h = torch.cuda.Stream(device=dev)
    def bidi_big():
        with torch.cuda.stream(s_h2d):
            h2d_whole(host, device_tensors)
        with torch.cuda.stream(s_d2h):
            d2h_whole(device_tensors, d2h_host)
    bidi = sync_ms(bidi_big, a.iters, dev)
    bidi["case"] = "h2d_d2h_big_full_duplex"
    bidi["ideal_ms"] = max(whole["median_ms"], d2h["median_ms"])
    bidi["overlap_efficiency"] = (whole["median_ms"] + d2h["median_ms"] - bidi["median_ms"]) / min(whole["median_ms"], d2h["median_ms"])
    results["rows"].append(bidi)
    print(f"bidi_big median={bidi['median_ms']:.4f}ms ideal={bidi['ideal_ms']:.4f} overlap_eff={bidi['overlap_efficiency']:.2f}", flush=True)

    small_host = torch.empty(1, a.d_model, dtype=torch.bfloat16, pin_memory=True)
    small_dev = torch.empty(1, a.d_model, dtype=torch.bfloat16, device=dev)
    small_d2h_host = torch.empty(1, a.d_model, dtype=torch.bfloat16, pin_memory=True)
    small_result_host = torch.empty(1, a.d_model, dtype=torch.bfloat16, pin_memory=True)
    small_result_dev = torch.empty(1, a.d_model, dtype=torch.bfloat16, device=dev)
    s_big = torch.cuda.Stream(device=dev)
    s_small = torch.cuda.Stream(device=dev)

    small_d2h = sync_ms(lambda: small_d2h_host.copy_(small_dev, non_blocking=True), a.iters, dev)
    small_d2h["case"] = "small_d2h_alone"
    results["rows"].append(small_d2h)
    def small_d2h_during_h2d():
        with torch.cuda.stream(s_big):
            h2d_whole(host, device_tensors)
        with torch.cuda.stream(s_small):
            small_d2h_host.copy_(small_dev, non_blocking=True)
    contended = sync_ms(small_d2h_during_h2d, a.iters, dev)
    contended["case"] = "small_d2h_during_big_h2d"
    results["rows"].append(contended)
    print(f"small_d2h alone={small_d2h['median_ms']:.5f}ms during_big_h2d_total={contended['median_ms']:.4f}ms", flush=True)

    small_h2d = sync_ms(lambda: small_result_dev.copy_(small_result_host, non_blocking=True), a.iters, dev)
    small_h2d["case"] = "small_h2d_alone"
    results["rows"].append(small_h2d)
    def small_h2d_during_h2d():
        with torch.cuda.stream(s_big):
            h2d_whole(host, device_tensors)
        with torch.cuda.stream(s_small):
            small_result_dev.copy_(small_result_host, non_blocking=True)
    h2d_contended = sync_ms(small_h2d_during_h2d, a.iters, dev)
    h2d_contended["case"] = "small_h2d_during_big_h2d"
    results["rows"].append(h2d_contended)
    print(f"small_h2d alone={small_h2d['median_ms']:.5f}ms during_big_h2d_total={h2d_contended['median_ms']:.4f}ms", flush=True)

    def small_completion_latency(big_enqueue):
        for _ in range(5):
            with torch.cuda.stream(s_big):
                big_enqueue()
            with torch.cuda.stream(s_small):
                small_result_dev.copy_(small_result_host, non_blocking=True)
            torch.cuda.synchronize(dev)
        samples = []
        for _ in range(a.iters):
            done = torch.cuda.Event()
            t0 = time.perf_counter()
            with torch.cuda.stream(s_big):
                big_enqueue()
            with torch.cuda.stream(s_small):
                small_result_dev.copy_(small_result_host, non_blocking=True)
                done.record()
            done.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
            torch.cuda.synchronize(dev)
        samples.sort()
        return {
            "median_ms": float(statistics.median(samples)),
            "mean_ms": float(sum(samples) / len(samples)),
            "p90_ms": float(samples[min(len(samples) - 1, int(0.9 * len(samples)))]),
            "min_ms": float(samples[0]),
            "max_ms": float(samples[-1]),
        }

    hol = small_completion_latency(lambda: h2d_whole(host, device_tensors))
    hol["case"] = "small_h2d_completion_after_whole_h2d"
    results["rows"].append(hol)
    print(f"small_h2d completion after whole_h2d={hol['median_ms']:.4f}ms", flush=True)

    for mb in [float(x) for x in a.tile_mb.split(",") if x]:
        tile_elems = max(1, int(mb * 1e6 / 2))
        lat = small_completion_latency(lambda tile_elems=tile_elems: h2d_first_tile(host, device_tensors, tile_elems))
        lat["case"] = "small_h2d_completion_after_one_tile"
        lat["tile_mb"] = mb
        results["rows"].append(lat)
        print(f"small_h2d completion after_one_tile tile={mb:g}MB latency={lat['median_ms']:.4f}ms", flush=True)

        def small_h2d_during_tiled_h2d(tile_elems=tile_elems):
            with torch.cuda.stream(s_big):
                h2d_tiled(host, device_tensors, [s_big], tile_elems)
            with torch.cuda.stream(s_small):
                small_result_dev.copy_(small_result_host, non_blocking=True)
        row = sync_ms(small_h2d_during_tiled_h2d, a.iters, dev)
        row["case"] = "small_h2d_during_tiled_h2d"
        row["tile_mb"] = mb
        results["rows"].append(row)
        print(f"small_h2d during_tiled_h2d tile={mb:g}MB total={row['median_ms']:.4f}ms", flush=True)

    for ns in [int(x) for x in a.streams.split(",") if x]:
        streams = [torch.cuda.Stream(device=dev) for _ in range(ns)]
        for mb in [float(x) for x in a.tile_mb.split(",") if x]:
            tile_elems = max(1, int(mb * 1e6 / 2))
            row = sync_ms(lambda: h2d_tiled(host, device_tensors, streams, tile_elems), a.iters, dev)
            row["case"] = "h2d_tiled"
            row["streams"] = ns
            row["tile_mb"] = mb
            row["gbps"] = expert_mb / row["median_ms"] / 1.024
            row["speedup_vs_whole"] = whole["median_ms"] / row["median_ms"]
            results["rows"].append(row)
            print(f"h2d_tiled streams={ns} tile={mb:g}MB median={row['median_ms']:.4f}ms "
                  f"gbps={row['gbps']:.2f} speedup={row['speedup_vs_whole']:.3f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
