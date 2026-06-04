"""CUDA microbench for SPICE shallow H2D software issuer.

This is the hardware kill-test for the event replay's main assumption:
SPICE should keep draft prefetches as software intents and submit only a
shallow number of low-priority 17MB H2D DMA copies.  Then residual miss fetches
or CPU-result H2D copies should wait behind at most that shallow queue, not an
entire token worth of already-enqueued draft prefetches.

The benchmark measures completion latency of a high-priority transfer after a
known number of low-priority expert-size H2D copies have already been submitted:

  deep_backlog_N: submit N low H2D copies, then submit one high transfer.
  shallow_d:      submit d low H2D copies, then submit one high transfer.
  chunk_tile_MB:  submit one low H2D tile, then submit one high transfer.

The measured latency starts after low copies have been enqueued, so it captures
copy-engine head-of-line blocking plus the high transfer itself, not Python
enqueue overhead.  This is a diagnostic hardware probe, not a model benchmark.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPICE shallow H2D issuer CUDA microbench")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--d_inter", type=int, required=True)
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--deep_backlog", type=int, default=32)
    p.add_argument("--depths", default="0,1,2,4,8")
    p.add_argument("--tile_mb", default="0.5,1,2,4")
    p.add_argument("--low_priority", type=int, default=0)
    p.add_argument("--high_priority", type=int, default=-1)
    p.add_argument("--out", required=True)
    return p.parse_args()


def quantiles(samples: list[float]) -> dict[str, float]:
    samples = sorted(samples)
    return {
        "median_ms": float(statistics.median(samples)),
        "mean_ms": float(sum(samples) / len(samples)),
        "p90_ms": float(samples[min(len(samples) - 1, int(0.9 * len(samples)))]),
        "p99_ms": float(samples[min(len(samples) - 1, int(0.99 * len(samples)))]),
        "min_ms": float(samples[0]),
        "max_ms": float(samples[-1]),
    }


def sync_bench(fn, iters: int, warmup: int, dev: torch.device) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(dev)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return quantiles(samples)


def completion_latency_after_lows(enqueue_lows, enqueue_high, iters: int, warmup: int,
                                  dev: torch.device) -> dict[str, float]:
    for _ in range(warmup):
        enqueue_lows()
        done = enqueue_high()
        done.synchronize()
        torch.cuda.synchronize(dev)
    samples = []
    for _ in range(iters):
        enqueue_lows()
        t0 = time.perf_counter()
        done = enqueue_high()
        done.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
        # Drain any low-priority work that the high-priority copy bypassed.
        torch.cuda.synchronize(dev)
    return quantiles(samples)


def safe_priority_range():
    for obj, name in ((torch.cuda, "get_stream_priority_range"), (torch.cuda.Stream, "priority_range")):
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        try:
            x = fn()
            return [int(x[0]), int(x[1])] if isinstance(x, (tuple, list)) else str(x)
        except Exception as exc:  # pragma: no cover - hardware/API dependent
            return f"unavailable: {exc}"
    return "unavailable"


def nvidia_smi_query(gpu: int) -> dict:
    out = {}
    queries = {
        "gpu": [
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=uuid,name,pstate,clocks.sm,clocks.mem,compute_mode,mig.mode.current",
            "--format=csv,noheader,nounits",
        ],
        "compute_apps": [
            "nvidia-smi", f"--id={gpu}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    }
    for key, cmd in queries.items():
        try:
            out[key] = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:  # pragma: no cover - nvidia-smi environment dependent
            out[key] = f"unavailable: {exc}"
    return out


def main() -> None:
    a = parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)

    # One bf16 fine-grained expert: W_gate, W_up, W_down.
    expert_elems = 3 * a.d_model * a.d_inter
    expert_bytes = expert_elems * 2
    expert_mb = expert_bytes / 1e6
    small_elems = a.d_model
    max_depth = max([int(x) for x in a.depths.split(",") if x] + [a.deep_backlog, 1])

    low_host = [torch.empty(expert_elems, dtype=torch.bfloat16, pin_memory=True).normal_()
                for _ in range(max_depth)]
    low_dev = [torch.empty(expert_elems, dtype=torch.bfloat16, device=dev)
               for _ in range(max_depth)]
    high_host = torch.empty(expert_elems, dtype=torch.bfloat16, pin_memory=True).normal_()
    high_dev = torch.empty(expert_elems, dtype=torch.bfloat16, device=dev)
    small_h2d_host = torch.empty(small_elems, dtype=torch.bfloat16, pin_memory=True).normal_()
    small_h2d_dev = torch.empty(small_elems, dtype=torch.bfloat16, device=dev)
    small_d2h_dev = torch.empty(small_elems, dtype=torch.bfloat16, device=dev).normal_()
    small_d2h_host = torch.empty(small_elems, dtype=torch.bfloat16, pin_memory=True)

    low_stream = torch.cuda.Stream(device=dev, priority=a.low_priority)
    high_stream = torch.cuda.Stream(device=dev, priority=a.high_priority)

    def low_copy(i: int, elems: int | None = None):
        n = expert_elems if elems is None else min(elems, expert_elems)
        with torch.cuda.stream(low_stream):
            low_dev[i % max_depth][:n].copy_(low_host[i % max_depth][:n], non_blocking=True)

    def high_big_h2d():
        done = torch.cuda.Event()
        with torch.cuda.stream(high_stream):
            high_dev.copy_(high_host, non_blocking=True)
            done.record()
        return done

    def high_small_h2d():
        done = torch.cuda.Event()
        with torch.cuda.stream(high_stream):
            small_h2d_dev.copy_(small_h2d_host, non_blocking=True)
            done.record()
        return done

    def high_small_d2h():
        done = torch.cuda.Event()
        with torch.cuda.stream(high_stream):
            small_d2h_host.copy_(small_d2h_dev, non_blocking=True)
            done.record()
        return done

    def enqueue_n_lows(n: int):
        for i in range(n):
            low_copy(i)

    rows = []
    results = {
        "config": vars(a),
        "expert_mb": expert_mb,
        "small_h2d_kb": small_elems * 2 / 1000.0,
        "device": {
            "torch_device_name": torch.cuda.get_device_name(dev),
            "torch_device_properties": str(torch.cuda.get_device_properties(dev)),
            "stream_priority_range": safe_priority_range(),
            "nvidia_smi_before": nvidia_smi_query(a.gpu),
        },
        "rows": rows,
    }
    print(f"[setup] expert={expert_mb:.2f}MB small={small_elems*2/1000.0:.2f}KB "
          f"low_pri={a.low_priority} high_pri={a.high_priority}", flush=True)

    row = sync_bench(lambda: high_dev.copy_(high_host, non_blocking=True), a.iters, a.warmup, dev)
    row.update({"case": "big_h2d_alone", "queued_lows": 0, "high_kind": "big_h2d",
                "gbps": expert_mb / row["median_ms"] / 1.024})
    rows.append(row)
    big_alone = row["median_ms"]
    print(f"big_h2d_alone median={row['median_ms']:.4f}ms gbps={row['gbps']:.2f}", flush=True)

    row = sync_bench(lambda: small_h2d_dev.copy_(small_h2d_host, non_blocking=True),
                     a.iters, a.warmup, dev)
    row.update({"case": "small_h2d_alone", "queued_lows": 0, "high_kind": "small_h2d"})
    rows.append(row)
    print(f"small_h2d_alone median={row['median_ms']:.5f}ms", flush=True)

    row = sync_bench(lambda: small_d2h_host.copy_(small_d2h_dev, non_blocking=True),
                     a.iters, a.warmup, dev)
    row.update({"case": "small_d2h_alone", "queued_lows": 0, "high_kind": "small_d2h"})
    rows.append(row)
    print(f"small_d2h_alone median={row['median_ms']:.5f}ms", flush=True)

    high_cases = [
        ("big_h2d", high_big_h2d),
        ("small_h2d", high_small_h2d),
        ("small_d2h", high_small_d2h),
    ]
    for high_kind, high_fn in high_cases:
        row = completion_latency_after_lows(lambda: enqueue_n_lows(a.deep_backlog), high_fn,
                                            a.iters, a.warmup, dev)
        row.update({"case": "deep_backlog", "queued_lows": a.deep_backlog, "high_kind": high_kind,
                    "fifo_bound_ms": a.deep_backlog * big_alone + (big_alone if high_kind == "big_h2d" else 0.0)})
        rows.append(row)
        print(f"deep_backlog N={a.deep_backlog:2d} high={high_kind:9s} median={row['median_ms']:.4f}ms "
              f"p90={row['p90_ms']:.4f}ms", flush=True)

    for depth in [int(x) for x in a.depths.split(",") if x]:
        for high_kind, high_fn in high_cases:
            row = completion_latency_after_lows(lambda depth=depth: enqueue_n_lows(depth), high_fn,
                                                a.iters, a.warmup, dev)
            row.update({"case": "shallow_depth", "queued_lows": depth, "high_kind": high_kind,
                        "fifo_bound_ms": depth * big_alone + (big_alone if high_kind == "big_h2d" else 0.0)})
            rows.append(row)
            print(f"shallow depth={depth:2d} high={high_kind:9s} median={row['median_ms']:.4f}ms "
                  f"p90={row['p90_ms']:.4f}ms", flush=True)

    for mb in [float(x) for x in a.tile_mb.split(",") if x]:
        tile_elems = max(1, int(mb * 1e6 / 2))
        for high_kind, high_fn in high_cases:
            row = completion_latency_after_lows(lambda tile_elems=tile_elems: low_copy(0, tile_elems),
                                                high_fn, a.iters, a.warmup, dev)
            row.update({"case": "one_low_tile", "tile_mb": mb, "queued_lows": 1, "high_kind": high_kind})
            rows.append(row)
            print(f"one_low_tile={mb:g}MB high={high_kind:9s} median={row['median_ms']:.4f}ms "
                  f"p90={row['p90_ms']:.4f}ms", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    results["device"]["nvidia_smi_after"] = nvidia_smi_query(a.gpu)
    Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
