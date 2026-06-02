from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import asdict

import torch

from common import build_arg_parser, device_from_arg, ensure_dir, set_seed, write_json
from prefetch_system_sim import SimConfig, make_trace, parse_policy_list, simulate_policy


def sample_power(gpu_id: int, interval_s: float, stop: threading.Event, samples: list[dict]) -> None:
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu_id}",
                    "--query-gpu=timestamp,power.draw,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            parts = [p.strip() for p in out.split(",", 3)]
            if len(parts) == 4:
                samples.append(
                    {
                        "t_wall": time.time(),
                        "timestamp": parts[0],
                        "power_w": float(parts[1]),
                        "mem_used_mib": float(parts[2]),
                        "utilization_gpu_pct": float(parts[3]),
                    }
                )
        except Exception:
            pass
        stop.wait(interval_s)


def compute_step(mat: torch.Tensor, rhs: torch.Tensor, iters: int) -> torch.Tensor:
    x = mat
    for _ in range(iters):
        x = torch.mm(x, rhs)
        x = torch.relu(x)
    return x


def replay_policy(policy: str, sim_row: dict, args, device: torch.device) -> dict:
    if device.type != "cuda":
        raise RuntimeError("energy replay requires CUDA")

    elems = args.expert_mb * 1024 * 1024 // 2
    src = torch.empty(elems, dtype=torch.float16, pin_memory=True)
    dst_a = torch.empty(elems, dtype=torch.float16, device=device)
    dst_b = torch.empty(elems, dtype=torch.float16, device=device)
    mat = torch.randn(args.mat_dim, args.mat_dim, device=device, dtype=torch.float16) * 0.01
    rhs = torch.randn(args.mat_dim, args.mat_dim, device=device, dtype=torch.float16) * 0.01
    copy_stream = torch.cuda.Stream(device=device)
    comp_stream = torch.cuda.Stream(device=device)

    copy_count = max(1, int(round(sim_row["h2d_gb"] * 1024 / args.expert_mb * args.replay_scale)))
    replay_steps = max(1, int(round(args.steps * args.replay_scale)))
    copies_per_step = max(1, copy_count // replay_steps)
    leftover = copy_count - copies_per_step * replay_steps
    sync_policy = policy in {"naive", "lru"}

    torch.cuda.synchronize()
    samples: list[dict] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=sample_power,
        args=(args.power_gpu, args.power_interval, stop, samples),
        daemon=True,
    )
    sampler.start()
    t0 = time.perf_counter()
    last = None
    for step in range(replay_steps):
        n_copies = copies_per_step + (1 if step < leftover else 0)
        if sync_policy:
            for i in range(n_copies):
                (dst_a if i % 2 == 0 else dst_b).copy_(src, non_blocking=True)
            last = compute_step(mat, rhs, args.compute_iters)
        else:
            with torch.cuda.stream(copy_stream):
                for i in range(n_copies):
                    (dst_a if i % 2 == 0 else dst_b).copy_(src, non_blocking=True)
            with torch.cuda.stream(comp_stream):
                last = compute_step(mat, rhs, args.compute_iters)
            if step % args.sync_every == 0:
                torch.cuda.synchronize()

        if policy == "spice" and args.draft_iters > 0:
            last = compute_step(mat, rhs, args.draft_iters)

    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    stop.set()
    sampler.join(timeout=2.0)
    checksum = float(last[0, 0].detach().float().cpu()) if last is not None else 0.0

    powers = [s["power_w"] for s in samples]
    avg_power = sum(powers) / len(powers) if powers else None
    peak_power = max(powers) if powers else None
    energy_j = avg_power * elapsed_s if avg_power is not None else None
    return {
        "policy": policy,
        "sync_policy": sync_policy,
        "replay_steps": replay_steps,
        "copy_count": copy_count,
        "replayed_h2d_gb": copy_count * args.expert_mb / 1024,
        "elapsed_s": elapsed_s,
        "measured_tpot_ms": elapsed_s * 1000 / replay_steps,
        "avg_power_w": avg_power,
        "peak_power_w": peak_power,
        "energy_j": energy_j,
        "energy_per_token_j": energy_j / replay_steps if energy_j is not None else None,
        "num_power_samples": len(samples),
        "checksum": checksum,
        "sim_row": sim_row,
    }


def main() -> None:
    parser = build_arg_parser("SPICE energy-per-token hardware replay")
    parser.add_argument("--policies", type=str, default="naive,lru,moe_offloading,pregated,spice")
    parser.add_argument("--power_gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--expert_mb", type=int, default=8)
    parser.add_argument("--cache_capacity", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--replay_scale", type=float, default=0.50)
    parser.add_argument("--mat_dim", type=int, default=1024)
    parser.add_argument("--compute_iters", type=int, default=2)
    parser.add_argument("--draft_iters", type=int, default=1)
    parser.add_argument("--sync_every", type=int, default=8)
    parser.add_argument("--power_interval", type=float, default=0.20)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = device_from_arg(args.gpu)
    cfg = SimConfig(
        steps=args.steps,
        expert_mb=args.expert_mb,
        cache_capacity=args.cache_capacity,
        top_k=args.top_k,
    )
    trace = make_trace(cfg, args.seed)
    rows = []
    for policy in parse_policy_list(args.policies):
        sim_row = simulate_policy(policy, trace, cfg, args.seed)
        rows.append(replay_policy(policy, sim_row, args, device))

    result = {
        "experiment": "energy_per_token_replay",
        "device": str(device),
        "power_gpu": args.power_gpu,
        "config": {**asdict(cfg), "replay_scale": args.replay_scale},
        "rows": rows,
    }
    write_json(out_dir / "energy_per_token.json", result)
    print(result)


if __name__ == "__main__":
    main()
