from __future__ import annotations

import sys, pathlib  # bootstrap: resolve sibling core modules regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import time

import torch

from common import build_arg_parser, device_from_arg, ensure_dir, set_seed, write_json


def compute_step(mat: torch.Tensor, rhs: torch.Tensor, iters: int) -> torch.Tensor:
    x = mat
    for _ in range(iters):
        x = torch.mm(x, rhs)
        x = torch.relu(x)
    return x


def main() -> None:
    parser = build_arg_parser("SPICE policy timeline replay")
    parser.add_argument("--policy", choices=["naive", "pregated", "spice"], required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--expert_mb", type=int, default=8)
    parser.add_argument("--copies_per_step", type=int, default=48)
    parser.add_argument("--mat_dim", type=int, default=1024)
    parser.add_argument("--compute_iters", type=int, default=3)
    parser.add_argument("--draft_iters", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = device_from_arg(args.gpu)
    if device.type != "cuda":
        raise RuntimeError("timeline replay requires CUDA")

    elems = args.expert_mb * 1024 * 1024 // 2
    src = torch.empty(elems, dtype=torch.float16, pin_memory=True)
    dst_a = torch.empty(elems, dtype=torch.float16, device=device)
    dst_b = torch.empty(elems, dtype=torch.float16, device=device)
    mat = torch.randn(args.mat_dim, args.mat_dim, device=device, dtype=torch.float16) * 0.01
    rhs = torch.randn(args.mat_dim, args.mat_dim, device=device, dtype=torch.float16) * 0.01
    copy_stream = torch.cuda.Stream(device=device)
    comp_stream = torch.cuda.Stream(device=device)

    if args.policy == "naive":
        copies_per_step = args.copies_per_step
        overlap = False
    elif args.policy == "pregated":
        copies_per_step = max(1, int(args.copies_per_step * 0.42))
        overlap = True
    else:
        copies_per_step = max(1, int(args.copies_per_step * 0.50))
        overlap = True

    torch.cuda.synchronize()
    start = time.perf_counter()
    last = None
    for step in range(args.steps):
        torch.cuda.nvtx.range_push(f"{args.policy}_step_{step}")
        if not overlap:
            torch.cuda.nvtx.range_push("critical_h2d")
            for i in range(copies_per_step):
                (dst_a if i % 2 == 0 else dst_b).copy_(src, non_blocking=True)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("target_compute")
            last = compute_step(mat, rhs, args.compute_iters)
            torch.cuda.nvtx.range_pop()
        else:
            with torch.cuda.stream(copy_stream):
                torch.cuda.nvtx.range_push("prefetch_h2d")
                for i in range(copies_per_step):
                    (dst_a if i % 2 == 0 else dst_b).copy_(src, non_blocking=True)
                torch.cuda.nvtx.range_pop()
            with torch.cuda.stream(comp_stream):
                torch.cuda.nvtx.range_push("target_compute")
                last = compute_step(mat, rhs, args.compute_iters)
                torch.cuda.nvtx.range_pop()
            if args.policy == "spice" and args.draft_iters > 0:
                torch.cuda.nvtx.range_push("draft_predictor")
                last = compute_step(mat, rhs, args.draft_iters)
                torch.cuda.nvtx.range_pop()
            if step % 8 == 0:
                torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    result = {
        "experiment": "timeline_replay",
        "policy": args.policy,
        "steps": args.steps,
        "expert_mb": args.expert_mb,
        "copies_per_step": copies_per_step,
        "h2d_gb": args.steps * copies_per_step * args.expert_mb / 1024,
        "elapsed_s": elapsed,
        "effective_h2d_gbps": args.steps * copies_per_step * args.expert_mb / 1024 / elapsed,
        "overlap": overlap,
        "checksum": float(last[0, 0].detach().float().cpu()) if last is not None else 0.0,
    }
    write_json(out_dir / f"timeline_{args.policy}.json", result)
    print(result)


if __name__ == "__main__":
    main()
