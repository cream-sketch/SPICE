from __future__ import annotations

import time

import torch

from common import build_arg_parser, device_from_arg, ensure_dir, set_seed, write_json


def main() -> None:
    parser = build_arg_parser("CUDA H2D copy overlap timeline workload")
    parser.add_argument("--expert_mb", type=int, default=8)
    parser.add_argument("--copies", type=int, default=2048)
    parser.add_argument("--compute_iters", type=int, default=24)
    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = device_from_arg(args.gpu)
    if device.type != "cuda":
        raise RuntimeError("copy_timeline requires CUDA")

    elems = args.expert_mb * 1024 * 1024 // 2
    src = torch.empty(elems, dtype=torch.float16, pin_memory=True)
    dst_a = torch.empty(elems, dtype=torch.float16, device=device)
    dst_b = torch.empty(elems, dtype=torch.float16, device=device)
    mat = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    vec = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    copy_stream = torch.cuda.Stream(device=device)
    comp_stream = torch.cuda.Stream(device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(args.copies):
        with torch.cuda.stream(copy_stream):
            (dst_a if i % 2 == 0 else dst_b).copy_(src, non_blocking=True)
        with torch.cuda.stream(comp_stream):
            x = mat
            for _ in range(args.compute_iters):
                x = torch.mm(x, vec)
                x = torch.relu(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_bytes = args.copies * args.expert_mb * 1024 * 1024
    result = {
        "experiment": "copy_timeline",
        "expert_mb": args.expert_mb,
        "copies": args.copies,
        "compute_iters": args.compute_iters,
        "elapsed_s": elapsed,
        "h2d_gb": total_bytes / 1024**3,
        "effective_h2d_gbps": total_bytes / elapsed / 1024**3,
    }
    write_json(out_dir / "copy_timeline.json", result)
    print(result)


if __name__ == "__main__":
    main()
