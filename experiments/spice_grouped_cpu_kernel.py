"""Grouped CPU MoE-serve kernel microbench (the strongest POSITIVE exact batch=1 lever, per gemini
peer-review + our analysis). batch=1, Qwen expert shape, EXACT same-precision (bf16/fp32).

The measured CPU multi-expert burst is super-linear (4 experts = 2.19ms = only ~31 GB/s effective,
far below DRAM peak ~200 GB/s) -> the per-expert loop kernel is inefficient (thread-launch overhead +
L3 contention). HYPOTHESIS: serving the top-k missed experts as ONE GROUPED operation (stack their
gate/up/down weights contiguously, one big GEMV sweep per matrix type) reaches DRAM bandwidth ->
cuts the burst toward the ~0.35-0.7ms floor -> +20-30% TPOT, exact.

Compares per N in {1,2,3,4}:
  loop_separate : N independent expert forwards (current Fiddler kernel)
  grouped       : stacked-weight grouped GEMV (gate: (1,dm)x(N*di,dm)^T, etc.)
Reports ms and effective GB/s. Also sweeps torch thread count. All printed English. No core defaults.
"""
import argparse, time, json
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="Grouped CPU MoE-serve kernel microbench")
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--max_experts", type=int, required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--threads", type=str, required=True, help="comma list of torch thread counts to sweep")
    ap.add_argument("--dtype", required=True, help="float32 or bfloat16")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def bench(fn, iters):
    for _ in range(5): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main():
    a = parse_args()
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    bytes_per = 2 if dt == torch.bfloat16 else 4
    dm, di = a.d_model, a.d_inter
    expert_bytes = 3 * di * dm * bytes_per
    res = {"d_model": dm, "d_inter": di, "dtype": a.dtype, "expert_mb": expert_bytes / 1e6, "rows": []}

    # weight pool for max_experts
    Wg = [torch.randn(di, dm, dtype=dt) for _ in range(a.max_experts)]
    Wu = [torch.randn(di, dm, dtype=dt) for _ in range(a.max_experts)]
    Wd = [torch.randn(dm, di, dtype=dt) for _ in range(a.max_experts)]
    x = torch.randn(1, dm, dtype=dt)

    def loop_separate(n):
        out = 0
        for i in range(n):
            z = F.silu(F.linear(x, Wg[i])) * F.linear(x, Wu[i])
            out = out + F.linear(z, Wd[i])
        return out

    def make_grouped(n):
        Wg_s = torch.cat(Wg[:n], dim=0).contiguous()   # (n*di, dm)
        Wu_s = torch.cat(Wu[:n], dim=0).contiguous()
        Wd_s = torch.cat(Wd[:n], dim=0).contiguous()   # (n*dm, di)
        def grouped():
            g = F.linear(x, Wg_s); u = F.linear(x, Wu_s)        # (1, n*di) each
            z = F.silu(g) * u                                    # (1, n*di)
            zr = z.view(n, di)                                   # per-expert intermediates
            # down: each expert's z_i (1,di) @ Wd_i (dm,di)^T -> sum. Use block-diag-free batched matmul
            out = torch.bmm(zr.unsqueeze(1), Wd_s.view(n, dm, di).transpose(1, 2)).sum(0)  # (1, dm)
            return out
        return grouped

    for nt in [int(t) for t in a.threads.split(",")]:
        torch.set_num_threads(nt)
        for n in range(1, a.max_experts + 1):
            t_loop = bench(lambda: loop_separate(n), a.iters)
            grouped = make_grouped(n)
            t_grp = bench(grouped, a.iters)
            read_mb = n * expert_bytes / 1e6
            row = {"threads": nt, "n_experts": n,
                   "loop_ms": t_loop, "grouped_ms": t_grp,
                   "loop_gbps": read_mb / t_loop / 1.024, "grouped_gbps": read_mb / t_grp / 1.024,
                   "speedup": t_loop / max(1e-9, t_grp)}
            res["rows"].append(row)
            print(f"threads={nt:>3} n={n} loop={t_loop:.3f}ms({row['loop_gbps']:.0f}GB/s) "
                  f"grouped={t_grp:.3f}ms({row['grouped_gbps']:.0f}GB/s) speedup={row['speedup']:.2f}x", flush=True)

    with open(a.out, "w") as f: json.dump(res, f, indent=2)
    # verdict at full threads, max experts
    best = [r for r in res["rows"] if r["n_experts"] == a.max_experts]
    best = min(best, key=lambda r: r["grouped_ms"])
    print(f"\n[verdict] {a.max_experts} experts: best grouped {best['grouped_ms']:.3f}ms "
          f"({best['grouped_gbps']:.0f}GB/s, {best['threads']}thr) vs loop {best['loop_ms']:.3f}ms "
          f"-> {best['speedup']:.2f}x. DRAM-floor ~{a.max_experts*res['expert_mb']/200/1.024:.3f}ms@200GB/s", flush=True)


if __name__ == "__main__":
    main()
