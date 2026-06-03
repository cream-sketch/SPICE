"""Windowed partial-split microbench: combines Direction B (partial CPU/GPU split: non-resident segment
CPU-computed) with Direction A (window of m positions routing to one expert -> amortize sync + CPU read).
EXACT same-precision (bf16), batch=1 request, m = positions in the speculative window routing to ONE expert.

Per unique expert serving m positions:
  full_cpu          : CPU computes whole expert over m positions (reads 17.3MB DRAM once, grouped).
  split_window      : GPU computes m_int = silu(h@Wg)*(h@Wu) for the m positions on RESIDENT Wg,Wu;
                      ONE D2H of (m,di); CPU computes y=(m,di)@Wd (reads only Wd 5.77MB once); ONE H2D (m,dm).
Key: split CPU reads only 1/3 the weight, and D2H/H2D sync amortizes over m. Sweep m. Go: split beats
full_cpu by >=15% at the typical window reuse (m~2-3 at K=16). Real CUDA+CPU ops. No core defaults.
"""
import argparse, time, json
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="Windowed partial-split microbench")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--ms", type=str, required=True, help="comma list of positions-per-expert window sizes m")
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--cpu_threads", type=int, required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev); dt = torch.bfloat16
    dm, di = a.d_model, a.d_inter
    WgG = torch.randn(di, dm, device=dev, dtype=dt); WuG = torch.randn(di, dm, device=dev, dtype=dt)
    WgC = torch.randn(di, dm, dtype=dt); WuC = torch.randn(di, dm, dtype=dt); WdC = torch.randn(dm, di, dtype=dt)

    def bench(fn):
        for _ in range(10): fn()
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(a.iters): fn()
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / a.iters * 1000.0

    rows = []
    for m in [int(x) for x in a.ms.split(",")]:
        h = torch.randn(m, dm, device=dev, dtype=dt)
        def full_cpu():
            h_c = h.cpu()                                                  # D2H m positions
            y = F.linear(F.silu(F.linear(h_c, WgC)) * F.linear(h_c, WuC), WdC)  # CPU whole expert, reads 17.3MB
            _ = y.to(dev)                                                  # H2D
            torch.cuda.synchronize(dev)
        def split_window():
            mi = F.silu(F.linear(h, WgG)) * F.linear(h, WuG)              # GPU resident Wg,Wu -> (m,di)
            mi_c = mi.cpu()                                                # ONE D2H (m,di)
            y = F.linear(mi_c, WdC)                                        # CPU reads only Wd 5.77MB, grouped over m
            _ = y.to(dev)                                                  # ONE H2D (m,dm)
            torch.cuda.synchronize(dev)
        tf = bench(full_cpu); ts = bench(split_window)
        rows.append({"m": m, "full_cpu_ms": tf, "split_window_ms": ts, "speedup": tf / ts})
        print(f"m={m:>2} full_cpu={tf:.4f}ms split_window={ts:.4f}ms speedup={tf/ts:.2f}x "
              f"{'GO' if tf/ts>=1.15 else 'no'}", flush=True)
    json.dump({"rows": rows}, open(a.out, "w"), indent=2)
    print(f"\n[verdict] partial-split CPU reads 1/3 weight + amortized sync. Best speedup "
          f"{max(r['speedup'] for r in rows):.2f}x", flush=True)


if __name__ == "__main__":
    main()
