"""Partial-expert CPU/GPU split execution microbench (user priority #1; EXACT same-precision, batch=1).
Differs from MoEpic (arXiv 2509.08342, which caches top segment + FETCHES bottom over PCIe): here the
NON-resident segment is COMPUTED on CPU (DRAM 200GB/s) instead of fetched (PCIe 21.6GB/s). SwiGLU:
  a=h@Wg, b=h@Wu, m=silu(a)*b, y=m@Wd
Paths (per missed expert, EXACT):
  full_cpu      : Fiddler -- CPU computes whole expert (reads 17.3MB DRAM). Baseline.
  split_gateup_resident : Wg,Wu resident on GPU -> GPU computes m; D2H m; CPU computes y=m@Wd (reads 5.77MB); H2D y.
  split_down_resident   : Wd resident on GPU; CPU computes m (reads 11.5MB Wg,Wu); H2D m; GPU computes y.
  moepic_fetch_bottom   : Wg,Wu resident; FETCH Wd over PCIe (0.26ms); GPU computes all (MoEpic-style).
Go line (user): split path beats full_cpu by >=15% per-expert latency at same HBM budget.
Real CUDA + CPU ops, CUDA events / perf_counter. All printed English. Core params: no defaults.
"""
import argparse, time, json
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="Partial-expert CPU/GPU split microbench")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--cpu_threads", type=int, required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev); dt = torch.bfloat16
    dm, di = a.d_model, a.d_inter

    # GPU-resident matrices (bf16) + CPU matrices (fp32, Fiddler-style upcast) + pinned for transfers
    WgG = torch.randn(di, dm, device=dev, dtype=dt); WuG = torch.randn(di, dm, device=dev, dtype=dt)
    WdG = torch.randn(dm, di, device=dev, dtype=dt)
    WgC = torch.randn(di, dm); WuC = torch.randn(di, dm); WdC = torch.randn(dm, di)
    WdC_pin = WdC.clone().pin_memory()  # for moepic fetch
    Wd_dst = torch.empty(dm, di, device=dev, dtype=dt)
    h = torch.randn(1, dm, device=dev, dtype=dt)
    m_pin = torch.empty(1, di, pin_memory=True)        # D2H intermediate
    y_pin = torch.empty(1, dm, pin_memory=True)        # H2D output
    m_dev = torch.empty(1, di, device=dev, dtype=dt)

    def bench_gpu(fn):
        for _ in range(10): fn()
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(a.iters): fn()
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / a.iters * 1000.0

    # full_cpu (Fiddler): D2H h -> CPU whole expert -> H2D y
    hc = h.float().cpu()
    def full_cpu():
        h_c = h.float().cpu()                                   # real D2H
        y = F.linear(F.silu(F.linear(h_c, WgC)) * F.linear(h_c, WuC), WdC)  # reads 17.3MB DRAM
        _ = y.to(dev, dtype=dt)                                 # real H2D
        torch.cuda.synchronize(dev)

    def split_gateup_resident():
        m = F.silu(F.linear(h, WgG)) * F.linear(h, WuG)         # GPU on resident Wg,Wu -> m
        m_c = m.float().cpu()                                    # D2H m (tiny)
        y = F.linear(m_c, WdC)                                   # CPU reads only Wd 5.77MB
        _ = y.to(dev, dtype=dt)                                  # H2D y (tiny)
        torch.cuda.synchronize(dev)

    def split_down_resident():
        h_c = h.float().cpu()                                    # D2H h
        m = F.silu(F.linear(h_c, WgC)) * F.linear(h_c, WuC)     # CPU reads Wg,Wu 11.5MB -> m
        m_d = m.to(dev, dtype=dt)                                # H2D m (tiny)
        _ = F.linear(m_d, WdG)                                   # GPU on resident Wd
        torch.cuda.synchronize(dev)

    def moepic_fetch_bottom():
        m = F.silu(F.linear(h, WgG)) * F.linear(h, WuG)         # GPU resident Wg,Wu
        Wd_dst.copy_(WdC_pin.to(dt), non_blocking=True)         # FETCH Wd over PCIe (5.77MB)
        _ = F.linear(m, Wd_dst)                                 # GPU computes y
        torch.cuda.synchronize(dev)

    res = {"d_model": dm, "d_inter": di,
           "full_cpu_ms": bench_gpu(full_cpu),
           "split_gateup_resident_ms": bench_gpu(split_gateup_resident),
           "split_down_resident_ms": bench_gpu(split_down_resident),
           "moepic_fetch_bottom_ms": bench_gpu(moepic_fetch_bottom)}
    base = res["full_cpu_ms"]
    for k in ["split_gateup_resident_ms", "split_down_resident_ms", "moepic_fetch_bottom_ms"]:
        res[k.replace("_ms", "_speedup")] = base / res[k]
    with open(a.out, "w") as f: json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print(f"\n[partial-split] full_cpu(Fiddler)={base:.4f}ms | "
          f"gate/up-resident+CPU-down={res['split_gateup_resident_ms']:.4f}ms "
          f"({res['split_gateup_resident_speedup']:.2f}x) | "
          f"down-resident+CPU-gateup={res['split_down_resident_ms']:.4f}ms "
          f"({res['split_down_resident_speedup']:.2f}x) | moepic-fetch={res['moepic_fetch_bottom_ms']:.4f}ms", flush=True)
    print(f"[verdict] split beats full_cpu by >=15%? gate/up:{res['split_gateup_resident_speedup']>=1.15} "
          f"down:{res['split_down_resident_speedup']>=1.15}", flush=True)


if __name__ == "__main__":
    main()
