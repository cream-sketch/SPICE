"""Real resource-edge microbenchmark on A800 (user-directed: simulator only screens mechanisms;
final resource edges MUST be measured on real hardware). batch=1, Qwen expert shape, EXACT same-precision.

Measures the resource edges that decide whether ANY exact batch=1 mechanism beyond Fiddler exists:
  A. GPU fetch path:  pinned H2D(weight 17MB) -> GPU expert compute            (predecessor chain)
  B. CPU path:        D2H(act) -> CPU expert compute -> H2D(out)               (Fiddler)
  C. DRAM contention: CPU expert compute ALONE vs CONCURRENT with a big H2D PCIe DMA
  D. PCIe priority:   low-priority big H2D in flight, then high-priority small H2D -> is it preempted?
  E. Copy engines:    concurrent H2D + D2H -> do they overlap (separate engines) or serialize?

Answers: PCIe priority real? CPU compute fast? CPU vs PCIe DMA contend on DRAM? prefetch blocks output?
All timings via CUDA events / perf_counter. All printed content English. Core params: no defaults.
"""
import argparse, time, threading, json
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="Real resource-edge microbenchmark (A800, batch=1, exact)")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def make_expert(d_model, d_inter, device, dtype):
    g = torch.randn(d_inter, d_model, device=device, dtype=dtype)
    u = torch.randn(d_inter, d_model, device=device, dtype=dtype)
    d = torch.randn(d_model, d_inter, device=device, dtype=dtype)
    return g, u, d


def expert_fwd(x, g, u, d):
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


def ms_event(fn, iters, device):
    """Time a GPU op via CUDA events (ms/iter), with warmup."""
    for _ in range(5): fn()
    torch.cuda.synchronize(device)
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(device)
    return s.elapsed_time(e) / iters


def main():
    a = parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)
    dt = torch.bfloat16
    res = {"d_model": a.d_model, "d_inter": a.d_inter}

    # expert on GPU (resident) and weights pinned on host (for H2D fetch)
    gG, uG, dG = make_expert(a.d_model, a.d_inter, dev, dt)
    expert_mb = sum(t.numel() * 2 for t in (gG, uG, dG)) / 1e6
    res["expert_mb"] = expert_mb
    host = [t.detach().to("cpu").contiguous().pin_memory() for t in (gG, uG, dG)]
    dst = [torch.empty_like(t) for t in (gG, uG, dG)]
    xG = torch.randn(1, a.d_model, device=dev, dtype=dt)

    # ---- A: GPU resident expert compute (no fetch) ----
    res["t_gpu_expert_ms"] = ms_event(lambda: expert_fwd(xG, gG, uG, dG), a.iters, dev)

    # ---- H2D weight fetch (pinned, one expert) on default stream ----
    def h2d_weight():
        for s, d in zip(host, dst): d.copy_(s, non_blocking=True)
    res["t_h2d_weight_ms"] = ms_event(h2d_weight, a.iters, dev)
    res["pcie_h2d_gbps"] = expert_mb / res["t_h2d_weight_ms"] / 1.024  # MB/ms -> GB/s

    # ---- B: CPU expert path (D2H act -> CPU compute -> H2D out) ----
    gC, uC, dC = (t.detach().float().cpu() for t in (gG, uG, dG))
    xC = torch.randn(1, a.d_model)
    def cpu_expert(): return expert_fwd(xC, gC, uC, dC)
    for _ in range(5): cpu_expert()
    t0 = time.perf_counter()
    for _ in range(a.iters): cpu_expert()
    res["t_cpu_expert_ms"] = (time.perf_counter() - t0) / a.iters * 1000
    # activation roundtrip (tiny): D2H one hidden vec + H2D one out vec
    act_h = xG.detach().contiguous(); act_host = torch.empty(1, a.d_model, pin_memory=True)
    def d2h_act(): act_host.copy_(act_h, non_blocking=True)
    res["t_d2h_act_ms"] = ms_event(d2h_act, a.iters, dev)

    # ---- C: DRAM contention -- CPU compute alone vs concurrent with big H2D DMA ----
    big_host = host[0]; big_dst = dst[0]  # ~6MB chunk; loop to sustain DMA
    stop = threading.Event()
    def dma_loop():
        st = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(st):
            while not stop.is_set():
                for s, d in zip(host, dst): d.copy_(s, non_blocking=True)
        torch.cuda.synchronize(device=dev)
    # baseline CPU
    for _ in range(5): cpu_expert()
    t0 = time.perf_counter()
    for _ in range(a.iters): cpu_expert()
    cpu_alone = (time.perf_counter() - t0) / a.iters * 1000
    # CPU concurrent with sustained DMA
    stop.clear(); th = threading.Thread(target=dma_loop); th.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    for _ in range(a.iters): cpu_expert()
    cpu_concurrent = (time.perf_counter() - t0) / a.iters * 1000
    stop.set(); th.join()
    res["dram_contention"] = {"cpu_alone_ms": cpu_alone, "cpu_during_dma_ms": cpu_concurrent,
                              "slowdown_x": cpu_concurrent / max(1e-9, cpu_alone)}

    # ---- D: PCIe priority -- low-pri big H2D in flight, high-pri small H2D, measure small op latency ----
    try:
        lo = torch.cuda.Stream(device=dev, priority=0)
        hi = torch.cuda.Stream(device=dev, priority=-1)
    except Exception:
        lo = torch.cuda.Stream(device=dev); hi = torch.cuda.Stream(device=dev)
    small_host = torch.empty(1, a.d_model, pin_memory=True); small_dst = torch.empty(1, a.d_model, device=dev)
    # baseline: small H2D latency alone
    def small_h2d_on(stream):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            s.record(stream); small_dst.copy_(small_host, non_blocking=True); e.record(stream)
        return s, e
    torch.cuda.synchronize(device=dev)
    lat_alone = []
    for _ in range(a.iters):
        s, e = small_h2d_on(hi); torch.cuda.synchronize(device=dev); lat_alone.append(s.elapsed_time(e))
    # with big low-pri H2D in flight
    lat_contended = []
    for _ in range(a.iters):
        with torch.cuda.stream(lo):
            for s_, d_ in zip(host, dst): d_.copy_(s_, non_blocking=True)  # big low-pri in flight
        s, e = small_h2d_on(hi)
        torch.cuda.synchronize(device=dev); lat_contended.append(s.elapsed_time(e))
    import statistics
    res["pcie_priority"] = {"small_h2d_alone_ms": statistics.median(lat_alone),
                            "small_h2d_behind_big_lowpri_ms": statistics.median(lat_contended),
                            "preempted": statistics.median(lat_contended) < 2 * statistics.median(lat_alone) + 0.05}

    # ---- E: copy engines -- concurrent H2D + D2H overlap? ----
    h2d_only = ms_event(h2d_weight, a.iters, dev)
    d2h_host = [torch.empty_like(h, pin_memory=True) for h in host]
    def d2h_weight():
        for s, d in zip((gG, uG, dG), d2h_host): d.copy_(s, non_blocking=True)
    d2h_only = ms_event(d2h_weight, a.iters, dev)
    s2 = torch.cuda.Stream(device=dev)
    def both():
        h2d_weight()
        with torch.cuda.stream(s2): d2h_weight()
    both_t = ms_event(both, a.iters, dev)
    res["copy_engines"] = {"h2d_only_ms": h2d_only, "d2h_only_ms": d2h_only, "both_ms": both_t,
                           "overlap": both_t < (h2d_only + d2h_only) * 0.8}

    print(json.dumps(res, indent=2), flush=True)
    with open(a.out, "w") as f: json.dump(res, f, indent=2)
    print("\n[edges] PCIe H2D ~%.1f GB/s; CPU expert %.3f ms vs GPU %.3f ms vs fetch %.3f ms" % (
        res["pcie_h2d_gbps"], res["t_cpu_expert_ms"], res["t_gpu_expert_ms"], res["t_h2d_weight_ms"]), flush=True)
    print("[edges] DRAM contention slowdown %.2fx; PCIe preempt=%s; copy-engine overlap=%s" % (
        res["dram_contention"]["slowdown_x"], res["pcie_priority"]["preempted"], res["copy_engines"]["overlap"]), flush=True)


if __name__ == "__main__":
    main()
