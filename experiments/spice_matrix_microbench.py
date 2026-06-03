"""Matrix-granular exact residency microbenchmark (user-directed, A800, batch=1, EXACT same-precision).

A SwiGLU expert is NOT atomic: E(h) = W_down( silu(W_gate h) * (W_up h) ). Decompose the fetch DAG
into exact matrix-level predecessors. Key question: does keeping W_gate/W_up resident and fetching
ONLY W_down let stage1 GPU compute OVERLAP the W_down H2D -> a cheaper exact miss than whole-expert
fetch (0.78ms) and competitive with Fiddler CPU-serve (0.167ms) but CACHEABLE?

Measures real edges (CUDA events):
  T_H2D(W_gate), T_H2D(W_up), T_H2D(W_down)          (each ~5.77MB)
  T_stage1 (W_gate/W_up resident -> intermediate)
  T_stage2 (W_down resident -> output)
  T_overlap(stage1 || H2D(W_down))                   (the crux: does fetch hide behind compute?)
  whole-expert path vs prefix-resident path vs suffix-resident path vs cpu_fiddler critical path
  grouped top-k stage1/stage2 (launch-overhead amortization)
All printed English. Core params: no defaults.
"""
import argparse, time, json
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="Matrix-granular exact residency microbenchmark")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def ev(fn, iters, dev):
    for _ in range(5): fn()
    torch.cuda.synchronize(dev)
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(dev)
    return s.elapsed_time(e) / iters


def main():
    a = parse_args()
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev); dt = torch.bfloat16
    dm, di = a.d_model, a.d_inter
    res = {"d_model": dm, "d_inter": di, "top_k": a.top_k}

    Wg = torch.randn(di, dm, device=dev, dtype=dt)   # gate
    Wu = torch.randn(di, dm, device=dev, dtype=dt)   # up
    Wd = torch.randn(dm, di, device=dev, dtype=dt)   # down
    mat_mb = Wg.numel() * 2 / 1e6
    res["matrix_mb"] = mat_mb; res["expert_mb"] = 3 * mat_mb
    h = torch.randn(1, dm, device=dev, dtype=dt)

    hg = Wg.detach().cpu().contiguous().pin_memory()
    hu = Wu.detach().cpu().contiguous().pin_memory()
    hd = Wd.detach().cpu().contiguous().pin_memory()
    dg = torch.empty_like(Wg); du = torch.empty_like(Wu); dd = torch.empty_like(Wd)

    def stage1(x, wg, wu): return F.silu(F.linear(x, wg)) * F.linear(x, wu)   # -> (1, di)
    def stage2(z, wd): return F.linear(z, wd)                                  # -> (1, dm)

    z = stage1(h, Wg, Wu)
    res["t_stage1_ms"] = ev(lambda: stage1(h, Wg, Wu), a.iters, dev)
    res["t_stage2_ms"] = ev(lambda: stage2(z, Wd), a.iters, dev)
    res["t_whole_expert_ms"] = ev(lambda: stage2(stage1(h, Wg, Wu), Wd), a.iters, dev)

    res["t_h2d_wgate_ms"] = ev(lambda: dg.copy_(hg, non_blocking=True), a.iters, dev)
    res["t_h2d_wup_ms"] = ev(lambda: du.copy_(hu, non_blocking=True), a.iters, dev)
    res["t_h2d_wdown_ms"] = ev(lambda: dd.copy_(hd, non_blocking=True), a.iters, dev)
    res["t_h2d_whole_ms"] = ev(lambda: (dg.copy_(hg, non_blocking=True), du.copy_(hu, non_blocking=True),
                                        dd.copy_(hd, non_blocking=True)), a.iters, dev)

    # CRUX: stage1 GPU compute on compute stream || H2D(W_down) on copy stream -- do they overlap?
    cs = torch.cuda.Stream(dev); ps = torch.cuda.Stream(dev)
    def overlap_prefix():
        # prefix-resident: Wg/Wu resident -> stage1 (compute) while fetching Wd (copy), then stage2
        torch.cuda.current_stream(dev).wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(ps):
            dd.copy_(hd, non_blocking=True)
        with torch.cuda.stream(cs):
            zz = stage1(h, Wg, Wu)
        cs.synchronize(); ps.synchronize()
        return stage2(zz, dd)
    res["t_prefix_overlap_ms"] = ev(overlap_prefix, a.iters, dev)
    # serial reference (no overlap): H2D(Wd) then stage1 then stage2
    def prefix_serial():
        dd.copy_(hd, non_blocking=True); torch.cuda.current_stream(dev).synchronize()
        return stage2(stage1(h, Wg, Wu), dd)
    res["t_prefix_serial_ms"] = ev(prefix_serial, a.iters, dev)

    # suffix-resident: Wd resident -> must fetch Wg/Wu BEFORE stage1
    def suffix_path():
        with torch.cuda.stream(ps):
            dg.copy_(hg, non_blocking=True); du.copy_(hu, non_blocking=True)
        ps.synchronize()
        return stage2(stage1(h, dg, du), Wd)
    res["t_suffix_ms"] = ev(suffix_path, a.iters, dev)

    # whole-expert miss path (fetch all 3 then compute)
    def whole_path():
        dg.copy_(hg, non_blocking=True); du.copy_(hu, non_blocking=True); dd.copy_(hd, non_blocking=True)
        torch.cuda.current_stream(dev).synchronize()
        return stage2(stage1(h, dg, du), dd)
    res["t_whole_path_ms"] = ev(whole_path, a.iters, dev)

    # grouped top-k stage compute (launch-overhead amortization): batch top_k experts' stage1 as one GEMM
    Wg_g = torch.randn(a.top_k * di, dm, device=dev, dtype=dt)
    Wd_g = torch.randn(a.top_k, dm, di, device=dev, dtype=dt)
    def grouped_stage1():  # one (1,dm)x(top_k*di,dm)^T GEMM
        return F.linear(h, Wg_g)
    res["t_grouped_stage1_ms"] = ev(grouped_stage1, a.iters, dev)
    res["t_per_expert_stage1_x_topk_ms"] = res["t_stage1_ms"] * a.top_k

    # critical-path summary
    cpu_fiddler = 0.167
    summary = {
        "whole_expert_cache_miss_ms": res["t_h2d_whole_ms"] + res["t_whole_expert_ms"],
        "prefix_resident_miss_ms": res["t_prefix_overlap_ms"],
        "suffix_resident_miss_ms": res["t_suffix_ms"],
        "cpu_fiddler_miss_ms": cpu_fiddler,
        "resident_hit_ms": res["t_whole_expert_ms"],
        "prefix_vs_whole_speedup": (res["t_h2d_whole_ms"] + res["t_whole_expert_ms"]) / max(1e-9, res["t_prefix_overlap_ms"]),
        "prefix_overlap_benefit_ms": res["t_prefix_serial_ms"] - res["t_prefix_overlap_ms"],
    }
    res["summary"] = summary
    print(json.dumps(res, indent=2), flush=True)
    with open(a.out, "w") as f: json.dump(res, f, indent=2)
    print(f"\n[matrix edges] matrix={mat_mb:.2f}MB H2D each ~{res['t_h2d_wdown_ms']:.3f}ms; "
          f"stage1={res['t_stage1_ms']:.3f} stage2={res['t_stage2_ms']:.3f}", flush=True)
    print(f"[miss cost] whole={summary['whole_expert_cache_miss_ms']:.3f} prefix={summary['prefix_resident_miss_ms']:.3f} "
          f"suffix={summary['suffix_resident_miss_ms']:.3f} fiddler={cpu_fiddler:.3f} | "
          f"prefix_overlap_saves={summary['prefix_overlap_benefit_ms']:.3f}ms", flush=True)


if __name__ == "__main__":
    main()
