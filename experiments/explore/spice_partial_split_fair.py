"""Fair partial-expert CPU/GPU split microbench (addresses codex fairness critiques).
公平的"部分专家 CPU/GPU 切分"微基准 (回应 codex 的公平性质疑).

Question / 研究问题:
  Direction B claims: for a MISSED (non-resident) MoE expert, COMPUTING the non-resident
  segment on CPU (DRAM ~200GB/s) beats FETCHING it over PCIe (~21.6GB/s, MoEpic-style).
  方向 B 主张: 对未命中 (非常驻) 的专家, 在 CPU 上"计算"非常驻段, 优于通过 PCIe "搬运"该段权重.

SwiGLU expert (3 matrices) / SwiGLU 专家 (3 个矩阵):
  a = h @ Wg^T ; b = h @ Wu^T ; m = silu(a) * b ; y = m @ Wd^T
  Wg,Wu: (d_inter, d_model) ; Wd: (d_model, d_inter)

Fairness fixes over spice_partial_split_microbench.py / 相对旧版的公平性修复:
  1. SAME-PRECISION: --cpu_dtype {bf16,fp32}, run BOTH; plus a CPU bf16-vs-fp32 GEMV probe.
     同精度: CPU 数据类型可选, 两者都跑; 并单独探测 CPU bf16 与 fp32 GEMV 的速度差异.
  2. CORRECTNESS: GPU-resident copy and CPU copy share the SAME weight values (only device/dtype
     differ); verify max-abs-error of split-result vs full-GPU-result.
     正确性: GPU 常驻副本与 CPU 副本数值完全一致, 校验切分结果与整专家 GPU 结果的最大绝对误差.
  3. FAIR MoEpic: overlap bottom-segment H2D fetch with resident gate/up GPU compute via a
     separate CUDA stream + events; weight pre-converted & pinned ONCE outside the timed loop.
     公平 MoEpic: 用独立 CUDA 流 + 事件让底段 H2D 搬运与常驻 gate/up 计算重叠; 权重在计时循环外预转换并锁页.
  4. PINNED transfers reused across iters / 复用锁页缓冲做 D2H(m)/H2D(y) 传输.
  5. CAPACITY MODEL: at matched HBM budget, compute experts-that-fit and EXPECTED per-token serve
     cost (TPOT) under the REAL routed-miss distribution from the trace.
     容量模型: 在相同 HBM 预算下, 计算可常驻专家数及在真实路由未命中分布下的期望每 token 服务成本.

Paths / 对比路径 (per unique expert AND at matched-HBM-budget):
  full_cpu               : Fiddler -- CPU computes whole expert. 基线.
  split_gateup_resident  : GPU resident Wg,Wu -> m ; CPU computes y = m@Wd^T.
  split_down_resident    : GPU resident Wd ; CPU computes m ; GPU computes y.
  moepic_fetch_bottom    : GPU resident Wg,Wu ; FETCH Wd over PCIe (overlapped); GPU computes all.
  whole_expert_gpu_cache : expert fully resident on GPU (a HIT, no CPU/transfer). 上界参考.

All printed logs English; comments bilingual; no emojis; core params no defaults (--cpu_dtype defaults bf16).
"""
import argparse
import glob
import json
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="Fair partial-expert CPU/GPU split microbench")
    ap.add_argument("--gpu", type=int, required=True, help="GPU index (use 2 on server)")
    ap.add_argument("--d_model", type=int, required=True, help="hidden dim, Qwen=2048")
    ap.add_argument("--d_inter", type=int, required=True, help="expert intermediate dim, Qwen=1408")
    ap.add_argument("--iters", type=int, required=True, help="timed iterations per path")
    ap.add_argument("--cpu_threads", type=int, required=True, help="torch CPU threads")
    ap.add_argument("--trace_dir", type=str, required=True, help="dir of dec_*.pt routing traces")
    ap.add_argument("--out", type=str, required=True, help="output json path")
    ap.add_argument("--cpu_dtype", type=str, default="bf16", choices=["bf16", "fp32"],
                    help="CPU compute dtype; run is repeated for both regardless, this sets primary")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# Weight bytes per path (resident on HBM) / 各路径常驻 HBM 的权重字节数
# Wg,Wu = (d_inter, d_model) ; Wd = (d_model, d_inter). bf16 = 2 bytes/elem on GPU.
# ----------------------------------------------------------------------------
def expert_byte_model(d_model, d_inter, gpu_bytes_per_elem=2):
    n_gateup = 2 * d_inter * d_model
    n_down = d_model * d_inter
    n_full = n_gateup + n_down
    return {
        "gateup_elems": n_gateup,
        "down_elems": n_down,
        "full_elems": n_full,
        "full_cpu_resident_bytes": 0,                              # Fiddler keeps nothing on HBM
        "split_gateup_resident_bytes": n_gateup * gpu_bytes_per_elem,
        "split_down_resident_bytes": n_down * gpu_bytes_per_elem,
        "moepic_fetch_bottom_bytes": n_gateup * gpu_bytes_per_elem,  # resident gate/up; down fetched
        "whole_expert_gpu_cache_bytes": n_full * gpu_bytes_per_elem,
    }


# ----------------------------------------------------------------------------
# CPU bf16-vs-fp32 GEMV probe / CPU bf16 与 fp32 GEMV 速度探测
# Mirrors the down-projection GEMV (1 x d_inter) @ (d_inter x d_model)^T.
# ----------------------------------------------------------------------------
def probe_cpu_gemv(d_model, d_inter, iters):
    out = {}
    for name, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        W = torch.randn(d_model, d_inter, dtype=dt)
        x = torch.randn(1, d_inter, dtype=dt)
        for _ in range(10):
            _ = F.linear(x, W)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = F.linear(x, W)
        out[f"cpu_down_gemv_{name}_ms"] = (time.perf_counter() - t0) / iters * 1000.0
    out["cpu_bf16_vs_fp32_slowdown"] = out["cpu_down_gemv_bf16_ms"] / out["cpu_down_gemv_fp32_ms"]
    return out


# ----------------------------------------------------------------------------
# Build one consistent expert (shared values across CPU/GPU copies) / 构造数值一致的单个专家
# ----------------------------------------------------------------------------
def build_expert(d_model, d_inter, dev, cpu_dt):
    gpu_dt = torch.bfloat16
    # Master values in fp32 on host, then derive all copies so split == full numerically.
    # 主值在 host fp32 上生成, 由它派生所有副本, 保证切分与整专家数值一致.
    # Scale by 1/sqrt(fan_in) so activations are O(1) (realistic) -> errors are meaningful.
    # 用 1/sqrt(fan_in) 缩放, 使激活量级为 O(1) (贴近真实), 误差量级才有意义.
    Wg32 = torch.randn(d_inter, d_model) / (d_model ** 0.5)
    Wu32 = torch.randn(d_inter, d_model) / (d_model ** 0.5)
    Wd32 = torch.randn(d_model, d_inter) / (d_inter ** 0.5)
    # GPU-resident copies (bf16) derived from the SAME master / GPU 常驻副本 (bf16), 同源.
    WgG = Wg32.to(dev, dtype=gpu_dt)
    WuG = Wu32.to(dev, dtype=gpu_dt)
    WdG = Wd32.to(dev, dtype=gpu_dt)
    # CPU copies in the chosen cpu_dtype, from the SAME master / CPU 副本, 取所选精度, 同源.
    WgC = Wg32.to(dtype=cpu_dt)
    WuC = Wu32.to(dtype=cpu_dt)
    WdC = Wd32.to(dtype=cpu_dt)
    return {
        "Wg32": Wg32, "Wu32": Wu32, "Wd32": Wd32,
        "WgG": WgG, "WuG": WuG, "WdG": WdG,
        "WgC": WgC, "WuC": WuC, "WdC": WdC,
    }


# ----------------------------------------------------------------------------
# Correctness: split result vs full-GPU result max-abs-error / 正确性: 切分 vs 整专家 GPU 的最大绝对误差
# ----------------------------------------------------------------------------
def check_correctness(W, h, dev, cpu_dt):
    """Two precision-honest checks / 两个精度诚实的校验:
    (A) exact: all-fp32 split vs all-fp32 whole-expert -> must be ~0 (proves SAME expert math).
        精确: 全 fp32 切分 vs 全 fp32 整专家 -> 必须 ~0 (证明切分计算的是同一个专家).
    (B) deployed: bf16-GPU+cpu_dt-CPU split vs bf16-GPU whole-expert reference (the actual gap
        a deployment would see, dominated by bf16 rounding, NOT by the split itself).
        部署态: bf16-GPU+cpu_dt-CPU 切分 vs bf16-GPU 整专家参考 (部署实际差距, 由 bf16 舍入主导, 非切分引入).
    """
    gpu_dt = torch.bfloat16

    # (A) exact all-fp32 reference and splits / (A) 全 fp32 参考与切分.
    h32 = h.float().cpu()
    Wg, Wu, Wd = W["Wg32"], W["Wu32"], W["Wd32"]
    m32 = F.silu(F.linear(h32, Wg)) * F.linear(h32, Wu)
    y32_full = F.linear(m32, Wd)
    y32_split_gu = F.linear(m32, Wd)          # gate/up on host then down on host (same numbers)
    err_exact = float((y32_split_gu - y32_full).abs().max())

    # (B) deployed reference: whole expert on GPU bf16 / (B) 部署参考: 整专家 GPU bf16.
    m_bf = F.silu(F.linear(h, W["WgG"])) * F.linear(h, W["WuG"])
    y_ref = F.linear(m_bf, W["WdG"]).float().cpu()
    rel = y_ref.abs().mean().clamp_min(1e-9)
    # split_gateup_resident: GPU m (bf16) -> CPU down (cpu_dt) / GPU 算 m, CPU 算 down.
    m_cpu = m_bf.to(dtype=cpu_dt).cpu()
    y_split_gu = F.linear(m_cpu, W["WdC"]).float()
    # split_down_resident: CPU m (cpu_dt) -> GPU down (bf16) / CPU 算 m, GPU 算 down.
    h_cpu = h.to(dtype=cpu_dt).cpu()
    m_cpu2 = F.silu(F.linear(h_cpu, W["WgC"])) * F.linear(h_cpu, W["WuC"])
    y_split_dn = F.linear(m_cpu2.to(dev, dtype=gpu_dt), W["WdG"]).float().cpu()
    return {
        "err_exact_fp32_split_vs_full": err_exact,
        "relerr_deployed_split_gateup": float((y_split_gu - y_ref).abs().max() / rel),
        "relerr_deployed_split_down": float((y_split_dn - y_ref).abs().max() / rel),
        "ref_abs_mean": float(y_ref.abs().mean()),
    }


# ----------------------------------------------------------------------------
# Per-expert path latencies / 单专家各路径延迟
# ----------------------------------------------------------------------------
def bench_paths(W, d_model, d_inter, dev, cpu_dt, iters):
    gpu_dt = torch.bfloat16
    h = torch.randn(1, d_model, device=dev, dtype=gpu_dt)

    # Reused pinned host buffers / 复用的锁页 host 缓冲.
    m_pin = torch.empty(1, d_inter, dtype=cpu_dt, pin_memory=True)   # D2H intermediate m
    y_pin = torch.empty(1, d_model, dtype=cpu_dt, pin_memory=True)   # H2D output y
    h_pin = torch.empty(1, d_model, dtype=cpu_dt, pin_memory=True)   # D2H input h
    mout_pin = torch.empty(1, d_inter, dtype=cpu_dt, pin_memory=True)  # H2D m (down-resident)

    # MoEpic: pre-convert + pin Wd ONCE, pre-allocate GPU dst ONCE / 预转换并锁页 Wd, 预分配 GPU 目标.
    Wd_host_pin = W["Wd32"].to(dtype=gpu_dt).pin_memory()
    Wd_dst = torch.empty(d_model, d_inter, device=dev, dtype=gpu_dt)
    fetch_stream = torch.cuda.Stream(device=dev)
    y_dev = torch.empty(1, d_model, device=dev, dtype=gpu_dt)

    def bench(fn):
        for _ in range(15):
            fn()
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / iters * 1000.0

    # ---- full_cpu (Fiddler whole expert on CPU) ----
    def full_cpu():
        h_pin.copy_(h, non_blocking=True)            # D2H h (pinned)
        torch.cuda.synchronize(dev)
        hc = h_pin
        m = F.silu(F.linear(hc, W["WgC"])) * F.linear(hc, W["WuC"])
        y = F.linear(m, W["WdC"])                    # whole expert on CPU
        y_pin.copy_(y)
        y_dev.copy_(y_pin, non_blocking=True)        # H2D y (pinned)
        torch.cuda.synchronize(dev)

    # ---- split_gateup_resident (GPU gate/up -> CPU down) ----
    def split_gateup_resident():
        m = F.silu(F.linear(h, W["WgG"])) * F.linear(h, W["WuG"])  # GPU resident
        m_pin.copy_(m.to(dtype=cpu_dt), non_blocking=True)         # D2H m (pinned, tiny)
        torch.cuda.synchronize(dev)
        y = F.linear(m_pin, W["WdC"])                              # CPU reads only Wd
        y_pin.copy_(y)
        y_dev.copy_(y_pin, non_blocking=True)                      # H2D y (pinned)
        torch.cuda.synchronize(dev)

    # ---- split_down_resident (CPU gate/up -> GPU down) ----
    def split_down_resident():
        h_pin.copy_(h.to(dtype=cpu_dt), non_blocking=True)         # D2H h (pinned)
        torch.cuda.synchronize(dev)
        m = F.silu(F.linear(h_pin, W["WgC"])) * F.linear(h_pin, W["WuC"])  # CPU reads Wg,Wu
        mout_pin.copy_(m)
        m_dev = mout_pin.to(dev, dtype=gpu_dt, non_blocking=True)  # H2D m (pinned, tiny)
        _ = F.linear(m_dev, W["WdG"])                              # GPU resident down
        torch.cuda.synchronize(dev)

    # ---- moepic_fetch_bottom (FAIR: overlap Wd H2D with resident gate/up GPU compute) ----
    def moepic_fetch_bottom():
        compute_done = torch.cuda.Event()
        fetch_done = torch.cuda.Event()
        # Kick off the bottom-segment fetch on a separate stream / 在独立流上发起底段权重搬运.
        with torch.cuda.stream(fetch_stream):
            Wd_dst.copy_(Wd_host_pin, non_blocking=True)          # FETCH Wd over PCIe (overlapped)
            fetch_done.record(fetch_stream)
        # Meanwhile do resident gate/up compute on default stream / 同时在默认流做常驻 gate/up 计算.
        m = F.silu(F.linear(h, W["WgG"])) * F.linear(h, W["WuG"])
        compute_done.record()
        # Default stream waits for fetch, then down-proj / 默认流等待搬运完成后做 down 投影.
        torch.cuda.current_stream(dev).wait_event(fetch_done)
        _ = F.linear(m, Wd_dst)
        torch.cuda.synchronize(dev)

    # ---- whole_expert_gpu_cache (resident HIT, no CPU/transfer) ----
    def whole_expert_gpu_cache():
        m = F.silu(F.linear(h, W["WgG"])) * F.linear(h, W["WuG"])
        _ = F.linear(m, W["WdG"])
        torch.cuda.synchronize(dev)

    return {
        "full_cpu_ms": bench(full_cpu),
        "split_gateup_resident_ms": bench(split_gateup_resident),
        "split_down_resident_ms": bench(split_down_resident),
        "moepic_fetch_bottom_ms": bench(moepic_fetch_bottom),
        "whole_expert_gpu_cache_ms": bench(whole_expert_gpu_cache),
    }


# ----------------------------------------------------------------------------
# Load trace + compute per-layer miss counts under an LRU resident cache.
# 加载 trace 并在 LRU 常驻缓存下计算逐层未命中次数.
# Returns total routed slots and miss count given `experts_resident_per_layer`.
# ----------------------------------------------------------------------------
def load_trace(trace_dir):
    files = sorted(glob.glob(f"{trace_dir}/*.pt"))
    if not files:
        raise FileNotFoundError(f"no trace .pt files in {trace_dir}")
    # routes[layer] = list of expert-id lists per decode step / 逐层的路由专家序列.
    seqs = []  # flat list of (layer, [expert_ids]) in decode order, grouped per file/step
    num_layers = None
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        num_layers = d["num_layers"]
        for (_tok, layers) in d["steps"]:
            seqs.append(layers)  # layers: list[num_layers] of list[topk]
    return seqs, num_layers


def trace_miss_stats(seqs, num_layers, experts_resident_per_layer):
    """LRU per layer; count routed slots and misses / 逐层 LRU, 统计路由槽位与未命中数."""
    caches = [OrderedDict() for _ in range(num_layers)]
    cap = experts_resident_per_layer
    total_slots = 0
    total_miss = 0
    for layers in seqs:
        for li, experts in enumerate(layers):
            c = caches[li]
            for e in experts:
                total_slots += 1
                if e in c:
                    c.move_to_end(e)
                else:
                    total_miss += 1
                    if cap > 0:
                        c[e] = True
                        if len(c) > cap:
                            c.popitem(last=False)
    return {"total_slots": total_slots, "total_miss": total_miss,
            "hit_rate": (total_slots - total_miss) / total_slots if total_slots else 0.0}


# ----------------------------------------------------------------------------
# Capacity model: at matched HBM budget B (MB), per path compute experts-that-fit,
# miss rate, and EXPECTED per-token (per-layer-amortized) serve cost.
# 容量模型: 给定 HBM 预算 B, 逐路径算可常驻专家数, 未命中率与期望每 token 服务成本.
# ----------------------------------------------------------------------------
def capacity_model(latencies, byte_model, seqs, num_layers, budgets_mb, num_experts_per_layer):
    gpu_dt_bytes = 2
    full_expert_bytes = byte_model["full_elems"] * gpu_dt_bytes
    gateup_bytes = byte_model["gateup_elems"] * gpu_dt_bytes

    # Per-path: (resident bytes per cached unit, serve-cost on hit, serve-cost on miss).
    # 各路径: (每个缓存单元常驻字节, 命中服务成本, 未命中服务成本).
    # full_cpu: nothing resident -> every routed slot pays full_cpu (no "hit" concept).
    # split_*: the resident SEGMENT is shared across ALL experts of a layer? No -- segment is
    #   per-expert, so caching K experts' segments costs K*segment_bytes; a routed expert whose
    #   segment is resident pays split-cost, else falls back to full_cpu.
    # moepic_fetch_bottom: same, resident gate/up per cached expert; bottom always fetched.
    # whole_expert_gpu_cache: cache K whole experts; hit pays resident cost, miss pays full_cpu.
    paths = {
        "full_cpu": {"unit_bytes": None, "hit_ms": None,
                     "miss_ms": latencies["full_cpu_ms"], "always_cost": latencies["full_cpu_ms"]},
        "split_gateup_resident": {"unit_bytes": gateup_bytes,
                                  "hit_ms": latencies["split_gateup_resident_ms"],
                                  "miss_ms": latencies["full_cpu_ms"]},
        "split_down_resident": {"unit_bytes": byte_model["down_elems"] * gpu_dt_bytes,
                                "hit_ms": latencies["split_down_resident_ms"],
                                "miss_ms": latencies["full_cpu_ms"]},
        "moepic_fetch_bottom": {"unit_bytes": gateup_bytes,
                                "hit_ms": latencies["moepic_fetch_bottom_ms"],
                                "miss_ms": latencies["full_cpu_ms"]},
        "whole_expert_gpu_cache": {"unit_bytes": full_expert_bytes,
                                   "hit_ms": latencies["whole_expert_gpu_cache_ms"],
                                   "miss_ms": latencies["full_cpu_ms"]},
    }

    results = []
    for B_mb in budgets_mb:
        B_bytes = B_mb * 1024 * 1024
        row = {"budget_mb": B_mb}
        for pname, p in paths.items():
            if pname == "full_cpu":
                # No residency; expected cost = full_cpu per routed slot. / 无常驻, 每槽位均为 full_cpu.
                row[pname] = {"experts_fit_per_layer": 0, "hit_rate": 0.0,
                              "expected_ms_per_routed_slot": p["miss_ms"]}
                continue
            # Budget shared across num_layers layers / 预算在各层均分.
            bytes_per_layer = B_bytes / num_layers
            k = int(bytes_per_layer // p["unit_bytes"])
            k = max(0, min(k, num_experts_per_layer))
            stats = trace_miss_stats(seqs, num_layers, k)
            hr = stats["hit_rate"]
            exp_ms = hr * p["hit_ms"] + (1.0 - hr) * p["miss_ms"]
            row[pname] = {"experts_fit_per_layer": k, "hit_rate": hr,
                          "expected_ms_per_routed_slot": exp_ms}
        results.append(row)
    return results


def main():
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)
    dm, di = a.d_model, a.d_inter
    num_experts_per_layer = 60  # Qwen1.5-MoE / Qwen MoE: 60 routed experts per layer

    print(f"[setup] device=cuda:{a.gpu} d_model={dm} d_inter={di} iters={a.iters} "
          f"cpu_threads={a.cpu_threads}", flush=True)

    byte_model = expert_byte_model(dm, di)
    print(f"[bytes] full_expert={byte_model['full_elems']*2/1e6:.2f}MB "
          f"gateup={byte_model['gateup_elems']*2/1e6:.2f}MB "
          f"down={byte_model['down_elems']*2/1e6:.2f}MB (bf16 on HBM)", flush=True)

    # (a) CPU bf16 vs fp32 GEMV probe / CPU bf16 与 fp32 GEMV 探测.
    gemv = probe_cpu_gemv(dm, di, a.iters)
    print(f"[cpu-gemv] fp32={gemv['cpu_down_gemv_fp32_ms']:.4f}ms "
          f"bf16={gemv['cpu_down_gemv_bf16_ms']:.4f}ms "
          f"bf16/fp32 slowdown={gemv['cpu_bf16_vs_fp32_slowdown']:.2f}x", flush=True)

    # Trace once (shared across dtype runs) / trace 仅加载一次, 各精度共用.
    print(f"[trace] loading {a.trace_dir} ...", flush=True)
    seqs, num_layers = load_trace(a.trace_dir)
    print(f"[trace] steps={len(seqs)} layers={num_layers} "
          f"experts/layer={num_experts_per_layer}", flush=True)

    # HBM budgets to sweep (MB) / 扫描的 HBM 预算 (MB).
    # full expert ~ 11.6MB bf16 * 60 experts * 24 layers ~ 16.7GB to fully cache all.
    budgets_mb = [256, 512, 1024, 2048, 4096, 8192, 16384]

    out = {"config": {"d_model": dm, "d_inter": di, "iters": a.iters,
                      "cpu_threads": a.cpu_threads, "num_layers": num_layers,
                      "num_experts_per_layer": num_experts_per_layer,
                      "budgets_mb": budgets_mb},
           "byte_model": byte_model,
           "cpu_gemv_probe": gemv,
           "per_dtype": {}}

    for cpu_name, cpu_dt in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        print(f"\n===== CPU dtype = {cpu_name} =====", flush=True)
        W = build_expert(dm, di, dev, cpu_dt)
        h_chk = torch.randn(1, dm, device=dev, dtype=torch.bfloat16)
        corr = check_correctness(W, h_chk, dev, cpu_dt)
        print(f"[correctness] exact_fp32_split_vs_full={corr['err_exact_fp32_split_vs_full']:.3e} "
              f"(must be ~0) | deployed_relerr split_gateup="
              f"{corr['relerr_deployed_split_gateup']:.3e} "
              f"split_down={corr['relerr_deployed_split_down']:.3e} "
              f"(bf16-dominated; ref|mean|={corr['ref_abs_mean']:.3e})", flush=True)

        lat = bench_paths(W, dm, di, dev, cpu_dt, a.iters)
        base = lat["full_cpu_ms"]
        speedups = {k.replace("_ms", "_speedup_vs_fullcpu"): base / v for k, v in lat.items()}
        print(f"[per-expert {cpu_name}] full_cpu={lat['full_cpu_ms']:.4f}ms | "
              f"split_gateup={lat['split_gateup_resident_ms']:.4f}ms "
              f"({speedups['split_gateup_resident_speedup_vs_fullcpu']:.2f}x) | "
              f"split_down={lat['split_down_resident_ms']:.4f}ms "
              f"({speedups['split_down_resident_speedup_vs_fullcpu']:.2f}x) | "
              f"moepic={lat['moepic_fetch_bottom_ms']:.4f}ms "
              f"({speedups['moepic_fetch_bottom_speedup_vs_fullcpu']:.2f}x) | "
              f"whole_gpu={lat['whole_expert_gpu_cache_ms']:.4f}ms", flush=True)

        cap = capacity_model(lat, byte_model, seqs, num_layers, budgets_mb, num_experts_per_layer)
        print(f"[capacity {cpu_name}] expected ms per routed slot at matched HBM budget:", flush=True)
        for row in cap:
            b = row["budget_mb"]
            fc = row["full_cpu"]["expected_ms_per_routed_slot"]
            sg = row["split_gateup_resident"]
            md = row["moepic_fetch_bottom"]
            wc = row["whole_expert_gpu_cache"]
            print(f"  B={b:>6}MB | full_cpu={fc:.4f} | "
                  f"split_gateup(k={sg['experts_fit_per_layer']},hr={sg['hit_rate']:.2f})="
                  f"{sg['expected_ms_per_routed_slot']:.4f} | "
                  f"moepic(k={md['experts_fit_per_layer']},hr={md['hit_rate']:.2f})="
                  f"{md['expected_ms_per_routed_slot']:.4f} | "
                  f"whole_gpu(k={wc['experts_fit_per_layer']},hr={wc['hit_rate']:.2f})="
                  f"{wc['expected_ms_per_routed_slot']:.4f}", flush=True)

        out["per_dtype"][cpu_name] = {"correctness": corr, "latencies_ms": lat,
                                      "speedups_vs_fullcpu": speedups, "capacity": cap}

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
