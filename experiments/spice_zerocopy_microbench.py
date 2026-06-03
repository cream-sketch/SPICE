"""GPU zero-copy remote expert GEMV microbench (user-directed hardware path we missed).
EXACT same-precision, batch=1. The GPU kernel reads expert weights DIRECTLY from host-mapped pinned
DRAM over PCIe and computes the GEMV in ONE kernel -- NO explicit H2D copy to HBM, no copy engine,
no HBM expert-cache occupancy. Compares against (A) cudaMemcpy H2D + HBM GEMV, (B) CPU Fiddler GEMV.

Decisive question: is zero-copy remote GEMV <= CPU Fiddler (0.167ms)? Physics predicts PCIe-bound
(~0.78ms for 17MB at 21.6 GB/s) so likely SLOWER than CPU (DRAM 200GB/s), but MEASURE not guess.
Inline CUDA via torch cpp_extension. All printed English. Core params: no defaults.
"""
import argparse, time, json
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

// COALESCED GEMV (web-confirmed: zero-copy needs coalesced single-read). One WARP per output row;
// the 32 lanes read CONSECUTIVE elements of the row (coalesced PCIe burst), then warp-reduce.
// out[r] = sum_c x[c] * W[r*C + c], W is a HOST-MAPPED device pointer (zero-copy over PCIe).
__global__ void gemv_zerocopy(const float* __restrict__ x, const float* __restrict__ Wdev,
                              float* __restrict__ out, int R, int C) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= R) return;
    const float* wrow = Wdev + (long)warp * C;
    float acc = 0.f;
    for (int c = lane; c < C; c += 32) acc += x[c] * wrow[c];   // coalesced: lanes read consecutive c
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) out[warp] = acc;
}

// register a PLAIN (non-pinned) host tensor as MAPPED, return its device pointer (as int64)
int64_t map_host(torch::Tensor w) {
    void* hp = w.data_ptr();
    cudaError_t e1 = cudaHostRegister(hp, w.numel()*sizeof(float), cudaHostRegisterMapped);
    TORCH_CHECK(e1 == cudaSuccess, "cudaHostRegister failed: ", cudaGetErrorString(e1));
    void* dp = nullptr;
    cudaError_t e2 = cudaHostGetDevicePointer(&dp, hp, 0);
    TORCH_CHECK(e2 == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(e2));
    return (int64_t)dp;
}
void unmap_host(torch::Tensor w) { cudaHostUnregister(w.data_ptr()); }

void run_zerocopy(torch::Tensor x, int64_t wdev, torch::Tensor out, int R, int C) {
    int threads = 256;                                  // 8 warps/block
    long total_warps = (long)R;
    int blocks = (int)((total_warps * 32 + threads - 1) / threads);  // one warp per output row
    gemv_zerocopy<<<blocks, threads>>>(x.data_ptr<float>(), (const float*)wdev, out.data_ptr<float>(), R, C);
}
"""
CPP_SRC = r"""
int64_t map_host(torch::Tensor w);
void unmap_host(torch::Tensor w);
void run_zerocopy(torch::Tensor x, int64_t wdev, torch::Tensor out, int R, int C);
"""


def parse_args():
    ap = argparse.ArgumentParser(description="Zero-copy remote expert GEMV microbench")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev)
    m = load_inline(name="zc", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                    functions=["map_host", "unmap_host", "run_zerocopy"], verbose=False)
    dm, di = a.d_model, a.d_inter
    # one matrix (di x dm), fp32 for the simple kernel. Expert = 3 matrices; we time per-matrix x3.
    R, C = di, dm
    mat_mb = R * C * 4 / 1e6
    Wh = torch.randn(R, C).contiguous()                        # PLAIN host tensor (not torch-pinned)
    # allocate ALL device tensors FIRST (set up context), register host LAST
    Wg = Wh.to(dev)                                             # HBM-resident copy for baseline
    x = torch.randn(C, device=dev); out = torch.empty(R, device=dev)
    Wh_dst = torch.empty(R, C, device=dev)                     # H2D dst
    Wh_pin = Wh.pin_memory()                                    # separate pinned copy for H2D baseline
    torch.cuda.synchronize(dev)
    wdev = m.map_host(Wh)                                       # map the PLAIN host tensor (zero-copy) LAST

    def zc(): m.run_zerocopy(x, wdev, out, R, C); torch.cuda.synchronize(dev)
    def hbm_gemv(): torch.mv(Wg, x); torch.cuda.synchronize(dev)
    def h2d_then_gemv(): Wh_dst.copy_(Wh_pin, non_blocking=True); torch.mv(Wh_dst, x); torch.cuda.synchronize(dev)

    def bench(fn):
        for _ in range(5): fn()
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(a.iters): fn()
        return (time.perf_counter() - t0) / a.iters * 1000.0

    # correctness sanity (zero-copy vs hbm)
    m.run_zerocopy(x, wdev, out, R, C); torch.cuda.synchronize(dev)
    ref = torch.mv(Wg, x)
    err = (out - ref).abs().max().item()

    res = {"matrix_mb": mat_mb, "expert_mb": 3 * mat_mb, "zc_vs_hbm_maxabs_err": err}
    res["t_zerocopy_1matrix_ms"] = bench(zc)
    res["t_hbm_gemv_1matrix_ms"] = bench(hbm_gemv)
    res["t_h2d_then_gemv_1matrix_ms"] = bench(h2d_then_gemv)
    # per-expert (3 matrices) estimates
    res["zerocopy_gbps"] = mat_mb / res["t_zerocopy_1matrix_ms"] / 1.024
    res["t_zerocopy_expert_ms"] = res["t_zerocopy_1matrix_ms"] * 3
    res["cpu_fiddler_expert_ms_ref"] = 0.167
    res["fetch_expert_ms_ref"] = 0.782
    m.unmap_host(Wh)
    print(json.dumps(res, indent=2), flush=True)
    with open(a.out, "w") as f: json.dump(res, f, indent=2)
    print(f"\n[zero-copy] 1 matrix ({mat_mb:.1f}MB): zerocopy={res['t_zerocopy_1matrix_ms']:.3f}ms "
          f"({res['zerocopy_gbps']:.1f}GB/s) vs hbm-gemv={res['t_hbm_gemv_1matrix_ms']:.3f} "
          f"vs h2d+gemv={res['t_h2d_then_gemv_1matrix_ms']:.3f}", flush=True)
    print(f"[verdict] zerocopy expert ~{res['t_zerocopy_expert_ms']:.3f}ms vs CPU-Fiddler 0.167ms vs fetch 0.782ms "
          f"(correctness err={err:.4f})", flush=True)


if __name__ == "__main__":
    main()
