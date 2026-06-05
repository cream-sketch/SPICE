"""Diagnostic fused lossless-decode + GEMV for real Qwen bf16 MoE weights.

This is NOT a source-only MoE baseline. It is a diagnostic wrapper experiment:
we keep one expert projection matrix compressed in a simple random-row-addressable
format and validate whether a Triton GEMV can decode each bf16 weight in-register
fast enough that the kernel is bounded by reading PACKED bytes.

Format / 格式:
  bf16 = sign(1) + exponent(8) + mantissa(7)
  common path stores sign+mantissa in one byte, and a 4-bit exponent code in a
  per-row nibble stream. Code 15 is ESC. A matrix-level codebook stores the 15
  most common exponents, and rare exponents are stored in sorted per-row escape
  side lists (column + exact exponent). Each row has fixed byte offsets:
      signmant[row] starts at row*K
      expcodes[row] starts at row*ceil(K/2)

Why k=4 / 为什么选 4 bit:
  A nibble stream is the simplest cheap GPU decode (one byte load, one shift,
  one mask) and the fixed payload is exactly 1.5 bytes/weight, i.e. 2/1.5 =
  1.333x before rare escapes. Real trained bf16 weights usually concentrate in
  far fewer than 15 exponent values, so the escape tax should keep the final
  ratio in the requested ~1.25-1.4x band. k=3 can be faster on paper but escape
  scans are much riskier; k=5 is cleaner but only 1.23x before metadata.

Exactness / 精确性:
  The decoded bf16 bytes are asserted bit-identical to W. GEMV exactness is
  asserted against a raw-bf16 Triton GEMV with the SAME one-row reduction order.
  torch.nn.functional.linear is also reported, but cuBLAS/TensorCore reduction
  order may differ, so it is treated as the full-HBM latency baseline rather than
  the bit-exact arithmetic oracle.
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open


EXPERT_TAGS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class PackedBf16Meta:
    """CPU/GPU metadata tensors for one packed matrix / 单个矩阵的元数据."""

    m: int
    k_dim: int
    exp_bits: int
    esc_code: int
    signmant_offset: int
    exp_offset: int
    exp_row_bytes: int
    fixed_nbytes: int
    original_nbytes: int
    compressed_nbytes: int
    num_escapes: int
    codebook: torch.Tensor
    row_escape_offsets: torch.Tensor
    escape_cols: torch.Tensor
    escape_exponents: torch.Tensor
    name: str = ""

    @property
    def ratio(self) -> float:
        return self.original_nbytes / max(1, self.compressed_nbytes)

    @property
    def escape_rate(self) -> float:
        return self.num_escapes / max(1, self.m * self.k_dim)

    def to(self, device: torch.device | str) -> "PackedBf16Meta":
        """Move tensor metadata to device / 将 tensor 元数据搬到指定设备."""
        return replace(
            self,
            codebook=self.codebook.to(device=device, non_blocking=True).contiguous(),
            row_escape_offsets=self.row_escape_offsets.to(device=device, non_blocking=True).contiguous(),
            escape_cols=self.escape_cols.to(device=device, non_blocking=True).contiguous(),
            escape_exponents=self.escape_exponents.to(device=device, non_blocking=True).contiguous(),
        )


def _metadata_nbytes(meta: PackedBf16Meta) -> int:
    return (
        meta.codebook.numel() * meta.codebook.element_size()
        + meta.row_escape_offsets.numel() * meta.row_escape_offsets.element_size()
        + meta.escape_cols.numel() * meta.escape_cols.element_size()
        + meta.escape_exponents.numel() * meta.escape_exponents.element_size()
    )


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _num_warps_for(block_k: int) -> int:
    if block_k >= 2048:
        return 8
    if block_k >= 1024:
        return 4
    return 1 if block_k <= 256 else 2


def _ensure_bf16_matrix(W: torch.Tensor) -> torch.Tensor:
    if W.dtype != torch.bfloat16:
        raise TypeError(f"pack expects torch.bfloat16, got {W.dtype}")
    if W.ndim != 2:
        raise ValueError(f"pack expects a 2D matrix, got shape={tuple(W.shape)}")
    return W.detach().cpu().contiguous()


def pack(W: torch.Tensor, exp_bits: int = 4, name: str = "") -> tuple[torch.Tensor, PackedBf16Meta]:
    """Pack bf16 W losslessly into (flat uint8 payload, metadata).

    pack(W) -> (packed_uint8_tensor, metadata)
    打包后 common-case 每个元素固定 1 byte sign/mant + 4-bit exponent code.
    """
    if exp_bits != 4:
        raise NotImplementedError("this diagnostic Triton path intentionally specializes exp_bits=4")

    W = _ensure_bf16_matrix(W)
    m, k_dim = W.shape
    esc_code = (1 << exp_bits) - 1
    common_slots = esc_code

    # bf16 bits live in a uint16 word: sign bit 15, exponent bits 14..7, mantissa bits 6..0.
    # bf16 位布局: sign=15, exponent=14..7, mantissa=6..0.
    u16 = W.view(torch.uint16).to(torch.int32)
    signmant = ((((u16 >> 8) & 0x80) | (u16 & 0x7F)).to(torch.uint8)).contiguous()
    exponents = (((u16 >> 7) & 0xFF).to(torch.int64)).contiguous()

    counts = torch.bincount(exponents.reshape(-1), minlength=256).tolist()
    # Deterministic frequency order: high count first, then low exponent value.
    # 频率排序固定化,避免 topk 平票导致不可复现.
    common_exps = sorted(range(256), key=lambda e: (-counts[e], e))[:common_slots]
    codebook = torch.zeros(1 << exp_bits, dtype=torch.uint8)
    code_lut = torch.full((256,), esc_code, dtype=torch.uint8)
    for code, exp in enumerate(common_exps):
        codebook[code] = exp
        code_lut[exp] = code

    codes = code_lut[exponents].contiguous()
    escape_mask = codes == esc_code
    if bool(escape_mask.any()):
        esc_rows, esc_cols = torch.nonzero(escape_mask, as_tuple=True)
        row_counts = torch.bincount(esc_rows.to(torch.int64), minlength=m).to(torch.int32)
        row_escape_offsets = torch.empty(m + 1, dtype=torch.int32)
        row_escape_offsets[0] = 0
        row_escape_offsets[1:] = torch.cumsum(row_counts, dim=0)
        escape_cols = esc_cols.to(torch.int32).contiguous()
        escape_exponents = exponents[escape_mask].to(torch.uint8).contiguous()
    else:
        row_escape_offsets = torch.zeros(m + 1, dtype=torch.int32)
        escape_cols = torch.empty(0, dtype=torch.int32)
        escape_exponents = torch.empty(0, dtype=torch.uint8)

    # Per-row nibble packing / 每行独立 nibble 打包,保证 row random access.
    exp_row_bytes = (k_dim + 1) // 2
    exp_bytes = torch.zeros((m, exp_row_bytes), dtype=torch.uint8)
    even_codes = codes[:, 0::2]
    exp_bytes[:, : even_codes.shape[1]] = even_codes & 0x0F
    odd_codes = codes[:, 1::2]
    if odd_codes.numel():
        exp_bytes[:, : odd_codes.shape[1]] |= (odd_codes << 4) & 0xF0
    exp_bytes = exp_bytes.contiguous()

    signmant_offset = 0
    exp_offset = signmant.numel()
    packed = torch.cat([signmant.reshape(-1), exp_bytes.reshape(-1)]).contiguous()
    fixed_nbytes = int(packed.numel())
    original_nbytes = int(W.numel() * 2)

    # Build temporary meta first so metadata byte accounting uses tensor element sizes.
    meta = PackedBf16Meta(
        m=m,
        k_dim=k_dim,
        exp_bits=exp_bits,
        esc_code=esc_code,
        signmant_offset=signmant_offset,
        exp_offset=exp_offset,
        exp_row_bytes=exp_row_bytes,
        fixed_nbytes=fixed_nbytes,
        original_nbytes=original_nbytes,
        compressed_nbytes=0,
        num_escapes=int(escape_exponents.numel()),
        codebook=codebook,
        row_escape_offsets=row_escape_offsets,
        escape_cols=escape_cols,
        escape_exponents=escape_exponents,
        name=name,
    )
    meta = replace(meta, compressed_nbytes=fixed_nbytes + _metadata_nbytes(meta))
    return packed, meta


def unpack_cpu(packed: torch.Tensor, meta: PackedBf16Meta) -> torch.Tensor:
    """Reference CPU unpack, used for packer exactness / CPU 侧解包校验."""
    if packed.device.type != "cpu":
        packed = packed.cpu()
    codebook = meta.codebook.cpu()
    row_escape_offsets = meta.row_escape_offsets.cpu()
    escape_cols = meta.escape_cols.cpu()
    escape_exponents = meta.escape_exponents.cpu()

    signmant = packed[
        meta.signmant_offset : meta.signmant_offset + meta.m * meta.k_dim
    ].view(meta.m, meta.k_dim)
    exp_bytes = packed[
        meta.exp_offset : meta.exp_offset + meta.m * meta.exp_row_bytes
    ].view(meta.m, meta.exp_row_bytes)

    codes = torch.empty((meta.m, meta.k_dim), dtype=torch.uint8)
    codes[:, 0::2] = exp_bytes[:, : codes[:, 0::2].shape[1]] & 0x0F
    if meta.k_dim > 1:
        codes[:, 1::2] = (exp_bytes[:, : codes[:, 1::2].shape[1]] >> 4) & 0x0F
    exponents = codebook[codes.to(torch.int64)].to(torch.int32)

    # Apply rare exact exponents / 应用逃逸表中的真实 exponent.
    for row in range(meta.m):
        lo = int(row_escape_offsets[row])
        hi = int(row_escape_offsets[row + 1])
        if lo != hi:
            cols = escape_cols[lo:hi].to(torch.int64)
            exponents[row, cols] = escape_exponents[lo:hi].to(torch.int32)

    sm = signmant.to(torch.int32)
    u16 = (((sm & 0x80) << 8) | ((exponents & 0xFF) << 7) | (sm & 0x7F)).to(torch.uint16)
    return u16.contiguous().view(torch.bfloat16)


def assert_packer_exact(W: torch.Tensor, packed: torch.Tensor, meta: PackedBf16Meta) -> None:
    """Assert decompress(pack(W)) == W bit-for-bit / 逐位无损断言."""
    W_cpu = _ensure_bf16_matrix(W)
    recon = unpack_cpu(packed, meta)
    if not torch.equal(recon.view(torch.uint16), W_cpu.view(torch.uint16)):
        mismatch = (recon.view(torch.uint16) != W_cpu.view(torch.uint16)).nonzero()
        first = tuple(int(v) for v in mismatch[0]) if mismatch.numel() else ()
        raise AssertionError(f"CPU unpack is not bit-identical for {meta.name}; first mismatch={first}")


def to_device(
    packed: torch.Tensor, meta: PackedBf16Meta, device: torch.device | str
) -> tuple[torch.Tensor, PackedBf16Meta]:
    """Move packed payload and metadata to CUDA / 搬运 packed payload + metadata."""
    return packed.to(device=device, non_blocking=True).contiguous(), meta.to(device)


@triton.jit
def _fused_decode_gemv_kernel(
    packed_ptr,
    codebook_ptr,
    row_escape_offsets_ptr,
    escape_cols_ptr,
    escape_exponents_ptr,
    x_ptr,
    y_ptr,
    k_dim: tl.constexpr,
    signmant_offset: tl.constexpr,
    exp_offset: tl.constexpr,
    exp_row_bytes: tl.constexpr,
    has_escapes: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < k_dim

    sm = tl.load(
        packed_ptr + signmant_offset + row * k_dim + offs,
        mask=mask,
        other=0,
    ).to(tl.uint32)
    exp_byte = tl.load(
        packed_ptr + exp_offset + row * exp_row_bytes + (offs >> 1),
        mask=mask,
        other=0,
    ).to(tl.uint32)
    exp_code = (exp_byte >> ((offs & 1) << 2)) & 0x0F
    exponent = tl.load(codebook_ptr + exp_code, mask=mask, other=0).to(tl.uint32)

    # Rare escape repair. Common path above is branch-free/coalesced; only rows with escapes scan.
    # 罕见 exponent 通过每行 side-list 修复; common path 不分支.
    if has_escapes:
        esc_i = tl.load(row_escape_offsets_ptr + row)
        esc_end = tl.load(row_escape_offsets_ptr + row + 1)
        while esc_i < esc_end:
            esc_col = tl.load(escape_cols_ptr + esc_i).to(tl.int32)
            esc_exp = tl.load(escape_exponents_ptr + esc_i).to(tl.uint32)
            exponent = tl.where(offs == esc_col, esc_exp, exponent)
            esc_i += 1

    w16 = (((sm & 0x80) << 8) | ((exponent & 0xFF) << 7) | (sm & 0x7F)).to(tl.uint32)
    # bf16 -> fp32 exactly by putting bf16 bits in the high 16 fp32 bits.
    # bf16 bitcast 到 fp32: 高 16 位放 bf16,低 16 位补 0.
    w = (w16 << 16).to(tl.float32, bitcast=True)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(tl.where(mask, w * x, 0.0), axis=0)
    tl.store(y_ptr + row, acc)


@triton.jit
def _decode_to_bf16_kernel(
    packed_ptr,
    codebook_ptr,
    row_escape_offsets_ptr,
    escape_cols_ptr,
    escape_exponents_ptr,
    out_ptr,
    k_dim: tl.constexpr,
    signmant_offset: tl.constexpr,
    exp_offset: tl.constexpr,
    exp_row_bytes: tl.constexpr,
    has_escapes: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < k_dim

    sm = tl.load(
        packed_ptr + signmant_offset + row * k_dim + offs,
        mask=mask,
        other=0,
    ).to(tl.uint32)
    exp_byte = tl.load(
        packed_ptr + exp_offset + row * exp_row_bytes + (offs >> 1),
        mask=mask,
        other=0,
    ).to(tl.uint32)
    exp_code = (exp_byte >> ((offs & 1) << 2)) & 0x0F
    exponent = tl.load(codebook_ptr + exp_code, mask=mask, other=0).to(tl.uint32)

    if has_escapes:
        esc_i = tl.load(row_escape_offsets_ptr + row)
        esc_end = tl.load(row_escape_offsets_ptr + row + 1)
        while esc_i < esc_end:
            esc_col = tl.load(escape_cols_ptr + esc_i).to(tl.int32)
            esc_exp = tl.load(escape_exponents_ptr + esc_i).to(tl.uint32)
            exponent = tl.where(offs == esc_col, esc_exp, exponent)
            esc_i += 1

    w16 = (((sm & 0x80) << 8) | ((exponent & 0xFF) << 7) | (sm & 0x7F)).to(tl.uint32)
    w = (w16 << 16).to(tl.float32, bitcast=True)
    tl.store(out_ptr + row * k_dim + offs, w.to(tl.bfloat16), mask=mask)


@triton.jit
def _raw_bf16_gemv_kernel(
    W_ptr,
    x_ptr,
    y_ptr,
    k_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < k_dim
    w = tl.load(W_ptr + row * k_dim + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(tl.where(mask, w * x, 0.0), axis=0)
    tl.store(y_ptr + row, acc)


def _check_cuda_inputs(packed: torch.Tensor, meta: PackedBf16Meta, x: torch.Tensor | None = None) -> None:
    tensors = [packed, meta.codebook, meta.row_escape_offsets, meta.escape_cols, meta.escape_exponents]
    if x is not None:
        tensors.append(x)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("packed, metadata tensors, and x must all be CUDA tensors")


def _flat_x(x: torch.Tensor, k_dim: int) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must be torch.bfloat16, got {x.dtype}")
    if x.ndim == 2 and x.shape[0] == 1:
        x = x.reshape(-1)
    if x.ndim != 1 or x.numel() != k_dim:
        raise ValueError(f"x must have shape [{k_dim}] or [1,{k_dim}], got {tuple(x.shape)}")
    return x.contiguous()


def fused_decode_gemv(
    packed: torch.Tensor,
    meta: PackedBf16Meta,
    x: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton fused decode-in-register GEMV / fused 解码并做 batch=1 GEMV."""
    x = _flat_x(x, meta.k_dim)
    _check_cuda_inputs(packed, meta, x)
    if out is None:
        out = torch.empty((meta.m,), device=x.device, dtype=torch.float32)
    elif out.shape != (meta.m,) or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("out must be CUDA float32 with shape [M]")
    block_k = _next_power_of_2(meta.k_dim)
    _fused_decode_gemv_kernel[(meta.m,)](
        packed,
        meta.codebook,
        meta.row_escape_offsets,
        meta.escape_cols,
        meta.escape_exponents,
        x,
        out,
        meta.k_dim,
        meta.signmant_offset,
        meta.exp_offset,
        meta.exp_row_bytes,
        meta.num_escapes > 0,
        BLOCK_K=block_k,
        num_warps=_num_warps_for(block_k),
    )
    return out


def decode_to_bf16(
    packed: torch.Tensor,
    meta: PackedBf16Meta,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode full W to bf16 tensor on GPU / 单独解码成完整 bf16 权重."""
    _check_cuda_inputs(packed, meta)
    if out is None:
        out = torch.empty((meta.m, meta.k_dim), device=packed.device, dtype=torch.bfloat16)
    elif out.shape != (meta.m, meta.k_dim) or out.dtype != torch.bfloat16 or not out.is_cuda:
        raise ValueError("out must be CUDA bf16 with shape [M,K]")
    block_k = _next_power_of_2(meta.k_dim)
    _decode_to_bf16_kernel[(meta.m,)](
        packed,
        meta.codebook,
        meta.row_escape_offsets,
        meta.escape_cols,
        meta.escape_exponents,
        out,
        meta.k_dim,
        meta.signmant_offset,
        meta.exp_offset,
        meta.exp_row_bytes,
        meta.num_escapes > 0,
        BLOCK_K=block_k,
        num_warps=_num_warps_for(block_k),
    )
    return out


def raw_bf16_gemv(W: torch.Tensor, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Same-reduction raw bf16 GEMV reference / 同 reduction 顺序的原始 bf16 GEMV."""
    if W.dtype != torch.bfloat16 or W.ndim != 2 or not W.is_cuda:
        raise ValueError("W must be CUDA bf16 [M,K]")
    x = _flat_x(x, W.shape[1])
    if not x.is_cuda:
        raise ValueError("x must be CUDA")
    if out is None:
        out = torch.empty((W.shape[0],), device=W.device, dtype=torch.float32)
    elif out.shape != (W.shape[0],) or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("out must be CUDA float32 with shape [M]")
    block_k = _next_power_of_2(W.shape[1])
    _raw_bf16_gemv_kernel[(W.shape[0],)](
        W.contiguous(),
        x,
        out,
        W.shape[1],
        BLOCK_K=block_k,
        num_warps=_num_warps_for(block_k),
    )
    return out


def separate_decode_then_gemv(
    packed: torch.Tensor,
    meta: PackedBf16Meta,
    x: torch.Tensor,
    decoded: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Separate pass baseline: decode full W, then GEMV / 分离解码再 GEMV."""
    decode_to_bf16(packed, meta, out=decoded)
    raw_bf16_gemv(decoded, x, out=out)
    return out


def _bench_cuda(fn, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / reps)


def _gbps(nbytes: int, ms: float) -> float:
    return nbytes / max(ms, 1e-12) / 1e6


@dataclass
class BenchResult:
    method: str
    ms: float
    physical_bytes: int
    logical_bytes: int

    @property
    def physical_gbps(self) -> float:
        return _gbps(self.physical_bytes, self.ms)

    @property
    def effective_gbps(self) -> float:
        return _gbps(self.logical_bytes, self.ms)


def _print_bench_table(results: list[BenchResult], full_ms: float) -> None:
    print(f"{'method':25s} {'ms':>9s} {'physGB/s':>10s} {'effGB/s':>10s} {'speedup':>8s}")
    for r in results:
        speedup = full_ms / r.ms if r.ms > 0 else float("inf")
        print(
            f"{r.method:25s} {r.ms:9.4f} {r.physical_gbps:10.1f} "
            f"{r.effective_gbps:10.1f} {speedup:8.3f}"
        )


def _verify_on_gpu(
    name: str,
    W: torch.Tensor,
    packed_gpu: torch.Tensor,
    meta_gpu: PackedBf16Meta,
    x: torch.Tensor,
) -> tuple[bool, float]:
    decoded = decode_to_bf16(packed_gpu, meta_gpu)
    torch.cuda.synchronize()
    W_bits = W.view(torch.uint16)
    decoded_bits = decoded.view(torch.uint16)
    if not torch.equal(decoded_bits, W_bits):
        mismatch = (decoded_bits != W_bits).nonzero()
        first = tuple(int(v) for v in mismatch[0]) if mismatch.numel() else ()
        raise AssertionError(f"GPU decode is not bit-identical for {name}; first mismatch={first}")

    raw = raw_bf16_gemv(W, x)
    fused = fused_decode_gemv(packed_gpu, meta_gpu, x)
    torch.cuda.synchronize()
    if not torch.equal(fused, raw):
        diff = (fused - raw).abs()
        raise AssertionError(
            f"fused GEMV != same-reduction raw GEMV for {name}; "
            f"max_abs={float(diff.max())}, mismatches={int((diff != 0).sum())}"
        )

    linear = F.linear(x.view(1, -1), W).reshape(-1)
    torch.cuda.synchronize()
    linear_equal_after_cast = torch.equal(fused.to(torch.bfloat16), linear)
    linear_max_abs = float((fused - linear.float()).abs().max())
    return linear_equal_after_cast, linear_max_abs


def bench_one_tensor(
    name: str,
    W_cpu: torch.Tensor,
    device: torch.device,
    warmup: int,
    reps: int,
    seed: int,
) -> tuple[PackedBf16Meta, list[BenchResult]]:
    packed_cpu, meta_cpu = pack(W_cpu, exp_bits=4, name=name)
    assert_packer_exact(W_cpu, packed_cpu, meta_cpu)

    W_gpu = W_cpu.to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()
    packed_gpu, meta_gpu = to_device(packed_cpu, meta_cpu, device)
    torch.manual_seed(seed)
    x = torch.randn((meta_cpu.k_dim,), device=device, dtype=torch.float32).to(torch.bfloat16)

    linear_equal, linear_max_abs = _verify_on_gpu(name, W_gpu, packed_gpu, meta_gpu, x)
    codebook_str = ",".join(str(int(v)) for v in meta_cpu.codebook[: meta_cpu.esc_code].tolist())
    print(
        f"\n[pack] {name}\n"
        f"       shape=({meta_cpu.m},{meta_cpu.k_dim}) ratio={meta_cpu.ratio:.3f}x "
        f"fixed={meta_cpu.fixed_nbytes/1e6:.3f}MB stored={meta_cpu.compressed_nbytes/1e6:.3f}MB "
        f"escapes={meta_cpu.num_escapes} ({meta_cpu.escape_rate*100:.3f}%)\n"
        f"       common_exponents={codebook_str}"
    )
    print(
        "[exact] decoded bf16 bytes == W bytes; fused == raw Triton same-reduction GEMV; "
        f"F.linear_equal_after_bf16_cast={linear_equal} F.linear_max_abs={linear_max_abs:.6g}"
    )

    y_fused = torch.empty((meta_cpu.m,), device=device, dtype=torch.float32)
    y_raw = torch.empty((meta_cpu.m,), device=device, dtype=torch.float32)
    decoded = torch.empty_like(W_gpu)
    x_row = x.view(1, -1)

    full_ms = _bench_cuda(lambda: F.linear(x_row, W_gpu), warmup, reps)
    fused_ms = _bench_cuda(lambda: fused_decode_gemv(packed_gpu, meta_gpu, x, out=y_fused), warmup, reps)
    separate_ms = _bench_cuda(
        lambda: separate_decode_then_gemv(packed_gpu, meta_gpu, x, decoded, y_raw),
        warmup,
        reps,
    )

    # Physical bytes are the intended HBM traffic model, ignoring the tiny x vector because it should
    # sit in cache across rows. 物理字节按权重主路径估算,忽略会被 cache 住的 x.
    results = [
        BenchResult("torch F.linear bf16", full_ms, meta_cpu.original_nbytes, meta_cpu.original_nbytes),
        BenchResult("fused decode+gemv", fused_ms, meta_cpu.compressed_nbytes, meta_cpu.original_nbytes),
        BenchResult(
            "separate decode+gemv",
            separate_ms,
            meta_cpu.compressed_nbytes + 2 * meta_cpu.original_nbytes,
            meta_cpu.original_nbytes,
        ),
    ]
    _print_bench_table(results, full_ms)
    print(
        f"[bound] fused_speedup_vs_F.linear={full_ms/fused_ms:.3f}x, "
        f"compression_ratio={meta_cpu.ratio:.3f}x, "
        f"speedup/ratio={(full_ms/fused_ms)/meta_cpu.ratio:.3f}"
    )
    return meta_cpu, results


def _parse_projection_arg(value: str) -> tuple[str, ...]:
    value = value.strip()
    if value == "all":
        return EXPERT_TAGS
    aliases = {
        "gate": "gate_proj",
        "up": "up_proj",
        "down": "down_proj",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
    }
    tags = []
    for part in value.split(","):
        key = part.strip()
        if key not in aliases:
            raise ValueError(f"unknown projection '{key}', expected one of gate,up,down,all")
        tags.append(aliases[key])
    # Preserve order and remove duplicates / 保序去重.
    seen = set()
    out = []
    for tag in tags:
        if tag not in seen:
            out.append(tag)
            seen.add(tag)
    return tuple(out)


def _is_expert_weight_key(key: str, tags: Iterable[str]) -> bool:
    return (
        ".mlp.experts." in key
        and key.endswith(".weight")
        and any(key.endswith(f".{tag}.weight") for tag in tags)
    )


def load_qwen_expert_tensors(
    model_dir: str,
    n_tensors: int,
    projection: str = "gate",
) -> list[tuple[str, torch.Tensor]]:
    """Load real Qwen expert projection weights from safetensors / 从 safetensors 读取真实专家权重."""
    tags = _parse_projection_arg(projection)
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files found in {model_dir}")

    key2file: dict[str, str] = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as h:
            for key in h.keys():
                if _is_expert_weight_key(key, tags):
                    key2file[key] = f
    keys = sorted(key2file)
    if not keys:
        raise RuntimeError(f"no Qwen expert weight keys found for projection={projection!r}")
    if n_tensors <= 0 or n_tensors >= len(keys):
        selected = keys
    else:
        step = max(1, len(keys) // n_tensors)
        selected = keys[::step][:n_tensors]

    by_file: dict[str, list[str]] = defaultdict(list)
    for key in selected:
        by_file[key2file[key]].append(key)

    tensors: list[tuple[str, torch.Tensor]] = []
    for f, keys_in_file in by_file.items():
        with safe_open(f, framework="pt", device="cpu") as h:
            for key in keys_in_file:
                W = h.get_tensor(key)
                if W.dtype != torch.bfloat16:
                    raise TypeError(f"{key} is {W.dtype}, expected torch.bfloat16")
                if W.ndim != 2:
                    raise ValueError(f"{key} has shape={tuple(W.shape)}, expected 2D")
                tensors.append((key, W.contiguous()))
    return tensors


def _print_aggregate(all_results: list[tuple[PackedBf16Meta, list[BenchResult]]]) -> None:
    if not all_results:
        return
    orig = sum(meta.original_nbytes for meta, _ in all_results)
    comp = sum(meta.compressed_nbytes for meta, _ in all_results)
    esc = sum(meta.num_escapes for meta, _ in all_results)
    elems = sum(meta.m * meta.k_dim for meta, _ in all_results)
    by_method: dict[str, list[BenchResult]] = defaultdict(list)
    for _, rows in all_results:
        for r in rows:
            by_method[r.method].append(r)

    print("\n[aggregate]")
    print(f"weighted_ratio={orig / max(1, comp):.3f}x escape_rate={esc / max(1, elems) * 100:.3f}% tensors={len(all_results)}")
    print(f"{'method':25s} {'avg_ms':>9s} {'avg_physGB/s':>13s} {'avg_effGB/s':>12s}")
    for method, rows in by_method.items():
        avg_ms = sum(r.ms for r in rows) / len(rows)
        avg_phys = sum(r.physical_gbps for r in rows) / len(rows)
        avg_eff = sum(r.effective_gbps for r in rows) / len(rows)
        print(f"{method:25s} {avg_ms:9.4f} {avg_phys:13.1f} {avg_eff:12.1f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnostic fused lossless decode-in-register GEMV for Qwen bf16 MoE expert weights"
    )
    p.add_argument("--model_dir", required=True, help="Qwen model directory containing .safetensors")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n", type=int, default=8, help="number of expert projection tensors to sample")
    p.add_argument(
        "--projection",
        default="gate",
        help="gate (default), up, down, all, or comma list such as gate,down",
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton kernels")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    tensors = load_qwen_expert_tensors(args.model_dir, args.n, args.projection)
    total_mb = sum(t.numel() * 2 for _, t in tensors) / 1e6
    print(
        "[load] diagnostic wrapper, not a source-only baseline; "
        f"loaded {len(tensors)} tensors ({total_mb:.1f} MB bf16) projection={args.projection}"
    )

    all_results = []
    for i, (name, W) in enumerate(tensors):
        all_results.append(bench_one_tensor(name, W, device, args.warmup, args.reps, args.seed + i))
    _print_aggregate(all_results)

    print(
        "\n[note] Expected ratio is near the 4-bit fixed payload limit (1.333x) minus escape metadata; "
        "on real Qwen bf16 expert weights this should usually land around 1.25-1.4x if the top 15 "
        "exponents dominate. The biggest memory-bound risk is not the nibble decode itself, but a high "
        "escape rate causing per-row side-list scans plus tiny-GEMV launch/reduction overhead to dominate "
        "the saved HBM bytes."
    )


if __name__ == "__main__":
    main()
