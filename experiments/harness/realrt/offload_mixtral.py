"""Single-GPU real offloaded Mixtral-8x7B decode runtime.

This is a benchmark harness, not a simulator.  Attention, norms, embedding,
LM head, and every router stay GPU-resident; routed expert weights live in CPU
pinned memory or in a packed CPU bank and are fetched/served according to the
selected policy.

真实单卡 offload Mixtral:非专家部分常驻 GPU, routed experts 在 CPU pinned bank,
MoE forward 使用真实 router + 按 expert 分组 dispatch,用于 TTFT/TPOT 测量。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from fused_decode_gemv import PackedBf16Meta, decode_to_bf16, fused_decode_gemv, pack
except ImportError:  # allow package-style imports from external harnesses
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from fused_decode_gemv import PackedBf16Meta, decode_to_bf16, fused_decode_gemv, pack


VALID_POLICIES = ("cpu_serve", "on_demand", "fused_compressed", "split_cpu_gpu")
EXPECTED = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 32,
}
PROMPT = "Mixture-of-experts language models are useful because"


class ExpertBank:
    """CPU pinned bf16 expert bank keyed by (layer, expert). / CPU pinned 专家权重库."""

    def __init__(self) -> None:
        self.w: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.bytes = 0

    def add(self, layer: int, eid: int, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> None:
        triple = tuple(
            w.detach().to("cpu", torch.bfloat16).contiguous().pin_memory()
            for w in (gate_w, up_w, down_w)
        )
        self.w[(layer, eid)] = triple
        self.bytes += sum(t.numel() * t.element_size() for t in triple)

    def get(self, layer: int, eid: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.w[(layer, eid)]

    def assert_cpu_only(self) -> None:
        bad = [(k, t.device) for k, triple in self.w.items() for t in triple if t.device.type != "cpu"]
        if bad:
            raise RuntimeError(f"expert bank must stay on CPU; first bad tensor={bad[0]}")


@dataclass
class PackedExpert:
    gate_packed: torch.Tensor
    gate_meta: PackedBf16Meta
    up_packed: torch.Tensor
    up_meta: PackedBf16Meta
    down_packed: torch.Tensor
    down_meta: PackedBf16Meta

    @property
    def payload_bytes(self) -> int:
        return int(self.gate_packed.numel() + self.up_packed.numel() + self.down_packed.numel())

    @property
    def metadata_bytes(self) -> int:
        metas = (self.gate_meta, self.up_meta, self.down_meta)
        total = 0
        for meta in metas:
            total += (
                meta.codebook.numel() * meta.codebook.element_size()
                + meta.row_escape_offsets.numel() * meta.row_escape_offsets.element_size()
                + meta.escape_cols.numel() * meta.escape_cols.element_size()
                + meta.escape_exponents.numel() * meta.escape_exponents.element_size()
            )
        return total


class PackedExpertBank:
    """CPU pinned packed payloads for lossless compressed policy. / 无损 packed CPU bank."""

    def __init__(self) -> None:
        self.w: dict[tuple[int, int], PackedExpert] = {}
        self.payload_bytes = 0
        self.metadata_bytes = 0
        self.max_gate_bytes = 0
        self.max_up_bytes = 0
        self.max_down_bytes = 0
        self.max_metadata_bytes = 0

    def add(self, layer: int, eid: int, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> None:
        packed = []
        for tag, weight in (("gate", gate_w), ("up", up_w), ("down", down_w)):
            p, m = pack(weight.detach().to("cpu", torch.bfloat16).contiguous(), name=f"L{layer}.E{eid}.{tag}")
            packed.append((p.contiguous().pin_memory(), m))
        expert = PackedExpert(
            gate_packed=packed[0][0],
            gate_meta=packed[0][1],
            up_packed=packed[1][0],
            up_meta=packed[1][1],
            down_packed=packed[2][0],
            down_meta=packed[2][1],
        )
        self.w[(layer, eid)] = expert
        self.payload_bytes += expert.payload_bytes
        self.metadata_bytes += expert.metadata_bytes
        self.max_gate_bytes = max(self.max_gate_bytes, int(expert.gate_packed.numel()))
        self.max_up_bytes = max(self.max_up_bytes, int(expert.up_packed.numel()))
        self.max_down_bytes = max(self.max_down_bytes, int(expert.down_packed.numel()))
        self.max_metadata_bytes = max(self.max_metadata_bytes, expert.metadata_bytes)

    def get(self, layer: int, eid: int) -> PackedExpert:
        return self.w[(layer, eid)]

    def avg_cache_bytes(self) -> int:
        if not self.w:
            raise RuntimeError("packed bank is empty")
        return int((self.payload_bytes + self.metadata_bytes) / len(self.w))

    def max_cache_bytes(self) -> int:
        if not self.w:
            raise RuntimeError("packed bank is empty")
        return self.max_gate_bytes + self.max_up_bytes + self.max_down_bytes + self.max_metadata_bytes


class GpuExpertCache:
    """LRU cache for full bf16 experts. / bf16 解压权重 GPU LRU cache."""

    def __init__(self, capacity: int, d_model: int, d_inter: int, dev: torch.device, h2d_stream: torch.cuda.Stream):
        if capacity < 1:
            raise ValueError("GpuExpertCache capacity must be >= 1")
        self.cap = capacity
        self.dev = dev
        self.h2d = h2d_stream
        self.free = list(range(capacity))
        self.gate = torch.empty(capacity, d_inter, d_model, device=dev, dtype=torch.bfloat16)
        self.up = torch.empty(capacity, d_inter, d_model, device=dev, dtype=torch.bfloat16)
        self.down = torch.empty(capacity, d_model, d_inter, device=dev, dtype=torch.bfloat16)
        self.map: dict[tuple[int, int], int] = {}
        self.lru: list[int] = []
        self.ready: list[torch.cuda.Event | None] = [None] * capacity
        self.stats = {"hit": 0, "miss": 0, "evict": 0}

    def _touch(self, sid: int) -> None:
        self.lru.remove(sid)
        self.lru.append(sid)

    def _evict(self) -> int:
        sid = self.lru.pop(0)
        key = next(k for k, v in self.map.items() if v == sid)
        del self.map[key]
        self.ready[sid] = None
        self.stats["evict"] += 1
        return sid

    def get_slot(self, layer: int, eid: int, bank: ExpertBank) -> int:
        key = (layer, eid)
        if key in self.map:
            sid = self.map[key]
            self._touch(sid)
            self.stats["hit"] += 1
            return sid
        self.stats["miss"] += 1
        sid = self.free.pop() if self.free else self._evict()
        self.map[key] = sid
        self.lru.append(sid)
        gw, uw, dw = bank.get(layer, eid)
        with torch.cuda.stream(self.h2d):
            self.gate[sid].copy_(gw, non_blocking=True)
            self.up[sid].copy_(uw, non_blocking=True)
            self.down[sid].copy_(dw, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.h2d)
        self.ready[sid] = ev
        return sid


class PackedGpuExpertCache:
    """LRU cache for packed expert payloads. / packed payload GPU LRU cache."""

    def __init__(self, capacity: int, bank: PackedExpertBank, dev: torch.device, h2d_stream: torch.cuda.Stream):
        if capacity < 1:
            raise ValueError("PackedGpuExpertCache capacity must be >= 1")
        self.cap = capacity
        self.dev = dev
        self.h2d = h2d_stream
        self.free = list(range(capacity))
        self.gate = torch.empty(capacity, bank.max_gate_bytes, device=dev, dtype=torch.uint8)
        self.up = torch.empty(capacity, bank.max_up_bytes, device=dev, dtype=torch.uint8)
        self.down = torch.empty(capacity, bank.max_down_bytes, device=dev, dtype=torch.uint8)
        self.meta: list[tuple[PackedBf16Meta, PackedBf16Meta, PackedBf16Meta] | None] = [None] * capacity
        self.map: dict[tuple[int, int], int] = {}
        self.lru: list[int] = []
        self.ready: list[torch.cuda.Event | None] = [None] * capacity
        self.stats = {"hit": 0, "miss": 0, "evict": 0}

    def _touch(self, sid: int) -> None:
        self.lru.remove(sid)
        self.lru.append(sid)

    def _evict(self) -> int:
        sid = self.lru.pop(0)
        key = next(k for k, v in self.map.items() if v == sid)
        del self.map[key]
        self.meta[sid] = None
        self.ready[sid] = None
        self.stats["evict"] += 1
        return sid

    def get_slot(self, layer: int, eid: int, bank: PackedExpertBank) -> int:
        key = (layer, eid)
        if key in self.map:
            sid = self.map[key]
            self._touch(sid)
            self.stats["hit"] += 1
            return sid
        self.stats["miss"] += 1
        sid = self.free.pop() if self.free else self._evict()
        self.map[key] = sid
        self.lru.append(sid)
        expert = bank.get(layer, eid)
        # Metadata is small but required by the Triton decode kernels; keep it slot-local with payload.
        # 元数据和 payload 一起随 slot 生命周期走,避免全量专家元数据常驻 GPU。
        self.meta[sid] = (
            expert.gate_meta.to(self.dev),
            expert.up_meta.to(self.dev),
            expert.down_meta.to(self.dev),
        )
        with torch.cuda.stream(self.h2d):
            self.gate[sid, : expert.gate_packed.numel()].copy_(expert.gate_packed, non_blocking=True)
            self.up[sid, : expert.up_packed.numel()].copy_(expert.up_packed, non_blocking=True)
            self.down[sid, : expert.down_packed.numel()].copy_(expert.down_packed, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.h2d)
        self.ready[sid] = ev
        return sid

    def tensors(self, sid: int) -> tuple[torch.Tensor, PackedBf16Meta, torch.Tensor, PackedBf16Meta, torch.Tensor, PackedBf16Meta]:
        metas = self.meta[sid]
        if metas is None:
            raise RuntimeError(f"packed cache slot {sid} has no metadata")
        gm, um, dm = metas
        return (
            self.gate[sid, : gm.fixed_nbytes],
            gm,
            self.up[sid, : um.fixed_nbytes],
            um,
            self.down[sid, : dm.fixed_nbytes],
            dm,
        )


def _swiglu(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)


def _expert_groups(topk_i: torch.Tensor, n_experts: int) -> Iterable[tuple[int, torch.Tensor, torch.Tensor]]:
    """Yield active experts in stock ascending expert order. / 按 stock expert 顺序分组."""
    for eid in range(n_experts):
        rows, pos = torch.where(topk_i == eid)
        if rows.numel():
            yield eid, rows, pos


def _cpu_grouped(
    bank: ExpertBank,
    layer: int,
    eids: list[int],
    x_cpu: torch.Tensor,
    topk_i_cpu: torch.Tensor,
    topk_w_cpu: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(x_cpu)
    for eid in eids:
        rows, pos = torch.where(topk_i_cpu == eid)
        if not rows.numel():
            continue
        gw, uw, dw = bank.get(layer, eid)
        y = _swiglu(x_cpu.index_select(0, rows), gw, uw, dw)
        y = y * topk_w_cpu[rows, pos].to(y.dtype).unsqueeze(1)
        out.index_add_(0, rows, y.to(out.dtype))
    return out


class Runtime:
    """Patched MoE runtime with all policy dispatches. / patched MoE runtime."""

    def __init__(
        self,
        *,
        policy: str,
        dev: torch.device,
        n_experts: int,
        raw_bank: ExpertBank | None,
        raw_cache: GpuExpertCache | None,
        packed_bank: PackedExpertBank | None,
        packed_cache: PackedGpuExpertCache | None,
        d_model: int,
        d_inter: int,
        split_g: float,
    ) -> None:
        if policy not in VALID_POLICIES:
            raise ValueError(f"policy must be one of {VALID_POLICIES}, got {policy!r}")
        if not (0.0 <= split_g <= 1.0):
            raise ValueError(f"split_g must be in [0,1], got {split_g}")
        self.policy = policy
        self.dev = dev
        self.n_experts = n_experts
        self.raw_bank = raw_bank
        self.raw_cache = raw_cache
        self.packed_bank = packed_bank
        self.packed_cache = packed_cache
        self.split_g = split_g
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.scratch_gate: torch.Tensor | None = None
        self.scratch_up: torch.Tensor | None = None
        self.scratch_down: torch.Tensor | None = None
        if policy == "fused_compressed":
            # Prefill uses full decode -> batched GEMM so many tokens amortize one decode per expert.
            # prefill N>1 时先 decode_to_bf16 再 GEMM,避免每 token 跑 GEMV kernel。
            self.scratch_gate = torch.empty(d_inter, d_model, device=dev, dtype=torch.bfloat16)
            self.scratch_up = torch.empty(d_inter, d_model, device=dev, dtype=torch.bfloat16)
            self.scratch_down = torch.empty(d_model, d_inter, device=dev, dtype=torch.bfloat16)

    def _run_cpu(self, layer: int, topk_i: torch.Tensor, topk_w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.raw_bank is None:
            raise RuntimeError("cpu_serve requires a raw ExpertBank")
        x_cpu = x.detach().to("cpu", torch.bfloat16)
        topk_i_cpu = topk_i.detach().to("cpu")
        topk_w_cpu = topk_w.detach().to("cpu", torch.bfloat16)
        return _cpu_grouped(self.raw_bank, layer, list(range(self.n_experts)), x_cpu, topk_i_cpu, topk_w_cpu).to(
            self.dev,
            non_blocking=True,
        )

    def _run_gpu_full(
        self,
        layer: int,
        topk_i: torch.Tensor,
        topk_w: torch.Tensor,
        x: torch.Tensor,
        eids: set[int] | None = None,
    ) -> torch.Tensor:
        if self.raw_bank is None or self.raw_cache is None:
            raise RuntimeError("on_demand GPU path requires raw bank and raw cache")
        out = torch.zeros_like(x)
        cur = torch.cuda.current_stream(self.dev)
        for eid, rows, pos in _expert_groups(topk_i, self.n_experts):
            if eids is not None and eid not in eids:
                continue
            sid = self.raw_cache.get_slot(layer, eid, self.raw_bank)
            ev = self.raw_cache.ready[sid]
            if ev is not None:
                cur.wait_event(ev)
            y = _swiglu(
                x.index_select(0, rows),
                self.raw_cache.gate[sid],
                self.raw_cache.up[sid],
                self.raw_cache.down[sid],
            )
            y = y * topk_w[rows, pos].to(y.dtype).unsqueeze(1)
            out.index_add_(0, rows, y.to(out.dtype))
        return out

    def _run_split(self, layer: int, topk_i: torch.Tensor, topk_w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        active = [eid for eid, _rows, _pos in _expert_groups(topk_i, self.n_experts)]
        if not active:
            return torch.zeros_like(x)
        n_gpu = int(round(self.split_g * len(active)))
        gpu_eids = set(active[:n_gpu])
        cpu_eids = active[n_gpu:]
        fut = None
        if cpu_eids:
            if self.raw_bank is None:
                raise RuntimeError("split_cpu_gpu requires a raw ExpertBank")
            x_cpu = x.detach().to("cpu", torch.bfloat16)
            topk_i_cpu = topk_i.detach().to("cpu")
            topk_w_cpu = topk_w.detach().to("cpu", torch.bfloat16)
            fut = self.pool.submit(_cpu_grouped, self.raw_bank, layer, cpu_eids, x_cpu, topk_i_cpu, topk_w_cpu)
        gpu_out = torch.zeros_like(x) if not gpu_eids else self._run_gpu_full(layer, topk_i, topk_w, x, gpu_eids)
        if fut is None:
            return gpu_out
        cpu_out = fut.result().to(self.dev, non_blocking=True)
        return gpu_out + cpu_out

    def _run_packed_decode(
        self,
        x1: torch.Tensor,
        gp: torch.Tensor,
        gm: PackedBf16Meta,
        up: torch.Tensor,
        um: PackedBf16Meta,
        dp: torch.Tensor,
        dm: PackedBf16Meta,
    ) -> torch.Tensor:
        gate = fused_decode_gemv(gp, gm, x1)
        upv = fused_decode_gemv(up, um, x1)
        mid = (F.silu(gate) * upv).to(torch.bfloat16)
        return fused_decode_gemv(dp, dm, mid).to(torch.bfloat16).view(1, -1)

    def _run_packed_prefill(
        self,
        xe: torch.Tensor,
        gp: torch.Tensor,
        gm: PackedBf16Meta,
        up: torch.Tensor,
        um: PackedBf16Meta,
        dp: torch.Tensor,
        dm: PackedBf16Meta,
    ) -> torch.Tensor:
        if self.scratch_gate is None or self.scratch_up is None or self.scratch_down is None:
            raise RuntimeError("fused_compressed prefill scratch was not allocated")
        decode_to_bf16(gp, gm, out=self.scratch_gate)
        decode_to_bf16(up, um, out=self.scratch_up)
        decode_to_bf16(dp, dm, out=self.scratch_down)
        return _swiglu(xe, self.scratch_gate, self.scratch_up, self.scratch_down)

    def _run_packed(self, layer: int, topk_i: torch.Tensor, topk_w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.packed_bank is None or self.packed_cache is None:
            raise RuntimeError("fused_compressed requires packed bank and packed cache")
        out = torch.zeros_like(x)
        cur = torch.cuda.current_stream(self.dev)
        decode_token = x.shape[0] == 1
        for eid, rows, pos in _expert_groups(topk_i, self.n_experts):
            sid = self.packed_cache.get_slot(layer, eid, self.packed_bank)
            ev = self.packed_cache.ready[sid]
            if ev is not None:
                cur.wait_event(ev)
            gp, gm, up, um, dp, dm = self.packed_cache.tensors(sid)
            if decode_token:
                y = self._run_packed_decode(x[0], gp, gm, up, um, dp, dm)
            else:
                y = self._run_packed_prefill(x.index_select(0, rows), gp, gm, up, um, dp, dm)
            y = y * topk_w[rows, pos].to(y.dtype).unsqueeze(1)
            out.index_add_(0, rows, y.to(out.dtype))
        return out

    def run_experts(self, layer: int, topk_i: torch.Tensor, topk_w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute grouped MoE dispatch for x=[N,d]. / 对 [N,d] 做按 expert 分组 dispatch."""
        if x.ndim != 2:
            raise ValueError(f"x must be [N,d], got {tuple(x.shape)}")
        if topk_i.shape != topk_w.shape or topk_i.shape[0] != x.shape[0]:
            raise ValueError("topk_i/topk_w must both be [N,top_k] and match x rows")
        if self.policy == "cpu_serve":
            return self._run_cpu(layer, topk_i, topk_w, x)
        if self.policy == "on_demand":
            return self._run_gpu_full(layer, topk_i, topk_w, x)
        if self.policy == "fused_compressed":
            return self._run_packed(layer, topk_i, topk_w, x)
        if self.policy == "split_cpu_gpu":
            return self._run_split(layer, topk_i, topk_w, x)
        raise RuntimeError(f"unhandled policy {self.policy}")

    def cache_stats(self) -> dict[str, int]:
        cache = self.packed_cache if self.policy == "fused_compressed" else self.raw_cache
        return dict(cache.stats) if cache is not None else {"hit": 0, "miss": 0, "evict": 0}


def _get_moe(layer: nn.Module) -> nn.Module:
    if hasattr(layer, "block_sparse_moe"):
        return layer.block_sparse_moe
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts"):
        return layer.mlp
    raise TypeError("Mixtral layer must expose block_sparse_moe (transformers 4.49) or compatible mlp")


def _router_logits(gate: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(gate, nn.Linear):
        return gate(x)
    if hasattr(gate, "weight"):
        return F.linear(x, gate.weight)
    raise TypeError(f"unsupported Mixtral gate type: {type(gate).__name__}")


def _expert_weight_triple(experts: nn.Module, eid: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(experts, nn.ModuleList):
        exp = experts[eid]
        return exp.w1.weight, exp.w3.weight, exp.w2.weight
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        gate_up = experts.gate_up_proj[eid]
        mid = gate_up.shape[0] // 2
        return gate_up[:mid], gate_up[mid:], experts.down_proj[eid]
    raise TypeError(f"unsupported Mixtral experts container: {type(experts).__name__}")


def _move_non_experts_to_gpu(model: nn.Module, dev: torch.device) -> None:
    """Move every non-routed-expert module to GPU. / 只搬非 routed expert 模块到 GPU."""
    model.model.embed_tokens.to(dev)
    model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.to(dev)
    model.lm_head.to(dev)
    for layer in model.model.layers:
        layer.self_attn.to(dev)
        layer.input_layernorm.to(dev)
        layer.post_attention_layernorm.to(dev)
        _get_moe(layer).gate.to(dev)


def _validate_config(model: nn.Module) -> tuple[int, int, int, int, int]:
    cfg = model.config
    d_model = int(cfg.hidden_size)
    d_inter = int(cfg.intermediate_size)
    n_layers = int(cfg.num_hidden_layers)
    n_experts = int(cfg.num_local_experts)
    top_k = int(cfg.num_experts_per_tok)
    got = {
        "hidden_size": d_model,
        "intermediate_size": d_inter,
        "num_local_experts": n_experts,
        "num_experts_per_tok": top_k,
        "num_hidden_layers": n_layers,
    }
    bad = {k: (got[k], v) for k, v in EXPECTED.items() if got[k] != v}
    if bad:
        raise ValueError(f"expected Mixtral-8x7B config; mismatches got/expected={bad}")
    return d_model, d_inter, n_layers, n_experts, top_k


def _offload_experts(model: nn.Module, policy: str) -> tuple[ExpertBank | None, PackedExpertBank | None, int]:
    """Bank routed experts then delete them from the HF module. / 建 bank 后删除原专家引用.

    fused_compressed banks directly into packed pinned payloads so host RAM does not hold both
    ~90GB raw bf16 experts and packed copies at once.  fused 策略直接建 packed bank,避免 CPU 侧双份专家。
    """
    raw_bank = ExpertBank() if policy != "fused_compressed" else None
    packed_bank = PackedExpertBank() if policy == "fused_compressed" else None
    offload_bytes = 0
    for li, layer in enumerate(model.model.layers):
        moe = _get_moe(layer)
        if not hasattr(moe, "experts"):
            raise TypeError(f"layer {li} MoE has no experts")
        n = int(model.config.num_local_experts)
        if isinstance(moe.experts, nn.ModuleList) and len(moe.experts) != n:
            raise ValueError(f"layer {li} expected {n} experts, got {len(moe.experts)}")
        for eid in range(n):
            gate_w, up_w, down_w = _expert_weight_triple(moe.experts, eid)
            offload_bytes += sum(w.numel() * 2 for w in (gate_w, up_w, down_w))
            if raw_bank is not None:
                raw_bank.add(li, eid, gate_w, up_w, down_w)
            if packed_bank is not None:
                packed_bank.add(li, eid, gate_w, up_w, down_w)
        moe.experts = nn.ModuleList()
    gc.collect()
    torch.cuda.empty_cache()
    if raw_bank is not None:
        raw_bank.assert_cpu_only()
    return raw_bank, packed_bank, offload_bytes


def _assert_model_resident_and_experts_gone(model: nn.Module, dev: torch.device) -> None:
    expert_gpu = [(n, p.device) for n, p in model.named_parameters() if ".experts." in n and p.device.type == "cuda"]
    if expert_gpu:
        raise RuntimeError(f"expert weights remain on GPU; first={expert_gpu[0]}")
    cpu_params = [(n, p.device) for n, p in model.named_parameters() if p.device != dev]
    if cpu_params:
        raise RuntimeError(f"non-expert parameter was not moved to {dev}; first={cpu_params[0]}")
    for li, layer in enumerate(model.model.layers):
        if len(_get_moe(layer).experts) != 0:
            raise RuntimeError(f"layer {li} experts were not removed after banking")


def _cache_capacity(
    *,
    policy: str,
    free_bytes: int,
    reserve_gb: float,
    total_experts: int,
    bf16_per_expert: int,
    packed_bank: PackedExpertBank | None,
) -> tuple[int, int, int]:
    usable = int(free_bytes - reserve_gb * (1024**3))
    if usable <= 0:
        raise RuntimeError(
            f"reserve_gb={reserve_gb} leaves no GPU memory for expert cache "
            f"(free={free_bytes / 1024**3:.2f} GiB)"
        )
    bf16_cap = max(0, min(total_experts, usable // bf16_per_expert))
    if policy == "cpu_serve":
        return 0, bf16_cap, bf16_per_expert
    if policy == "fused_compressed":
        if packed_bank is None:
            raise RuntimeError("fused_compressed requires packed_bank before sizing")
        per_cache = packed_bank.max_cache_bytes()
        cap = min(total_experts, usable // per_cache)
        if cap < 1:
            raise RuntimeError(
                f"not enough GPU memory for one packed expert cache slot after reserve "
                f"(usable={usable / 1024**2:.1f}MiB, per_slot={per_cache / 1024**2:.1f}MiB)"
            )
        return cap, bf16_cap, per_cache
    if bf16_cap < 1:
        raise RuntimeError(
            f"not enough GPU memory for one bf16 expert cache slot after reserve "
            f"(usable={usable / 1024**2:.1f}MiB, per_slot={bf16_per_expert / 1024**2:.1f}MiB)"
        )
    cap = bf16_cap
    return cap, bf16_cap, bf16_per_expert


def _make_patched_forward(block: nn.Module, layer_idx: int, rt: Runtime, return_router_logits: bool):
    top_k = int(getattr(block, "top_k", EXPECTED["num_experts_per_tok"]))
    gate = block.gate

    def forward(hidden_states: torch.Tensor):
        bsz, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)
        router_logits = _router_logits(gate, x)
        # Stock Mixtral routing: softmax float -> top2 -> renormalize -> cast to activation dtype.
        # 严格复现 stock Mixtral routing,权重归一化后再 cast 回 bf16。
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        out = rt.run_experts(layer_idx, selected_experts, routing_weights, x)
        out = out.reshape(bsz, seq_len, hidden_dim)
        if return_router_logits:
            return out, router_logits
        return out

    return forward


def _patch_model(model: nn.Module, rt: Runtime) -> None:
    for li, layer in enumerate(model.model.layers):
        moe = _get_moe(layer)
        return_router_logits = hasattr(layer, "block_sparse_moe")
        moe.forward = _make_patched_forward(moe, li, rt, return_router_logits)


def setup(
    model_dir: str,
    gpu: int,
    policy: str,
    reserve_gb: float = 4.0,
    split_g: float = 0.5,
    budget_gb: float = 0.0,
) -> tuple[nn.Module, object, dict[str, object]]:
    """Load, offload, patch, and return a normal HF-callable model. / 加载并 patch 模型."""
    if policy not in VALID_POLICIES:
        raise ValueError(f"policy must be one of {VALID_POLICIES}, got {policy!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Mixtral offload runtime")
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(dev)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=False,
        local_files_only=True,
    ).eval()
    d_model, d_inter, n_layers, n_experts, _top_k = _validate_config(model)

    _move_non_experts_to_gpu(model, dev)
    raw_bank, packed_bank, offload_bytes = _offload_experts(model, policy)
    _assert_model_resident_and_experts_gone(model, dev)

    torch.cuda.synchronize(dev)
    free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
    total_experts = n_layers * n_experts
    bf16_per_expert = 3 * d_inter * d_model * 2
    cache_experts, bf16_cap, per_cache_expert = _cache_capacity(
        policy=policy,
        free_bytes=int(free_bytes),
        reserve_gb=reserve_gb,
        total_experts=total_experts,
        bf16_per_expert=bf16_per_expert,
        packed_bank=packed_bank,
    )
    print(
        f"[cache] free={free_bytes / 1024**3:.2f}GiB reserve={reserve_gb:.2f}GiB "
        f"bf16_per_expert={bf16_per_expert / 1024**2:.1f}MiB -> "
        f"bf16_capacity={bf16_cap}/{total_experts} ({100 * bf16_cap / total_experts:.1f}%)"
    )
    print(
        f"[cache] policy={policy} cache_experts={cache_experts}/{total_experts} "
        f"({100 * cache_experts / total_experts:.1f}%) "
        f"cache_bytes_per_expert={per_cache_expert / 1024**2:.1f}MiB"
    )

    h2d = torch.cuda.Stream(device=dev)
    raw_cache = None
    packed_cache = None
    if policy in ("on_demand", "split_cpu_gpu"):
        raw_cache = GpuExpertCache(cache_experts, d_model, d_inter, dev, h2d)
    elif policy == "fused_compressed":
        if packed_bank is None:
            raise RuntimeError("packed_bank missing for fused_compressed")
        packed_cache = PackedGpuExpertCache(cache_experts, packed_bank, dev, h2d)

    rt = Runtime(
        policy=policy,
        dev=dev,
        n_experts=n_experts,
        raw_bank=raw_bank,
        raw_cache=raw_cache,
        packed_bank=packed_bank,
        packed_cache=packed_cache,
        d_model=d_model,
        d_inter=d_inter,
        split_g=split_g,
    )
    _patch_model(model, rt)
    model._offload_mixtral_runtime = rt

    info = {
        "policy": policy,
        "device": str(dev),
        "cache_experts": cache_experts,
        "cache_percent": cache_experts / total_experts,
        "bf16_capacity_experts": bf16_cap,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "total_experts": total_experts,
        "hidden_size": d_model,
        "intermediate_size": d_inter,
        "offload_bytes": offload_bytes,
        "bf16_per_expert_bytes": bf16_per_expert,
        "cache_bytes_per_expert": per_cache_expert,
        "gpu_total_bytes": int(total_bytes),
        "gpu_free_after_offload_bytes": int(free_bytes),
        "reserve_gb": reserve_gb,
        "split_g": split_g,
        "packed_payload_bytes": 0 if packed_bank is None else packed_bank.payload_bytes,
        "packed_metadata_bytes": 0 if packed_bank is None else packed_bank.metadata_bytes,
    }
    return model, tokenizer, info


@torch.inference_mode()
def _smoke(model: nn.Module, tokenizer: object, dev: torch.device, decode_tokens: int) -> tuple[float, float]:
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(dev)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize(dev)
    ttft_ms = (time.perf_counter() - t0) * 1000.0
    past = out.past_key_values
    cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    step_ms = []
    for _ in range(decode_tokens):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        torch.cuda.synchronize(dev)
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return ttft_ms, sum(step_ms) / max(1, len(step_ms))


def main() -> None:
    p = argparse.ArgumentParser(description="single-GPU offloaded Mixtral-8x7B TTFT/TPOT smoke")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--policy", choices=VALID_POLICIES, required=True)
    p.add_argument("--reserve_gb", type=float, default=4.0)
    p.add_argument("--split_g", type=float, default=0.5)
    p.add_argument("--decode_tokens", type=int, default=4)
    p.add_argument("--cpu_threads", type=int, default=16)
    args = p.parse_args()

    torch.set_num_threads(args.cpu_threads)
    model, tokenizer, info = setup(args.model_dir, args.gpu, args.policy, args.reserve_gb, args.split_g)
    dev = torch.device(info["device"])
    ttft_ms, tpot_ms = _smoke(model, tokenizer, dev, args.decode_tokens)
    rt = model._offload_mixtral_runtime
    print(
        f"[smoke] policy={args.policy} TTFT_prefill_ms={ttft_ms:.3f} "
        f"TPOT_mean_decode_ms={tpot_ms:.3f} cache={rt.cache_stats()}"
    )


if __name__ == "__main__":
    main()
