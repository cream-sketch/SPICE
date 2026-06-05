"""FP16 exact continuous-batch offloaded Qwen1.5-MoE runtime.

This harness is the first positive-regime test for SPICE after the batch=1
negative results: continuous batching creates expert reuse.  SPICE is used as a
future expert-union oracle over the current batch, not as a single-token
fetch-all prefetcher.

Policies:
  on_demand      : actual batch expert union; each unique expert fetched once.
  cpu_serve      : actual batch expert union; each unique expert computed on CPU.
  spice_prefetch : real SPICE fcast issues low-priority transient staging for
                   future batch expert union; actual routes are still verified.
  spice_dummy    : same forecast H2D traffic as spice_prefetch, never consumed.

The timed path is teacher-forced when --text_file is provided.  This is required
for real SPICE forecast dumps: fcast[layer, horizon, sequence_position] only
aligns with the original token stream used by spice_draft.cli.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


_TEXT = (
    "The history of mixture of experts models in large language modeling spans several decades. "
    "In economics the theory of comparative advantage explains how trade between nations creates value. "
    "Photosynthesis converts sunlight carbon dioxide and water into glucose and oxygen in plant cells. "
    "The French Revolution began in 1789 and reshaped the political landscape of modern Europe. "
    "Quantum entanglement links the states of particles so that measuring one affects the other instantly. "
    "Basketball strategy relies on spacing ball movement and reading the defense to create open shots. "
    "The mitochondria is the powerhouse of the cell generating ATP through oxidative phosphorylation. "
    "Climate models simulate the atmosphere oceans and ice to project future global temperature change. "
    "Jazz improvisation builds on chord progressions while soloists explore melody rhythm and harmony. "
    "The Roman aqueducts carried water across vast distances using a precise gradient and stone arches. "
    "Neural networks learn representations by adjusting weights through gradient descent on a loss surface. "
    "Volcanic eruptions release ash gas and lava shaping landscapes and influencing the global climate. "
    "Shakespeare wrote tragedies comedies and histories that still define much of English literature. "
    "Supply and demand determine market prices as buyers and sellers respond to incentives and scarcity. "
    "The human immune system defends the body with innate barriers and adaptive antibody responses. "
    "Deep sea creatures survive crushing pressure and darkness using bioluminescence and slow metabolism."
)


VALID_POLICIES = ("on_demand", "cpu_serve", "spice_prefetch", "spice_dummy")


def parse_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    raise ValueError(f"this batch harness is intentionally FP16-only, got {name!r}")


def swiglu(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)


class ExpertBank:
    """CPU pinned expert weights keyed by (model_layer, expert)."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.w: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.bytes = 0

    def add(self, layer: int, eid: int, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> None:
        triple = tuple(
            t.detach().to("cpu", self.dtype).contiguous().pin_memory()
            for t in (gate_w, up_w, down_w)
        )
        self.w[(layer, eid)] = triple
        self.bytes += sum(t.numel() * t.element_size() for t in triple)

    def get(self, layer: int, eid: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.w[(layer, eid)]


class GpuExpertCache:
    """Resident LRU cache for demand-fetched full FP16 experts."""

    def __init__(self, capacity: int, d_model: int, d_inter: int, dev: torch.device,
                 dtype: torch.dtype, h2d_stream: torch.cuda.Stream) -> None:
        if capacity < 1:
            raise ValueError("cache capacity must be >= 1")
        self.cap = capacity
        self.dev = dev
        self.dtype = dtype
        self.h2d = h2d_stream
        self.free = list(range(capacity))
        self.gate = torch.empty(capacity, d_inter, d_model, device=dev, dtype=dtype)
        self.up = torch.empty(capacity, d_inter, d_model, device=dev, dtype=dtype)
        self.down = torch.empty(capacity, d_model, d_inter, device=dev, dtype=dtype)
        self.map: dict[tuple[int, int], int] = {}
        self.lru: list[int] = []
        self.ready: list[torch.cuda.Event | None] = [None] * capacity
        self.stats = defaultdict(float)

    def _touch(self, sid: int) -> None:
        self.lru.remove(sid)
        self.lru.append(sid)

    def _evict(self) -> int:
        sid = self.lru.pop(0)
        victim = next(k for k, v in self.map.items() if v == sid)
        del self.map[victim]
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
        if self.free:
            sid = self.free.pop()
            wait_prior_use = None
        else:
            sid = self._evict()
            # Do not let the H2D stream overwrite an evicted slot before prior
            # main-stream GEMMs that read that slot have completed.
            wait_prior_use = torch.cuda.Event()
            wait_prior_use.record(torch.cuda.current_stream(self.dev))

        self.map[key] = sid
        self.lru.append(sid)
        gate_w, up_w, down_w = bank.get(layer, eid)
        with torch.cuda.stream(self.h2d):
            if wait_prior_use is not None:
                self.h2d.wait_event(wait_prior_use)
            self.gate[sid].copy_(gate_w, non_blocking=True)
            self.up[sid].copy_(up_w, non_blocking=True)
            self.down[sid].copy_(down_w, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.h2d)
        self.ready[sid] = ev
        return sid

    def tensors(self, sid: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.gate[sid], self.up[sid], self.down[sid]

    def reset_state(self) -> None:
        self.free = list(range(self.cap))
        self.map.clear()
        self.lru.clear()
        self.ready = [None] * self.cap


class TransientStaging:
    """Low-priority transient staging slots for SPICE forecasted future experts."""

    def __init__(self, slots: int, d_model: int, d_inter: int, dev: torch.device,
                 dtype: torch.dtype, low_stream: torch.cuda.Stream) -> None:
        self.slots = slots
        self.dev = dev
        self.low = low_stream
        self.free = list(range(slots))
        self.gate = torch.empty(slots, d_inter, d_model, device=dev, dtype=dtype) if slots else None
        self.up = torch.empty(slots, d_inter, d_model, device=dev, dtype=dtype) if slots else None
        self.down = torch.empty(slots, d_model, d_inter, device=dev, dtype=dtype) if slots else None
        self.map: dict[tuple[int, int], int] = {}
        self.ready: list[torch.cuda.Event | None] = [None] * slots
        self.retiring: list[tuple[int, torch.cuda.Event]] = []
        self.stats = defaultdict(float)

    def reclaim(self) -> None:
        kept = []
        for sid, ev in self.retiring:
            if ev.query():
                self.free.append(sid)
            else:
                kept.append((sid, ev))
        self.retiring = kept

    def issue(self, layer: int, eid: int, bank: ExpertBank, resident: GpuExpertCache) -> bool:
        self.reclaim()
        key = (layer, eid)
        self.stats["forecast_candidates"] += 1
        if self.slots <= 0:
            self.stats["drop_capacity"] += 1
            return False
        if key in resident.map:
            self.stats["skip_resident"] += 1
            return False
        if key in self.map:
            self.stats["skip_duplicate"] += 1
            return False
        if not self.free:
            self.stats["drop_capacity"] += 1
            return False
        sid = self.free.pop()
        self.map[key] = sid
        gate_w, up_w, down_w = bank.get(layer, eid)
        with torch.cuda.stream(self.low):
            self.gate[sid].copy_(gate_w, non_blocking=True)
            self.up[sid].copy_(up_w, non_blocking=True)
            self.down[sid].copy_(down_w, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.low)
        self.ready[sid] = ev
        self.stats["issued"] += 1
        return True

    def expire_before(self, layer: int) -> None:
        stale = [key for key in self.map if key[0] < layer]
        for key in stale:
            self._retire_key(key, used=False)

    def flush(self) -> None:
        for key in list(self.map):
            self._retire_key(key, used=False)
        self.reclaim()

    def reset_state(self) -> None:
        self.free = list(range(self.slots))
        self.map.clear()
        self.ready = [None] * self.slots
        self.retiring.clear()

    def _retire_key(self, key: tuple[int, int], used: bool) -> None:
        sid = self.map.pop(key)
        ev = self.ready[sid]
        self.ready[sid] = None
        if used:
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(self.dev))
            self.retiring.append((sid, done))
        elif ev is not None and not ev.query():
            self.retiring.append((sid, ev))
            self.stats["expired_pending"] += 1
        else:
            self.free.append(sid)
            self.stats["expired_ready"] += 1

    def get_ready(self, layer: int, eid: int):
        key = (layer, eid)
        sid = self.map.get(key)
        if sid is None:
            return None
        ev = self.ready[sid]
        if ev is None or not ev.query():
            self.stats["late"] += 1
            return None
        self.stats["hit"] += 1
        return sid, (self.gate[sid], self.up[sid], self.down[sid])

    def mark_consumed(self, layer: int, eid: int) -> None:
        key = (layer, eid)
        if key in self.map:
            self._retire_key(key, used=True)


def offload_experts(model, dtype: torch.dtype) -> tuple[ExpertBank, int, int, int, int]:
    bank = ExpertBank(dtype)
    layers = model.model.layers
    d_model = model.config.hidden_size
    d_inter = model.config.moe_intermediate_size
    n_exp = model.config.num_experts
    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        for eid, expert in enumerate(mlp.experts):
            bank.add(layer_idx, eid, expert.gate_proj.weight, expert.up_proj.weight, expert.down_proj.weight)
        mlp.experts = torch.nn.ModuleList()
    torch.cuda.empty_cache()
    return bank, d_model, d_inter, len(layers), n_exp


@dataclass
class ForecastItem:
    name: str
    true_top: torch.Tensor
    fcast: torch.Tensor


def load_forecast_items(root: str) -> tuple[list[ForecastItem], dict]:
    froot = Path(root)
    manifest = json.loads((froot / "manifest.json").read_text())
    files = manifest.get("files") or sorted(p.name for p in froot.glob("fc_*.pt"))
    items = []
    for name in files:
        d = torch.load(froot / name, map_location="cpu", weights_only=False)
        items.append(ForecastItem(name=name, true_top=d["true_top"].long(), fcast=d["fcast"].long()))
    if not items:
        raise ValueError(f"no forecast files in {root}")
    return items, manifest


def read_texts(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize_texts(tok, texts: list[str], max_len: int) -> list[torch.Tensor]:
    out = []
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]
        out.append(ids)
    return out


def build_text_batch(tokenized: list[torch.Tensor], batch: int, prompt_len: int, decode_tokens: int,
                     draw: int, dev: torch.device, forecasts: list[ForecastItem] | None = None):
    seq_ids = []
    prompts = []
    continuations = []
    chosen_fc = []
    need = prompt_len + decode_tokens
    start = (draw * batch) % len(tokenized)
    probe = 0
    idx = start
    while len(prompts) < batch and probe < len(tokenized) * 2:
        ids = tokenized[idx % len(tokenized)]
        fc = forecasts[idx % len(forecasts)] if forecasts else None
        enough_tokens = ids.numel() >= need
        enough_forecast = fc is None or fc.fcast.shape[2] >= need
        if enough_tokens and enough_forecast:
            seq_ids.append(idx % len(tokenized))
            prompts.append(ids[:prompt_len])
            continuations.append(ids[prompt_len:need])
            if fc is not None:
                chosen_fc.append(fc)
        idx += 1
        probe += 1
    if len(prompts) < batch:
        raise ValueError(
            f"not enough texts with >= {need} tokens and matching forecast length; got {len(prompts)}/{batch}"
        )
    return (
        torch.stack(prompts).to(dev),
        torch.stack(continuations).to(dev),
        chosen_fc if forecasts else None,
        seq_ids,
    )


def load_token_stream(tok, dataset_dir: str, split: str) -> torch.Tensor:
    if not dataset_dir:
        return tok(_TEXT, return_tensors="pt").input_ids[0]
    from datasets import load_from_disk

    ds = load_from_disk(dataset_dir)[split]
    text = "\n".join(t for t in ds["text"] if t and t.strip())
    return tok(text, return_tensors="pt", truncation=False).input_ids[0]


def build_stream_batch(stream: torch.Tensor, batch: int, prompt_len: int, decode_tokens: int,
                       draw: int, dev: torch.device):
    need = prompt_len + decode_tokens
    chunks = []
    conts = []
    span = batch * need
    base = (draw * span) % max(1, stream.numel() - span)
    for b in range(batch):
        s = base + b * need
        seq = stream[s:s + need]
        if seq.numel() < need:
            raise ValueError("token stream too short for requested prompt/decode window")
        chunks.append(seq[:prompt_len])
        conts.append(seq[prompt_len:need])
    return torch.stack(chunks).to(dev), torch.stack(conts).to(dev), None, list(range(batch))


class BatchRuntime:
    def __init__(self, bank: ExpertBank, cache: GpuExpertCache, staging: TransientStaging | None,
                 dev: torch.device, policy: str, fallback: str, max_lead: int,
                 prefetch_per_layer: int, forecast_alignment: str,
                 prefetch_budget_per_step: int) -> None:
        self.bank = bank
        self.cache = cache
        self.staging = staging
        self.dev = dev
        self.policy = policy
        self.fallback = fallback
        self.max_lead = max_lead
        self.prefetch_per_layer = prefetch_per_layer
        self.forecast_alignment = forecast_alignment
        self.prefetch_budget_per_step = prefetch_budget_per_step
        self.prefetch_budget_remaining = prefetch_budget_per_step
        self.prefetch_enabled = False
        self.current_positions: list[int] = []
        self.current_forecasts: list[ForecastItem] = []
        self.stats = defaultdict(float)

    def set_step(self, positions: list[int], forecasts: list[ForecastItem] | None) -> None:
        self.current_positions = positions
        self.current_forecasts = forecasts or []
        self.prefetch_budget_remaining = self.prefetch_budget_per_step

    def finish_step(self) -> None:
        if self.staging is not None:
            self.staging.flush()

    def issue_forecast(self, anchor_layer: int) -> None:
        if self.policy not in ("spice_prefetch", "spice_dummy") or not self.prefetch_enabled:
            return
        if not self.current_forecasts or self.staging is None:
            return
        self.staging.expire_before(anchor_layer)
        scored: dict[tuple[int, int], list[float]] = {}
        true_future_union = set()
        for pos, fc in zip(self.current_positions, self.current_forecasts):
            fcast = fc.fcast
            if anchor_layer >= fcast.shape[0] or pos >= fcast.shape[2]:
                continue
            max_h = min(self.max_lead + 1, fcast.shape[1])
            for h in range(1, max_h):
                target_layer = anchor_layer + h
                if target_layer >= fcast.shape[0]:
                    break
                if pos < fc.true_top.shape[1]:
                    for true_eid in fc.true_top[target_layer, pos].tolist():
                        if true_eid >= 0:
                            true_future_union.add((target_layer, int(true_eid)))
                added = 0
                for eid in fcast[anchor_layer, h, pos].tolist():
                    if eid < 0:
                        continue
                    key = (target_layer, int(eid))
                    # Score by cross-batch multiplicity and deadline.  A global
                    # budget is useful only if the scarce PCIe slots go to
                    # experts that are close and likely shared.
                    if key not in scored:
                        scored[key] = [0.0, float(h)]
                    scored[key][0] += 1.0 / float(h)
                    scored[key][1] = min(scored[key][1], float(h))
                    added += 1
                    if added >= self.prefetch_per_layer:
                        break
        candidates = sorted(scored, key=lambda k: (scored[k][1], -scored[k][0], k[0], k[1]))
        self.stats["forecast_candidate_union"] += len(candidates)

        union: list[tuple[int, int]] = []
        free_slots = len(self.staging.free)
        budget_left = self.prefetch_budget_remaining
        for key in candidates:
            if key in self.cache.map:
                self.stats["forecast_plan_skip_resident"] += 1
                continue
            if key in self.staging.map:
                self.stats["forecast_plan_skip_duplicate"] += 1
                continue
            if free_slots <= 0:
                self.stats["forecast_plan_drop_capacity"] += 1
                continue
            if budget_left >= 0 and budget_left <= 0:
                self.stats["forecast_budget_drop"] += 1
                continue
            union.append(key)
            free_slots -= 1
            if budget_left >= 0:
                budget_left -= 1

        issued: list[tuple[int, int]] = []
        for layer, eid in union:
            if self.staging.issue(layer, eid, self.bank, self.cache):
                issued.append((layer, eid))
                if self.prefetch_budget_remaining >= 0:
                    self.prefetch_budget_remaining -= 1
            else:
                self.stats["forecast_issue_failed_after_plan"] += 1
        self.stats["forecast_union"] += len(issued)
        self.stats["forecast_selected"] += len(issued)
        self.stats["forecast_true_union"] += len(true_future_union)
        self.stats["forecast_true_positive"] += len(set(issued) & true_future_union)
        self.stats["forecast_selected_true_positive"] += len(set(issued) & true_future_union)

    def run_experts_batched(self, layer: int, topk_i: torch.Tensor, topk_w: torch.Tensor,
                            x: torch.Tensor) -> torch.Tensor:
        n_rows, _dim = x.shape
        out = torch.zeros_like(x)
        groups: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
        ids = topk_i.tolist()
        weights = topk_w.tolist()
        for row in range(n_rows):
            for rank, eid in enumerate(ids[row]):
                groups[int(eid)].append((row, float(weights[row][rank]), rank))
        self.stats["actual_unique"] += len(groups)
        self.stats["assignments"] += int(topk_i.numel())
        cur = torch.cuda.current_stream(self.dev)
        cpu_groups: list[tuple[int, list[tuple[int, float, int]]]] = []

        # HF Qwen MoE accumulates by expert index order.  Keep the same expert
        # order here; the benefit still comes from grouping rows per expert.
        for eid in sorted(groups):
            toks = groups[eid]
            used_stage = False
            staged = None
            if self.policy == "spice_prefetch" and self.staging is not None:
                staged = self.staging.get_ready(layer, eid)
            elif self.policy == "spice_dummy" and self.staging is not None and (layer, eid) in self.staging.map:
                self.staging.stats["dummy_available"] += 1

            rows = torch.tensor([r for r, _w, _rank in toks], device=self.dev)
            w = torch.tensor([wt for _r, wt, _rank in toks], device=self.dev, dtype=x.dtype).unsqueeze(1)
            xe = x.index_select(0, rows)

            if staged is not None:
                _sid, weights_gpu = staged
                y = swiglu(xe, *weights_gpu) * w
                out.index_add_(0, rows, y)
                self.staging.mark_consumed(layer, eid)
                used_stage = True
                self.stats["staged_used"] += 1

            if used_stage:
                continue

            if self.policy == "cpu_serve" or (self.policy in ("spice_prefetch", "spice_dummy")
                                              and self.fallback == "cpu"):
                cpu_groups.append((eid, toks))
                self.stats["cpu_groups"] += 1
                continue

            sid = self.cache.get_slot(layer, eid, self.bank)
            ev = self.cache.ready[sid]
            if ev is not None:
                cur.wait_event(ev)
            y = swiglu(xe, *self.cache.tensors(sid)) * w
            out.index_add_(0, rows, y)
            self.stats["gpu_groups"] += 1
        if cpu_groups:
            self._run_cpu_groups(layer, cpu_groups, x, out)
        return out

    def check_forecast_alignment(self, layer: int, topk_i: torch.Tensor) -> bool:
        if self.forecast_alignment == "off" or not self.prefetch_enabled or not self.current_forecasts:
            return True
        if topk_i.shape[0] != len(self.current_forecasts):
            self.stats["forecast_align_skipped_shape"] += 1
            return False
        got = topk_i.detach().cpu()
        ok = True
        for row, (pos, fc) in enumerate(zip(self.current_positions, self.current_forecasts)):
            if layer >= fc.true_top.shape[0] or pos >= fc.true_top.shape[1]:
                self.stats["forecast_align_skipped_oob"] += 1
                ok = False
                continue
            expect = fc.true_top[layer, pos].detach().cpu()
            self.stats["forecast_align_checked"] += 1
            if set(got[row].tolist()) != set(expect.tolist()):
                ok = False
                self.stats["forecast_align_mismatch"] += 1
                if self.forecast_alignment == "fail":
                    raise RuntimeError(
                        f"forecast true_top mismatch at layer={layer} row={row} pos={pos}: "
                        f"runtime={got[row].tolist()} dump={expect.tolist()} "
                        f"(regenerate forecast with matching tokenizer/model/dtype)"
                    )
        if not ok:
            self.stats["forecast_align_blocked_prefetch"] += 1
        return ok

    def _run_cpu_group(self, layer: int, eid: int, toks: list[tuple[int, float, int]],
                       x: torch.Tensor, out_gpu: torch.Tensor) -> None:
        x_cpu = x.detach().to("cpu", self.bank.dtype)
        out_cpu = torch.zeros_like(x_cpu)
        self._run_cpu_groups_from_cpu(layer, [(eid, toks)], x_cpu, out_cpu)
        out_gpu.add_(out_cpu.to(self.dev, non_blocking=True))

    def _run_cpu_groups(self, layer: int, groups: list[tuple[int, list[tuple[int, float, int]]]],
                        x: torch.Tensor, out_gpu: torch.Tensor) -> None:
        x_cpu = x.detach().to("cpu", self.bank.dtype)
        out_cpu = torch.zeros_like(x_cpu)
        self._run_cpu_groups_from_cpu(layer, groups, x_cpu, out_cpu)
        out_gpu.add_(out_cpu.to(self.dev, non_blocking=True))
        self.stats["cpu_layers"] += 1

    def _run_cpu_groups_from_cpu(self, layer: int, groups: list[tuple[int, list[tuple[int, float, int]]]],
                                 x_cpu: torch.Tensor, out_cpu: torch.Tensor) -> None:
        for eid, toks in groups:
            rows_cpu = torch.tensor([r for r, _w, _rank in toks], dtype=torch.long)
            weights_cpu = torch.tensor([wt for _r, wt, _rank in toks], dtype=self.bank.dtype).unsqueeze(1)
            xe = x_cpu.index_select(0, rows_cpu)
            y_cpu = swiglu(xe, *self.bank.get(layer, eid)) * weights_cpu
            out_cpu.index_add_(0, rows_cpu, y_cpu)


def make_batched_forward(block, layer_idx: int, rt: BatchRuntime):
    gate = block.gate
    shared_expert = block.shared_expert
    shared_gate = block.shared_expert_gate
    top_k = block.top_k
    norm_topk = block.norm_topk_prob

    def forward(hidden_states):
        bsz, seq_len, dim = hidden_states.shape
        x = hidden_states.reshape(-1, dim)
        router_logits = gate(x)
        routing = F.softmax(router_logits, dim=-1, dtype=torch.float)
        topk_w, topk_i = torch.topk(routing, top_k, dim=-1)
        if norm_topk:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        topk_w = topk_w.to(x.dtype)
        shared = F.sigmoid(shared_gate(x)) * shared_expert(x)
        if rt.check_forecast_alignment(layer_idx, topk_i):
            rt.issue_forecast(layer_idx)
        routed = rt.run_experts_batched(layer_idx, topk_i, topk_w, x)
        return (routed + shared).view(bsz, seq_len, dim), router_logits

    return forward


def patch_model(model, rt: BatchRuntime) -> None:
    for layer_idx, layer in enumerate(model.model.layers):
        layer.mlp.forward = make_batched_forward(layer.mlp, layer_idx, rt)


@torch.inference_mode()
def teacher_forced_logits(model, input_ids: torch.Tensor, token_ids: torch.Tensor,
                          rt: BatchRuntime | None = None,
                          forecasts: list[ForecastItem] | None = None,
                          prompt_len: int = 0):
    if rt is not None:
        rt.prefetch_enabled = False
    out = model(input_ids=input_ids, use_cache=True)
    kv = out.past_key_values
    logits = []
    for step in range(token_ids.shape[1]):
        cur = token_ids[:, step:step + 1]
        if rt is not None:
            rt.set_step([prompt_len + step] * token_ids.shape[0], forecasts)
            rt.prefetch_enabled = True
        out = model(input_ids=cur, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        logits.append(out.logits[:, -1, :].float().cpu())
        if rt is not None:
            rt.finish_step()
            rt.prefetch_enabled = False
    return logits


@torch.inference_mode()
def timed_teacher_forced(model, input_ids: torch.Tensor, token_ids: torch.Tensor, warmup: int,
                         dev: torch.device, rt: BatchRuntime | None = None,
                         forecasts: list[ForecastItem] | None = None,
                         prompt_len: int = 0) -> float:
    if rt is not None:
        rt.prefetch_enabled = False
    out = model(input_ids=input_ids, use_cache=True)
    kv = out.past_key_values
    start = None
    for step in range(token_ids.shape[1]):
        if step == warmup:
            if rt is not None:
                reset_stats(rt, rt.cache, rt.staging)
            torch.cuda.synchronize(dev)
            start = time.perf_counter()
        cur = token_ids[:, step:step + 1]
        if rt is not None:
            rt.set_step([prompt_len + step] * token_ids.shape[0], forecasts)
            rt.prefetch_enabled = True
        out = model(input_ids=cur, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        if rt is not None:
            rt.finish_step()
            rt.prefetch_enabled = False
    torch.cuda.synchronize(dev)
    if start is None:
        return 0.0
    return (time.perf_counter() - start) * 1000.0 / max(1, token_ids.shape[1] - warmup)


def summarize_stats(rt: BatchRuntime, cache: GpuExpertCache, staging: TransientStaging | None) -> dict:
    out = {f"rt_{k}": float(v) for k, v in rt.stats.items()}
    out.update({f"cache_{k}": float(v) for k, v in cache.stats.items()})
    if staging is not None:
        out.update({f"staging_{k}": float(v) for k, v in staging.stats.items()})
    return out


def reset_stats(rt: BatchRuntime, cache: GpuExpertCache, staging: TransientStaging | None) -> None:
    rt.stats.clear()
    cache.stats.clear()
    if staging is not None:
        staging.stats.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="FP16 exact continuous-batch offloaded Qwen1.5-MoE decode")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--decode_tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--cache_experts", type=int, required=True)
    parser.add_argument("--policy", choices=VALID_POLICIES, default="on_demand")
    parser.add_argument("--fallback", choices=["on_demand", "cpu"], default="on_demand",
                        help="fallback for spice_prefetch/spice_dummy when staged expert is absent or late")
    parser.add_argument("--dtype", choices=["fp16"], default="fp16")
    parser.add_argument("--cpu_threads", type=int, default=16)
    parser.add_argument("--dataset_dir", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text_file", default="", help="text file used to build the SPICE forecast dump")
    parser.add_argument("--forecast_dir", default="", help="real SPICE forecast dump dir; required for spice_*")
    parser.add_argument("--max_samples", type=int, default=0, help="optional cap for text_file/forecast rows")
    parser.add_argument("--n_batches", type=int, default=4)
    parser.add_argument("--max_lead", type=int, default=3)
    parser.add_argument("--prefetch_per_layer", type=int, default=4)
    parser.add_argument("--prefetch_budget_per_step", type=int, default=-1,
                        help="-1 means unlimited; otherwise max forecast H2D expert issues per decode step")
    parser.add_argument("--staging_slots", type=int, default=256)
    parser.add_argument("--forecast_alignment", choices=["off", "warn", "fail"], default="fail")
    parser.add_argument("--persistent_state", action="store_true",
                        help="keep HBM cache/staging state across n_batches draws")
    parser.add_argument("--check_exact", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.warmup < 0 or args.warmup >= args.decode_tokens:
        raise ValueError("--warmup must satisfy 0 <= warmup < decode_tokens")
    if args.policy in ("spice_prefetch", "spice_dummy") and not (args.forecast_dir and args.text_file):
        raise ValueError("spice_prefetch/spice_dummy require --forecast_dir and --text_file for position alignment")

    dtype = parse_dtype(args.dtype)
    torch.set_num_threads(args.cpu_threads)
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(dev).eval()

    forecasts = None
    tokenized = None
    manifest = None
    if args.text_file:
        texts = read_texts(args.text_file)
        if args.max_samples:
            texts = texts[:args.max_samples]
        tokenized = tokenize_texts(tokenizer, texts, args.prompt_len + args.decode_tokens)
    if args.forecast_dir:
        forecasts, manifest = load_forecast_items(args.forecast_dir)
        if args.max_samples:
            forecasts = forecasts[:args.max_samples]
        if tokenized is not None and len(forecasts) != len(tokenized):
            n = min(len(forecasts), len(tokenized))
            forecasts = forecasts[:n]
            tokenized = tokenized[:n]
            print(f"[align] truncated text/forecast rows to {n}")
        f_dtype = manifest.get("dtype")
        if f_dtype is None:
            raise ValueError("forecast manifest has no dtype; regenerate with spice_draft --dtype fp16")
        elif f_dtype != args.dtype:
            raise ValueError(f"forecast dtype {f_dtype!r} does not match runtime dtype {args.dtype!r}")
        print(f"[forecast] files={len(forecasts)} top_k={manifest.get('top_k')} "
              f"max_horizon={manifest.get('max_horizon')} oracle={manifest.get('oracle_fcast')}")

    if tokenized is None:
        stream = load_token_stream(tokenizer, args.dataset_dir, args.split)
    else:
        stream = None

    exact_batch = None
    ref_logits = None
    if args.check_exact:
        if tokenized is not None:
            exact_batch = build_text_batch(
                tokenized, args.batch, args.prompt_len, args.decode_tokens, 0, dev, forecasts
            )
        else:
            exact_batch = build_stream_batch(stream, args.batch, args.prompt_len, args.decode_tokens, 0, dev)
        print("[exact] collecting full-resident FP16 reference before expert offload", flush=True)
        ref_logits = teacher_forced_logits(model, exact_batch[0], exact_batch[1])

    bank, d_model, d_inter, n_layers, n_exp = offload_experts(model, dtype)
    h2d = torch.cuda.Stream(device=dev)
    low = torch.cuda.Stream(device=dev)
    cache = GpuExpertCache(args.cache_experts, d_model, d_inter, dev, dtype, h2d)
    staging = TransientStaging(args.staging_slots, d_model, d_inter, dev, dtype, low) \
        if args.policy in ("spice_prefetch", "spice_dummy") else None
    rt = BatchRuntime(
        bank=bank,
        cache=cache,
        staging=staging,
        dev=dev,
        policy=args.policy,
        fallback=args.fallback,
        max_lead=args.max_lead,
        prefetch_per_layer=args.prefetch_per_layer,
        forecast_alignment=args.forecast_alignment,
        prefetch_budget_per_step=args.prefetch_budget_per_step,
    )
    patch_model(model, rt)
    expert_mb = (3 * d_model * d_inter * torch.tensor([], dtype=dtype).element_size()) / 1e6

    rows = []
    step_times = []
    per_tok_times = []
    for draw in range(args.n_batches):
        if tokenized is not None:
            input_ids, token_ids, batch_fc, seq_ids = build_text_batch(
                tokenized, args.batch, args.prompt_len, args.decode_tokens, draw, dev, forecasts
            )
        else:
            input_ids, token_ids, batch_fc, seq_ids = build_stream_batch(
                stream, args.batch, args.prompt_len, args.decode_tokens, draw, dev
            )

        if not args.persistent_state:
            torch.cuda.synchronize(dev)
            cache.reset_state()
            if staging is not None:
                staging.reset_state()
        reset_stats(rt, cache, staging)
        step_ms = timed_teacher_forced(model, input_ids, token_ids, args.warmup, dev, rt, batch_fc, args.prompt_len)
        stats = summarize_stats(rt, cache, staging)
        measured_tokens = max(1, (args.decode_tokens - args.warmup) * args.batch)
        demand_h2d_mb_tok = stats.get("cache_miss", 0.0) * expert_mb / measured_tokens
        prefetch_h2d_mb_tok = stats.get("staging_issued", 0.0) * expert_mb / measured_tokens
        pred = max(1.0, stats.get("rt_forecast_union", 0.0))
        true_u = max(1.0, stats.get("rt_forecast_true_union", 0.0))
        tp = stats.get("rt_forecast_true_positive", 0.0)
        stats["demand_h2d_mb_per_token"] = demand_h2d_mb_tok
        stats["prefetch_h2d_mb_per_token"] = prefetch_h2d_mb_tok
        stats["forecast_union_precision"] = tp / pred
        stats["forecast_union_recall"] = tp / true_u
        per_tok = step_ms / args.batch
        step_times.append(step_ms)
        per_tok_times.append(per_tok)
        row = {
            "draw": draw,
            "seq_ids": seq_ids,
            "step_ms": step_ms,
            "per_token_ms": per_tok,
            "stats": stats,
        }
        rows.append(row)
        unique = stats.get("rt_actual_unique", 0.0) / max(1, args.decode_tokens - args.warmup)
        staged = stats.get("rt_staged_used", 0.0) / max(1, args.decode_tokens - args.warmup)
        align_bad = stats.get("rt_forecast_align_mismatch", 0.0)
        print(f"[draw] {draw} step_ms={step_ms:.2f} per_token_ms={per_tok:.2f} "
              f"unique/step={unique:.1f} staged_used/step={staged:.1f} "
              f"align_mismatch={align_bad:.0f} seq_ids={seq_ids}", flush=True)

    exact_report = None
    if args.check_exact and exact_batch is not None and ref_logits is not None:
        torch.cuda.synchronize(dev)
        cache.reset_state()
        if staging is not None:
            staging.reset_state()
        reset_stats(rt, cache, staging)
        patched_logits = teacher_forced_logits(model, exact_batch[0], exact_batch[1], rt, exact_batch[2], args.prompt_len)
        maxdiff = max(float((a - b).abs().max()) for a, b in zip(ref_logits, patched_logits))
        argmatch = all(torch.equal(a.argmax(-1), b.argmax(-1)) for a, b in zip(ref_logits, patched_logits))
        exact_report = {"max_logit_diff": maxdiff, "argmax_match": bool(argmatch)}
        print(f"[exact] vs_full_resident_fp16 max_logit_diff={maxdiff:.6g} argmax_match={argmatch}", flush=True)

    mean_step = sum(step_times) / len(step_times)
    mean_per_tok = sum(per_tok_times) / len(per_tok_times)
    result = {
        "policy": args.policy,
        "fallback": args.fallback,
        "dtype": args.dtype,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "decode_tokens": args.decode_tokens,
        "warmup": args.warmup,
        "cache_experts": args.cache_experts,
        "staging_slots": args.staging_slots if staging is not None else 0,
        "prefetch_budget_per_step": args.prefetch_budget_per_step,
        "persistent_state": bool(args.persistent_state),
        "forecast_alignment": args.forecast_alignment,
        "mean_step_ms": mean_step,
        "mean_per_token_ms": mean_per_tok,
        "n_layers": n_layers,
        "n_exp": n_exp,
        "exact": exact_report,
        "rows": rows,
    }
    print(f"[tpot] policy={args.policy} fallback={args.fallback} dtype={args.dtype} "
          f"batch={args.batch} cache_experts={args.cache_experts} "
          f"step_ms={mean_step:.2f} per_token_ms={mean_per_tok:.2f} "
          f"draws={[round(x, 2) for x in step_times]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
