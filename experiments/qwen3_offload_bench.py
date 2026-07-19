"""Offloaded MoE inference benchmark for Qwen3-30B-A3B on a single memory-limited GPU.

All expert weights live in host (CPU) memory; non-expert weights (attention,
router gates, embeddings, norms, lm_head) reside on the GPU. A fixed-budget GPU
expert cache is managed by a pluggable policy:

  naive    MoE-On-Demand: no cache; every selected expert is fetched
           synchronously over PCIe when the router selects it.
  lru      Reactive LRU expert cache; misses fetch synchronously.
  collab   elsa-lab style CPU-GPU collaborative inference: GPU cache holds hot
           experts (frequency-promoted); cache misses are computed on the CPU
           (activations move, weights do not).
  adapmoe  AdapMoE-style lossless subset: cross-layer transition-based
           prefetching (statistics from collected routing traces) on top of an
           LRU cache; misses fetch synchronously.
  spice    SPICE draft-driven speculative prefetching with verified fallback
           (requires a trained draft predictor; see --draft_checkpoint).

Metrics: TPOT (per decode step), TTFT, expert cache hit/fallback rates, H2D
expert-weight traffic, optional GPU power sampling (nvidia-smi) for J/token.

The target router always remains authoritative: prefetching only schedules
weight movement, so outputs are bit-identical across policies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Expert weight store (host side)
# --------------------------------------------------------------------------- #


class ExpertStore:
    """Holds every expert's weights in host memory as one flat tensor per expert.

    Flat layout per expert: [gate_proj | up_proj | down_proj], all bf16,
    so a single contiguous H2D copy moves one expert.
    """

    def __init__(self, hidden: int, inter: int, dtype: torch.dtype):
        self.hidden = hidden
        self.inter = inter
        self.dtype = dtype
        self.gate_numel = inter * hidden
        self.up_numel = inter * hidden
        self.down_numel = hidden * inter
        self.expert_numel = self.gate_numel + self.up_numel + self.down_numel
        self.expert_bytes = self.expert_numel * dtype.itemsize
        self.cpu_flat: dict[tuple[int, int], torch.Tensor] = {}
        # experts are stored in one slab per layer (48 big allocations instead
        # of 6144 small ones: far less allocator overhead/fragmentation)
        self._slabs: dict[int, torch.Tensor] = {}
        self._slab_next: dict[int, int] = {}
        self._num_experts_hint = 128
        # Three placement tiers, assigned by routing popularity:
        #   hot  -> GPU VRAM on `hot_device` (P2P prefetch, fully async, no
        #           pinned host memory, so no systemd-oomd pressure kills)
        #   warm -> pageable host RAM slabs (bounded by the 25GB/user share)
        #   cold -> one flat NVMe file, mmap'd (written once, reused across runs)
        self.hot_set: set[tuple[int, int]] = set()
        self.hot_device: torch.device | None = None
        self._hot_per_layer: dict[int, int] = {}
        self.cold_index: dict[tuple[int, int], int] = {}
        self._cold_per_layer: dict[int, int] = {}
        self.cold_file: Path | None = None
        self._cold_fh = None
        self._cold_cached = False
        self._cold_written = 0
        self._cold_mmap: torch.Tensor | None = None
        self._gpu_slabs: dict[int, torch.Tensor] = {}
        self._gpu_next: dict[int, int] = {}

    def configure_tiers(self, hot_set: set, device: torch.device | None,
                        cold_keys: list, cold_file: str | None) -> None:
        self.hot_set = hot_set
        self.hot_device = device
        for layer, _ in hot_set:
            self._hot_per_layer[layer] = self._hot_per_layer.get(layer, 0) + 1
        self.cold_index = {k: i for i, k in enumerate(cold_keys)}
        for layer, _ in cold_keys:
            self._cold_per_layer[layer] = self._cold_per_layer.get(layer, 0) + 1
        if cold_keys:
            self.cold_file = Path(cold_file)
            expected = len(cold_keys) * self.expert_bytes
            marker = self.cold_file.with_suffix(".ok")
            # size check alone is unsafe: the file is pre-truncated to full
            # size, so a run killed mid-write leaves a hole-filled file behind
            self._cold_cached = (self.cold_file.exists()
                                 and self.cold_file.stat().st_size == expected
                                 and marker.exists())
            if not self._cold_cached:
                marker.unlink(missing_ok=True)
                self._cold_fh = open(self.cold_file, "wb")
                self._cold_fh.truncate(expected)
                self._cold_written = 0

    def tier(self, key: tuple[int, int]) -> str:
        if key in self.hot_set:
            return "gpu_store"
        if key in self.cold_index:
            return "nvme"
        return "ram"

    def finalize_cold(self) -> None:
        if self._cold_fh is not None:
            self._cold_fh.flush()
            import os as _os
            _os.fsync(self._cold_fh.fileno())
            self._cold_fh.close()
            self._cold_fh = None
            self.cold_file.with_suffix(".ok").touch()
        if self.cold_index:
            n = len(self.cold_index)
            self._cold_mmap = torch.from_file(
                str(self.cold_file), shared=True,
                size=n * self.expert_numel, dtype=self.dtype,
            )
            for key, i in self.cold_index.items():
                self.cpu_flat[key] = self._cold_mmap.narrow(
                    0, i * self.expert_numel, self.expert_numel)

    def put(self, layer: int, eid: int, gate_w: torch.Tensor, up_w: torch.Tensor,
            down_w: torch.Tensor, pin: bool) -> None:
        key = (layer, eid)
        if key in self.cold_index:
            if not self._cold_cached:
                tmp = torch.empty(self.expert_numel, dtype=self.dtype)
                tmp[: self.gate_numel].copy_(gate_w.reshape(-1))
                tmp[self.gate_numel : self.gate_numel + self.up_numel].copy_(up_w.reshape(-1))
                tmp[self.gate_numel + self.up_numel :].copy_(down_w.reshape(-1))
                self._cold_fh.seek(self.cold_index[key] * self.expert_bytes)
                self._cold_fh.write(tmp.view(torch.int16).numpy().tobytes())
                self._cold_written += 1
                if self._cold_written % 256 == 0:
                    # bound dirty page cache so the container cgroup stays calm
                    import os as _os
                    self._cold_fh.flush()
                    _os.fsync(self._cold_fh.fileno())
                    try:
                        _os.posix_fadvise(self._cold_fh.fileno(), 0, 0,
                                          _os.POSIX_FADV_DONTNEED)
                    except (AttributeError, OSError):
                        pass
            return  # mmap views are registered in finalize_cold()
        if key in self.hot_set:
            slab = self._gpu_slabs.get(layer)
            if slab is None:
                n = self._hot_per_layer.get(layer, 0)
                slab = torch.empty(n * self.expert_numel, dtype=self.dtype,
                                   device=self.hot_device)
                self._gpu_slabs[layer] = slab
                self._gpu_next[layer] = 0
            off = self._gpu_next[layer]
            self._gpu_next[layer] = off + 1
            flat = slab.narrow(0, off * self.expert_numel, self.expert_numel)
        else:
            slab = self._slabs.get(layer)
            if slab is None:
                warm = (self._num_experts_hint
                        - self._hot_per_layer.get(layer, 0)
                        - self._cold_per_layer.get(layer, 0))
                slab = torch.empty(max(1, warm) * self.expert_numel,
                                   dtype=self.dtype, pin_memory=pin)
                self._slabs[layer] = slab
                self._slab_next[layer] = 0
            off = self._slab_next[layer]
            self._slab_next[layer] = off + 1
            flat = slab.narrow(0, off * self.expert_numel, self.expert_numel)
        flat[: self.gate_numel].copy_(gate_w.reshape(-1))
        flat[self.gate_numel : self.gate_numel + self.up_numel].copy_(up_w.reshape(-1))
        flat[self.gate_numel + self.up_numel :].copy_(down_w.reshape(-1))
        self.cpu_flat[key] = flat

    def views(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = flat[: self.gate_numel].view(self.inter, self.hidden)
        u = flat[self.gate_numel : self.gate_numel + self.up_numel].view(self.inter, self.hidden)
        d = flat[self.gate_numel + self.up_numel :].view(self.hidden, self.inter)
        return g, u, d


# --------------------------------------------------------------------------- #
# GPU expert cache
# --------------------------------------------------------------------------- #


class ExpertCache:
    """Fixed-capacity GPU expert cache with LRU eviction and async prefetch."""

    def __init__(self, store: ExpertStore, capacity: int, device: torch.device):
        self.store = store
        self.capacity = capacity
        self.device = device
        self.slots: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
        self.pending: dict[tuple[int, int], torch.cuda.Event] = {}
        self.copy_stream = torch.cuda.Stream(device=device)
        self.free_pool: list[torch.Tensor] = []
        self.lock = threading.RLock()  # prefetches may arrive from a worker thread
        self._inflight: set[tuple[int, int]] = set()
        self.miss_by_tier: dict[str, int] = {"gpu_store": 0, "ram": 0, "nvme": 0}
        # counters
        self.hits = 0
        self.misses = 0
        self.prefetch_issued = 0
        self.prefetch_unused = 0
        self.h2d_bytes = 0

    def _alloc(self) -> torch.Tensor:
        if self.free_pool:
            return self.free_pool.pop()
        return torch.empty(self.store.expert_numel, dtype=self.store.dtype, device=self.device)

    def _evict_if_full(self) -> None:
        while len(self.slots) >= self.capacity and self.slots:
            key, buf = self.slots.popitem(last=False)
            ev = self.pending.pop(key, None)
            if ev is not None:
                ev.synchronize()  # never recycle a buffer with an in-flight copy
            self.free_pool.append(buf)

    def contains(self, key: tuple[int, int]) -> bool:
        return key in self.slots or key in self.pending

    def prefetch(self, key: tuple[int, int]) -> None:
        with self.lock:
            if self.capacity <= 0 or self.contains(key) or key in self._inflight:
                return
            self._evict_if_full()
            buf = self._alloc()
            self._inflight.add(key)
            self.prefetch_issued += 1
            self.h2d_bytes += self.store.expert_bytes
        # The actual copy happens outside the lock: with pageable source
        # memory this blocks the *calling* thread (the draft worker), which is
        # fine — it must never block the target model's thread in get().
        with torch.cuda.stream(self.copy_stream):
            buf.copy_(self.store.cpu_flat[key], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
        with self.lock:
            self._inflight.discard(key)
            if key in self.slots:  # a demand fetch raced us; drop our copy
                self.free_pool.append(buf)
                return
            self.slots[key] = buf
            self.pending[key] = ev

    def get(self, key: tuple[int, int], insert_on_miss: bool = True) -> torch.Tensor | None:
        """Return GPU flat tensor for expert, fetching synchronously on miss.

        Returns None only when capacity==0 and insert_on_miss is False.
        """
        with self.lock:
            if key in self.slots:
                if key in self.pending:
                    torch.cuda.current_stream().wait_event(self.pending.pop(key))
                self.slots.move_to_end(key)
                self.hits += 1
                return self.slots[key]
            # miss
            self.misses += 1
            self.miss_by_tier[self.store.tier(key)] += 1
            self.h2d_bytes += self.store.expert_bytes
            if self.capacity > 0 and insert_on_miss:
                self._evict_if_full()
                buf = self._alloc()
                buf.copy_(self.store.cpu_flat[key], non_blocking=False)
                self.slots[key] = buf
                return buf
            # scratch path (no caching): reuse a single scratch buffer
            if not hasattr(self, "_scratch"):
                self._scratch = torch.empty(
                    self.store.expert_numel, dtype=self.store.dtype, device=self.device
                )
            self._scratch.copy_(self.store.cpu_flat[key], non_blocking=False)
            return self._scratch

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "fallback_rate": self.misses / max(1, total),
            "hit_rate": self.hits / max(1, total),
            "prefetch_issued": self.prefetch_issued,
            "h2d_gb": self.h2d_bytes / 1024**3,
            "cached_experts": len(self.slots),
            "miss_by_tier": dict(self.miss_by_tier),
        }


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #


class Policy:
    """Base: reactive LRU."""

    name = "lru"
    cpu_compute_on_miss = False
    insert_on_miss = True

    def __init__(self, args, num_layers: int, num_experts: int):
        self.args = args
        self.num_layers = num_layers
        self.num_experts = num_experts

    def on_layer_routed(self, cache: ExpertCache, layer: int,
                        router_probs: torch.Tensor, selected: list[int]) -> None:
        """Called after layer `layer` routing is known; may issue prefetches."""


class NaivePolicy(Policy):
    name = "naive"
    insert_on_miss = False  # no reuse: every selection is an on-demand fetch


class CollabPolicy(Policy):
    """elsa-lab style: frequency-promoted GPU cache; CPU computes on miss."""

    name = "collab"
    cpu_compute_on_miss = True
    insert_on_miss = False

    def __init__(self, args, num_layers, num_experts):
        super().__init__(args, num_layers, num_experts)
        self.freq = torch.zeros(num_layers, num_experts, dtype=torch.long)
        self.promote_threshold = args.collab_promote_threshold

    def on_layer_routed(self, cache, layer, router_probs, selected):
        for eid in selected:
            self.freq[layer, eid] += 1
            key = (layer, eid)
            if (not cache.contains(key)
                    and int(self.freq[layer, eid]) >= self.promote_threshold):
                cache.prefetch(key)  # async promote; CPU handles current miss


class AdapMoEPolicy(Policy):
    """Cross-layer transition prefetching (lossless subset of AdapMoE)."""

    name = "adapmoe"

    def __init__(self, args, num_layers, num_experts):
        super().__init__(args, num_layers, num_experts)
        if not args.transition_stats:
            raise SystemExit("adapmoe policy requires --transition_stats (from traces)")
        blob = torch.load(args.transition_stats, map_location="cpu")
        self.transition = blob["transition"]  # [L-1, E, E] P(next e' | cur e)
        self.prefetch_k = args.prefetch_k

    def on_layer_routed(self, cache, layer, router_probs, selected):
        if layer + 1 >= self.num_layers:
            return
        scores = self.transition[layer][selected].sum(dim=0)  # [E]
        top = torch.topk(scores, self.prefetch_k).indices.tolist()
        for eid in top:
            cache.prefetch((layer + 1, eid))


class SpicePolicy(Policy):
    """Draft-model speculative prefetching with confidence-adaptive depth.

    Prediction runs from decoder-layer forward hooks (anchor re-initialization
    on every layer's true output), not from on_layer_routed: the runtime keeps
    a short rolling window of per-layer hidden states as causal context and
    rolls the LoRE draft forward up to l_max layers per anchor.
    """

    name = "spice"

    def __init__(self, args, num_layers, num_experts):
        super().__init__(args, num_layers, num_experts)
        if not args.draft_checkpoint:
            raise SystemExit("spice policy requires --draft_checkpoint")
        self.l_max = args.l_max
        self.tau = args.confidence_threshold
        self.prefetch_k = args.prefetch_k


POLICIES = {p.name: p for p in [Policy, NaivePolicy, CollabPolicy, AdapMoEPolicy, SpicePolicy]}


# --------------------------------------------------------------------------- #
# Patched MoE block
# --------------------------------------------------------------------------- #


class OffloadedMoeBlock(torch.nn.Module):
    """Replaces Qwen3MoeSparseMoeBlock: original TopK router stays on GPU,
    expert weights are served through the ExpertCache."""

    def __init__(self, layer_idx: int, gate: torch.nn.Module, cfg, runtime: "Runtime"):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = gate  # Qwen3MoeTopKRouter (returns logits, scores, indices)
        self.num_experts = cfg.num_experts
        self.rt = runtime

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        router_logits, routing_weights, selected_experts = self.gate(x)
        rt = self.rt

        unique = torch.unique(selected_experts).tolist()
        probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        rt.policy.on_layer_routed(rt.cache, self.layer_idx, probs.mean(dim=0), unique)

        out = torch.zeros_like(x)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for eid in unique:
            top_k_pos, token_idx = torch.where(expert_mask[eid])
            cur = x[token_idx]
            key = (self.layer_idx, eid)
            if rt.policy.cpu_compute_on_miss and not rt.cache.contains(key):
                y = rt.expert_forward_cpu(key, cur)
                rt.cache.misses += 1  # miss handled by CPU compute: no H2D traffic
                rt.cpu_computed += int(token_idx.numel())
            else:
                flat = rt.cache.get(key, insert_on_miss=rt.policy.insert_on_miss)
                g, u, d = rt.store.views(flat)
                y = F.linear(F.silu(F.linear(cur, g)) * F.linear(cur, u), d)
            out.index_add_(0, token_idx, (y * routing_weights[token_idx, top_k_pos, None]).to(out.dtype))
        return out.view(b, s, h)


# --------------------------------------------------------------------------- #
# Runtime: loading, surgery, benchmark
# --------------------------------------------------------------------------- #


class Runtime:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}")
        self.cpu_computed = 0

    # -- model surgery ------------------------------------------------------ #

    @staticmethod
    def _read_shard_sequential(path: Path, consume) -> None:
        """Read a safetensors shard with plain sequential read() calls.

        Avoids safetensors' mmap: under host-RAM pressure, mmap page faults
        degrade into repeated re-reads (observed 5x file re-read on this box).
        Tensors are consumed in file-offset order and the page cache for the
        shard is dropped afterwards, so the file is read exactly once.
        """
        import json as _json
        import struct

        dtypes = {"BF16": torch.bfloat16, "F16": torch.float16,
                  "F32": torch.float32, "I64": torch.int64}
        with open(path, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = _json.loads(fh.read(hlen))
            base = 8 + hlen
            metas = sorted(
                ((k, v) for k, v in header.items() if k != "__metadata__"),
                key=lambda kv: kv[1]["data_offsets"][0],
            )
            for key, meta in metas:
                off0, off1 = meta["data_offsets"]
                fh.seek(base + off0)
                raw = bytearray(fh.read(off1 - off0))
                t = torch.frombuffer(raw, dtype=dtypes[meta["dtype"]]).reshape(meta["shape"])
                consume(key, t)
            try:
                import os as _os
                _os.posix_fadvise(fh.fileno(), 0, 0, _os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass

    def load(self):
        import re

        from accelerate import init_empty_weights
        from transformers import AutoConfig

        args = self.args
        model_dir = Path(args.model)
        print(f"[load] tokenizer + config from {args.model}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
        self.num_layers = cfg.num_hidden_layers
        self.num_experts = cfg.num_experts
        self.store = ExpertStore(cfg.hidden_size, cfg.moe_intermediate_size, torch.bfloat16)
        self.store._num_experts_hint = cfg.num_experts
        if args.store_gpu is not None:
            if not args.transition_stats:
                raise SystemExit("--store_gpu requires --transition_stats for popularity ranking")
            pop = torch.load(args.transition_stats, map_location="cpu")["popularity"]
            order = torch.argsort(pop.flatten(), descending=True)
            keys = [(int(i) // cfg.num_experts, int(i) % cfg.num_experts) for i in order]
            n_hot = min(len(keys), int(args.store_gpu_gib * 1024**3) // self.store.expert_bytes)
            n_warm = min(len(keys) - n_hot,
                         int(args.ram_tier_gib * 1024**3) // self.store.expert_bytes)
            hot = set(keys[:n_hot])
            cold = keys[n_hot + n_warm:]
            cold_file = args.cold_file or str(Path(args.out_dir).parent / "qwen3_cold_experts.bin")
            self.store.configure_tiers(hot, torch.device(f"cuda:{args.store_gpu}"),
                                       cold, cold_file)
            gib = self.store.expert_bytes / 1024**3
            print(f"[load] tiers: gpu={n_hot} ({n_hot*gib:.1f}G) ram={n_warm} "
                  f"({n_warm*gib:.1f}G) nvme={len(cold)} ({len(cold)*gib:.1f}G, "
                  f"cached={self.store._cold_cached})", flush=True)

        # Stream safetensors shards exactly once: expert weights go straight
        # into the host store (one flat tensor per expert), everything else is
        # collected for the skeleton. Avoids materializing the model twice —
        # essential on this RAM-constrained shared host.
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        by_shard: dict[str, list[str]] = {}
        for key, shard in index["weight_map"].items():
            by_shard.setdefault(shard, []).append(key)
        exp_re = re.compile(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
        )
        pending: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        nonexpert_sd: dict[str, torch.Tensor] = {}
        t0 = time.perf_counter()

        def consume(key: str, t: torch.Tensor) -> None:
            m = exp_re.match(key)
            if m:
                li, eid, which = int(m.group(1)), int(m.group(2)), m.group(3)
                slot = pending.setdefault((li, eid), {})
                slot[which] = t
                if len(slot) == 3:
                    self.store.put(li, eid, slot["gate_proj"], slot["up_proj"],
                                   slot["down_proj"], pin=args.pin_memory)
                    del pending[(li, eid)]
            else:
                nonexpert_sd[key] = t.clone()  # detach from the read buffer

        shards = sorted(by_shard)
        for si, shard in enumerate(shards):
            self._read_shard_sequential(model_dir / shard, consume)
            print(f"[load] shard {si+1}/{len(shards)} done "
                  f"({time.perf_counter()-t0:.0f}s, store={len(self.store.cpu_flat)} experts)",
                  flush=True)
        assert not pending, f"incomplete experts: {list(pending)[:4]}"
        self.store.finalize_cold()
        assert len(self.store.cpu_flat) == self.num_layers * self.num_experts

        print("[load] building skeleton and loading non-expert weights", flush=True)
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(cfg)
        for li, layer in enumerate(model.model.layers):
            layer.mlp = OffloadedMoeBlock(li, layer.mlp.gate, cfg, self)
        missing, unexpected = model.load_state_dict(nonexpert_sd, strict=False, assign=True)
        assert not unexpected, f"unexpected keys: {unexpected[:4]}"
        real_missing = [k for k in missing if "lm_head" not in k]
        assert not real_missing, f"missing keys: {real_missing[:4]}"
        if getattr(cfg, "tie_word_embeddings", False) or "lm_head.weight" not in nonexpert_sd:
            model.tie_weights()
        del nonexpert_sd
        model.eval()
        model.to(self.device)
        self.model = model

        cap = args.cache_experts
        self.cache = ExpertCache(self.store, cap, self.device)
        pol_cls = POLICIES[args.policy]
        self.policy = pol_cls(args, self.num_layers, self.num_experts)
        if args.policy == "spice":
            self._setup_spice_draft(cfg)
        print(f"[load] done. policy={args.policy} cache={cap} experts "
              f"({cap * self.store.expert_bytes / 1024**3:.1f} GiB)", flush=True)

    # -- SPICE draft integration --------------------------------------------- #

    def _setup_spice_draft(self, cfg):
        from transformers import AutoConfig
        from qwen3_train_draft import Qwen3Draft, load_nonexpert_weights

        args = self.args
        ckpt = torch.load(args.draft_checkpoint, map_location="cpu", weights_only=False)
        dcfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
        dcfg._attn_implementation = "eager"
        print("[spice] building draft model", flush=True)
        draft = Qwen3Draft(dcfg, ckpt["config"]["rank"], ckpt["config"]["d_route"])
        draft.load_frozen(load_nonexpert_weights(args.model, dcfg.num_hidden_layers),
                          torch.bfloat16)
        missing, unexpected = draft.load_state_dict(ckpt["state_dict"], strict=False)
        assert not unexpected, unexpected
        draft = draft.to(self.device).eval()  # keeps per-param dtypes (LoRE fp32)
        self.draft = draft
        self.hist_window = args.draft_history_window
        self.layer_hist: list[list[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        self.draft_ms_total = 0.0

        # Worker thread: takes finished rollout predictions, waits for their
        # GPU results, and issues prefetches — keeping every sync off the
        # target model's launch thread so draft compute overlaps fetch stalls.
        import queue
        self._draft_q: "queue.Queue" = queue.Queue(maxsize=64)

        def drain():
            while True:
                item = self._draft_q.get()
                if item is None:
                    return
                depth_top, tau, ev = item
                ev.synchronize()  # rollout ran on draft_stream
                confs = torch.stack([c for _, _, c in depth_top]).cpu()
                for j, (li, topi, _) in enumerate(depth_top):
                    for eid in topi.tolist():
                        self.cache.prefetch((li, eid))
                    if j >= 1 and float(confs[j]) < tau:
                        break

        self._draft_worker = threading.Thread(target=drain, daemon=True)
        self._draft_worker.start()
        self.draft_stream = torch.cuda.Stream(device=self.device)

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                hist = self.layer_hist[layer_idx]
                hist.append(h[:, -self.hist_window:].detach())
                if len(hist) > 1 and sum(t.shape[1] for t in hist) > self.hist_window:
                    self.layer_hist[layer_idx] = [torch.cat(hist, dim=1)[:, -self.hist_window:]]
                if layer_idx % self.args.draft_anchor_stride == 0:
                    self._spice_prefetch(layer_idx)
            return hook

        for li, layer in enumerate(self.model.model.layers):
            layer.register_forward_hook(make_hook(li))

    @torch.no_grad()
    def _spice_prefetch(self, anchor_layer: int) -> None:
        pol = self.policy
        if anchor_layer + 1 >= self.num_layers:
            return
        t0 = time.perf_counter()
        hist = self.layer_hist[anchor_layer]
        h = hist[0] if len(hist) == 1 else torch.cat(hist, dim=1)
        self.draft_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.draft_stream):
            self._rollout_and_enqueue(anchor_layer, h)
        self.draft_ms_total += (time.perf_counter() - t0) * 1000

    @torch.no_grad()
    def _rollout_and_enqueue(self, anchor_layer: int, h: torch.Tensor) -> None:
        pol = self.policy
        T = h.shape[1]
        pos = torch.arange(T, device=h.device).unsqueeze(0)
        pe = self.draft.rotary(h, pos)
        mask = torch.full((T, T), torch.finfo(h.dtype).min, device=h.device,
                          dtype=h.dtype).triu(diagonal=1)[None, None]
        alpha = torch.sigmoid(self.draft.alpha_logit)
        ctx = None
        cur = h
        # Roll the full lookahead on-GPU first, then do a single host sync:
        # per-depth .tolist() calls would serialize the draft into the target's
        # critical path once per depth.
        depth_top: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for d in range(1, pol.l_max + 1):
            li = anchor_layer + d
            if li >= self.num_layers:
                break
            cur, _logits, probs = self.draft.layers[li](cur, pe, mask, ctx, self.draft.Wc)
            p_last = probs[0, -1]
            topv, topi = torch.topk(p_last, max(pol.prefetch_k, 8))
            depth_top.append((li, topi[: pol.prefetch_k], topv[:8].sum()))
            desc = F.linear(probs.to(self.draft.Wr.dtype), self.draft.Wr)
            ctx = desc if ctx is None else alpha * ctx + (1 - alpha) * desc
        if depth_top:
            # Single batched readback inside the draft_stream context: syncs
            # only the draft's own tiny compute — never the target pipeline.
            packed = torch.cat(
                [torch.stack([c for _, _, c in depth_top])]
                + [topi.float() for _, topi, _ in depth_top]
            ).cpu()
            D = len(depth_top)
            confs = packed[:D]
            k = pol.prefetch_k
            for j, (li, _, _) in enumerate(depth_top):
                for eid in packed[D + j * k : D + (j + 1) * k].long().tolist():
                    self.cache.prefetch((li, eid))
                if j >= 1 and float(confs[j]) < pol.tau:
                    break

    def expert_forward_cpu(self, key, xt_gpu: torch.Tensor) -> torch.Tensor:
        src = self.store.cpu_flat[key]
        g, u, d = self.store.views(src)
        if src.device.type == "cuda":  # hot-tier expert: compute on the store GPU
            x = xt_gpu.to(src.device)
            y = F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)
            return y.to(xt_gpu.device)
        x = xt_gpu.float().cpu()
        y = F.linear(F.silu(F.linear(x, g.float())) * F.linear(x, u.float()), d.float())
        return y.to(xt_gpu.dtype).to(xt_gpu.device, non_blocking=False)

    # -- power sampling ------------------------------------------------------ #

    def _sample_power(self, stop: threading.Event, samples: list):
        gid = self.args.power_gpu
        while not stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--id={gid}",
                     "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
                samples.append(float(out))
            except Exception:
                pass
            stop.wait(self.args.power_interval)

    # -- benchmark ----------------------------------------------------------- #

    @torch.no_grad()
    def benchmark(self) -> dict:
        args = self.args
        prompts = Path(args.prompt_file).read_text(encoding="utf-8").splitlines() \
            if args.prompt_file else ["The key challenge of deploying large mixture-of-experts models is"]
        prompts = [p for p in prompts if p.strip()][: args.max_prompts]

        results = []
        power_samples: list[float] = []
        stop = threading.Event()
        sampler = threading.Thread(target=self._sample_power, args=(stop, power_samples), daemon=True)
        if args.measure_power:
            sampler.start()
        t_bench0 = time.perf_counter()

        for pi, prompt in enumerate(prompts):
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            past = None
            step_times = []
            tok = enc.input_ids
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for step in range(args.new_tokens):
                out = self.model(
                    input_ids=tok if past is None else tok[:, -1:],
                    past_key_values=past, use_cache=True,
                )
                past = out.past_key_values
                tok = torch.cat([tok, out.logits[:, -1:].argmax(-1)], dim=-1)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                step_times.append(t1 - t0)
                t0 = t1
            ttft = step_times[0]
            decode = step_times[1:]
            results.append({
                "prompt_tokens": int(enc.input_ids.shape[1]),
                "ttft_s": ttft,
                "tpot_ms_mean": 1000 * sum(decode) / max(1, len(decode)),
                "tpot_ms_p50": 1000 * sorted(decode)[len(decode) // 2] if decode else None,
                "generated": self.tokenizer.decode(tok[0, enc.input_ids.shape[1]:][:32]),
            })
            print(f"[bench] prompt {pi}: ttft={ttft:.2f}s "
                  f"tpot={results[-1]['tpot_ms_mean']:.1f}ms", flush=True)

        elapsed = time.perf_counter() - t_bench0
        stop.set()
        if args.measure_power:
            sampler.join(timeout=2)

        tpots = [r["tpot_ms_mean"] for r in results]
        avg_power = sum(power_samples) / len(power_samples) if power_samples else None
        decode_steps = max(1, len(prompts) * (args.new_tokens - 1))
        return {
            "experiment": "qwen3_offload_bench",
            "model": args.model,
            "policy": args.policy,
            "cache_experts": args.cache_experts,
            "cache_gib": args.cache_experts * self.store.expert_bytes / 1024**3,
            "new_tokens": args.new_tokens,
            "num_prompts": len(prompts),
            "tpot_ms_mean": sum(tpots) / len(tpots),
            "ttft_s_mean": sum(r["ttft_s"] for r in results) / len(results),
            "cache": self.cache.stats(),
            "cpu_computed_experts": self.cpu_computed,
            "draft_prefetch_ms_total": getattr(self, "draft_ms_total", 0.0),
            "elapsed_s": elapsed,
            "avg_power_w": avg_power,
            "energy_per_token_j": (avg_power * elapsed / decode_steps) if avg_power else None,
            "per_prompt": results,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--policy", choices=sorted(POLICIES), default="naive")
    ap.add_argument("--cache_experts", type=int, default=0,
                    help="GPU cache capacity in experts (0 = no cache, pure on-demand)")
    ap.add_argument("--new_tokens", type=int, default=32)
    ap.add_argument("--max_prompts", type=int, default=4)
    ap.add_argument("--prompt_file", default=None)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--store_gpu", type=int, default=None,
                    help="device index for the hot expert tier (P2P prefetch source)")
    ap.add_argument("--store_gpu_gib", type=float, default=26.0)
    ap.add_argument("--ram_tier_gib", type=float, default=14.0)
    ap.add_argument("--cold_file", default=None,
                    help="flat file for the NVMe tier (default: <out_dir>/../qwen3_cold_experts.bin)")
    ap.add_argument("--measure_power", action="store_true")
    ap.add_argument("--power_gpu", type=int, default=0)
    ap.add_argument("--power_interval", type=float, default=0.1)
    # policy knobs
    ap.add_argument("--collab_promote_threshold", type=int, default=3)
    ap.add_argument("--cpu_threads", type=int, default=16)
    ap.add_argument("--transition_stats", default=None)
    ap.add_argument("--prefetch_k", type=int, default=8)
    ap.add_argument("--draft_checkpoint", default=None)
    ap.add_argument("--draft_history_window", type=int, default=64)
    ap.add_argument("--draft_anchor_stride", type=int, default=1,
                    help="run the draft rollout every N layers (decode only)")
    ap.add_argument("--l_max", type=int, default=6)
    ap.add_argument("--confidence_threshold", type=float, default=0.7)
    args = ap.parse_args()

    torch.set_num_threads(args.cpu_threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = Runtime(args)
    rt.load()
    result = rt.benchmark()
    path = out_dir / f"bench_{args.policy}_c{args.cache_experts}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_prompt"}, indent=2))


if __name__ == "__main__":
    main()
