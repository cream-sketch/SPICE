"""REAL offloaded-MoE decode for Qwen1.5-MoE (NOT a simulator).

Stage 1: routed experts live in CPU pinned RAM; a fixed GPU LRU cache holds a
subset; the REAL Qwen2MoE router decides top-k each token; missing experts are
fetched on-demand (H2D) or CPU-served; shared expert stays GPU-resident.
Attention + router + shared expert are stock (unchanged) -> output is exact.

This file: model load, expert offload (CPU pinned bank + GPU LRU cache), patched
MoE forward (policies: on_demand_fetch / cpu_serve), manual decode loop, exactness
check (vs full-resident reference, max_logit_diff), and per-token TPOT.

Bilingual note: 真实 offload 解码,非模拟器;routed 专家在 CPU pinned,GPU LRU cache。
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- expert offload: CPU pinned bank + GPU LRU cache ----------

class ExpertBank:
    """CPU pinned weights for ALL routed experts, keyed by (layer, expert)."""
    def __init__(self):
        self.w = {}  # (layer, eid) -> (gate_w, up_w, down_w) pinned CPU bf16

    def add(self, layer, eid, gate_w, up_w, down_w):
        self.w[(layer, eid)] = (
            gate_w.detach().to("cpu", torch.bfloat16).contiguous().pin_memory(),
            up_w.detach().to("cpu", torch.bfloat16).contiguous().pin_memory(),
            down_w.detach().to("cpu", torch.bfloat16).contiguous().pin_memory(),
        )

    def get(self, layer, eid):
        return self.w[(layer, eid)]


class GpuExpertCache:
    """Fixed-size LRU expert cache as CONTIGUOUS GPU tensors (so the top-k experts can be
    gathered + computed in ONE grouped GEMM per layer instead of a python per-expert loop)."""
    def __init__(self, capacity, d_model, d_inter, dev, dtype, h2d_stream):
        self.cap = capacity
        self.dev = dev
        self.h2d = h2d_stream
        self.free = list(range(capacity))
        self.gate = torch.empty(capacity, d_inter, d_model, device=dev, dtype=dtype)
        self.up = torch.empty(capacity, d_inter, d_model, device=dev, dtype=dtype)
        self.down = torch.empty(capacity, d_model, d_inter, device=dev, dtype=dtype)
        self.map = {}          # (layer,eid) -> slot_id
        self.lru = []          # slot_ids, most-recent last
        self.ready = [None] * capacity  # cuda event per slot
        self.stats = {"hit": 0, "miss": 0, "evict": 0}

    def _evict(self):
        sid = self.lru.pop(0)
        key = next(k for k, v in self.map.items() if v == sid)
        del self.map[key]
        self.stats["evict"] += 1
        return sid

    def get_slot(self, layer, eid, bank, prefetch=False):
        """Return slot_id. On miss, async H2D from the pinned bank on the h2d stream."""
        key = (layer, eid)
        if key in self.map:
            sid = self.map[key]
            self.lru.remove(sid); self.lru.append(sid)
            if not prefetch:
                self.stats["hit"] += 1
            return sid
        if not prefetch:
            self.stats["miss"] += 1
        sid = self.free.pop() if self.free else self._evict()
        self.map[key] = sid; self.lru.append(sid)
        gw, uw, dw = bank.get(layer, eid)
        with torch.cuda.stream(self.h2d):
            self.gate[sid].copy_(gw, non_blocking=True)
            self.up[sid].copy_(uw, non_blocking=True)
            self.down[sid].copy_(dw, non_blocking=True)
            ev = torch.cuda.Event(); ev.record(self.h2d)
        self.ready[sid] = ev
        return sid


def _swiglu(x, gate_w, up_w, down_w):
    return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)


# ---------- patched Qwen2MoE forward ----------

def make_patched_forward(block, layer_idx, rt):
    """Return a forward for Qwen2MoeSparseMoeBlock using offloaded routed experts."""
    gate = block.gate
    shared_expert = block.shared_expert
    shared_gate = block.shared_expert_gate
    top_k = block.top_k
    norm_topk = block.norm_topk_prob

    def forward(hidden_states):
        b, s, d = hidden_states.shape
        x = hidden_states.view(-1, d)
        router_logits = gate(x)
        routing = F.softmax(router_logits, dim=-1, dtype=torch.float)
        topk_w, topk_i = torch.topk(routing, top_k, dim=-1)
        if norm_topk:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        topk_w = topk_w.to(x.dtype)
        shared = shared_expert(x)
        shared = F.sigmoid(shared_gate(x)) * shared
        if rt.draft_mode:
            # DRAFT pass: record predicted top-k (last token only for decode) + issue prefetch;
            # propagate hidden with SHARED EXPERT ONLY (the validated training-free surrogate),
            # do NOT run routed experts. 仅 shared-only 传播 + 记录预测 + 预取.
            rt.on_forecast(layer_idx, topk_i[-1].tolist())
            return shared.view(b, s, d), router_logits
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):  # 1 iter in decode (s==1); prefill loops tokens
            eids = [int(topk_i[t, j]) for j in range(top_k)]
            out[t] = rt.run_experts(layer_idx, eids, x[t], topk_w[t])
        out = out + shared
        return out.view(b, s, d), router_logits
    return forward


class Runtime:
    def __init__(self, bank, cache, dev, policy):
        self.bank = bank; self.cache = cache; self.dev = dev; self.policy = policy
        self.draft_mode = False

    def on_forecast(self, layer, eids):
        """Draft forward predicted these experts for `layer` -> stage them now (low-stream H2D)."""
        if self.policy != "gos_transient":
            return
        for eid in eids:
            self.cache.get_slot(layer, eid, self.bank, prefetch=True)

    def run_experts(self, layer, eids, x0, probs):
        """Compute sum_k probs_k * expert_k(x0) for the top-k experts of one token.
        x0: [d_model]; probs: [k]."""
        if self.policy == "cpu_serve":
            xc = x0.unsqueeze(0).to("cpu", torch.bfloat16)
            acc = None
            for eid, w in zip(eids, probs.tolist()):
                gw, uw, dw = self.bank.get(layer, eid)
                y = _swiglu(xc, gw, uw, dw)[0] * w
                acc = y if acc is None else acc + y
            return acc.to(self.dev)
        if self.policy == "hybrid_resident_cpu":
            # resource-DAG: resident (popular) experts -> GPU (0 PCIe); the rest -> CPU-serve.
            # No fetch, no eviction during timing (the resident set is pre-warmed by popularity).
            cur = torch.cuda.current_stream(self.dev)
            x1 = x0.unsqueeze(0)
            out = None
            for eid, w in zip(eids, probs.tolist()):
                sid = self.cache.map.get((layer, eid))
                if sid is not None:                       # resident hit -> GPU
                    self.cache.stats["hit"] += 1
                    ev = self.cache.ready[sid]
                    if ev is not None:
                        cur.wait_event(ev)
                    y = _swiglu(x1, self.cache.gate[sid], self.cache.up[sid], self.cache.down[sid])[0] * w
                else:                                     # miss -> CPU serve (no PCIe)
                    self.cache.stats["miss"] += 1
                    gw, uw, dw = self.bank.get(layer, eid)
                    y = _swiglu(x1.to("cpu", torch.bfloat16), gw, uw, dw)[0].to(self.dev) * w
                out = y if out is None else out + y
            return out
        # on_demand_fetch / gos_transient: per-expert F.linear from the contiguous cache.
        # (cuBLAS F.linear matches the stock Qwen MoE accumulation -> argmax-exact; a batched
        # einsum was tried but its bf16 reduction flipped argmax, and the runtime is H2D-bound
        # at batch=1 so batching gave no speedup. Correctness-first.)
        cur = torch.cuda.current_stream(self.dev)
        x1 = x0.unsqueeze(0)
        out = None
        for eid, w in zip(eids, probs.tolist()):
            sid = self.cache.get_slot(layer, eid, self.bank)
            ev = self.cache.ready[sid]
            if ev is not None:
                cur.wait_event(ev)
            y = _swiglu(x1, self.cache.gate[sid], self.cache.up[sid], self.cache.down[sid])[0] * w
            out = y if out is None else out + y
        return out


# ---------- offload setup ----------

def offload_experts(model, dev):
    """Move routed experts to a CPU pinned bank; return (bank, d_model, d_inter, n_layers, n_exp)."""
    bank = ExpertBank()
    layers = model.model.layers
    d_model = model.config.hidden_size
    d_inter = model.config.moe_intermediate_size
    n_exp = model.config.num_experts
    for li, layer in enumerate(layers):
        mlp = layer.mlp
        for eid, exp in enumerate(mlp.experts):
            bank.add(li, eid, exp.gate_proj.weight, exp.up_proj.weight, exp.down_proj.weight)
        # free GPU expert params (replace experts with an empty list-like to drop refs)
        mlp.experts = torch.nn.ModuleList()
    torch.cuda.empty_cache()
    return bank, d_model, d_inter, len(layers), n_exp


def patch_model(model, rt):
    for li, layer in enumerate(model.model.layers):
        layer.mlp.forward = make_patched_forward(layer.mlp, li, rt)


# ---------- decode loop + exactness + TPOT ----------

@torch.inference_mode()
def reference_logits(model, input_ids, n_tokens):
    """Full-resident greedy decode; return chosen token ids + per-step last logits."""
    out = model(input_ids=input_ids, use_cache=True)
    kv = out.past_key_values
    logits = [out.logits[:, -1, :].float().cpu()]
    ids = [int(out.logits[:, -1, :].argmax(-1))]
    cur = torch.tensor([[ids[-1]]], device=input_ids.device)
    for _ in range(n_tokens - 1):
        out = model(input_ids=cur, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        logits.append(out.logits[:, -1, :].float().cpu())
        ids.append(int(out.logits[:, -1, :].argmax(-1)))
        cur = torch.tensor([[ids[-1]]], device=input_ids.device)
    return ids, logits


@torch.inference_mode()
def replay_logits(model, input_ids, token_ids):
    """Teacher-forced replay of token_ids; return per-step last logits (for exactness)."""
    out = model(input_ids=input_ids, use_cache=True)
    kv = out.past_key_values
    logits = [out.logits[:, -1, :].float().cpu()]
    for tid in token_ids[:-1]:
        cur = torch.tensor([[tid]], device=input_ids.device)
        out = model(input_ids=cur, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        logits.append(out.logits[:, -1, :].float().cpu())
    return logits


@torch.inference_mode()
def timed_decode(model, input_ids, token_ids, warmup, dev):
    """Replay token_ids; measure per-token TPOT over the post-warmup window."""
    out = model(input_ids=input_ids, use_cache=True); kv = out.past_key_values
    seq = token_ids
    for k, tid in enumerate(seq):
        if k == warmup:
            torch.cuda.synchronize(dev); t0 = time.perf_counter()
        cur = torch.tensor([[tid]], device=dev)
        out = model(input_ids=cur, past_key_values=kv, use_cache=True); kv = out.past_key_values
    torch.cuda.synchronize(dev)
    measured = len(seq) - warmup
    return (time.perf_counter() - t0) * 1000.0 / max(1, measured)


@torch.inference_mode()
def timed_decode_gos(model, rt, input_ids, token_ids, warmup, dev):
    """GOS: per token run a SHALLOW-ONLY draft forward (separate KV) that forecasts + prefetches
    future experts, then the REAL forward (gos policy) consuming staged experts. Both counted."""
    rt.draft_mode = True
    draft_kv = model(input_ids=input_ids, use_cache=True).past_key_values
    rt.draft_mode = False
    real_kv = model(input_ids=input_ids, use_cache=True).past_key_values
    for k, tid in enumerate(token_ids):
        if k == warmup:
            torch.cuda.synchronize(dev); t0 = time.perf_counter()
        cur = torch.tensor([[tid]], device=dev)
        rt.draft_mode = True   # draft forward: shared-only, fills forecast + issues prefetch
        draft_kv = model(input_ids=cur, past_key_values=draft_kv, use_cache=True).past_key_values
        rt.draft_mode = False  # real forward: gos consumes staged experts
        real_kv = model(input_ids=cur, past_key_values=real_kv, use_cache=True).past_key_values
    torch.cuda.synchronize(dev)
    measured = len(token_ids) - warmup
    return (time.perf_counter() - t0) * 1000.0 / max(1, measured)


@torch.inference_mode()
def count_expert_freq(model, input_ids, token_ids):
    """Count (layer, expert) routing frequency over the reference decode (full model, gate hooks)
    -> popularity for warming the resident set."""
    from collections import defaultdict
    freq = defaultdict(int); handles = []

    def mk(li, tk):
        def hook(m, inp, out):
            for e in out.float().topk(tk, dim=-1).indices.view(-1).tolist():
                freq[(li, e)] += 1
        return hook

    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if hasattr(mlp, "experts") and len(mlp.experts) > 0:
            handles.append(mlp.gate.register_forward_hook(mk(li, mlp.top_k)))
    replay_logits(model, input_ids, token_ids)
    for h in handles:
        h.remove()
    return freq


def warm_resident(cache, bank, freq, cap):
    """Load the top-`cap` most popular experts into the resident cache (one-time fetch)."""
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:cap]
    for (li, e), _c in top:
        cache.get_slot(li, e, bank, prefetch=True)
    torch.cuda.synchronize(cache.dev)
    cache.stats.update(hit=0, miss=0, evict=0)
    return len(top)


def main():
    p = argparse.ArgumentParser(description="REAL offloaded Qwen1.5-MoE decode (stage 1)")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--prompt", default="The history of mixture-of-experts models in large language modeling")
    p.add_argument("--decode_tokens", type=int, default=16)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--cache_experts", type=int, required=True, help="GPU resident expert-cache capacity")
    p.add_argument("--policy", choices=["on_demand_fetch", "cpu_serve", "gos_transient", "hybrid_resident_cpu"], default="on_demand_fetch")
    p.add_argument("--cpu_threads", type=int, default=16)
    p.add_argument("--check_exact", action="store_true")
    p.add_argument("--calib_prompt", default="In economics, the theory of comparative advantage explains how nations",
                   help="hybrid only: a DIFFERENT text used to estimate static expert popularity "
                        "(deployable proxy; avoids same-sequence oracle leakage)")
    p.add_argument("--oracle_resident", action="store_true",
                   help="hybrid only: estimate popularity on the EVAL sequence itself (oracle "
                        "upper bound, NOT deployable). Default uses --calib_prompt (honest).")
    args = p.parse_args()

    torch.set_num_threads(args.cpu_threads)
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
    ).to(dev).eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)

    # reference (full-resident) decode -> token ids + reference logits
    ref_ids, ref_logits = reference_logits(model, ids, args.decode_tokens)
    # hybrid popularity source: honest (separate calibration text) by default; oracle (eval seq) only
    # under --oracle_resident. Both must run BEFORE offload (need full GPU experts + gate hooks).
    freq = None
    if args.policy == "hybrid_resident_cpu":
        if args.oracle_resident:
            freq = count_expert_freq(model, ids, ref_ids)
        else:
            calib_in = tok(args.calib_prompt, return_tensors="pt").input_ids.to(dev)
            calib_ids, _ = reference_logits(model, calib_in, args.decode_tokens)
            freq = count_expert_freq(model, calib_in, calib_ids)

    # offload + patch
    bank, d_model, d_inter, n_layers, n_exp = offload_experts(model, dev)
    h2d = torch.cuda.Stream(device=dev)
    cache = GpuExpertCache(args.cache_experts, d_model, d_inter, dev, torch.bfloat16, h2d)
    rt = Runtime(bank, cache, dev, args.policy)
    patch_model(model, rt)
    if args.policy == "hybrid_resident_cpu":
        nres = warm_resident(cache, bank, freq, args.cache_experts)
        src = "ORACLE(eval-seq)" if args.oracle_resident else "calib-text(honest)"
        print(f"[hybrid] warmed {nres} popular experts resident on GPU (popularity={src}); rest CPU-served")

    if args.check_exact:
        rep = replay_logits(model, ids, ref_ids)
        maxdiff = max(float((a - b).abs().max()) for a, b in zip(ref_logits, rep))
        argmatch = all(int(a.argmax(-1)) == int(b.argmax(-1)) for a, b in zip(ref_logits, rep))
        print(f"[exact] policy={args.policy} max_logit_diff={maxdiff:.6f} argmax_match={argmatch} "
              f"cache(hit={cache.stats['hit']},miss={cache.stats['miss']},evict={cache.stats['evict']})")

    cache.stats.update(hit=0, miss=0, evict=0)  # reset after exactness replay
    if args.policy == "gos_transient":
        tpot = timed_decode_gos(model, rt, ids, ref_ids, args.warmup, dev)
    else:
        tpot = timed_decode(model, ids, ref_ids, args.warmup, dev)
    print(f"[tpot] policy={args.policy} cache_experts={args.cache_experts} "
          f"n_layers={n_layers} n_exp={n_exp} TPOT_ms={tpot:.3f} "
          f"cache(hit={cache.stats['hit']},miss={cache.stats['miss']},evict={cache.stats['evict']})")


if __name__ == "__main__":
    main()
