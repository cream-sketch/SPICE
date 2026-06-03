"""SPICE-HX kill-shot component calibration (codex decisive-experiment step D, part 1).

Goal / 目标: at batch=1 decode on real Qwen1.5-MoE, measure whether CPU-miss-compute is
already hidden under GPU work. If t_expert_cpu * n_miss_per_layer <= hideable_gpu_per_layer,
then there is NO exposed stall to schedule away -> NO-GO (honest negative for paper I).
Otherwise there is exposed stall -> proceed to deadline-prefetch replay.

Measures (CUDA events on real decode forward via module hooks):
  t_attn, t_gate, t_shared, t_expert_gpu per layer; t_expert_cpu; t_copy_h2d (17MB pinned async);
  CPU||GPU overlap_factor. Plus per-layer miss-count distribution from real decode traces at
  10%/30% routed-expert residency (global-popularity placement, matching Fiddler set_expert_loc).

All printed content in English. Core params: no defaults.
"""
import argparse, json, glob, time, threading
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="SPICE-HX component calibration + miss-count (batch=1 decode)")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--trace_dir", required=True, help="dir of dec_*.pt decode traces")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--decode_tokens", type=int, required=True, help="decode steps to time for component calibration")
    ap.add_argument("--max_traces", type=int, required=True)
    ap.add_argument("--residency", type=str, required=True, help="comma list of routed-expert GPU residency fractions, e.g. 0.1,0.3")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


class Timer:
    """Record CUDA-event pairs around module forward via hooks; sum per module-type after one sync."""
    def __init__(self):
        self.events = {}  # name -> list of (start_event, end_event)

    def hook(self, module, name):
        self.events.setdefault(name, [])

        def pre(mod, inp):
            s = torch.cuda.Event(enable_timing=True); s.record(); mod._hx_start = s
        def post(mod, inp, out):
            e = torch.cuda.Event(enable_timing=True); e.record()
            self.events[name].append((mod._hx_start, e))
        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    def summary_ms(self):
        torch.cuda.synchronize()
        res = {}
        for name, pairs in self.events.items():
            if not pairs: continue
            ts = [s.elapsed_time(e) for s, e in pairs]  # ms
            res[name] = {"mean_ms": float(np.mean(ts)), "count": len(ts), "total_ms": float(np.sum(ts))}
        return res


def measure_components(model, tok, device, decode_tokens):
    """DECODE-ONLY per-call GPU times. Critical: warm up (prefill + several decode steps) BEFORE
    registering hooks, so neither prefill (multi-token) nor CUDA/cuBLAS warmup contaminates the mean.
    Hooks fire once per decode step (batch=1, seq=1) -> clean per-token component cost."""
    l0 = model.model.layers[0]
    ids = tok("The history of computing began in the early twentieth century with", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, use_cache=True)           # prefill (NOT hooked)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(8):                          # warmup decode (NOT hooked)
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        timer = Timer()                             # register hooks AFTER warmup
        timer.hook(l0.self_attn, "attn"); timer.hook(l0.mlp.gate, "gate"); timer.hook(l0.mlp.shared_expert, "shared")
        for j, ex in enumerate(l0.mlp.experts): timer.hook(ex, f"exp_{j}")
        for _ in range(decode_tokens):              # timed decode steps (seq_len=1)
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
    s = timer.summary_ms()
    exp_calls = [v for k, v in s.items() if k.startswith("exp_")]
    expert_gpu_ms = float(np.mean([v["mean_ms"] for v in exp_calls])) if exp_calls else 0.0
    return {"t_attn_ms": s["attn"]["mean_ms"], "t_gate_ms": s["gate"]["mean_ms"],
            "t_shared_ms": s["shared"]["mean_ms"], "t_expert_gpu_ms": expert_gpu_ms}


def measure_cpu_expert(model, device):
    """One routed expert on CPU, input (1, hidden). Returns ms/expert (batch=1)."""
    ex = model.model.layers[0].mlp.experts[0]
    g = ex.gate_proj.weight.detach().float().cpu()
    u = ex.up_proj.weight.detach().float().cpu()
    d = ex.down_proj.weight.detach().float().cpu()
    x = torch.randn(1, g.shape[1])
    def run(): return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)
    for _ in range(5): run()
    t0 = time.perf_counter()
    n = 50
    for _ in range(n): run()
    return (time.perf_counter() - t0) / n * 1000.0


def measure_copy_h2d(model, device):
    """Pinned async H2D of one expert's weights (bf16). Returns (ms, MB)."""
    ex = model.model.layers[0].mlp.experts[0]
    tensors = [ex.gate_proj.weight, ex.up_proj.weight, ex.down_proj.weight]
    cpu_pinned = [t.detach().to("cpu", torch.bfloat16).contiguous().pin_memory() for t in tensors]
    dst = [torch.empty_like(t, device=device) for t in cpu_pinned]
    mb = sum(t.numel() * 2 for t in cpu_pinned) / 1e6
    stream = torch.cuda.Stream(device=device)
    for _ in range(3):
        with torch.cuda.stream(stream):
            for s, dt in zip(cpu_pinned, dst): dt.copy_(s, non_blocking=True)
    torch.cuda.synchronize()
    n = 30
    s_ev = torch.cuda.Event(enable_timing=True); e_ev = torch.cuda.Event(enable_timing=True)
    s_ev.record(stream)
    with torch.cuda.stream(stream):
        for _ in range(n):
            for s, dt in zip(cpu_pinned, dst): dt.copy_(s, non_blocking=True)
    e_ev.record(stream)
    torch.cuda.synchronize()
    return s_ev.elapsed_time(e_ev) / n, mb


def measure_overlap(model, device):
    """Does CPU expert compute overlap GPU work? Compare serial vs concurrent wall time.
    GPU work = a routed expert on GPU run many times; CPU work = a routed expert on CPU.
    overlap_factor = (t_gpu + t_cpu) / t_concurrent ; ~2 means full overlap, ~1 means serialized."""
    ex = model.model.layers[0].mlp.experts[0]
    xg = torch.randn(1, ex.gate_proj.weight.shape[1], device=device, dtype=ex.gate_proj.weight.dtype)
    def gpu_work(rounds):
        for _ in range(rounds): ex(xg)
        torch.cuda.synchronize()
    g = ex.gate_proj.weight.detach().float().cpu(); u = ex.up_proj.weight.detach().float().cpu(); d = ex.down_proj.weight.detach().float().cpu()
    xc = torch.randn(1, g.shape[1])
    def cpu_work(rounds):
        for _ in range(rounds): F.linear(F.silu(F.linear(xc, g)) * F.linear(xc, u), d)
    RG, RC = 2000, 3000  # balanced so t_gpu ~ t_cpu (~0.4s each); imbalanced test hides true overlap
    gpu_work(10); cpu_work(10)
    t0 = time.perf_counter(); gpu_work(RG); t_gpu = time.perf_counter() - t0
    t0 = time.perf_counter(); cpu_work(RC); t_cpu = time.perf_counter() - t0
    # concurrent: GPU in main thread (async launches), CPU in a worker thread (torch CPU ops release GIL)
    th = threading.Thread(target=cpu_work, args=(RC,))
    t0 = time.perf_counter(); th.start(); gpu_work(RG); th.join(); t_conc = time.perf_counter() - t0
    return {"t_gpu_s": t_gpu, "t_cpu_s": t_cpu, "t_concurrent_s": t_conc,
            "overlap_factor": (t_gpu + t_cpu) / max(t_conc, 1e-9)}


def miss_counts_from_traces(trace_dir, max_traces, residencies, n_experts, n_layers):
    """Global-popularity placement (Fiddler-style): per residency, mark top-N popular experts per layer
    resident; count routed misses per (step, layer). Returns per-residency miss stats."""
    files = sorted(glob.glob(str(Path(trace_dir) / "dec_*.pt")))[:max_traces]
    # popularity per (layer, expert)
    pop = np.zeros((n_layers, n_experts), dtype=np.int64)
    all_steps = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        for (_tok, per_layer) in d["steps"]:
            all_steps.append(per_layer)
            for l, topk in enumerate(per_layer):
                for e in topk: pop[l, int(e)] += 1
    res = {}
    for r in residencies:
        n_res = max(1, int(round(r * n_experts)))
        resident = np.zeros((n_layers, n_experts), dtype=bool)
        for l in range(n_layers):
            top = np.argsort(-pop[l])[:n_res]
            resident[l, top] = True
        miss_per_layer = []
        for per_layer in all_steps:
            for l, topk in enumerate(per_layer):
                m = sum(0 if resident[l, int(e)] else 1 for e in topk)
                miss_per_layer.append(m)
        arr = np.array(miss_per_layer)
        res[f"{r}"] = {"n_resident_per_layer": n_res, "mean_miss_per_layer": float(arr.mean()),
                       "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)),
                       "p99": float(np.percentile(arr, 99)), "max": int(arr.max())}
    return res


def main():
    a = parse_args()
    device = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {a.model_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device).eval()
    cfg = model.config
    n_experts = cfg.num_experts; n_layers = cfg.num_hidden_layers; top_k = cfg.num_experts_per_tok
    print(f"[cfg] layers={n_layers} experts={n_experts} top_k={top_k}", flush=True)

    comp = measure_components(model, tok, device, a.decode_tokens)
    comp["t_expert_cpu_ms"] = measure_cpu_expert(model, device)
    t_copy, mb = measure_copy_h2d(model, device)
    comp["t_copy_h2d_ms"] = t_copy; comp["expert_MB"] = mb
    ov = measure_overlap(model, device)
    residencies = [float(x) for x in a.residency.split(",")]
    miss = miss_counts_from_traces(a.trace_dir, a.max_traces, residencies, n_experts, n_layers)

    # verdict: per-layer critical path. CORRECTED: attn+gate are SEQUENTIAL before experts (expert
    # input = post-attention hidden) -> NOT hideable behind CPU-miss compute. Only same-layer GPU
    # expert work (shared + resident routed) overlaps CPU misses.
    L = n_layers
    a_, g_, s_, eg, ec, cp = (comp["t_attn_ms"], comp["t_gate_ms"], comp["t_shared_ms"],
                              comp["t_expert_gpu_ms"], comp["t_expert_cpu_ms"], comp["t_copy_h2d_ms"])
    pre = a_ + g_  # sequential prefix per layer
    verdict = {}
    for r, st in miss.items():
        nm = st["mean_miss_per_layer"]; nres = top_k - nm  # resident routed hits
        hideable = s_ + eg * max(0.0, nres)                # GPU expert work parallel to CPU misses
        cpu_miss = ec * nm
        # per-layer expert-region cost under each policy
        reg_fetch = s_ + top_k * eg + nm * cp              # B0: fetch miss weight, compute on GPU (fetch on crit path)
        reg_cpu_overlap = max(hideable, cpu_miss)          # B1: CPU miss compute overlaps GPU expert work
        reg_cpu_serial = hideable + cpu_miss               # B1 if overlap fails (GIL serialize)
        reg_prefetch_bound = s_ + top_k * eg               # oracle: all misses prefetched resident (free)
        tpot = lambda reg: L * (pre + reg)
        verdict[r] = {
            "mean_miss_per_layer": nm,
            "exposed_cpu_stall_per_layer_ms": max(0.0, cpu_miss - hideable),
            "TPOT_B0_fetch_all_ms": tpot(reg_fetch),
            "TPOT_B1_cpu_overlap_ms": tpot(reg_cpu_overlap),
            "TPOT_B1_cpu_serial_ms": tpot(reg_cpu_serial),
            "TPOT_prefetch_bound_ms": tpot(reg_prefetch_bound),
            "headroom_B1overlap_vs_prefetchbound_pct": 100.0 * (tpot(reg_cpu_overlap) - tpot(reg_prefetch_bound)) / tpot(reg_cpu_overlap),
            "GO_headroom_over_5pct": (tpot(reg_cpu_overlap) - tpot(reg_prefetch_bound)) / tpot(reg_cpu_overlap) > 0.05,
        }

    result = {"components_ms": comp, "overlap": ov, "miss_counts": miss, "verdict": verdict}
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("\n===== SPICE-HX KILL-SHOT RESULT =====", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print("\n[interpretation] If GO_has_exposed_stall is False at all residencies, CPU-miss compute is", flush=True)
    print("hidden under GPU work at batch=1 -> NO scheduling headroom -> honest negative (paper I).", flush=True)


if __name__ == "__main__":
    main()
