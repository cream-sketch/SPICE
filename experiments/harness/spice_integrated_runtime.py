"""Diagnostic timing scaffold: SPICE prefetch + capacity-aware verified-fallback.

诊断用 timing scaffold: SPICE prefetch + capacity-aware verified-fallback.

SPICE prefetch and the capacity-aware fallback are ORTHOGONAL (user's point):
  - SPICE prefetch (paper: 74.24% slot hit, up to 2.86x TPOT) reduces the MISS RATE by predicting
    future-layer experts and async-prefetching them overlapped with compute. Prefetched/resident slots
    are served by a GPU GEMM (the weight is already in HBM). SPICE also issues the prefetch H2D traffic.
  - The capacity-aware fallback improves how the RESIDUAL misses (paper ~25.76%) are served, replacing
    SPICE's SYNCHRONOUS fetch-all (SPICE.pdf line 262/548) with CPU||PCIe split + use!=residency admission.

FAIRNESS (codex review): the per-(token,layer) HIT/RESIDUAL mask is PRE-GENERATED ONCE and shared by all
policies, so the residual workload is IDENTICAL; only the residual-miss SERVICE differs. Prefetch H2D
traffic for the hit slots is issued identically for all policies (real, overlapped on a low-pri stream,
contends PCIe with the capacity fallback's fetches -- a real effect that favors CPU-serve). Activation
D2H is done ONCE per layer. Each policy is warmed up separately. No RNG inside the timed loop.

Policies (prefetch ON for all; only residual fallback differs):
  prefetch_sync     : SPICE original timing = residual miss -> synchronous per-expert H2D fetch + GPU GEMM.
  prefetch_cpu      : SPICE + Fiddler timing = residual miss -> all-CPU serve.
  prefetch_capacity : residual miss -> fixed CPU/PCIe split.
  prefetch_pressure_dp : residual miss -> layer-serial pressure-aware CPU/PCIe split.

All ops are real tensor ops (attn/shared/router/experts), real KV, bf16 path timing, batch=1 decode.
This remains diagnostic, not a source-only baseline and not a full exact-logit model replay:
resident-hit weights are synthetic timing weights, residual masks are synthetic unless supplied by
another harness, and HBM admission/cache feedback is handled by trace replay.
target_residual_rate calibrates the prefetch operating point (paper 0.2576). All printed English.
Core params: no defaults (cpu_dtype/seed have defaults).
"""
import argparse, time, json, glob, random
from pathlib import Path
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description="Diagnostic SPICE prefetch + capacity-aware fallback timing scaffold")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--n_layers", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--d_inter", type=int, required=True)
    ap.add_argument("--n_experts", type=int, required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--n_heads", type=int, required=True)
    ap.add_argument("--ctx_len", type=int, required=True)
    ap.add_argument("--decode_tokens", type=int, required=True)
    ap.add_argument("--cpu_threads", type=int, required=True)
    ap.add_argument("--target_residual_rate", type=float, required=True,
                    help="fraction of routed slots that miss after prefetch (paper 0.2576)")
    ap.add_argument("--fetch_frac", type=float, required=True,
                    help="capacity policy: fraction of residual misses sent to PCIe fetch (rest to CPU)")
    ap.add_argument("--substitute_ranks", type=str, default="",
                    help="controller policy: comma list of top-k RANKS (0=highest gate weight) whose residual "
                         "misses are SHARED-EXPERT-SUBSTITUTED (free, lossy); other residuals CPU-served (exact). "
                         "Empty string = no substitution (all CPU-serve = exact).")
    ap.add_argument("--scheduler_cost_json", default="",
                    help="miss_assignment_microbench JSON for exact pressure-aware DP")
    ap.add_argument("--scheduler_cost_metric", choices=["ms", "mean_ms", "p90_ms"], default="ms")
    ap.add_argument("--scheduler_dense_ms_per_layer", type=float, default=0.687,
                    help="estimated dense+shared/router layer time used only by DP planning")
    ap.add_argument("--scheduler_t_gpu_ms", type=float, default=0.079,
                    help="estimated resident/fetched expert GPU compute ms used only by DP planning")
    ap.add_argument("--scheduler_t_fetch_h2d_ms", type=float, default=0.769,
                    help="estimated one-expert H2D ms used only by DP planning")
    ap.add_argument("--scheduler_prefetch_floor_ms_per_tok", type=float, default=0.0,
                    help="SPICE predicted-hit H2D pressure already occupying PCIe, used only by DP planning")
    ap.add_argument("--policies", default="prefetch_sync,prefetch_cpu,prefetch_capacity,prefetch_pressure_dp",
                    help="comma list from prefetch_sync,prefetch_cpu,prefetch_capacity,prefetch_pressure_dp,prefetch_controller")
    ap.add_argument("--trace_dir", required=True, help="real decode traces dir, or 'synthetic'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu_dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filler_compute_dim", type=int, default=0,
                    help="if >0, run a REAL [dim,dim]x[dim,dim] matmul per layer on the MAIN stream to create a "
                         "realistic per-layer COMPUTE WINDOW (models heavier model / weaker GPU where compute~PCIe). "
                         "Prefetch H2D overlaps this window; lets the real runtime expose SPICE's overlap benefit. 0=A800-native.")
    return ap.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    rng = random.Random(a.seed)
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev); dt = torch.bfloat16
    cpu_dt = torch.bfloat16 if a.cpu_dtype == "bf16" else torch.float32
    dm, di, H = a.d_model, a.d_inter, a.n_heads; hd = dm // H
    L, E = a.n_layers, a.n_experts
    # controller: which top-k ranks (0=highest gate weight) to shared-expert-substitute when residual-missed
    substitute_ranks = set(int(x) for x in a.substitute_ranks.split(",") if x.strip() != "")
    policies = [x.strip() for x in a.policies.split(",") if x.strip()]
    valid_policies = {"no_prefetch", "prefetch_sync", "prefetch_cpu", "prefetch_capacity",
                      "prefetch_pressure_dp", "prefetch_controller"}
    unknown = sorted(set(policies) - valid_policies)
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    if "prefetch_pressure_dp" in policies and not a.scheduler_cost_json:
        raise ValueError("--scheduler_cost_json is required when policies include prefetch_pressure_dp")
    if "prefetch_pressure_dp" in policies and substitute_ranks:
        raise ValueError("prefetch_pressure_dp is exact; use empty --substitute_ranks or run prefetch_controller separately")

    cost_table = {}
    if a.scheduler_cost_json:
        data = json.loads(Path(a.scheduler_cost_json).read_text())
        for row in data["rows"]:
            metric = a.scheduler_cost_metric
            if metric not in row:
                raise KeyError(f"{a.scheduler_cost_json} row missing metric {metric}")
            cost_table[(int(row["n_miss"]), int(row["n_fetch"]))] = float(row[metric])

    Wq = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wk = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wv = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wo = [torch.randn(dm, dm, device=dev, dtype=dt) for _ in range(L)]
    Wsh = [(torch.randn(di, dm, device=dev, dtype=dt), torch.randn(di, dm, device=dev, dtype=dt),
            torch.randn(dm, di, device=dev, dtype=dt)) for _ in range(L)]
    Wrt = [torch.randn(E, dm, device=dev, dtype=dt) for _ in range(L)]
    Kc = [torch.randn(H, a.ctx_len, hd, device=dev, dtype=dt) for _ in range(L)]
    Vc = [torch.randn(H, a.ctx_len, hd, device=dev, dtype=dt) for _ in range(L)]

    # distinct host (pinned, for fetch H2D) + CPU (for CPU-serve) banks to avoid L3 artifacts
    NBANK = 256
    host_bank = [(torch.randn(di, dm, dtype=dt).pin_memory(), torch.randn(di, dm, dtype=dt).pin_memory(),
                  torch.randn(dm, di, dtype=dt).pin_memory()) for _ in range(NBANK)]
    # CPU and fetch paths share the same missed-expert weights; only resident hits use a synthetic GPU timing weight.
    cpu_bank = [(g.to(dtype=cpu_dt).contiguous(), u.to(dtype=cpu_dt).contiguous(), d.to(dtype=cpu_dt).contiguous())
                for g, u, d in host_bank]
    resident_w = (torch.randn(di, dm, device=dev, dtype=dt), torch.randn(di, dm, device=dev, dtype=dt),
                  torch.randn(dm, di, device=dev, dtype=dt))  # shared GPU weight for hit-GEMM timing

    fetch_stage = [(torch.empty(di, dm, device=dev, dtype=dt), torch.empty(di, dm, device=dev, dtype=dt),
                    torch.empty(dm, di, device=dev, dtype=dt)) for _ in range(8)]
    pf_stage = (torch.empty(di, dm, device=dev, dtype=dt), torch.empty(di, dm, device=dev, dtype=dt),
                torch.empty(dm, di, device=dev, dtype=dt))  # reused target for prefetch H2D traffic
    fetch_stream = torch.cuda.Stream(dev, priority=-1)
    pf_stream = torch.cuda.Stream(dev, priority=0)

    def hkey(l, e):
        return (l * E + e) % NBANK

    def attn(l, h):
        q = (h @ Wq[l].T).view(H, 1, hd); k = (h @ Wk[l].T).view(H, 1, hd); v = (h @ Wv[l].T).view(H, 1, hd)
        K = torch.cat([Kc[l], k], dim=1); V = torch.cat([Vc[l], v], dim=1)
        o = F.scaled_dot_product_attention(q, K, V)
        return (o.reshape(1, dm) @ Wo[l].T)

    def shared(l, h):
        g, u, d = Wsh[l]; return F.linear(F.silu(F.linear(h, g)) * F.linear(h, u), d)

    def expert_gpu(w, h):
        g, u, d = w; return F.linear(F.silu(F.linear(h, g)) * F.linear(h, u), d)

    def expert_cpu(hc, l, e):
        g, u, d = cpu_bank[hkey(l, e)]
        return F.linear(F.silu(F.linear(hc, g)) * F.linear(hc, u), d)

    # REAL per-layer compute window (filler matmul) to model heavier-model / weaker-GPU regime where
    # compute ~ PCIe, so SPICE's prefetch overlap can hide transfers. fcd=0 -> A800-native (tiny compute).
    fcd = a.filler_compute_dim
    filler_a = torch.randn(fcd, fcd, device=dev, dtype=dt) if fcd > 0 else None
    filler_b = torch.randn(fcd, fcd, device=dev, dtype=dt) if fcd > 0 else None

    def filler_compute():
        if filler_a is not None:
            return (filler_a @ filler_b).sum()  # real GPU matmul on the main stream (a compute window)
        return None

    # ---- routing trace ----
    if a.trace_dir == "synthetic":
        import numpy as np
        nr = np.random.default_rng(a.seed)
        routes = [[nr.choice(E, a.top_k, replace=False).tolist() for _ in range(L)] for _ in range(a.decode_tokens)]
    else:
        f = sorted(glob.glob(str(Path(a.trace_dir) / "dec_*.pt")))[0]
        d = torch.load(f, map_location="cpu", weights_only=False)
        steps = d["steps"]
        routes = [[[int(e) for e in pl[l]] for l in range(L)] for (_t, pl) in steps if all(x is not None for x in pl)][:a.decode_tokens]
        if len(routes) < a.decode_tokens:
            routes = (routes * (a.decode_tokens // max(1, len(routes)) + 1))[:a.decode_tokens]
    T = len(routes)

    # ---- PRE-GENERATE shared per-slot mask (identical across policies) ----
    # slot_info[ti][l] = list of (expert, rank, is_residual). rank = index in the token-layer top-k
    # (descending gate weight, 0=highest). is_residual: prefetch missed this slot (prob target_residual_rate);
    # else it is a prefetch/resident HIT. The controller may SUBSTITUTE low-rank slots (skip prefetch+serve).
    slot_info = [[[] for _ in range(L)] for _ in range(T)]
    for ti in range(T):
        for l in range(L):
            for rank, e in enumerate(routes[ti][l]):  # routes top-k ordered by gate weight (rank 0 = highest)
                is_resid = rng.random() < a.target_residual_rate
                slot_info[ti][l].append((e, rank, is_resid))

    def cpu_plan_cost(n_cpu):
        if n_cpu <= 0:
            return 0.0
        return cost_table[(n_cpu, 0)]

    def layer_serial_plan_step(clock, prev_fetches, hit_count, nmiss, n_fetch):
        n_fetch = max(0, min(nmiss, n_fetch))
        n_cpu = nmiss - n_fetch
        base_done = clock + a.scheduler_dense_ms_per_layer + hit_count * a.scheduler_t_gpu_ms
        if nmiss == 0:
            return base_done, prev_fetches, 0.0
        if n_fetch == 0:
            return base_done + cpu_plan_cost(n_cpu), prev_fetches, 0.0
        cpu_ms = cpu_plan_cost(n_cpu)
        no_backlog_service = max(cost_table[(nmiss, n_fetch)], cpu_ms,
                                 n_fetch * (a.scheduler_t_fetch_h2d_ms + a.scheduler_t_gpu_ms))
        copy_ready = a.scheduler_prefetch_floor_ms_per_tok + prev_fetches * a.scheduler_t_fetch_h2d_ms
        pcie_wait = max(0.0, copy_ready - base_done)
        blocked_fetch_service = pcie_wait + n_fetch * a.scheduler_t_fetch_h2d_ms + n_fetch * a.scheduler_t_gpu_ms
        service = max(no_backlog_service, cpu_ms, blocked_fetch_service)
        return base_done + service, prev_fetches + n_fetch, pcie_wait

    def dp_fetch_counts_for_token(ti):
        """Layer-serial timing planner for this token's residual misses."""
        dp = {0: (0.0, [], 0.0)}
        for l in range(L):
            hit_count = 0
            nmiss = 0
            for _e, rank, is_resid in slot_info[ti][l]:
                if rank in substitute_ranks:
                    continue
                if is_resid:
                    nmiss += 1
                else:
                    hit_count += 1
            ndp = {}
            for prev_fetches, (clock, prev_choices, wait_sum) in dp.items():
                for nf in range(nmiss + 1):
                    total_clock, total_fetches, wait = layer_serial_plan_step(clock, prev_fetches, hit_count, nmiss, nf)
                    old = ndp.get(total_fetches)
                    if old is None or total_clock < old[0]:
                        ndp[total_fetches] = (total_clock, prev_choices + [nf], wait_sum + wait)
            dp = ndp
        best = None
        for total_fetches, (clock_ms, choices, wait_sum) in dp.items():
            pcie_ms = a.scheduler_prefetch_floor_ms_per_tok + total_fetches * a.scheduler_t_fetch_h2d_ms
            item = (max(clock_ms, pcie_ms), total_fetches, clock_ms, pcie_ms, choices, wait_sum)
            if best is None or item < best:
                best = item
        assert best is not None
        return best

    def serve_residual(policy, misses, l, hn_gpu, sub_stats, n_fetch_override=None):
        # misses: list of (expert, rank). Returns residual contribution tensor on GPU; timed fetch/cpu service.
        # sub_stats: [substituted, cpu_served, fetched] counters (mutated). hn_gpu: (1, dm) on GPU.
        if not misses:
            return None
        if policy == "prefetch_sync":
            acc = None
            for e, _r in misses:  # synchronous per-expert: H2D -> wait -> GEMM (SPICE original fallback)
                with torch.cuda.stream(fetch_stream):
                    for s, dd in zip(host_bank[hkey(l, e)], fetch_stage[0]):
                        dd.copy_(s, non_blocking=True)
                torch.cuda.current_stream(dev).wait_stream(fetch_stream)
                oe = expert_gpu(fetch_stage[0], hn_gpu)
                acc = oe if acc is None else acc + oe
                sub_stats[3] += 1
            return acc
        if policy == "prefetch_cpu":
            hc = hn_gpu.to(dtype=cpu_dt).cpu()  # ONE D2H per layer (codex fix)
            acc = None
            for e, _r in misses:
                oc = expert_cpu(hc, l, e)
                acc = oc if acc is None else acc + oc
                sub_stats[2] += 1
            return acc.to(dev, dtype=dt)
        # prefetch_capacity / prefetch_pressure_dp: split fetch (PCIe) || cpu (DRAM), concurrent.
        # fetch_frac=0 -> all CPU; fetch_frac=1 -> all fetch. round() allows exact 0 (no forced fetch).
        n = len(misses)
        nf = int(round(a.fetch_frac * n)) if n_fetch_override is None else int(n_fetch_override)
        nf = max(0, min(n, nf))
        fetch_set = misses[:nf]; cpu_set = misses[nf:]
        fetch_outs = []
        for i, (e, _r) in enumerate(fetch_set):  # launch async H2D (concurrent with CPU below)
            slot = fetch_stage[i % len(fetch_stage)]
            with torch.cuda.stream(fetch_stream):
                for s, dd in zip(host_bank[hkey(l, e)], slot):
                    dd.copy_(s, non_blocking=True)
            fetch_outs.append((slot, e)); sub_stats[3] += 1
        acc = None
        if cpu_set:
            hc = hn_gpu.to(dtype=cpu_dt).cpu()  # ONE D2H
            for e, _r in cpu_set:
                oc = expert_cpu(hc, l, e)
                acc = oc if acc is None else acc + oc
                sub_stats[2] += 1
            acc = acc.to(dev, dtype=dt)
        if fetch_outs:
            torch.cuda.current_stream(dev).wait_stream(fetch_stream)
            for slot, e in fetch_outs:
                oe = expert_gpu(slot, hn_gpu)
                acc = oe if acc is None else acc + oe
        return acc

    def run_decode(policy):
        sub_stats = [0, 0, 0, 0, 0, 0.0, 0.0, 0.0]
        # [sub_hit, sub_resid, cpu, fetch, dp_fetch, dp_layer_clock, dp_pcie, dp_pcie_wait]
        # controller substitutes low-gate-mass (high-rank) slots: skip BOTH prefetch H2D AND serve,
        # relying on the always-on shared expert. This removes their PCIe traffic -> lowers the floor.
        do_sub = (policy == "prefetch_controller")
        no_pf = (policy == "no_prefetch")  # on-demand baseline: EVERY routed expert sync-fetched, no prefetch
        serve_policy = "prefetch_sync" if no_pf else ("prefetch_cpu" if policy == "prefetch_controller" else policy)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for ti in range(T):
            h = torch.randn(1, dm, device=dev, dtype=dt)
            dp_plan = None
            if policy == "prefetch_pressure_dp":
                _tok_ms, total_fetches, layer_clock_ms, pcie_ms, choices, wait_ms = dp_fetch_counts_for_token(ti)
                dp_plan = choices
                sub_stats[4] += total_fetches
                sub_stats[5] += layer_clock_ms
                sub_stats[6] += pcie_ms
                sub_stats[7] += wait_ms
            for l in range(L):
                h = h + attn(l, h)
                hn = h
                _ = F.linear(hn, Wrt[l])
                out = shared(l, hn)
                filler_compute()  # REAL per-layer compute window (prefetch H2D overlaps this on pf_stream)
                residual = []
                for e, rank, is_resid in slot_info[ti][l]:
                    if do_sub and rank in substitute_ranks:
                        # substituted: no prefetch, no serve (shared covers it). Split hit vs residual:
                        # substituted HIT saves a prefetch H2D (PCIe, the bottleneck); residual saves CPU serve.
                        sub_stats[1 if is_resid else 0] += 1
                        continue
                    if no_pf:
                        # on-demand baseline: this slot (hit OR residual) is sync-fetched on the critical path
                        residual.append((e, rank))
                    elif is_resid:
                        residual.append((e, rank))
                    else:
                        # hit: GPU GEMM + SPICE prefetch H2D traffic (overlapped on low-pri stream, hides behind compute)
                        out = out + expert_gpu(resident_w, hn)
                        with torch.cuda.stream(pf_stream):
                            for s, dd in zip(host_bank[hkey(l, e)], pf_stage):
                                dd.copy_(s, non_blocking=True)
                n_fetch_override = dp_plan[l] if dp_plan is not None else None
                r = serve_residual(serve_policy, residual, l, hn, sub_stats, n_fetch_override)
                if r is not None:
                    out = out + r
                h = h + out
        torch.cuda.synchronize(dev)
        ms = (time.perf_counter() - t0) / T * 1000.0
        return ms, sub_stats

    residual_per_tok = sum(1 for ti in range(T) for l in range(L) for (_e, _r, ir) in slot_info[ti][l] if ir) / T
    hits_per_tok = sum(1 for ti in range(T) for l in range(L) for (_e, _r, ir) in slot_info[ti][l] if not ir) / T
    res = {"config": vars(a), "T": T, "residual_miss_per_tok": residual_per_tok, "hits_per_tok": hits_per_tok,
           "effective_residual_rate": residual_per_tok / max(1e-9, residual_per_tok + hits_per_tok),
           "substitute_ranks": sorted(substitute_ranks)}
    import statistics
    for pol in policies:
        run_decode(pol)  # per-policy warmup
        runs = [run_decode(pol) for _ in range(5)]  # 5 timed runs, take median (reduce single-run noise)
        ms = statistics.median([r[0] for r in runs]); st = runs[-1][1]
        res[f"tpot_{pol}_ms"] = ms
        res[f"stats_{pol}"] = {"substituted_hit_per_tok": st[0] / T, "substituted_residual_per_tok": st[1] / T,
                               "cpu_served_per_tok": st[2] / T, "fetched_per_tok": st[3] / T,
                               "dp_planned_fetches_per_tok": st[4] / T,
                               "dp_planned_layer_clock_ms_per_tok": st[5] / T,
                               "dp_planned_pcie_ms_per_tok": st[6] / T,
                               "dp_planned_pcie_wait_ms_per_tok": st[7] / T}
        print(f"[{pol:>20}] TPOT={ms:8.3f} ms/token  subst_hit/tok={st[0]/T:5.2f} subst_resid/tok={st[1]/T:5.2f} "
              f"cpu/tok={st[2]/T:5.2f} fetch/tok={st[3]/T:5.2f} dp_fetch/tok={st[4]/T:5.2f}", flush=True)
    base = res["tpot_prefetch_sync_ms"]
    if "tpot_prefetch_cpu_ms" in res:
        res["cpu_vs_sync_pct"] = 100 * (base - res["tpot_prefetch_cpu_ms"]) / base
    if "tpot_prefetch_capacity_ms" in res:
        res["capacity_vs_sync_pct"] = 100 * (base - res["tpot_prefetch_capacity_ms"]) / base
    if "tpot_prefetch_pressure_dp_ms" in res:
        res["pressure_dp_vs_sync_pct"] = 100 * (base - res["tpot_prefetch_pressure_dp_ms"]) / base
        if "tpot_prefetch_cpu_ms" in res:
            res["pressure_dp_vs_cpu_pct"] = 100 * (res["tpot_prefetch_cpu_ms"] - res["tpot_prefetch_pressure_dp_ms"]) / res["tpot_prefetch_cpu_ms"]
    if "tpot_prefetch_controller_ms" in res:
        res["controller_vs_sync_pct"] = 100 * (base - res["tpot_prefetch_controller_ms"]) / base
        if "tpot_prefetch_cpu_ms" in res:
            res["controller_vs_cpu_pct"] = 100 * (res["tpot_prefetch_cpu_ms"] - res["tpot_prefetch_controller_ms"]) / res["tpot_prefetch_cpu_ms"]
    print(f"\n[diagnostic] residual_miss/tok={residual_per_tok:.2f} (rate={res['effective_residual_rate']:.3f}) "
          f"substitute_ranks={sorted(substitute_ranks)} | cpu vs sync={res.get('cpu_vs_sync_pct', float('nan')):+.1f}%  "
          f"pressure_dp vs sync={res.get('pressure_dp_vs_sync_pct', float('nan')):+.1f}%  "
          f"pressure_dp vs cpu={res.get('pressure_dp_vs_cpu_pct', float('nan')):+.1f}%", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
