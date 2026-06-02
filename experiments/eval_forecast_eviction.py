"""Experiment 2: forecast-driven eviction vs LRU/LS/oracle, with the 2x2
(prefetch x eviction) ablation that isolates the eviction-only benefit.

实验2：forecast 驱动驱逐 vs LRU/LS/oracle, 用 2x2 (预取 x 驱逐) ablation 隔离
"驱逐本身"的收益 (codex Q4 反 "只是更好的预取器" 守门).

Consumes per-text dumps from qwen_spice_draft.py --dump_forecast:
  true_top [L, S, top_k]  : real per-layer per-position top-K experts (demand)
  fcast    [L, H, S, top_k]: draft forecast; fcast[a, h, p] = predicted top-K for
                             layer a+h at position p (anchored at true state a).
每条序列独立(冷启动). decode 流 = 各 position p 顺序走 L 层.

Deadline-aware overlap model per (position, layer):
  - resolve DEMAND (true_top[l,p]); a miss synchronously fetches (stall += fetch_ms);
  - COMPUTE layer gives an overlap budget = bandwidth * t_layer_ms bytes;
  - if prefetch=draft: prefetch forecast experts for upcoming layers l+1..l+H
    (fcast[l+1, :, p]) within the overlap budget (these are 'pending');
  - a pending (prefetched, not-yet-used) expert evicted before use = collision miss
    (the SpecMD failure mode). Eviction policy decides victims.
  - lossless: forecast only schedules movement; demand is always verified+fetched.

Policies (eviction): lru, ls (cyclic), forecast (protect near forecast, evict
non-forecast first), oracle (Belady over this text's remaining demand stream).
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch


def load_dumps(dump_dir: Path) -> list[dict]:
    manifest = json.loads((dump_dir / "manifest.json").read_text(encoding="utf-8"))
    texts = []
    for f in manifest["files"]:
        d = torch.load(dump_dir / f, map_location="cpu", weights_only=False)
        texts.append({
            "true_top": d["true_top"].tolist(),   # [L][S][k]
            "fcast": d["fcast"].tolist(),          # [L][H][S][k]
            "num_layers": int(d["num_layers"]),
        })
    return texts, manifest


def build_next_use(true_top: list, num_layers: int, seq_len: int):
    """Precompute Belady next-use per (layer,expert) over the demand stream.

    预计算 Belady 下一次使用位置 (global idx = p*L + l). 返回 occ 字典 + 指针.
    """
    occ: dict[tuple[int, int], list[int]] = {}
    for p in range(seq_len):
        for l in range(num_layers):
            gidx = p * num_layers + l
            for e in true_top[l][p]:
                occ.setdefault((l, e), []).append(gidx)
    return occ


def simulate_text(text: dict, seq_len: int, num_layers: int, top_k: int, horizon: int,
                  prefetch: str, evict: str, capacity: int,
                  expert_bytes: int, bandwidth_gbps: float, t_layer_ms: float) -> dict:
    """Deadline-aware DMA model. A transfer of expert_bytes takes fetch_ms; transfers
    span MULTIPLE layer-compute windows (an expert is much larger than one window).
    Demand misses preempt the PCIe channel (demand-priority DMA); speculative
    prefetches use channel time and become resident only when 'ready' time passes.

    时间模型: now=计算时钟, dma_free=通道空闲时刻; 传输跨多层; demand 抢占通道.
    cache[key] = {ready: 到位时刻ms, last: step, pending: bool, layer}.
    """
    true_top = text["true_top"]
    fcast = text["fcast"]
    bytes_per_ms = bandwidth_gbps * (1024 ** 3) / 1000.0
    fetch_ms = expert_bytes / bytes_per_ms

    occ = build_next_use(true_top, num_layers, seq_len) if evict == "oracle" else None
    occ_ptr = {k: 0 for k in occ} if occ else None

    cache: "OrderedDict[tuple[int,int], dict]" = OrderedDict()
    # running per-(layer,expert) frequency = realizable cross-token reuse prior (layer_prior)
    layer_freq: dict[tuple[int, int], int] = {}
    step = 0
    now = 0.0
    dma_free = 0.0

    hits = 0
    demand_misses = 0
    total_demand = 0
    h2d_prefetch_bytes = 0
    h2d_demand_bytes = 0
    stall_ms = 0.0
    late_prefetch_wait_ms = 0.0
    collisions = 0
    prefetch_useful = 0
    prefetch_issued = 0

    def next_use(key, cur):
        positions = occ.get(key, ())
        ptr = occ_ptr.get(key, 0)
        while ptr < len(positions) and positions[ptr] <= cur:
            ptr += 1
        occ_ptr[key] = ptr
        return positions[ptr] if ptr < len(positions) else (seq_len * num_layers + 10)

    def forecast_set(l, p):
        out = {}
        if l + 1 >= num_layers:
            return out
        for h in range(horizon):
            tl = l + 1 + h
            if tl >= num_layers:
                break
            out[tl] = set(e for e in fcast[l + 1][h][p] if e >= 0)
        return out

    cur_layer_forecast: dict[tuple[int, int], int] = {}

    def evict_one(protect: set, cur_layer: int, cur_gidx: int):
        candidates = [k for k in cache if k not in protect]
        if not candidates:
            candidates = list(cache.keys())
        if evict == "lru":
            victim = min(candidates, key=lambda k: cache[k]["last"])
        elif evict == "ls":
            victim = max(candidates, key=lambda k: (((cache[k]["layer"] - cur_layer) % num_layers) or num_layers,
                                                    -cache[k]["last"]))
        elif evict == "oracle":
            victim = max(candidates, key=lambda k: next_use(k, cur_gidx))
        elif evict == "forecast":
            # codex Q3: keep-value = draft near-forecast (precise, this token) + layer_prior
            # frequency (cross-token reuse for residents beyond the lookahead window).
            # evict MIN keep-value; tie-break LRU.
            def keep_value(k):
                if k in cur_layer_forecast:
                    horizon_d = max(1, cur_layer_forecast[k] - cur_layer)
                    return 1000.0 + 1.0 / horizon_d   # predicted very soon this token -> protect
                return float(layer_freq.get(k, 0))     # else cross-token frequency prior
            victim = min(candidates, key=lambda k: (keep_value(k), cache[k]["last"]))
        else:
            raise ValueError(evict)
        ent = cache.pop(victim)
        return victim, ent

    def admit(key, ready, pending, layer, protect, cur_layer, cur_gidx):
        nonlocal collisions
        while len(cache) >= capacity and cache:
            _, ent = evict_one(protect, cur_layer, cur_gidx)
            if ent["pending"] and ent["ready"] > now:
                collisions += 1  # in-flight prefetch evicted before delivery
        cache[key] = {"ready": ready, "last": step, "pending": pending, "layer": layer}
        cache.move_to_end(key)

    for p in range(seq_len):
        for l in range(num_layers):
            step += 1
            gidx = p * num_layers + l
            demand_keys = [(l, e) for e in true_top[l][p]]
            demand_set = set(demand_keys)
            cur_layer_forecast = {}
            fs = forecast_set(l, p)
            for tl, experts in fs.items():
                for e in experts:
                    cur_layer_forecast[(tl, e)] = tl

            # ---- DEMAND (deadline = now) ----
            for key in demand_keys:
                total_demand += 1
                layer_freq[key] = layer_freq.get(key, 0) + 1  # online cross-token reuse prior
                ent = cache.get(key)
                if ent is not None and ent["ready"] <= now:
                    hits += 1
                    if ent["pending"]:
                        ent["pending"] = False
                        prefetch_useful += 1
                    ent["last"] = step
                    cache.move_to_end(key)
                elif ent is not None and ent["ready"] > now:
                    # in-flight prefetch not yet delivered: wait the remainder (partial benefit)
                    wait = ent["ready"] - now
                    stall_ms += wait
                    late_prefetch_wait_ms += wait
                    now = ent["ready"]
                    hits += 1
                    if ent["pending"]:
                        ent["pending"] = False
                        prefetch_useful += 1
                    ent["last"] = step
                    cache.move_to_end(key)
                else:
                    # demand miss: preempt channel, fetch now on critical path
                    demand_misses += 1
                    h2d_demand_bytes += expert_bytes
                    arrive = now + fetch_ms
                    stall_ms += fetch_ms
                    dma_free = max(dma_free, now) + fetch_ms  # demand pushes any in-flight spec back
                    now = arrive
                    admit(key, ready=arrive, pending=False, layer=l,
                          protect=demand_set, cur_layer=l, cur_gidx=gidx)

            # ---- COMPUTE this layer ----
            now += t_layer_ms

            # ---- SPECULATIVE PREFETCH (uses channel; becomes ready at arrive) ----
            if prefetch == "draft":
                for tl in sorted(fs.keys()):
                    for e in fs[tl]:
                        key = (tl, e)
                        if key in cache:
                            continue
                        start = max(now, dma_free)
                        # bandwidth-aware throttle: skip if channel can't start within the lookahead window
                        if start - now > horizon * t_layer_ms:
                            break
                        arrive = start + fetch_ms
                        dma_free = arrive
                        admit(key, ready=arrive, pending=True, layer=tl,
                              protect=demand_set | {key}, cur_layer=l, cur_gidx=gidx)
                        h2d_prefetch_bytes += expert_bytes
                        prefetch_issued += 1
                    else:
                        continue
                    break

    return {
        "late_prefetch_wait_ms": late_prefetch_wait_ms,
        "total_demand": total_demand,
        "hits": hits,
        "demand_misses": demand_misses,
        "stall_ms": stall_ms,
        "h2d_prefetch_bytes": h2d_prefetch_bytes,
        "h2d_demand_bytes": h2d_demand_bytes,
        "collisions": collisions,
        "prefetch_useful": prefetch_useful,
        "prefetch_issued": prefetch_issued,
    }


def run(texts, seq_caps, num_layers, top_k, horizon, prefetch, evict, capacity,
        expert_bytes, bandwidth_gbps, t_layer_ms) -> dict:
    agg = {"total_demand": 0, "hits": 0, "demand_misses": 0, "stall_ms": 0.0,
           "h2d_prefetch_bytes": 0, "h2d_demand_bytes": 0, "collisions": 0,
           "prefetch_useful": 0, "prefetch_issued": 0}
    tokens = 0
    for text in texts:
        S = len(text["true_top"][0])
        L = text["num_layers"]
        r = simulate_text(text, S, L, top_k, horizon, prefetch, evict, capacity,
                          expert_bytes, bandwidth_gbps, t_layer_ms)
        for k in agg:
            agg[k] += r[k]
        tokens += S
    total_h2d_gb = (agg["h2d_prefetch_bytes"] + agg["h2d_demand_bytes"]) / (1024 ** 3)
    return {
        "prefetch": prefetch, "evict": evict, "capacity": capacity,
        "bandwidth_gbps": bandwidth_gbps,
        "hit_rate": agg["hits"] / max(1, agg["total_demand"]),
        "demand_miss_rate": agg["demand_misses"] / max(1, agg["total_demand"]),
        "exposed_stall_ms_per_token": agg["stall_ms"] / max(1, tokens),
        "h2d_gb": total_h2d_gb,
        "collisions": agg["collisions"],
        "prefetch_useful": agg["prefetch_useful"],
        "prefetch_issued": agg["prefetch_issued"],
        "prefetch_waste_rate": 1.0 - agg["prefetch_useful"] / max(1, agg["prefetch_issued"]),
        "decode_tokens": tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2: forecast-driven eviction 2x2 ablation")
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--expert_mb", type=float, default=17.0)
    ap.add_argument("--t_layer_ms", type=float, default=0.40)
    ap.add_argument("--cache_slots", type=str, default="72,144,288,720")
    ap.add_argument("--bandwidth_gbps", type=float, default=5.0)
    args = ap.parse_args()

    texts, manifest = load_dumps(Path(args.dump_dir))
    num_layers = texts[0]["num_layers"]
    expert_bytes = int(args.expert_mb * 1024 * 1024)
    caps = [int(x) for x in args.cache_slots.split(",") if x.strip()]

    combos = [(pf, ev) for pf in ("off", "draft") for ev in ("lru", "ls", "forecast", "oracle")]
    rows = []
    for cap in caps:
        for pf, ev in combos:
            rows.append(run(texts, None, num_layers, args.top_k, args.horizon, pf, ev, cap,
                            expert_bytes, args.bandwidth_gbps, args.t_layer_ms))
    out = {"experiment": "forecast_eviction_2x2", "dump_dir": args.dump_dir,
           "num_layers": num_layers, "num_texts": len(texts),
           "config": {"top_k": args.top_k, "horizon": args.horizon, "expert_mb": args.expert_mb,
                      "t_layer_ms": args.t_layer_ms, "bandwidth_gbps": args.bandwidth_gbps},
           "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"layers={num_layers} texts={len(texts)} tokens={rows[0]['decode_tokens']} bw={args.bandwidth_gbps}")
    print(f"{'cap':>5} {'prefetch':>9} {'evict':>9} {'hit':>6} {'miss':>6} {'stall/tok':>10} {'h2d_gb':>7} {'collis':>7} {'pf_waste':>8}")
    for r in rows:
        print(f"{r['capacity']:>5} {r['prefetch']:>9} {r['evict']:>9} {r['hit_rate']:>6.3f} "
              f"{r['demand_miss_rate']:>6.3f} {r['exposed_stall_ms_per_token']:>10.3f} {r['h2d_gb']:>7.2f} "
              f"{r['collisions']:>7} {r['prefetch_waste_rate']:>8.3f}")
    # key Q4 contrast: draft prefetch + lru vs draft prefetch + forecast
    print("\n[Q4 eviction-only benefit: draft-prefetch, forecast vs lru]")
    for cap in caps:
        lru = next(r for r in rows if r["capacity"] == cap and r["prefetch"] == "draft" and r["evict"] == "lru")
        fc = next(r for r in rows if r["capacity"] == cap and r["prefetch"] == "draft" and r["evict"] == "forecast")
        if lru["exposed_stall_ms_per_token"] > 0:
            gain = (lru["exposed_stall_ms_per_token"] - fc["exposed_stall_ms_per_token"]) / lru["exposed_stall_ms_per_token"]
            print(f"  cap={cap}: lru_stall={lru['exposed_stall_ms_per_token']:.3f} "
                  f"forecast_stall={fc['exposed_stall_ms_per_token']:.3f} eviction_gain={gain*100:.1f}%")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
