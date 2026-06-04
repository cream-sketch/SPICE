"""Unified SPICE prefetch + residual-miss event scheduler replay.

This harness is the unified event-replay successor to the scalar PCIe-floor replays.  It
uses SPICE forecast dumps (`true_top`, `fcast`) and simulates one shared H2D copy
queue, CPU expert service, GPU layer time, HBM main-cache residency, and an
optional deadline-protected staging buffer.  All routed experts are computed:
there is no drop, quantization, or approximation.

Important modeling choices:
  * Expert H2D transfers are atomic at the measured 17MB expert granularity.
  * Low-priority draft prefetch can be reordered behind high-priority fallback
    fetches / CPU-result H2D if it has not started.  An in-flight expert copy is
    not preempted.  `fifo_deadline` disables this bypass and models a deep CUDA
    FIFO where already-issued prefetch DMA blocks later fallback traffic.
  * `shallow_scheduler` models a software issuer that keeps only a small number
    of low-priority prefetch DMA requests submitted.  Cancellations in this mode
    are cancellations of software intents that have not started, not aborts of
    already-running CUDA copies.
  * The deadline scheduler issues draft prefetches only when the transfer can
    complete before the target-layer lower-bound deadline.  Baselines issue the
    original forecast stream without this throttle.
  * The staging buffer prevents late/unused prefetches from becoming a persistent
    second cache.  `no_staging` is an ablation that admits prefetched experts
    directly into the main cache.

Diagnostic replay, not an upstream baseline reproduction.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from miss_assignment_replay import evict_ls, load_costs, popularity, warm_cache  # noqa: E402


@dataclass
class Transfer:
    start: float
    end: float
    duration: float
    kind: str
    key: tuple[int, int] | None
    target_token: int | None
    target_layer: int | None
    deadline: float | None
    priority: str
    cancelable: bool
    direct_main: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified event replay for SPICE prefetch + residual miss handling")
    p.add_argument("--forecast_dir", required=True)
    p.add_argument("--cost_json", required=True)
    p.add_argument("--pcie_json", default="", help="optional pcie_topology_microbench JSON for small-copy costs")
    p.add_argument("--out", required=True)
    p.add_argument("--train_frac", type=float, required=True)
    p.add_argument("--residency", required=True, help="comma list of HBM routed-expert residency fractions")
    p.add_argument("--max_test_tokens", type=int, required=True)
    p.add_argument("--t_attn", type=float, required=True)
    p.add_argument("--t_gate", type=float, required=True)
    p.add_argument("--t_shared", type=float, required=True)
    p.add_argument("--t_gpu", type=float, required=True)
    p.add_argument("--t_fetch_h2d", type=float, default=0.0, help="ms per bf16 expert H2D; inferred if 0")
    p.add_argument("--min_lead_layers", type=int, default=1)
    p.add_argument("--max_lead_layers", type=int, default=6)
    p.add_argument("--staging_slots", type=int, default=64)
    p.add_argument("--cost_metric", choices=["ms", "mean_ms", "p90_ms"], default="ms")
    p.add_argument("--cpu_scale", type=float, default=1.0)
    p.add_argument("--fetch_scale", type=float, default=1.0)
    p.add_argument("--dense_deadline_scale", type=float, default=1.0,
                   help="target-layer deadline lower bound = layer_start + lead*dense_ms*scale")
    p.add_argument("--shallow_low_depth", type=int, default=2,
                   help="max active low-priority H2D prefetch copies for shallow_scheduler")
    p.add_argument("--policies", default="spice_fetch_all,fiddler_cpu,deadline_scheduler,shallow_scheduler,"
                                       "fifo_deadline,no_cancel,no_staging,oracle_deadline")
    return p.parse_args()


def load_forecast_sequences(forecast_dir: str):
    root = Path(forecast_dir)
    man = json.loads((root / "manifest.json").read_text())
    files = man.get("files") or sorted(p.name for p in root.glob("fc_*.pt"))
    seqs = []
    n_layers = top_k = max_horizon = None
    n_experts = 0
    for name in files:
        d = torch.load(root / name, map_location="cpu", weights_only=False)
        true_top = d["true_top"].long()  # [L,S,K]
        fcast = d["fcast"].long()        # [L,H,S,K]
        if n_layers is None:
            n_layers = int(d.get("num_layers", true_top.shape[0]))
            top_k = int(d.get("top_k", true_top.shape[-1]))
            max_horizon = int(d.get("max_horizon", fcast.shape[1]))
        n_experts = max(n_experts, int(true_top.max().item()) + 1)
        valid = fcast[fcast >= 0]
        if valid.numel():
            n_experts = max(n_experts, int(valid.max().item()) + 1)
        seqs.append({"name": name, "true_top": true_top, "fcast": fcast})
    if n_layers is None:
        raise ValueError(f"no forecast files in {forecast_dir}")
    return seqs, int(n_layers), int(n_experts), int(top_k), int(max_horizon), man


def seq_for_popularity(item):
    true_top = item["true_top"]
    layers, tokens, _top_k = true_top.shape
    seq = []
    for t in range(tokens):
        seq.append((t, [[int(x) for x in true_top[l, t].tolist()] for l in range(layers)]))
    return seq


def build_future_positions(true_top: torch.Tensor):
    fut = defaultdict(list)
    layers, tokens, _top_k = true_top.shape
    for t in range(tokens):
        for l in range(layers):
            for e in true_top[l, t].tolist():
                fut[(l, int(e))].append(t)
    return fut


def oracle_next_use(fut, key, ti: int):
    arr = fut.get(key, [])
    idx = bisect.bisect_right(arr, ti)
    return arr[idx] if idx < len(arr) else 10**12


def pcie_small_costs(path: str) -> tuple[float, float]:
    if not path:
        return 0.014, 0.015
    data = json.loads(Path(path).read_text())
    by_case = {row["case"]: row for row in data.get("rows", [])}
    d2h = by_case.get("small_d2h_alone", {}).get("median_ms", 0.014)
    h2d = by_case.get("small_h2d_alone", {}).get("median_ms", 0.015)
    return float(d2h), float(h2d)


def cpu_cost_ms(cost_table, n_cpu: int, cpu_scale: float) -> float:
    if n_cpu <= 0:
        return 0.0
    return cpu_scale * cost_table[(n_cpu, 0)]


class H2DQueue:
    def __init__(self):
        self.transfers: list[Transfer] = []
        self.cancelled = 0
        self.cancelled_late = 0
        self.cancelled_mb = 0.0
        self.cancelled_keys: list[tuple[int, int]] = []

    def _sort(self):
        self.transfers.sort(key=lambda x: (x.start, x.end))

    def _assert_no_overlap(self):
        self._sort()
        for a, b in zip(self.transfers, self.transfers[1:]):
            if a.end > b.start + 1e-9:
                raise AssertionError(
                    f"H2D overlap: {a.kind}@[{a.start:.6f},{a.end:.6f}] "
                    f"and {b.kind}@[{b.start:.6f},{b.end:.6f}]"
                )

    def tail(self) -> float:
        return max((x.end for x in self.transfers), default=0.0)

    def active_low_count(self, now: float) -> int:
        return sum(1 for x in self.transfers if x.kind == "prefetch" and x.end > now + 1e-12)

    def pop_completed(self, now: float) -> list[Transfer]:
        done = [x for x in self.transfers if x.end <= now + 1e-12]
        if done:
            self.transfers = [x for x in self.transfers if x.end > now + 1e-12]
        return done

    def _repack_future(self, now: float, future: list[Transfer], after: float):
        t = after
        for tr in future:
            t = max(t, now)
            tr.start = t
            tr.end = t + tr.duration
            t = tr.end

    def schedule_low(self, now: float, duration: float, key: tuple[int, int], target_token: int,
                     target_layer: int, deadline: float, direct_main: bool,
                     deadline_filter: bool) -> Transfer | None:
        start = max(now, self.tail())
        end = start + duration
        if deadline_filter and end > deadline + 1e-12:
            return None
        tr = Transfer(start, end, duration, "prefetch", key, target_token, target_layer,
                      deadline, "low", True, direct_main)
        self.transfers.append(tr)
        self._sort()
        self._assert_no_overlap()
        return tr

    def schedule_high(self, now: float, duration: float, kind: str, key: tuple[int, int] | None,
                      cancel_future: bool, reorder_future: bool, expert_mb: float) -> Transfer:
        if not reorder_future:
            # Conservative FIFO mode: once a low-priority H2D has been submitted,
            # later fallback traffic waits behind the queue.  This is the lower
            # bound if CUDA stream priority cannot bypass already-issued copies.
            start = max(now, self.tail())
            tr = Transfer(start, start + duration, duration, kind, key, None, None, None,
                          "high", False, True)
            self.transfers.append(tr)
            self._sort()
            self._assert_no_overlap()
            return tr

        inflight = [x for x in self.transfers if x.start < now < x.end]
        future = [x for x in self.transfers if x.start >= now]
        # Keep completed transfers in the list so EventState.process_ready can materialize
        # their cache/staging effects later; they do not block the new high-priority copy.
        past_or_inflight = [x for x in self.transfers if x.start < now]
        if cancel_future:
            kept_future = []
            for tr in future:
                if tr.cancelable and tr.kind == "prefetch":
                    self.cancelled += 1
                    self.cancelled_mb += expert_mb
                    if tr.key is not None:
                        self.cancelled_keys.append(tr.key)
                    if tr.deadline is not None and tr.end > tr.deadline:
                        self.cancelled_late += 1
                else:
                    kept_future.append(tr)
            future = kept_future
        start = max(now, max((x.end for x in inflight), default=now))
        tr = Transfer(start, start + duration, duration, kind, key, None, None, None, "high", False, True)
        self.transfers = past_or_inflight + [tr]
        self._repack_future(now, future, tr.end)
        self.transfers.extend(future)
        self._sort()
        self._assert_no_overlap()
        return tr

    def cancel_late_prefetches(self, now: float, expert_mb: float):
        kept = []
        for tr in self.transfers:
            if (tr.kind == "prefetch" and tr.cancelable and tr.start >= now and
                    tr.deadline is not None and tr.end > tr.deadline + 1e-12):
                self.cancelled += 1
                self.cancelled_late += 1
                self.cancelled_mb += expert_mb
                if tr.key is not None:
                    self.cancelled_keys.append(tr.key)
            else:
                kept.append(tr)
        self.transfers = kept
        self._assert_no_overlap()

    def drain_cancelled_keys(self) -> list[tuple[int, int]]:
        out = self.cancelled_keys
        self.cancelled_keys = []
        return out


class EventState:
    def __init__(self, n_layers, capacity, staging_slots, pop, policy, use_staging, n_experts):
        self.n_layers = n_layers
        self.capacity = capacity
        self.staging_slots = staging_slots
        self.policy = policy
        self.use_staging = use_staging
        self.cache = warm_cache(pop, capacity)
        self.last_used = {k: 0 for k in self.cache}
        self.main_live: dict[tuple[int, int], list] = {}  # key -> [used, source: prefetch|residual]
        self.staging: dict[tuple[int, int], tuple[float, int, int, bool]] = {}
        self.pending_keys: set[tuple[int, int]] = set()
        self.h2d = H2DQueue()
        self.cpu_free = 0.0
        self.d2h_free = 0.0
        self.pos = 0
        self.n_experts = n_experts
        self.stats = defaultdict(float)

    def _evict_main_if_needed(self, cur_layer: int):
        while len(self.cache) > self.capacity:
            victim = evict_ls(self.cache, self.last_used, cur_layer, self.n_layers)
            live = self.main_live.pop(victim, None)
            if live is not None:
                used, source = live
                if not used:
                    self.stats[f"{source}_main_admit_unused"] += 1

    def add_main(self, key: tuple[int, int], cur_layer: int, source: str | None = None,
                 already_used: bool = False):
        if key not in self.cache and source is not None:
            self.main_live[key] = [already_used, source]
            self.stats[f"{source}_main_admit"] += 1
            if already_used:
                self.stats[f"{source}_main_admit_used"] += 1
        self.cache.add(key)
        self.last_used[key] = self.pos
        self._evict_main_if_needed(cur_layer)

    def add_staging(self, key: tuple[int, int], deadline: float, target_token: int, target_layer: int):
        if key in self.cache:
            return
        if self.staging_slots <= 0:
            self.stats["staging_drop_capacity"] += 1
            return
        if key not in self.staging and len(self.staging) >= self.staging_slots:
            # Keep nearer-deadline entries; evict the farthest deadline first.
            victim = max(self.staging, key=lambda k: self.staging[k][0])
            _deadline, _tt, _tl, used = self.staging.pop(victim)
            if not used:
                self.stats["staging_evicted_unused"] += 1
        self.staging[key] = (deadline, target_token, target_layer, False)

    def expire_staging(self, now: float, ti: int, layer: int):
        kept = {}
        for key, (deadline, target_token, target_layer, used) in self.staging.items():
            # A dense-only wall-clock deadline is only an issue-time filter; the
            # real target layer may be delayed by residual misses.  Expire only
            # when the logical demand point has passed.
            expired = target_token < ti or (target_token == ti and target_layer < layer)
            if expired:
                if used:
                    self.stats["staging_expired_used"] += 1
                else:
                    self.stats["staging_expired_unused"] += 1
            else:
                kept[key] = (deadline, target_token, target_layer, used)
        self.staging = kept

    def process_ready(self, now: float, ti: int, layer: int):
        for tr in self.h2d.pop_completed(now):
            if tr.key is not None:
                self.pending_keys.discard(tr.key)
            if tr.kind == "prefetch" and tr.key is not None:
                self.stats["prefetch_completed"] += 1
                if tr.deadline is not None and tr.end > tr.deadline + 1e-12:
                    self.stats["prefetch_completed_late"] += 1
                if tr.direct_main or not self.use_staging:
                    self.add_main(tr.key, layer, source="prefetch")
                else:
                    target_token = tr.target_token if tr.target_token is not None else ti
                    target_layer = tr.target_layer if tr.target_layer is not None else layer
                    self.add_staging(tr.key, tr.deadline or tr.end, target_token, target_layer)
            elif tr.kind == "residual_fetch" and tr.key is not None:
                self.add_main(tr.key, layer, source="residual")
        self.expire_staging(now, ti, layer)

    def clear_cancelled_pending(self):
        for key in self.h2d.drain_cancelled_keys():
            self.pending_keys.discard(key)

    def is_hit(self, key: tuple[int, int], ti: int, layer: int) -> bool:
        if key in self.cache:
            if key in self.main_live and not self.main_live[key][0]:
                self.main_live[key][0] = True
                source = self.main_live[key][1]
                self.stats[f"{source}_main_admit_used"] += 1
                if source == "prefetch":
                    self.stats["prefetch_useful"] += 1
            self.last_used[key] = self.pos
            return True
        if key in self.staging:
            deadline, target_token, target_layer, _used = self.staging.pop(key)
            self.stats["staging_useful"] += 1
            self.stats["prefetch_useful"] += 1
            self.add_main(key, layer, source="prefetch", already_used=True)
            return True
        return False

    def finalize(self, final_time: float):
        h2d_tail_before_flush = self.h2d.tail()
        self.process_ready(float("inf"), 10**12, self.n_layers - 1)
        for _key, (_deadline, _tt, _tl, used) in self.staging.items():
            if used:
                self.stats["staging_expired_used"] += 1
            else:
                self.stats["staging_expired_unused"] += 1
        self.staging.clear()
        for used, source in self.main_live.values():
            if used is False:
                self.stats[f"{source}_main_admit_unused"] += 1
        self.stats["h2d_tail_ms"] = max(final_time, h2d_tail_before_flush)


def policy_config(policy: str):
    if policy == "spice_fetch_all":
        return {"deadline_filter": False, "cancel": False, "use_staging": False, "oracle": False,
                "reorder_high": True, "low_queue_depth": None}
    if policy == "fiddler_cpu":
        return {"deadline_filter": False, "cancel": False, "use_staging": False, "oracle": False,
                "reorder_high": True, "low_queue_depth": None}
    if policy == "deadline_scheduler":
        return {"deadline_filter": True, "cancel": True, "use_staging": True, "oracle": False,
                "reorder_high": True, "low_queue_depth": None}
    if policy == "shallow_scheduler":
        return {"deadline_filter": True, "cancel": True, "use_staging": True, "oracle": False,
                "reorder_high": True, "low_queue_depth": "arg"}
    if policy == "fifo_deadline":
        return {"deadline_filter": True, "cancel": False, "use_staging": True, "oracle": False,
                "reorder_high": False, "low_queue_depth": None}
    if policy == "no_cancel":
        return {"deadline_filter": True, "cancel": False, "use_staging": True, "oracle": False,
                "reorder_high": True, "low_queue_depth": None}
    if policy == "no_staging":
        return {"deadline_filter": True, "cancel": True, "use_staging": False, "oracle": False,
                "reorder_high": True, "low_queue_depth": None}
    if policy == "oracle_deadline":
        return {"deadline_filter": True, "cancel": True, "use_staging": True, "oracle": True,
                "reorder_high": True, "low_queue_depth": None}
    raise ValueError(policy)


def choose_fetch_count(policy: str, nmiss: int, base_done: float, now: float, state: EventState,
                       cost_table, t_fetch: float, t_gpu: float, small_h2d: float, cpu_scale: float,
                       reorder_high: bool) -> int:
    if nmiss <= 0:
        return 0
    if policy == "spice_fetch_all":
        return nmiss
    if policy == "fiddler_cpu":
        return 0
    best = None
    if reorder_high:
        inflight_block = max((tr.end for tr in state.h2d.transfers if tr.start < now < tr.end), default=now)
        h2d_ready = max(now, inflight_block)
    else:
        h2d_ready = max(now, state.h2d.tail())
    cpu_ready = max(now, state.cpu_free)
    for nf in range(nmiss + 1):
        nc = nmiss - nf
        fetch_done = base_done
        if nf:
            fetch_done = max(base_done, h2d_ready + nf * t_fetch) + nf * t_gpu
        cpu_done = base_done
        if nc:
            cpu_done = max(cpu_ready, now) + cpu_cost_ms(cost_table, nc, cpu_scale) + small_h2d
        item = (max(base_done, fetch_done, cpu_done), nf)
        if best is None or item < best:
            best = item
    assert best is not None
    return best[1]


def issue_prefetches(state: EventState, item, ti: int, layer: int, now: float, dense_ms: float,
                     max_horizon: int, n_layers: int, args, cfg, t_fetch: float, expert_mb: float):
    true_top = item["true_top"]
    fcast = item["fcast"]
    max_lead = min(args.max_lead_layers, max_horizon - 1, n_layers - layer - 1)
    if max_lead < args.min_lead_layers:
        return
    for lead in range(args.min_lead_layers, max_lead + 1):
        target_layer = layer + lead
        deadline = now + lead * dense_ms * args.dense_deadline_scale
        if cfg["oracle"]:
            pred = [int(x) for x in true_top[target_layer, ti].tolist()]
        else:
            pred = [int(x) for x in fcast[layer, lead, ti].tolist() if int(x) >= 0]
        seen = set()
        true_set = set(int(x) for x in true_top[target_layer, ti].tolist())
        for e in pred:
            if e in seen:
                continue
            seen.add(e)
            key = (target_layer, e)
            if key in state.cache or key in state.staging or key in state.pending_keys:
                continue
            low_depth = args.shallow_low_depth if cfg["low_queue_depth"] == "arg" else cfg["low_queue_depth"]
            if low_depth is not None and state.h2d.active_low_count(now) >= low_depth:
                state.stats["prefetch_skipped_depth"] += 1
                continue
            tr = state.h2d.schedule_low(now, t_fetch, key, ti, target_layer, deadline,
                                        direct_main=not cfg["use_staging"],
                                        deadline_filter=cfg["deadline_filter"])
            if tr is None:
                state.stats["prefetch_skipped_deadline"] += 1
                continue
            state.pending_keys.add(key)
            state.stats["prefetch_issued"] += 1
            state.stats["prefetch_wrong"] += int(e not in true_set)
            state.stats["prefetch_h2d_mb"] += expert_mb


def schedule_cpu_service(state: EventState, now: float, n_cpu: int, cost_table, cpu_scale: float,
                         small_d2h: float, small_h2d: float, cancel: bool,
                         reorder_high: bool, expert_mb: float) -> float:
    if n_cpu <= 0:
        return now
    d2h_start = max(state.d2h_free, now)
    d2h_done = d2h_start + small_d2h
    state.d2h_free = d2h_done
    cpu_ms = cpu_cost_ms(cost_table, n_cpu, cpu_scale)
    cpu_compute_ms = max(0.0, cpu_ms - small_d2h - small_h2d)
    cpu_start = max(state.cpu_free, d2h_done)
    cpu_done = cpu_start + cpu_compute_ms
    state.cpu_free = cpu_done
    out = state.h2d.schedule_high(cpu_done, small_h2d, "cpu_result_h2d", None,
                                  cancel, reorder_high, expert_mb)
    state.clear_cancelled_pending()
    state.stats["cpu_layers"] += 1
    state.stats["cpu_act_roundtrip_h2d_ms"] += small_h2d
    state.stats["cpu_act_roundtrip_d2h_ms"] += small_d2h
    return out.end


def schedule_residual_fetches(state: EventState, now: float, layer: int, fetched: list[int], t_fetch: float,
                              t_gpu: float, cancel: bool, reorder_high: bool, expert_mb: float) -> float:
    ready = now
    for e in fetched:
        key = (layer, e)
        # Residual expert fetches for a layer are serialized on the same H2D
        # engine.  Use the previous fetch ready time as the next insertion time;
        # otherwise repeated high-priority insertions at the same timestamp can
        # incorrectly appear to complete in parallel.
        tr = state.h2d.schedule_high(ready, t_fetch, "residual_fetch", key,
                                     cancel, reorder_high, expert_mb)
        state.clear_cancelled_pending()
        ready = max(ready, tr.end)
        state.stats["fallback_fetches"] += 1
        state.stats["fallback_h2d_mb"] += expert_mb
    return ready + len(fetched) * t_gpu


def select_fetched(layer_misses: list[int], n_fetch: int, layer: int, ti: int, fut, oracle: bool):
    if n_fetch <= 0:
        return []
    if oracle:
        return sorted(layer_misses, key=lambda e: oracle_next_use(fut, (layer, e), ti))[:n_fetch]
    # Realizable fallback: `layer_misses` preserves router top-k order.
    return layer_misses[:n_fetch]


def simulate_item(item, capacity: int, policy: str, cost_table, pop, expert_mb: float, args,
                  small_d2h: float, small_h2d: float):
    cfg = policy_config(policy)
    true_top = item["true_top"]
    n_layers, tokens, _top_k = true_top.shape
    max_horizon = item["fcast"].shape[1]
    dense_ms = args.t_attn + args.t_gate + args.t_shared
    fut = build_future_positions(true_top)
    state = EventState(n_layers, capacity, args.staging_slots, pop, policy, cfg["use_staging"],
                       n_experts=max(int(true_top.max().item()) + 1, capacity))
    clock = 0.0
    routed = hits = misses = 0
    tokens_done = 0

    for ti in range(tokens):
        if tokens_done >= args.max_test_tokens:
            break
        for layer in range(n_layers):
            state.process_ready(clock, ti, layer)
            layer_hits = 0
            layer_misses = []
            for e in [int(x) for x in true_top[layer, ti].tolist()]:
                key = (layer, e)
                routed += 1
                if state.is_hit(key, ti, layer):
                    hits += 1
                    layer_hits += 1
                else:
                    misses += 1
                    layer_misses.append(e)
                state.pos += 1

            base_done = clock + dense_ms + layer_hits * args.t_gpu
            n_fetch = choose_fetch_count(policy, len(layer_misses), base_done, clock, state, cost_table,
                                         args.t_fetch_h2d, args.t_gpu, small_h2d, args.cpu_scale,
                                         cfg["reorder_high"])
            fetched = select_fetched(layer_misses, n_fetch, layer, ti, fut, oracle=cfg["oracle"])
            n_cpu = len(layer_misses) - len(fetched)
            state.stats["cpu_served"] += n_cpu
            fetch_done = schedule_residual_fetches(state, clock, layer, fetched, args.t_fetch_h2d,
                                                   args.t_gpu, cfg["cancel"], cfg["reorder_high"], expert_mb)
            cpu_done = schedule_cpu_service(state, clock, n_cpu, cost_table, args.cpu_scale,
                                            small_d2h, small_h2d, cfg["cancel"], cfg["reorder_high"], expert_mb)
            layer_end = max(base_done, fetch_done, cpu_done)

            issue_prefetches(state, item, ti, layer, clock, dense_ms, max_horizon, n_layers,
                             args, cfg, args.t_fetch_h2d, expert_mb)
            if cfg["cancel"]:
                state.h2d.cancel_late_prefetches(clock, expert_mb)
                state.clear_cancelled_pending()
            # Materialize copies that completed during this layer only after all
            # H2D work for the layer has been scheduled.  Popping residual fetches
            # earlier would shorten the queue tail seen by draft prefetches and
            # create impossible overlap on the single H2D engine.
            state.process_ready(layer_end, ti, layer)
            clock = layer_end
        tokens_done += 1

    state.finalize(clock)
    stats = state.stats
    tokens_done = max(1, tokens_done)
    routed = max(1, routed)
    prefetch_main_admit = max(1.0, stats["prefetch_main_admit"])
    residual_main_admit = max(1.0, stats["residual_main_admit"])
    issued = max(1.0, stats["prefetch_issued"])
    completed = max(1.0, stats["prefetch_completed"])
    return {
        "total_ms": clock,
        "tail_total_ms": stats["h2d_tail_ms"],
        "tokens": tokens_done,
        "hits": hits,
        "misses": misses,
        "routed": routed,
        "hit_rate": hits / routed,
        "residual_rate": misses / routed,
        "fallback_fetches": stats["fallback_fetches"],
        "cpu_served": stats["cpu_served"],
        "cpu_layers": stats["cpu_layers"],
        "prefetch_issued": stats["prefetch_issued"],
        "prefetch_completed": stats["prefetch_completed"],
        "prefetch_completed_late": stats["prefetch_completed_late"],
        "prefetch_wrong": stats["prefetch_wrong"],
        "prefetch_skipped_deadline": stats["prefetch_skipped_deadline"],
        "prefetch_skipped_depth": stats["prefetch_skipped_depth"],
        "prefetch_late_dropped": stats["prefetch_late_dropped"],
        "prefetch_useful": stats["prefetch_useful"],
        "staging_useful": stats["staging_useful"],
        "staging_evicted_unused": stats["staging_evicted_unused"],
        "staging_expired_unused": stats["staging_expired_unused"],
        "prefetch_main_admit": stats["prefetch_main_admit"],
        "prefetch_main_admit_used": stats["prefetch_main_admit_used"],
        "prefetch_main_admit_unused": stats["prefetch_main_admit_unused"],
        "residual_main_admit": stats["residual_main_admit"],
        "residual_main_admit_used": stats["residual_main_admit_used"],
        "residual_main_admit_unused": stats["residual_main_admit_unused"],
        "cancelled_prefetches": state.h2d.cancelled,
        "cancelled_late_prefetches": state.h2d.cancelled_late,
        "prefetch_h2d_mb": stats["prefetch_h2d_mb"],
        "actual_prefetch_h2d_mb": max(0.0, (stats["prefetch_issued"] - state.h2d.cancelled) * expert_mb),
        "fallback_h2d_mb": stats["fallback_h2d_mb"],
        "cancelled_h2d_mb": state.h2d.cancelled_mb,
        "cpu_result_h2d_ms": stats["cpu_act_roundtrip_h2d_ms"],
        "cpu_input_d2h_ms": stats["cpu_act_roundtrip_d2h_ms"],
        "tpot_ms": clock / tokens_done,
        "tail_tpot_ms": stats["h2d_tail_ms"] / tokens_done,
        "h2d_backlog_ms_per_tok": max(0.0, stats["h2d_tail_ms"] - clock) / tokens_done,
        "prefetch_wrong_frac": stats["prefetch_wrong"] / issued,
        "prefetch_useful_frac": stats["prefetch_useful"] / issued,
        "prefetch_useful_completed_frac": stats["prefetch_useful"] / completed,
        "prefetch_main_admit_used_frac": stats["prefetch_main_admit_used"] / prefetch_main_admit,
        "residual_main_admit_used_frac": stats["residual_main_admit_used"] / residual_main_admit,
    }


def main() -> None:
    args = parse_args()
    cost_table, _best_table, cost_meta, expert_mb, _act_roundtrip_mb = load_costs(args.cost_json, args.cost_metric)
    if args.t_fetch_h2d <= 0:
        bw = float(cost_meta.get("config", {}).get("bw_gbps", 0.0))
        args.t_fetch_h2d = (expert_mb / (bw * 1024.0 / 1000.0)) if bw else 0.792
    args.t_fetch_h2d *= args.fetch_scale
    small_d2h, small_h2d = pcie_small_costs(args.pcie_json)

    seqs, n_layers, n_experts, top_k, max_horizon, manifest = load_forecast_sequences(args.forecast_dir)
    split = max(1, int(round(len(seqs) * args.train_frac)))
    train_items = seqs[:split]
    test_items = seqs[split:] or seqs
    train_for_pop = [seq_for_popularity(x) for x in train_items]
    pop = popularity(train_for_pop, n_layers, n_experts)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]

    rows = []
    print(f"[data] files={len(seqs)} train={len(train_items)} test={len(test_items)} "
          f"layers={n_layers} experts={n_experts} top_k={top_k} horizon={max_horizon}", flush=True)
    print(f"[resource] expert={expert_mb:.2f}MB t_fetch={args.t_fetch_h2d:.3f}ms "
          f"small_d2h={small_d2h:.4f}ms small_h2d={small_h2d:.4f}ms metric={args.cost_metric} "
          f"cpu_scale={args.cpu_scale:.2f}", flush=True)

    for r in [float(x) for x in args.residency.split(",")]:
        cap = max(1, int(round(r * n_layers * n_experts)))
        for policy in policies:
            agg = defaultdict(float)
            remaining = args.max_test_tokens
            for item in test_items:
                if remaining <= 0:
                    break
                old = args.max_test_tokens
                args.max_test_tokens = remaining
                res = simulate_item(item, cap, policy, cost_table, pop, expert_mb, args, small_d2h, small_h2d)
                args.max_test_tokens = old
                remaining -= int(res["tokens"])
                for k, v in res.items():
                    agg[k] += v
            tokens = max(1.0, agg["tokens"])
            routed = max(1.0, agg["routed"])
            issued = max(1.0, agg["prefetch_issued"])
            completed = max(1.0, agg["prefetch_completed"])
            prefetch_main_admit = max(1.0, agg["prefetch_main_admit"])
            residual_main_admit = max(1.0, agg["residual_main_admit"])
            row = {
                "policy": policy,
                "residency": r,
                "capacity": cap,
                "tokens": agg["tokens"],
                "tpot_ms": agg["total_ms"] / tokens,
                "tail_tpot_ms": agg["tail_total_ms"] / tokens,
                "h2d_backlog_ms_per_tok": max(0.0, agg["tail_total_ms"] - agg["total_ms"]) / tokens,
                "hit_rate": agg["hits"] / routed,
                "residual_rate": agg["misses"] / routed,
                "residual_misses_per_tok": agg["misses"] / tokens,
                "fallback_fetches_per_tok": agg["fallback_fetches"] / tokens,
                "cpu_served_per_tok": agg["cpu_served"] / tokens,
                "cpu_layers_per_tok": agg["cpu_layers"] / tokens,
                "prefetch_issued_per_tok": agg["prefetch_issued"] / tokens,
                "prefetch_completed_per_tok": agg["prefetch_completed"] / tokens,
                "prefetch_completed_late_per_tok": agg["prefetch_completed_late"] / tokens,
                "prefetch_skipped_deadline_per_tok": agg["prefetch_skipped_deadline"] / tokens,
                "prefetch_skipped_depth_per_tok": agg["prefetch_skipped_depth"] / tokens,
                "cancelled_prefetches_per_tok": agg["cancelled_prefetches"] / tokens,
                "prefetch_wrong_frac": agg["prefetch_wrong"] / issued,
                "prefetch_useful_frac": agg["prefetch_useful"] / issued,
                "prefetch_useful_completed_frac": agg["prefetch_useful"] / completed,
                "staging_useful_per_tok": agg["staging_useful"] / tokens,
                "staging_evicted_unused_per_tok": agg["staging_evicted_unused"] / tokens,
                "staging_expired_unused_per_tok": agg["staging_expired_unused"] / tokens,
                "prefetch_main_admit_used_frac": agg["prefetch_main_admit_used"] / prefetch_main_admit,
                "residual_main_admit_used_frac": agg["residual_main_admit_used"] / residual_main_admit,
                "prefetch_h2d_mb_per_tok": agg["prefetch_h2d_mb"] / tokens,
                "actual_prefetch_h2d_mb_per_tok": agg["actual_prefetch_h2d_mb"] / tokens,
                "fallback_h2d_mb_per_tok": agg["fallback_h2d_mb"] / tokens,
                "actual_total_h2d_mb_per_tok": (agg["actual_prefetch_h2d_mb"] + agg["fallback_h2d_mb"]) / tokens,
                "cancelled_h2d_mb_per_tok": agg["cancelled_h2d_mb"] / tokens,
                "cpu_result_h2d_ms_per_tok": agg["cpu_result_h2d_ms"] / tokens,
                "cpu_input_d2h_ms_per_tok": agg["cpu_input_d2h_ms"] / tokens,
            }
            rows.append(row)
            print(f"res={r:.3f} {policy:>18} TPOT={row['tpot_ms']:7.2f} tail={row['tail_tpot_ms']:7.2f} "
                  f"hit={row['hit_rate']:.3f} "
                  f"miss/tok={row['residual_misses_per_tok']:6.2f} fb_fetch/tok={row['fallback_fetches_per_tok']:5.2f} "
                  f"pf/tok={row['prefetch_issued_per_tok']:6.2f} useful={row['prefetch_useful_frac']:.3f} "
                  f"cancel/tok={row['cancelled_prefetches_per_tok']:5.2f}", flush=True)

    by = defaultdict(dict)
    for row in rows:
        by[row["residency"]][row["policy"]] = row
    verdict = {}
    for r, d in by.items():
        sched = d.get("deadline_scheduler")
        fid = d.get("fiddler_cpu")
        spice = d.get("spice_fetch_all")
        verdict[str(r)] = {
            "deadline_scheduler": sched["tpot_ms"] if sched else None,
            "deadline_scheduler_tail": sched["tail_tpot_ms"] if sched else None,
            "fiddler_cpu": fid["tpot_ms"] if fid else None,
            "fiddler_cpu_tail": fid["tail_tpot_ms"] if fid else None,
            "spice_fetch_all": spice["tpot_ms"] if spice else None,
            "spice_fetch_all_tail": spice["tail_tpot_ms"] if spice else None,
            "gain_vs_fiddler_pct": 100.0 * (fid["tpot_ms"] - sched["tpot_ms"]) / fid["tpot_ms"]
            if fid and sched else None,
            "tail_gain_vs_fiddler_pct": 100.0 * (fid["tail_tpot_ms"] - sched["tail_tpot_ms"]) / fid["tail_tpot_ms"]
            if fid and sched else None,
            "gain_vs_spice_fetch_all_pct": 100.0 * (spice["tpot_ms"] - sched["tpot_ms"]) / spice["tpot_ms"]
            if spice and sched else None,
            "tail_gain_vs_spice_fetch_all_pct": 100.0 * (spice["tail_tpot_ms"] - sched["tail_tpot_ms"]) / spice["tail_tpot_ms"]
            if spice and sched else None,
        }

    out = {
        "config": vars(args),
        "forecast_manifest": manifest,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "max_horizon": max_horizon,
        "rows": rows,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
