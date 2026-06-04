"""Real CUDA timing harness for SPICE shallow H2D issuer + residual miss scheduler.

This is the runtime bridge between:
  1. `spice_event_scheduler_replay.py` (resource-DAG replay), and
  2. `shallow_h2d_issuer_microbench.py` (A800 copy-engine queue probe).

It consumes a SPICE forecast dump (`true_top[L,T,K]`, `fcast[L,H,T,K]`) and
executes a trace-driven timing loop with real CUDA H2D copies and real CPU expert
compute.  It is still a diagnostic timing harness, not a source-only baseline and
not a full exact-logit model replay: dense/attention is represented by a real
optional filler GEMM, and HBM/cache admission uses trace keys rather than the
model's full logits. Warm resident experts, completed prefetches, and residual
fetches nevertheless occupy real device slots and run real GPU expert GEMMs from
those slots; CPU fallback runs real same-precision CPU expert GEMMs. No drop,
quantization, or compression.

Policies:
  deep_fetch_all      : dispatch all forecast prefetch H2D immediately; residual
                        misses fetch all over H2D (SPICE-style weak fallback).
  deep_cpu            : dispatch all forecast prefetch H2D immediately; residual
                        misses all CPU-served (Fiddler-like diagnostic wrapper).
  shallow_cpu         : keep forecast prefetch in a shallow software issuer;
                        residual misses all CPU-served.
  shallow_scheduler   : shallow issuer + per-layer resource split between
                        residual H2D fetch and CPU service.
  deep_dummy_cpu /
  shallow_dummy_cpu   : same forecast traffic as deep/shallow, but prefetched
                        experts are not consumed as hits; this keeps miss counts
                        identical and isolates copy-queue interference.
  gos_cpu             : SPICE global overflow scheduler. Forecasted experts are
                        admitted to H2D only when they reduce a future CPU miss
                        burst and can finish before that future-layer deadline.
                        Staged hits can then be promoted to the resident cache
                        using always/never/diagnostic selective-admission rules.
  gos_dummy_cpu       : GOS-admitted H2D perturbation control. Prefetched experts
                        are not consumed as hits, so this is a state-divergent
                        control, not an identical-traffic replay.

The key question is whether a real software issuer that limits submitted low
H2D depth can preserve SPICE prefetch utility without letting draft traffic
block exact residual miss service.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from miss_assignment_replay import evict_ls, load_costs, popularity, warm_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPICE shallow H2D issuer runtime")
    p.add_argument("--forecast_dir", required=True)
    p.add_argument("--cost_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--train_frac", type=float, required=True)
    p.add_argument("--residency", type=float, required=True)
    p.add_argument("--max_test_tokens", type=int, required=True)
    p.add_argument("--policies", default="deep_fetch_all,deep_cpu,shallow_cpu,shallow_scheduler")
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--d_inter", type=int, required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--cpu_threads", type=int, required=True)
    p.add_argument("--cpu_dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--cost_metric", choices=["ms", "mean_ms", "p90_ms"], default="ms")
    p.add_argument("--shallow_depth", type=int, default=2)
    p.add_argument("--low_slots", type=int, default=128)
    p.add_argument("--high_slots", type=int, default=8)
    p.add_argument("--bank", type=int, default=256,
                   help="distinct host/CPU expert bank size; keys map modulo this bank for timing")
    p.add_argument("--max_lead_layers", type=int, default=5)
    p.add_argument("--min_prefetch_lead", type=int, default=1,
                   help="do not submit predictions closer than this lead; near-deadline misses fall back to CPU")
    p.add_argument("--prefetch_per_layer", type=int, default=4)
    p.add_argument("--substitute_ranks", type=str, default="",
                   help="verified-gate NEGATIVE admission: comma top-k RANKS (0=highest gate weight) whose "
                        "MISSED experts are shared-expert-substituted -- skipped entirely (no CPU, no fetch). "
                        "Lossy (PPL cost from drop_quality curve); '' = exact (serve every expert).")
    p.add_argument("--timed_repeats", type=int, default=3)
    p.add_argument("--filler_compute_dim", type=int, default=0,
                   help="optional real GPU GEMM size per layer to create a compute window")
    p.add_argument("--filler_repeats", type=int, default=1,
                   help="repeat the filler GEMM this many times per layer; ignored when filler_compute_dim=0")
    p.add_argument("--fetch_margin_ms", type=float, default=0.25,
                   help="scheduler only residual-fetches when measured split cost beats all-CPU by this margin")
    p.add_argument("--gos_layer_slack_ms", type=float, default=0.58,
                   help="GOS deadline model: per-future-layer GPU slack available to hide low H2D")
    p.add_argument("--gos_cpu_overlap_ms", type=float, default=0.58,
                   help="GOS value model: target-layer non-miss GPU window that can hide CPU fallback")
    p.add_argument("--gos_value_margin_ms", type=float, default=0.10,
                   help="GOS admission margin: CPU critical-path saving must exceed this")
    p.add_argument("--gos_max_prefetch_per_target", type=int, default=4,
                   help="GOS cap on admitted experts per predicted future target layer")
    p.add_argument("--gos_scheduler", choices=["greedy", "dp"], default="greedy",
                   help="GOS admission optimizer: greedy preserves old per-target behavior; "
                        "dp globally optimizes admitted forecast copies under backlog/deadline/slot constraints")
    p.add_argument("--t_gpu_ms", type=float, default=0.079,
                   help="calibrated resident GPU expert compute time used by GOS value model")
    p.add_argument("--no_admit_prefetch_hits", action="store_true",
                   help="serve low-prefetch hits from staging slots and release them without main-stream D2D cache admission")
    p.add_argument("--prefetch_hit_admission",
                   choices=["always", "never", "hotter_than_victim", "recent_reuse", "oracle_value"],
                   default="always",
                   help="resident-cache promotion policy after a staged prefetch hit is consumed. "
                        "'always' preserves old behavior; 'never' is equivalent to --no_admit_prefetch_hits; "
                        "'hotter_than_victim' promotes only when train popularity beats the LS victim; "
                        "'recent_reuse' uses online same-layer recent route reuse against the LS victim; "
                        "'oracle_value' is a true-future upper-bound kill-test, not deployable.")
    p.add_argument("--residual_fetch_admission",
                   choices=["always", "never", "same_as_prefetch", "oracle_value"],
                   default="always",
                   help="resident-cache admission after a residual demand fetch. Default preserves old behavior; "
                        "same_as_prefetch/oracle_value are diagnostics for global HBM admission.")
    p.add_argument("--allow_oracle_admission", action="store_true",
                   help="required guard for oracle_value admission policies; marks the run as a true-future upper bound")
    p.add_argument("--admit_recent_window", type=int, default=8,
                   help="token window for --prefetch_hit_admission=recent_reuse")
    p.add_argument("--admit_value_horizon_tokens", type=int, default=16,
                   help="true-future token horizon for --prefetch_hit_admission=oracle_value")
    p.add_argument("--resident_value_margin_ms", type=float, default=0.25,
                   help="oracle resident admission requires future saved critical-path value above this margin")
    p.add_argument("--resident_admit_cost_ms", type=float, default=0.0,
                   help="one-time cost charged to each resident admission (main-stream D2D promotion copy + "
                        "low-stream fence/opportunity cost). Subtracted from oracle_value before the margin test; "
                        "sweep this to reflect measured promotion overhead.")
    p.add_argument("--seed", type=int, default=0)
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
        true_top = d["true_top"].long()
        fcast = d["fcast"].long()
        if n_layers is None:
            n_layers = int(d.get("num_layers", true_top.shape[0]))
            top_k = int(d.get("top_k", true_top.shape[-1]))
            max_horizon = int(d.get("max_horizon", fcast.shape[1]))
        n_experts = max(n_experts, int(true_top.max().item()) + 1)
        valid = fcast[fcast >= 0]
        if valid.numel():
            n_experts = max(n_experts, int(valid.max().item()) + 1)
        seqs.append({"name": name, "true_top": true_top, "fcast": fcast})
    if not seqs:
        raise ValueError(f"no forecast files in {forecast_dir}")
    return seqs, int(n_layers), int(n_experts), int(top_k), int(max_horizon), man


def seq_for_popularity(item):
    true_top = item["true_top"]
    layers, tokens, _top_k = true_top.shape
    return [
        (t, [[int(x) for x in true_top[l, t].tolist()] for l in range(layers)])
        for t in range(tokens)
    ]


def make_host_bank(bank: int, dm: int, di: int, dtype: torch.dtype):
    scale_g = dm ** -0.5
    scale_d = di ** -0.5
    return [
        (
            (torch.randn(di, dm) * scale_g).to(dtype=dtype).pin_memory(),
            (torch.randn(di, dm) * scale_g).to(dtype=dtype).pin_memory(),
            (torch.randn(dm, di) * scale_d).to(dtype=dtype).pin_memory(),
        )
        for _ in range(bank)
    ]


def make_cpu_bank(host_bank, dtype: torch.dtype):
    return [(g.to(dtype=dtype).contiguous(), u.to(dtype=dtype).contiguous(), d.to(dtype=dtype).contiguous())
            for g, u, d in host_bank]


def make_dev_slots(n: int, dm: int, di: int, dev: torch.device, dtype: torch.dtype):
    return [
        (
            torch.empty(di, dm, device=dev, dtype=dtype),
            torch.empty(di, dm, device=dev, dtype=dtype),
            torch.empty(dm, di, device=dev, dtype=dtype),
        )
        for _ in range(n)
    ]


def expert_gpu(x: torch.Tensor, w):
    g, u, d = w
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


def expert_cpu(x: torch.Tensor, w):
    g, u, d = w
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


class ShallowIssuer:
    def __init__(self, dev, host_bank, low_slots, high_slots, bank_size, max_depth: int | None):
        self.dev = dev
        self.host_bank = host_bank
        self.low_slots = low_slots
        self.high_slots = high_slots
        self.bank_size = bank_size
        self.max_depth = max_depth
        self.low_stream = torch.cuda.Stream(device=dev, priority=0)
        # PyTorch accepts negative priority values and clamps unsupported ranges.
        # Shallow software depth is still the primary protection; priority is
        # only a best-effort hint for residual demand copies.
        self.high_stream = torch.cuda.Stream(device=dev, priority=-1)
        self.free_low = deque(range(len(low_slots)))
        self.active_low = []  # dict(event,key,slot,target_token,target_layer,expired)
        self.staged = {}      # key -> dict(slot,target_token,target_layer)
        self.intents = deque()
        self.pending = set()
        self.high_rr = 0
        self.stats = defaultdict(float)

    def hkey(self, key):
        layer, expert = key
        return (layer * 4099 + expert) % self.bank_size

    def _copy_to_slot(self, key, slot, stream):
        src = self.host_bank[self.hkey(key)]
        with torch.cuda.stream(stream):
            for s, d in zip(src, slot):
                d.copy_(s, non_blocking=True)

    @staticmethod
    def expired(item, ti, layer):
        tt = item["target_token"]
        tl = item["target_layer"]
        return tt < ti or (tt == ti and tl < layer)

    def release_low_slot(self, slot_id):
        # The slot may have just been read on the main stream for expert compute
        # or D2D admission; make future low-stream H2D reuse wait on that work.
        self.low_stream.wait_stream(torch.cuda.current_stream(self.dev))
        self.free_low.append(slot_id)

    def expire(self, ti, layer):
        kept = deque()
        for item in self.intents:
            if self.expired(item, ti, layer):
                self.pending.discard(item["key"])
                self.stats["prefetch_intent_expired"] += 1
            else:
                kept.append(item)
        self.intents = kept

        for key, item in list(self.staged.items()):
            if self.expired(item, ti, layer):
                self.staged.pop(key, None)
                self.pending.discard(key)
                self.release_low_slot(item["slot"])
                self.stats["prefetch_staged_expired"] += 1

        for item in self.active_low:
            if self.expired(item, ti, layer):
                item["expired"] = True

    def poll(self, ti=None, layer=None):
        if ti is not None and layer is not None:
            self.expire(ti, layer)
        kept = []
        for item in self.active_low:
            if item["event"].query():
                self.pending.discard(item["key"])
                if item.get("expired", False):
                    self.release_low_slot(item["slot"])
                    self.stats["prefetch_completed_expired"] += 1
                else:
                    self.staged[item["key"]] = {
                        "slot": item["slot"],
                        "target_token": item["target_token"],
                        "target_layer": item["target_layer"],
                    }
                    self.stats["prefetch_completed"] += 1
            else:
                kept.append(item)
        self.active_low = kept

    def active_low_count(self):
        self.poll()
        return len(self.active_low)

    def low_backlog_count(self):
        self.poll()
        return len(self.active_low) + len(self.intents)

    def low_reserved_count(self):
        self.poll()
        return len(self.active_low) + len(self.staged) + len(self.intents)

    def staged_keys(self):
        return set(self.staged)

    def add_intent(self, key, target_token, target_layer, resident_or_staged):
        if key in resident_or_staged or key in self.pending:
            return
        self.intents.append({"key": key, "target_token": target_token, "target_layer": target_layer})
        self.pending.add(key)
        self.stats["prefetch_intents"] += 1

    def pump(self):
        self.poll()
        while self.intents and self.free_low:
            if self.max_depth is not None and len(self.active_low) >= self.max_depth:
                break
            item = self.intents.popleft()
            key = item["key"]
            slot_id = self.free_low.popleft()
            self._copy_to_slot(key, self.low_slots[slot_id], self.low_stream)
            ev = torch.cuda.Event()
            with torch.cuda.stream(self.low_stream):
                ev.record()
            self.active_low.append({
                "event": ev,
                "key": key,
                "slot": slot_id,
                "target_token": item["target_token"],
                "target_layer": item["target_layer"],
                "expired": False,
            })
            self.stats["prefetch_submitted"] += 1

    def pop_staged(self, key):
        self.poll()
        item = self.staged.pop(key, None)
        if item is not None:
            self.pending.discard(key)
            self.stats["prefetch_useful"] += 1
            return item["slot"], self.low_slots[item["slot"]]
        return None

    def wait_active(self, key):
        for i, item in enumerate(self.active_low):
            if item["key"] == key:
                if item.get("expired", False):
                    self.stats["prefetch_wait_active_expired"] += 1
                    return None
                t0 = time.perf_counter()
                item["event"].synchronize()
                self.stats["prefetch_waited_active_ms"] += (time.perf_counter() - t0) * 1000.0
                self.active_low.pop(i)
                self.pending.discard(key)
                self.stats["prefetch_useful"] += 1
                self.stats["prefetch_waited_active"] += 1
                return item["slot"], self.low_slots[item["slot"]]
        return None

    def cancel_intent(self, key):
        kept = deque()
        removed = 0
        for item in self.intents:
            if item["key"] == key:
                removed += 1
            else:
                kept.append(item)
        if removed:
            self.intents = kept
            self.pending.discard(key)
            self.stats["prefetch_intent_cancelled"] += removed

    def fetch_residual_async(self, keys):
        if len(keys) > len(self.high_slots):
            raise ValueError(f"residual fetch fanout {len(keys)} exceeds high_slots={len(self.high_slots)}")
        events = []
        for key in keys:
            slot = self.high_slots[self.high_rr % len(self.high_slots)]
            self.high_rr += 1
            self.high_stream.wait_stream(torch.cuda.current_stream(self.dev))
            self._copy_to_slot(key, slot, self.high_stream)
            ev = torch.cuda.Event()
            with torch.cuda.stream(self.high_stream):
                ev.record()
            events.append({"key": key, "event": ev, "slot": slot})
            self.stats["residual_fetch_submitted"] += 1
        return events

    def wait_high(self, fetches):
        if fetches:
            t0 = time.perf_counter()
            fetches[-1]["event"].synchronize()
            self.stats["residual_fetch_wait_ms"] += (time.perf_counter() - t0) * 1000.0

    def flush(self):
        torch.cuda.synchronize(self.dev)
        self.poll()


def choose_fetch_count(nmiss: int, active_low: int, cost_table, t_fetch: float, fetch_margin_ms: float) -> int:
    if nmiss <= 0:
        return 0
    all_cpu = cost_table.get((nmiss, 0))
    if all_cpu is None:
        return 0
    best_cost = all_cpu
    best_fetch = 0
    h2d_wait = active_low * t_fetch
    for nf in range(1, nmiss + 1):
        measured = cost_table.get((nmiss, nf))
        if measured is None:
            continue
        # The split microbench does not include already-submitted low-prefetch
        # head-of-line wait, nor the runtime's admission/event overhead. Require
        # a real margin before choosing residual H2D; thin wins disappeared in
        # the CUDA harness.
        cost = measured + h2d_wait
        if all_cpu - cost >= fetch_margin_ms and cost < best_cost:
            best_cost = cost
            best_fetch = nf
    return best_fetch


def measured_cpu_cost_ms(nmiss: int, cost_table):
    if nmiss <= 0:
        return 0.0
    if (nmiss, 0) in cost_table:
        return cost_table[(nmiss, 0)]
    return None


def gos_target_options(ranked, resident_or_staged, reserved: int, backlog: int, low_slots: int, lead: int,
                       cost_table, t_fetch: float, t_gpu: float, args, stats=None):
    """Return feasible prefix choices for one future target.

    Option `f=0` is always present.  Positive options are local feasibility
    candidates; the DP still enforces cumulative deadline/slot feasibility.
    """
    unique = []
    seen = set()
    for key in ranked:
        if key in seen or key in resident_or_staged:
            continue
        seen.add(key)
        unique.append(key)
    info = {
        "unique": unique,
        "base_unmeasured": False,
        "slot_feasible": 0,
        "deadline_feasible": 0,
        "value_feasible": 0,
        "unmeasured": False,
        "empty": False,
    }
    options = [{"f": 0, "value": 0.0, "keys": []}]
    m = len(unique)
    if m <= 0:
        info["empty"] = True
        return options, info
    base_cpu = measured_cpu_cost_ms(m, cost_table)
    if base_cpu is None:
        info["base_unmeasured"] = True
        return options, info
    max_f = min(m, args.gos_max_prefetch_per_target)
    for f in range(1, max_f + 1):
        if reserved + f > low_slots:
            continue
        info["slot_feasible"] += 1
        finish_ms = (backlog + f) * t_fetch
        deadline_ms = lead * args.gos_layer_slack_ms
        if finish_ms > deadline_ms:
            continue
        info["deadline_feasible"] += 1
        after_cpu = measured_cpu_cost_ms(m - f, cost_table)
        if after_cpu is None:
            info["unmeasured"] = True
            continue
        base_inc = max(0.0, base_cpu - args.gos_cpu_overlap_ms)
        after_inc = max(f * t_gpu, after_cpu - args.gos_cpu_overlap_ms)
        value = base_inc - after_inc
        if value >= args.gos_value_margin_ms:
            info["value_feasible"] += 1
            options.append({"f": f, "value": value, "keys": unique[:f]})
    return options, info


def choose_gos_admissions_greedy(candidates_by_target, resident_or_staged, issuer: ShallowIssuer,
                                 cost_table, t_fetch: float, t_gpu: float, args, stats=None):
    admitted = []
    backlog = issuer.low_backlog_count()
    reserved = issuer.low_reserved_count()
    used = set()
    for (target_token, target_layer, lead), ranked in sorted(candidates_by_target.items(), key=lambda x: x[0]):
        if stats is not None:
            stats["gos_targets"] += 1
            stats["gos_backlog_copies"] += backlog
            stats["gos_slot_reserved"] += reserved
        options, info = gos_target_options([k for k in ranked if k not in used], resident_or_staged, reserved,
                                           backlog, len(issuer.low_slots), lead, cost_table, t_fetch, t_gpu,
                                           args, stats)
        best = max(options, key=lambda o: (o["value"], -o["f"]))
        best_f = best["f"]
        if info["empty"]:
            continue
        if stats is not None and best_f == 0:
            if info["base_unmeasured"] or info["unmeasured"]:
                stats["gos_reject_unmeasured_targets"] += 1
            elif info["slot_feasible"] == 0:
                stats["gos_reject_slot_targets"] += 1
            elif info["deadline_feasible"] == 0:
                stats["gos_reject_deadline_targets"] += 1
            else:
                stats["gos_reject_value_targets"] += 1
        for key in best["keys"]:
            admitted.append((key, target_token, target_layer))
            used.add(key)
        backlog += best_f
        reserved += best_f
    return admitted


def choose_gos_admissions_dp(candidates_by_target, resident_or_staged, issuer: ShallowIssuer,
                             cost_table, t_fetch: float, t_gpu: float, args, stats=None):
    backlog0 = issuer.low_backlog_count()
    reserved0 = issuer.low_reserved_count()
    low_slots = len(issuer.low_slots)
    targets = []
    for target, ranked in sorted(candidates_by_target.items(), key=lambda x: x[0]):
        target_token, target_layer, lead = target
        if stats is not None:
            stats["gos_targets"] += 1
            stats["gos_backlog_copies"] += backlog0
            stats["gos_slot_reserved"] += reserved0
        options, info = gos_target_options(ranked, resident_or_staged, reserved0, backlog0, low_slots, lead,
                                           cost_table, t_fetch, t_gpu, args, stats)
        deadline_copies = int((lead * args.gos_layer_slack_ms) // t_fetch) if t_fetch > 0 else low_slots
        targets.append({
            "target": target,
            "options": options,
            "info": info,
            "deadline_copies": deadline_copies,
        })

    # State: (cumulative newly admitted H2D copies, selected keys) -> (value, path).
    # Tracking keys prevents duplicate value/copy accounting while still allowing
    # a repeated key to be chosen by a later target if an earlier target skipped it.
    dp = {(0, frozenset()): (0.0, [])}
    for idx, target in enumerate(targets):
        next_dp = {}
        for (used_copies, used_keys), (value, path) in dp.items():
            for option in target["options"]:
                option_keys = frozenset(option["keys"])
                if option_keys & used_keys:
                    continue
                nc = used_copies + option["f"]
                if reserved0 + nc > low_slots:
                    continue
                if backlog0 + nc > target["deadline_copies"]:
                    continue
                nv = value + option["value"]
                state = (nc, used_keys | option_keys)
                old = next_dp.get(state)
                if old is None or nv > old[0]:
                    next_dp[state] = (nv, path + [(idx, option)])
        dp = next_dp or {(0, frozenset()): (0.0, [])}
        if stats is not None:
            stats["gos_dp_states"] += len(dp)

    if dp:
        (_best_copies, _best_keys), (best_value, best_path) = max(dp.items(), key=lambda x: (x[1][0], -x[0][0]))
    else:
        best_value, best_path = 0.0, []
    selected = {idx: option for idx, option in best_path if option["f"] > 0}
    admitted = []
    for idx, target in enumerate(targets):
        option = selected.get(idx)
        if option is None:
            info = target["info"]
            if info["empty"]:
                continue
            if stats is not None:
                if info["base_unmeasured"] or info["unmeasured"]:
                    stats["gos_reject_unmeasured_targets"] += 1
                elif info["slot_feasible"] == 0:
                    stats["gos_reject_slot_targets"] += 1
                elif info["deadline_feasible"] == 0:
                    stats["gos_reject_deadline_targets"] += 1
                elif info["value_feasible"] == 0:
                    stats["gos_reject_value_targets"] += 1
                else:
                    stats["gos_reject_global_targets"] += 1
            continue
        target_token, target_layer, _lead = target["target"]
        for key in option["keys"]:
            admitted.append((key, target_token, target_layer))
    if stats is not None:
        stats["gos_dp_selected_value_ms"] += best_value
        stats["gos_dp_selected_copies"] += len(admitted)
    return admitted


def choose_gos_admissions(candidates_by_target, resident_or_staged, issuer: ShallowIssuer,
                          cost_table, t_fetch: float, t_gpu: float, args, stats=None):
    """Global overflow admission over forecast jobs.

    A target is a future (token, layer).  For each target, SPICE forecasts a set
    of experts that may miss. GOS submits only the prefix that (1) can complete
    before the target deadline under the current low-stream backlog and (2)
    reduces predicted CPU burst more than it costs as future GPU hit compute.
    """
    if args.gos_scheduler == "dp":
        return choose_gos_admissions_dp(candidates_by_target, resident_or_staged, issuer,
                                        cost_table, t_fetch, t_gpu, args, stats)
    return choose_gos_admissions_greedy(candidates_by_target, resident_or_staged, issuer,
                                        cost_table, t_fetch, t_gpu, args, stats)


def main() -> None:
    args = parse_args()
    if (
        args.prefetch_hit_admission == "oracle_value"
        or args.residual_fetch_admission == "oracle_value"
    ) and not args.allow_oracle_admission:
        raise ValueError(
            "oracle_value admission reads true future routes and is an upper-bound diagnostic; "
            "rerun with --allow_oracle_admission to make that explicit"
        )
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    # verified-gate negative admission: top-k ranks (0=highest gate) whose MISSES are shared-substituted
    substitute_ranks = set(int(x) for x in args.substitute_ranks.split(",") if x.strip() != "")
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    dt = torch.bfloat16
    cpu_dt = torch.bfloat16 if args.cpu_dtype == "bf16" else torch.float32

    cost_table, _best_table, cost_meta, expert_mb, _act_mb = load_costs(args.cost_json, args.cost_metric)
    bw = float(cost_meta.get("config", {}).get("bw_gbps", 0.0))
    t_fetch = expert_mb / (bw * 1024.0 / 1000.0) if bw else 0.792
    t_gpu = args.t_gpu_ms

    seqs, n_layers, n_experts, dump_top_k, max_horizon, manifest = load_forecast_sequences(args.forecast_dir)
    if args.top_k != dump_top_k:
        raise ValueError(f"--top_k={args.top_k} does not match dump top_k={dump_top_k}")
    if args.high_slots < args.top_k:
        raise ValueError(f"--high_slots={args.high_slots} must be >= --top_k={args.top_k}")
    split = max(1, int(round(len(seqs) * args.train_frac)))
    # Fail-fast: a non-empty HELD-OUT test split is required. The old `seqs[split:] or seqs`
    # fallback silently reused the TRAIN sequences as test, leaking popularity/warm-cache
    # priors (and, for oracle_value admission, true future routes) into evaluation.
    train, test = seqs[:split], seqs[split:]
    if not test:
        raise ValueError(
            f"empty held-out test split (len(seqs)={len(seqs)}, train_frac={args.train_frac}); "
            "need seqs[split:] non-empty for a valid train/test separation"
        )
    pop = popularity([seq_for_popularity(x) for x in train], n_layers, n_experts)
    cap = max(1, int(round(args.residency * n_layers * n_experts)))

    bank_size = max(args.bank, args.top_k * 8)
    host_bank = make_host_bank(bank_size, args.d_model, args.d_inter, dt)
    cpu_bank = make_cpu_bank(host_bank, cpu_dt)
    low_slots = make_dev_slots(args.low_slots, args.d_model, args.d_inter, dev, dt)
    high_slots = make_dev_slots(args.high_slots, args.d_model, args.d_inter, dev, dt)
    resident_slots = make_dev_slots(cap, args.d_model, args.d_inter, dev, dt)
    x_gpu = torch.randn(1, args.d_model, device=dev, dtype=dt)
    x_cpu_pin = torch.empty(1, args.d_model, dtype=cpu_dt, pin_memory=True)
    y_cpu_pin = torch.empty(1, args.d_model, dtype=cpu_dt, pin_memory=True)
    y_gpu = torch.empty(1, args.d_model, device=dev, dtype=dt)
    act_stream = torch.cuda.Stream(device=dev, priority=0)

    fdim = args.filler_compute_dim
    filler_a = torch.randn(fdim, fdim, device=dev, dtype=dt) if fdim > 0 else None
    filler_b = torch.randn(fdim, fdim, device=dev, dtype=dt) if fdim > 0 else None
    filler_c = torch.empty(fdim, fdim, device=dev, dtype=dt) if fdim > 0 else None
    filler_repeats = max(0, args.filler_repeats)

    def filler_compute():
        if filler_a is not None:
            for _ in range(filler_repeats):
                torch.mm(filler_a, filler_b, out=filler_c)

    def hkey(key):
        layer, expert = key
        return (layer * 4099 + expert) % bank_size

    def copy_host_to_slot(key, slot, stream=None):
        stream = stream or torch.cuda.current_stream(dev)
        src = host_bank[hkey(key)]
        with torch.cuda.stream(stream):
            for s, d in zip(src, slot):
                d.copy_(s, non_blocking=True)

    def setup_resident(cache):
        setup_stream = torch.cuda.Stream(device=dev)
        resident_map = {}
        free_resident = deque(range(len(resident_slots)))
        for key in sorted(cache):
            if not free_resident:
                break
            sid = free_resident.popleft()
            copy_host_to_slot(key, resident_slots[sid], setup_stream)
            resident_map[key] = sid
        setup_stream.synchronize()
        return resident_map, free_resident

    def start_cpu_activation(has_cpu):
        if not has_cpu:
            return None
        ev = torch.cuda.Event()
        with torch.cuda.stream(act_stream):
            x_cpu_pin.copy_(x_gpu, non_blocking=True)
            ev.record()
        return ev

    def finish_cpu_serve(keys, d2h_event):
        if not keys:
            return None
        d2h_event.synchronize()
        out = None
        for key in keys:
            y = expert_cpu(x_cpu_pin, cpu_bank[hkey(key)])
            out = y if out is None else out + y
        y_cpu_pin.copy_(out)
        ev = torch.cuda.Event()
        with torch.cuda.stream(act_stream):
            y_gpu.copy_(y_cpu_pin, non_blocking=True)
            ev.record()
        return ev

    def run_policy(policy: str):
        max_depth = None if policy.startswith("deep_") else args.shallow_depth
        ignore_prefetch_hits = "dummy" in policy
        issuer = ShallowIssuer(dev, host_bank, low_slots, high_slots, bank_size, max_depth=max_depth)
        cache = warm_cache(pop, cap)
        resident_map, free_resident = setup_resident(cache)
        last_used = {k: 0 for k in cache}
        recent_by_layer = [deque() for _ in range(n_layers)]
        recent_count = [[0 for _ in range(n_experts)] for _ in range(n_layers)]
        stats = defaultdict(float)
        pos = 0
        tokens_done = 0

        def evict_for_resident_slot(cur_layer):
            if free_resident:
                return free_resident.popleft()
            victim = evict_ls(cache, last_used, cur_layer, n_layers)
            sid = resident_map.pop(victim, None)
            cache.discard(victim)
            last_used.pop(victim, None)
            if sid is None:
                sid = 0
            stats["cache_evictions"] += 1
            return sid

        def ls_victim(cur_layer):
            if free_resident or not cache:
                return None
            return max(cache, key=lambda k: ((k[0] - cur_layer) % n_layers or n_layers, -last_used.get(k, -1)))

        def count_future_hits(true_top, key, ti):
            horizon = max(0, args.admit_value_horizon_tokens)
            if horizon <= 0:
                return 0
            layer, expert = key
            start = ti + 1
            end = min(true_top.shape[1], start + horizon)
            if layer < 0 or layer >= true_top.shape[0] or start >= end:
                return 0
            hits = (true_top[layer, start:end] == int(expert)).any(dim=-1)
            return int(hits.sum().item())

        def record_resident_admit_stats(source, accept):
            stats["resident_admit_decisions"] += 1
            stats[f"{source}_admit_decisions"] += 1
            if accept:
                stats["resident_admit_accepted"] += 1
                stats[f"{source}_admit_accepted"] += 1
            else:
                stats["resident_admit_rejected"] += 1
                stats[f"{source}_admit_rejected"] += 1

        def oracle_value_admit(key, cur_layer, true_top, ti):
            # OPTIMISTIC oracle upper-bound on resident-admission value (NOT a deployable model).
            # Per future hit, the no-residency fallback is policy/forecast dependent: re-stage via a
            # low-slot H2D copy (if LoRE re-predicts it), residual high-fetch, or CPU serve. We credit
            # the FULL H2D transfer (t_fetch) as saved -- an UPPER bound, since a well-overlapped low
            # stream hides part of that transfer so the true EXPOSED saving is <= t_fetch (the harness
            # has no per-key exposed-wait column to measure it exactly). GPU compute (t_gpu) is paid in
            # BOTH the resident and the re-stage case, so it cancels. Admitting also pays a one-time
            # main-stream D2D promotion + low-stream fence (resident_admit_cost_ms, sweep it) and evicts
            # a victim whose own future hits then re-stage. Interpretation: if even this optimistic
            # ceiling cannot beat never-admit, resident admission is robustly not worth it.
            victim = ls_victim(cur_layer)
            cand_hits = count_future_hits(true_top, key, ti)
            victim_hits = 0 if victim is None else count_future_hits(true_top, victim, ti)
            saved_per_hit = t_fetch  # H2D re-stage avoided per future resident hit
            value_ms = (cand_hits - victim_hits) * saved_per_hit - args.resident_admit_cost_ms
            stats["resident_admit_oracle_cand_hits"] += cand_hits
            stats["resident_admit_oracle_victim_hits"] += victim_hits
            stats["resident_admit_oracle_value_ms"] += value_ms
            return value_ms >= args.resident_value_margin_ms

        def resolve_resident_admission_policy(source):
            prefetch_policy = "never" if args.no_admit_prefetch_hits else args.prefetch_hit_admission
            if source == "prefetch":
                return prefetch_policy
            if args.residual_fetch_admission == "same_as_prefetch":
                return prefetch_policy
            return args.residual_fetch_admission

        def should_admit_resident(key, cur_layer, current_routed_keys, true_top, ti, source):
            policy_admit = resolve_resident_admission_policy(source)
            if policy_admit == "always":
                record_resident_admit_stats(source, True)
                return True
            if policy_admit == "never":
                record_resident_admit_stats(source, False)
                return False
            if policy_admit == "oracle_value":
                accept = oracle_value_admit(key, cur_layer, true_top, ti)
                record_resident_admit_stats(source, accept)
                return accept
            victim = ls_victim(cur_layer)
            cand_pop = pop[key[0]][key[1]]
            victim_pop = -1 if victim is None else pop[victim[0]][victim[1]]
            if policy_admit == "recent_reuse":
                cand_recent = recent_count[key[0]][key[1]] + (1 if key in current_routed_keys else 0)
                victim_recent = -1 if victim is None else (
                    recent_count[victim[0]][victim[1]] + (1 if victim in current_routed_keys else 0)
                )
                accept = (
                    victim is None
                    or cand_recent > victim_recent
                    or (cand_recent == victim_recent and cand_pop > victim_pop)
                )
                stats["resident_admit_cand_recent"] += cand_recent
                stats["resident_admit_victim_recent"] += max(0, victim_recent)
                stats[f"{source}_admit_cand_recent"] += cand_recent
                stats[f"{source}_admit_victim_recent"] += max(0, victim_recent)
                if source == "prefetch":
                    stats["prefetch_admit_cand_recent"] += cand_recent
                    stats["prefetch_admit_victim_recent"] += max(0, victim_recent)
            else:
                accept = victim is None or cand_pop > victim_pop
            record_resident_admit_stats(source, accept)
            return accept

        def update_recent(layer, routed_keys):
            if args.admit_recent_window <= 0:
                return
            uniq = tuple(sorted(set(e for _l, e in routed_keys)))
            recent_by_layer[layer].append(uniq)
            for e in uniq:
                recent_count[layer][e] += 1
            while len(recent_by_layer[layer]) > args.admit_recent_window:
                old = recent_by_layer[layer].popleft()
                for e in old:
                    recent_count[layer][e] -= 1

        def admit_to_resident(key, src_slot, cur_layer):
            sid = resident_map.get(key)
            if sid is None:
                sid = evict_for_resident_slot(cur_layer)
                for s, d in zip(src_slot, resident_slots[sid]):
                    d.copy_(s, non_blocking=True)
                resident_map[key] = sid
            cache.add(key)
            last_used[key] = pos
            return sid

        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for item in test:
            true_top = item["true_top"]
            fcast = item["fcast"]
            tokens = min(true_top.shape[1], args.max_test_tokens - tokens_done)
            for ti in range(tokens):
                for layer in range(n_layers):
                    issuer.poll(ti, layer)

                    # Issue SPICE future prefetch intents before dense/router work
                    # so low-priority H2D can overlap the current layer compute
                    # window. Shallow depth, not magic stream priority, limits how
                    # much low traffic can be ahead of a future demand miss.
                    max_lead = min(args.max_lead_layers, max_horizon - 1, n_layers - layer - 1)
                    resident_or_staged = set(resident_map) | issuer.staged_keys() | set(issuer.pending)
                    if policy in ("gos_cpu", "gos_dummy_cpu"):
                        candidates_by_target = defaultdict(list)
                        for lead in range(max(1, args.min_prefetch_lead), max_lead + 1):
                            target_layer = layer + lead
                            pred = [int(x) for x in fcast[layer, lead, ti].tolist() if int(x) >= 0]
                            target = (ti, target_layer, lead)
                            for e in pred:
                                key = (target_layer, e)
                                candidates_by_target[target].append(key)
                                stats["gos_candidates"] += 1
                        admitted = choose_gos_admissions(candidates_by_target, resident_or_staged, issuer,
                                                         cost_table, t_fetch, t_gpu, args, stats)
                        stats["gos_admitted"] += len(admitted)
                        for key, target_token, target_layer in admitted:
                            issuer.add_intent(key, target_token, target_layer, resident_or_staged)
                            resident_or_staged.add(key)
                    else:
                        for lead in range(max(1, args.min_prefetch_lead), max_lead + 1):
                            target_layer = layer + lead
                            pred = [int(x) for x in fcast[layer, lead, ti].tolist() if int(x) >= 0]
                            for e in pred[:args.prefetch_per_layer]:
                                issuer.add_intent((target_layer, e), ti, target_layer, resident_or_staged)
                    issuer.pump()

                    # Dense/attention/router window.  The host cannot make the
                    # exact residual-miss decision until this work has produced
                    # the true router result, but draft prefetch is already live.
                    filler_compute()
                    torch.cuda.current_stream(dev).synchronize()
                    issuer.poll(ti, layer)

                    routed = [int(x) for x in true_top[layer, ti].tolist()]
                    routed_keys = {(layer, e) for e in routed}
                    hit_items = []  # (key, slot_id_or_None, slot, from_low_slot)
                    misses = []
                    for rank, e in enumerate(routed):  # true_top is gate-descending: rank 0 = highest gate
                        key = (layer, e)
                        stats["routed"] += 1
                        if key in resident_map:
                            hit_items.append((key, None, resident_slots[resident_map[key]], False))
                            last_used[key] = pos
                            stats["hits"] += 1
                            stats["resident_hits"] += 1
                        else:
                            staged = None if ignore_prefetch_hits else issuer.pop_staged(key)
                            if staged is None:
                                staged = None if ignore_prefetch_hits else issuer.wait_active(key)
                            if staged is not None:
                                slot_id, slot = staged
                                hit_items.append((key, slot_id, slot, True))
                                stats["hits"] += 1
                                stats["staging_hits"] += 1
                            elif rank in substitute_ranks:
                                # verified-gate NEGATIVE admission: low gate-mass miss -> shared-expert
                                # substitute (skip CPU+fetch). Lossy; PPL cost from drop_quality curve.
                                if not ignore_prefetch_hits:
                                    issuer.cancel_intent(key)
                                stats["substituted"] += 1
                            else:
                                if not ignore_prefetch_hits:
                                    issuer.cancel_intent(key)
                                misses.append(key)
                                stats["misses"] += 1
                        pos += 1

                    if policy.endswith("fetch_all"):
                        n_fetch = len(misses)
                    elif policy.endswith("cpu"):
                        n_fetch = 0
                    elif policy == "shallow_scheduler":
                        n_fetch = choose_fetch_count(len(misses), issuer.active_low_count(), cost_table,
                                                     t_fetch, args.fetch_margin_ms)
                    else:
                        raise ValueError(policy)
                    fetch_keys = misses[:n_fetch]
                    cpu_keys = misses[n_fetch:]

                    # Demand H2D, activation D2H, and resident expert GEMMs are
                    # launched together after the true router decision. CPU
                    # compute then runs while high-priority H2D and GPU resident
                    # expert compute are in flight.
                    high_fetches = issuer.fetch_residual_async(fetch_keys)
                    cpu_d2h_event = start_cpu_activation(bool(cpu_keys))
                    low_admit_after_hits = []
                    for key, slot_id, slot, from_low_slot in hit_items:
                        expert_gpu(x_gpu, slot)
                        if from_low_slot:
                            low_admit_after_hits.append((key, slot_id, slot))
                    for key, slot_id, slot in low_admit_after_hits:
                        if should_admit_resident(key, layer, routed_keys, true_top, ti, "prefetch"):
                            admit_to_resident(key, slot, layer)
                        issuer.release_low_slot(slot_id)

                    cpu_done_event = finish_cpu_serve(cpu_keys, cpu_d2h_event)
                    issuer.wait_high(high_fetches)
                    if high_fetches:
                        torch.cuda.current_stream(dev).wait_stream(issuer.high_stream)
                    for rec in high_fetches:
                        expert_gpu(x_gpu, rec["slot"])
                        if should_admit_resident(rec["key"], layer, routed_keys, true_top, ti, "residual"):
                            admit_to_resident(rec["key"], rec["slot"], layer)
                    if cpu_done_event is not None:
                        cpu_done_event.synchronize()
                    stats["residual_fetches"] += len(fetch_keys)
                    stats["cpu_served"] += len(cpu_keys)
                    update_recent(layer, routed_keys)
                tokens_done += 1
                if tokens_done >= args.max_test_tokens:
                    break
            if tokens_done >= args.max_test_tokens:
                break
        issuer.flush()
        torch.cuda.synchronize(dev)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stats.update(issuer.stats)
        stats["tokens"] = tokens_done
        stats["elapsed_ms"] = elapsed_ms
        stats["tpot_ms"] = elapsed_ms / max(1, tokens_done)
        return dict(stats)

    rows = []
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    for policy in policies:
        run_policy(policy)  # warmup per policy
        samples = [run_policy(policy) for _ in range(args.timed_repeats)]
        tpots = [x["tpot_ms"] for x in samples]
        med_idx = sorted(range(len(tpots)), key=lambda i: tpots[i])[len(tpots) // 2]
        row = samples[med_idx]
        row["policy"] = policy
        row["tpot_ms_median"] = float(statistics.median(tpots))
        row["tpot_ms_min"] = float(min(tpots))
        row["tpot_ms_max"] = float(max(tpots))
        for k in ["routed", "hits", "resident_hits", "staging_hits", "misses", "substituted",
                  "residual_fetches", "cpu_served", "cache_evictions",
                  "resident_admit_decisions", "resident_admit_accepted", "resident_admit_rejected",
                  "resident_admit_cand_recent", "resident_admit_victim_recent",
                  "resident_admit_oracle_cand_hits", "resident_admit_oracle_victim_hits",
                  "resident_admit_oracle_value_ms",
                  "prefetch_admit_decisions", "prefetch_admit_accepted", "prefetch_admit_rejected",
                  "prefetch_admit_cand_recent", "prefetch_admit_victim_recent",
                  "residual_admit_decisions", "residual_admit_accepted", "residual_admit_rejected",
                  "residual_admit_cand_recent", "residual_admit_victim_recent",
                  "gos_candidates", "gos_admitted", "gos_targets", "gos_backlog_copies", "gos_slot_reserved",
                  "gos_reject_slot_targets", "gos_reject_deadline_targets",
                  "gos_reject_value_targets", "gos_reject_unmeasured_targets",
                  "gos_reject_global_targets", "gos_dp_states", "gos_dp_selected_value_ms",
                  "gos_dp_selected_copies",
                  "prefetch_intents", "prefetch_submitted", "prefetch_completed", "prefetch_useful",
                  "prefetch_waited_active", "prefetch_intent_cancelled", "prefetch_intent_expired",
                  "prefetch_staged_expired", "prefetch_completed_expired", "prefetch_wait_active_expired",
                  "prefetch_waited_active_ms", "residual_fetch_wait_ms"]:
            row[f"{k}_per_tok"] = row.get(k, 0.0) / max(1, row["tokens"])
        rows.append(row)
        print(f"{policy:18s} TPOT={row['tpot_ms_median']:8.3f} "
              f"hit/tok={row['hits_per_tok']:6.2f} miss/tok={row['misses_per_tok']:6.2f} "
              f"fb_fetch/tok={row['residual_fetches_per_tok']:6.2f} cpu/tok={row['cpu_served_per_tok']:6.2f} "
              f"pf_sub/tok={row['prefetch_submitted_per_tok']:6.2f} pf_use/tok={row['prefetch_useful_per_tok']:6.2f} "
              f"pf_wait/tok={row['prefetch_waited_active_per_tok']:6.2f} "
              f"gos_admit/tok={row['gos_admitted_per_tok']:6.2f} "
              f"res_admit/tok={row['resident_admit_accepted_per_tok']:6.2f}/{row['resident_admit_decisions_per_tok']:6.2f} "
              f"gos_deadline_rej/tok={row['gos_reject_deadline_targets_per_tok']:6.2f} "
              f"gos_value_rej/tok={row['gos_reject_value_targets_per_tok']:6.2f} "
              f"gos_global_rej/tok={row['gos_reject_global_targets_per_tok']:6.2f} "
              f"gos_unmeasured_rej/tok={row['gos_reject_unmeasured_targets_per_tok']:6.2f} "
              f"pf_wait_ms/tok={row['prefetch_waited_active_ms_per_tok']:6.2f} "
              f"resid_wait_ms/tok={row['residual_fetch_wait_ms_per_tok']:6.2f}",
              flush=True)

    by = {r["policy"]: r for r in rows}
    verdict = {}
    if "deep_cpu" in by and "shallow_scheduler" in by:
        verdict["shallow_scheduler_vs_deep_cpu_pct"] = 100.0 * (
            by["deep_cpu"]["tpot_ms_median"] - by["shallow_scheduler"]["tpot_ms_median"]
        ) / by["deep_cpu"]["tpot_ms_median"]
    if "deep_fetch_all" in by and "shallow_scheduler" in by:
        verdict["shallow_scheduler_vs_deep_fetch_all_pct"] = 100.0 * (
            by["deep_fetch_all"]["tpot_ms_median"] - by["shallow_scheduler"]["tpot_ms_median"]
        ) / by["deep_fetch_all"]["tpot_ms_median"]
    out = {
        "config": vars(args),
        "forecast_manifest": manifest,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "top_k": dump_top_k,
        "max_horizon": max_horizon,
        "capacity": cap,
        "expert_mb": expert_mb,
        "t_fetch_ms": t_fetch,
        "rows": rows,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
