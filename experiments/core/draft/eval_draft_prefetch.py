from __future__ import annotations

import sys, pathlib  # bootstrap: resolve sibling core modules regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
from collections import OrderedDict
from pathlib import Path

import torch

from common import build_arg_parser, device_from_arg, ensure_dir, set_seed, write_json
from draft_model import draft_losses, load_checkpoint, routing_metrics, topk_sets


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.store: OrderedDict[tuple[int, int], None] = OrderedDict()

    def contains(self, key: tuple[int, int]) -> bool:
        ok = key in self.store
        if ok:
            self.store.move_to_end(key)
        return ok

    def add(self, key: tuple[int, int]) -> bool:
        if key in self.store:
            self.store.move_to_end(key)
            return False
        self.store[key] = None
        while len(self.store) > self.capacity:
            self.store.popitem(last=False)
        return True


def sample_batch(batch: int, hidden: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch, hidden, device=device)


def l_min_from_hardware(top_k: int, expert_mb: int, pcie_gbps: float, compute_ms: float) -> int:
    expert_gb = expert_mb / 1024.0
    window_gb = pcie_gbps * (compute_ms / 1000.0)
    if window_gb <= 0:
        return 1
    return max(1, math.ceil((top_k * expert_gb) / window_gb))


def mean_topk_conf(conf: torch.Tensor) -> float:
    return float(conf.detach().float().mean().item())


@torch.no_grad()
def context_from_observed_routes(draft, target_route_probs: list[torch.Tensor], upto_layer: int) -> torch.Tensor:
    batch = target_route_probs[0].shape[0]
    device = target_route_probs[0].device
    context = draft.initial_context(batch, device)
    for layer in range(upto_layer):
        context = draft.update_context(target_route_probs[layer], context)
    return context


@torch.no_grad()
def eval_static(target, draft, batches: int, batch: int, device: torch.device, align_lambda: float) -> dict:
    target.eval()
    draft.eval()
    route = 0.0
    align = 0.0
    slot_hit = 0.0
    fallback = 0.0
    exact = 0.0
    conf = 0.0
    for _ in range(batches):
        x = sample_batch(batch, target.cfg.hidden, device)
        target_out = target(x)
        draft_out = draft(x)
        losses = draft_losses(draft_out, target_out, align_lambda=align_lambda)
        metrics = routing_metrics(draft_out, target_out, target.cfg.top_k)
        route += float(losses["route_loss"].item())
        align += float(losses["align_loss"].item())
        slot_hit += metrics["slot_hit_rate"]
        fallback += metrics["fallback_slot_rate"]
        exact += metrics["exact_set_match_rate"]
        conf += metrics["mean_confidence"]
    denom = max(1, batches)
    return {
        "route_kl": route / denom,
        "align_mse": align / denom,
        "slot_hit_rate": slot_hit / denom,
        "fallback_slot_rate": fallback / denom,
        "exact_set_match_rate": exact / denom,
        "mean_confidence": conf / denom,
    }


@torch.no_grad()
def eval_verified_prefetch(
    target,
    draft,
    steps: int,
    batch: int,
    cache_capacity: int,
    l_min: int,
    l_max: int,
    confidence_threshold: float,
    expert_mb: int,
    pcie_gbps: float,
    compute_ms: float,
    device: torch.device,
    anchor_reinit: bool = True,
    observed_route_history: bool = True,
) -> dict:
    cache = LRUCache(cache_capacity)
    total_slots = 0
    hit_slots = 0
    fallback_slots = 0
    wrong_prefetches = 0
    issued_prefetches = 0
    predicted_slot_attempts = 0
    depth_sum = 0
    depth_count = 0
    depth_hist: dict[str, int] = {}
    confidence_by_depth: dict[str, list[float]] = {}

    for _ in range(steps):
        x = sample_batch(batch, target.cfg.hidden, device)
        target_out = target(x)
        target_sets = [topk_sets(t) for t in target_out["topk_indices"]]
        if anchor_reinit:
            full_draft_out = None
            full_pred_sets = None
        else:
            full_draft_out = draft(x)
            full_pred_sets = [topk_sets(p) for p in full_draft_out["topk_indices"]]

        for anchor in range(target.cfg.layers):
            if anchor_reinit:
                anchor_hidden = x if anchor == 0 else target_out["hidden_states"][anchor - 1]
                if observed_route_history:
                    context = context_from_observed_routes(draft, target_out["route_probs"], anchor)
                else:
                    context = None
                draft_out = draft(anchor_hidden, start_layer=anchor, context=context)
                pred_sets = [topk_sets(p) for p in draft_out["topk_indices"]]
            else:
                assert full_draft_out is not None and full_pred_sets is not None
                draft_out = full_draft_out
                pred_sets = full_pred_sets

            depth = 0
            for future in range(anchor, min(target.cfg.layers, anchor + l_max)):
                depth += 1
                draft_index = future - anchor if anchor_reinit else future
                conf = mean_topk_conf(draft_out["confidences"][draft_index])
                confidence_by_depth.setdefault(str(depth), []).append(conf)
                for b in range(batch):
                    for e in pred_sets[draft_index][b]:
                        predicted_slot_attempts += 1
                        issued_prefetches += int(cache.add((future, e)))
                    wrong_prefetches += len(pred_sets[draft_index][b].difference(target_sets[future][b]))
                if depth >= l_min and conf < confidence_threshold:
                    break
            depth_sum += depth
            depth_count += 1
            depth_hist[str(depth)] = depth_hist.get(str(depth), 0) + 1

            for b in range(batch):
                for e in target_sets[anchor][b]:
                    total_slots += 1
                    if cache.contains((anchor, e)):
                        hit_slots += 1
                    else:
                        fallback_slots += 1
                        cache.add((anchor, e))

    copy_ms = expert_mb / 1024.0 / max(1e-9, pcie_gbps) * 1000.0
    stall_ms = fallback_slots * copy_ms
    draft_ms = steps * target.cfg.layers * 0.06
    compute_total_ms = steps * target.cfg.layers * compute_ms
    sim_total_ms = compute_total_ms + stall_ms + draft_ms
    conf_summary = {
        depth: {
            "mean": sum(vals) / max(1, len(vals)),
            "count": len(vals),
        }
        for depth, vals in sorted(confidence_by_depth.items(), key=lambda x: int(x[0]))
    }
    return {
        "steps": steps,
        "batch": batch,
        "cache_capacity": cache_capacity,
        "l_min": l_min,
        "l_max": l_max,
        "confidence_threshold": confidence_threshold,
        "expert_mb": expert_mb,
        "pcie_gbps": pcie_gbps,
        "compute_ms": compute_ms,
        "anchor_reinit": anchor_reinit,
        "observed_route_history": observed_route_history,
        "total_slots": total_slots,
        "prefetch_slot_hit_rate": hit_slots / max(1, total_slots),
        "fallback_slot_rate": fallback_slots / max(1, total_slots),
        "fallback_slots": fallback_slots,
        "predicted_slot_attempts": predicted_slot_attempts,
        "issued_prefetches": issued_prefetches,
        "wrong_prefetches": wrong_prefetches,
        "wrong_prefetch_rate": wrong_prefetches / max(1, predicted_slot_attempts),
        "avg_lookahead_depth": depth_sum / max(1, depth_count),
        "depth_hist": depth_hist,
        "confidence_by_depth": conf_summary,
        "sim_stall_ms": stall_ms,
        "sim_total_ms": sim_total_ms,
        "sim_tpot_ms": sim_total_ms / max(1, steps),
    }


def online_self_correction(target, draft, steps: int, batch: int, lr: float, align_lambda: float, device: torch.device) -> list[dict]:
    if steps <= 0:
        return []
    draft.train()
    optimizer = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    logs = []
    for step in range(steps):
        x = sample_batch(batch, target.cfg.hidden, device)
        with torch.no_grad():
            target_out = target(x)
        draft_out = draft(x)
        losses = draft_losses(draft_out, target_out, align_lambda=align_lambda)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), max_norm=1.0)
        optimizer.step()
        if (step + 1) % max(1, steps // 4) == 0 or step == steps - 1:
            metrics = routing_metrics(draft_out, target_out, target.cfg.top_k)
            logs.append(
                {
                    "step": step + 1,
                    "loss": float(losses["loss"].item()),
                    "route_kl": float(losses["route_loss"].item()),
                    "align_mse": float(losses["align_loss"].item()),
                    **metrics,
                }
            )
            print(logs[-1])
    draft.eval()
    return logs


def main() -> None:
    parser = build_arg_parser("Evaluate SPICE draft-model-driven prefetching")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=16)
    parser.add_argument("--cache_capacity", type=int, default=512)
    parser.add_argument("--expert_mb", type=int, default=64)
    parser.add_argument("--pcie_gbps", type=float, default=48.0)
    parser.add_argument("--compute_ms", type=float, default=2.5)
    parser.add_argument("--l_min", type=int, default=0)
    parser.add_argument("--l_max", type=int, default=6)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--no_anchor_reinit", action="store_true")
    parser.add_argument("--no_observed_route_history", action="store_true")
    parser.add_argument("--online_steps", type=int, default=0)
    parser.add_argument("--online_lr", type=float, default=5e-5)
    parser.add_argument("--align_lambda", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = device_from_arg(args.gpu)
    cfg, target, draft, ckpt_extra = load_checkpoint(args.checkpoint, device)
    l_min = args.l_min or l_min_from_hardware(cfg.top_k, args.expert_mb, args.pcie_gbps, args.compute_ms)

    before = eval_static(target, draft, args.eval_batches, args.batch, device, args.align_lambda)
    online_logs = online_self_correction(target, draft, args.online_steps, args.batch, args.online_lr, args.align_lambda, device)
    after = eval_static(target, draft, args.eval_batches, args.batch, device, args.align_lambda)
    verified = eval_verified_prefetch(
        target=target,
        draft=draft,
        steps=args.steps,
        batch=args.batch,
        cache_capacity=args.cache_capacity,
        l_min=l_min,
        l_max=args.l_max,
        confidence_threshold=args.confidence_threshold,
        expert_mb=args.expert_mb,
        pcie_gbps=args.pcie_gbps,
        compute_ms=args.compute_ms,
        device=device,
        anchor_reinit=not args.no_anchor_reinit,
        observed_route_history=not args.no_observed_route_history,
    )
    result = {
        "experiment": "spice_draft_prefetch_eval",
        "device": str(device),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": cfg.__dict__,
        "checkpoint_extra": ckpt_extra,
        "before_online": before,
        "online_logs": online_logs,
        "after_online": after,
        "verified_prefetch": verified,
        "correctness_invariant": (
            "Draft predictions only issue prefetches; target router selections are verified, "
            "and missing target experts are synchronously fetched before execution."
        ),
    }
    write_json(Path(out_dir) / "draft_prefetch_eval.json", result)
    print(result)


if __name__ == "__main__":
    main()
