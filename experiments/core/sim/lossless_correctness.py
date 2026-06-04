from __future__ import annotations

import sys, pathlib  # bootstrap: resolve sibling core modules regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from common import build_arg_parser, device_from_arg, ensure_dir, set_seed, write_json


@dataclass
class CorrectnessConfig:
    layers: int = 8
    experts: int = 16
    top_k: int = 2
    hidden: int = 128
    batch: int = 16
    steps: int = 96
    vocab: int = 256
    prediction_accuracy: float = 0.72


class TinyMoE:
    def __init__(self, cfg: CorrectnessConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.router_w = [
            torch.randn(cfg.hidden, cfg.experts, device=device) / math.sqrt(cfg.hidden)
            for _ in range(cfg.layers)
        ]
        self.experts = [
            [
                torch.randn(cfg.hidden, cfg.hidden, device=device) / math.sqrt(cfg.hidden)
                for _ in range(cfg.experts)
            ]
            for _ in range(cfg.layers)
        ]
        self.head = torch.randn(cfg.hidden, cfg.vocab, device=device) / math.sqrt(cfg.hidden)

    def route(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        logits = h @ self.router_w[layer]
        return torch.topk(logits, k=self.cfg.top_k, dim=-1).indices

    def apply_layer(self, h: torch.Tensor, layer: int, selected: torch.Tensor) -> torch.Tensor:
        outs = []
        for b in range(h.shape[0]):
            acc = torch.zeros_like(h[b])
            for e in selected[b].tolist():
                acc = acc + torch.tanh(h[b] @ self.experts[layer][e])
            outs.append(h[b] + acc / self.cfg.top_k)
        return torch.stack(outs, dim=0)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.head


def corrupt_prediction(target: torch.Tensor, experts: int, accuracy: float) -> torch.Tensor:
    pred = target.clone()
    mask = torch.rand_like(pred.float()) > accuracy
    replacement = torch.randint(0, experts, pred.shape, device=pred.device)
    pred[mask] = replacement[mask]
    return pred


def run_baseline(model: TinyMoE, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    h = x.clone()
    routes = []
    for l in range(model.cfg.layers):
        selected = model.route(h, l)
        routes.append(selected)
        h = model.apply_layer(h, l, selected)
    return model.logits(h), routes


def run_spice_verified(model: TinyMoE, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    h = x.clone()
    total_slots = 0
    predicted_hits = 0
    fallback_slots = 0
    for l in range(model.cfg.layers):
        target = model.route(h, l)
        pred = corrupt_prediction(target, model.cfg.experts, model.cfg.prediction_accuracy)
        target_sets = [set(row.tolist()) for row in target]
        pred_sets = [set(row.tolist()) for row in pred]
        for ts, ps in zip(target_sets, pred_sets):
            total_slots += len(ts)
            predicted_hits += len(ts.intersection(ps))
            fallback_slots += len(ts.difference(ps))
        # The verified mode always executes the target-selected experts.
        h = model.apply_layer(h, l, target)
    metrics = {
        "total_expert_slots": total_slots,
        "predicted_hit_slots": predicted_hits,
        "fallback_slots": fallback_slots,
        "prefetch_slot_hit_rate": predicted_hits / max(1, total_slots),
        "fallback_slot_rate": fallback_slots / max(1, total_slots),
    }
    return model.logits(h), metrics


def main() -> None:
    parser = build_arg_parser("Lossless SPICE correctness harness")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--prediction_accuracy", type=float, default=0.72)
    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = device_from_arg(args.gpu)

    cfg = CorrectnessConfig(steps=args.steps, prediction_accuracy=args.prediction_accuracy)
    model = TinyMoE(cfg, device)
    max_abs = 0.0
    exact_token_matches = 0
    total_tokens = 0
    losses = []
    spice_losses = []
    aggregate = {"fallback_slots": 0, "predicted_hit_slots": 0, "total_expert_slots": 0}
    for _ in range(cfg.steps):
        x = torch.randn(cfg.batch, cfg.hidden, device=device)
        labels = torch.randint(0, cfg.vocab, (cfg.batch,), device=device)
        base_logits, _ = run_baseline(model, x)
        spice_logits, metrics = run_spice_verified(model, x)
        max_abs = max(max_abs, float((base_logits - spice_logits).abs().max().item()))
        exact_token_matches += int((base_logits.argmax(dim=-1) == spice_logits.argmax(dim=-1)).sum().item())
        total_tokens += cfg.batch
        losses.append(float(F.cross_entropy(base_logits, labels).item()))
        spice_losses.append(float(F.cross_entropy(spice_logits, labels).item()))
        for k in aggregate:
            aggregate[k] += metrics[k]

    result = {
        "experiment": "lossless_correctness",
        "device": str(device),
        "config": cfg.__dict__,
        "max_abs_logit_diff": max_abs,
        "exact_argmax_match_rate": exact_token_matches / max(1, total_tokens),
        "baseline_cross_entropy": sum(losses) / len(losses),
        "spice_verified_cross_entropy": sum(spice_losses) / len(spice_losses),
        "baseline_pseudo_ppl": math.exp(sum(losses) / len(losses)),
        "spice_verified_pseudo_ppl": math.exp(sum(spice_losses) / len(spice_losses)),
        "prefetch_slot_hit_rate": aggregate["predicted_hit_slots"] / max(1, aggregate["total_expert_slots"]),
        "fallback_slot_rate": aggregate["fallback_slots"] / max(1, aggregate["total_expert_slots"]),
        "interpretation": "SPICE verified mode executes the target router's experts; nonzero fallback affects latency, not logits.",
    }
    write_json(out_dir / "lossless_correctness.json", result)
    print(result)


if __name__ == "__main__":
    main()
