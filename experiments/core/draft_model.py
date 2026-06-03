from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DraftConfig:
    layers: int = 8
    experts: int = 16
    top_k: int = 2
    hidden: int = 256
    expert_hidden: int = 512
    rank: int = 16
    route_context: int = 64
    shared_down: bool = True
    history: str = "gru"


def topk_sets(topk: torch.Tensor) -> list[set[int]]:
    flat = topk.detach().cpu().tolist()
    return [set(row) for row in flat]


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_total(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class FrozenTargetMoE(nn.Module):
    """Small target MoE used to reproduce SPICE's draft-training path.

    It exposes the same ingredients required by the paper method: frozen
    attention-like state transitions, frozen routers, and physical experts.
    """

    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = nn.ModuleList(
            [nn.Linear(cfg.hidden, cfg.hidden, bias=False) for _ in range(cfg.layers)]
        )
        self.routers = nn.ModuleList(
            [nn.Linear(cfg.hidden, cfg.experts, bias=False) for _ in range(cfg.layers)]
        )
        self.experts = nn.ModuleList()
        for _ in range(cfg.layers):
            layer_experts = nn.ModuleList()
            for _ in range(cfg.experts):
                layer_experts.append(
                    nn.Sequential(
                        nn.Linear(cfg.hidden, cfg.expert_hidden, bias=False),
                        nn.SiLU(),
                        nn.Linear(cfg.expert_hidden, cfg.hidden, bias=False),
                    )
                )
            self.experts.append(layer_experts)
        self._init_weights()
        for p in self.parameters():
            p.requires_grad_(False)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.in_features))

    def route(self, z: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.routers[layer](z)
        probs = F.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, k=self.cfg.top_k, dim=-1)
        gate = top_prob / top_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs, top_idx, gate

    def apply_experts(self, z: torch.Tensor, layer: int, top_idx: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        mix = torch.zeros_like(z)
        for e, expert in enumerate(self.experts[layer]):
            expert_out = None
            for slot in range(self.cfg.top_k):
                weight = torch.where(top_idx[:, slot] == e, gate[:, slot], torch.zeros_like(gate[:, slot]))
                if torch.any(weight != 0):
                    if expert_out is None:
                        expert_out = expert(z)
                    mix = mix + weight.unsqueeze(-1) * expert_out
        return z + mix

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        h = x
        hidden_states = []
        route_probs = []
        topk_indices = []
        router_inputs = []
        for layer in range(self.cfg.layers):
            z = h + torch.tanh(self.attn[layer](h))
            probs, top_idx, gate = self.route(z, layer)
            h = self.apply_experts(z, layer, top_idx, gate)
            router_inputs.append(z)
            hidden_states.append(h)
            route_probs.append(probs)
            topk_indices.append(top_idx)
        return {
            "hidden_states": hidden_states,
            "route_probs": route_probs,
            "topk_indices": topk_indices,
            "router_inputs": router_inputs,
        }


class LoRELayer(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.shared_down:
            self.shared_down = nn.Linear(cfg.hidden, cfg.rank, bias=False)
            self.down = None
        else:
            self.shared_down = None
            self.down = nn.ModuleList(
                [nn.Linear(cfg.hidden, cfg.rank, bias=False) for _ in range(cfg.experts)]
            )
        self.up = nn.ModuleList(
            [nn.Linear(cfg.rank, cfg.hidden, bias=False) for _ in range(cfg.experts)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.shared_down is not None:
            nn.init.normal_(self.shared_down.weight, mean=0.0, std=0.02)
        if self.down is not None:
            for down in self.down:
                nn.init.normal_(down.weight, mean=0.0, std=0.02)
        for up in self.up:
            nn.init.zeros_(up.weight)

    def expert_delta(self, z: torch.Tensor, expert_id: int) -> torch.Tensor:
        if self.shared_down is not None:
            low = self.shared_down(z)
        else:
            assert self.down is not None
            low = self.down[expert_id](z)
        return self.up[expert_id](low)

    def forward(self, z: torch.Tensor, top_idx: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(z)
        for e in range(self.cfg.experts):
            expert_delta = None
            for slot in range(self.cfg.top_k):
                weight = torch.where(top_idx[:, slot] == e, gate[:, slot], torch.zeros_like(gate[:, slot]))
                if torch.any(weight != 0):
                    if expert_delta is None:
                        expert_delta = self.expert_delta(z, e)
                    delta = delta + weight.unsqueeze(-1) * expert_delta
        return z + delta


class SPICEDraftModel(nn.Module):
    """SPICE draft model: frozen attn/router + trainable LoRE + route history."""

    def __init__(self, cfg: DraftConfig, target: FrozenTargetMoE):
        super().__init__()
        self.cfg = cfg
        self.target = target
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.lore = nn.ModuleList([LoRELayer(cfg) for _ in range(cfg.layers)])
        if cfg.history == "gru":
            self.history_cell = nn.GRUCell(cfg.experts, cfg.route_context)
            self.route_in = None
            self.alpha_logit = None
        elif cfg.history == "ema":
            self.history_cell = None
            self.route_in = nn.Linear(cfg.experts, cfg.route_context, bias=False)
            self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(f"unknown history mode: {cfg.history}")
        self.context_to_logits = nn.Linear(cfg.route_context, cfg.experts, bias=False)

    def initial_context(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.route_context, device=device)

    def update_context(self, probs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.history_cell is not None:
            return self.history_cell(probs, context)
        assert self.route_in is not None and self.alpha_logit is not None
        route_desc = self.route_in(probs)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * context + (1.0 - alpha) * route_desc

    def forward(self, x: torch.Tensor, start_layer: int = 0, context: torch.Tensor | None = None) -> dict[str, Any]:
        h = x
        batch = x.shape[0]
        if context is None:
            context = self.initial_context(batch, x.device)
        hidden_states = []
        route_probs = []
        topk_indices = []
        confidences = []
        for layer in range(start_layer, self.cfg.layers):
            z = h + torch.tanh(self.target.attn[layer](h))
            logits = self.target.routers[layer](z) + self.context_to_logits(context)
            probs = F.softmax(logits, dim=-1)
            top_prob, top_idx = torch.topk(probs, k=self.cfg.top_k, dim=-1)
            gate = top_prob / top_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            h = self.lore[layer](z, top_idx, gate)
            context = self.update_context(probs, context)
            hidden_states.append(h)
            route_probs.append(probs)
            topk_indices.append(top_idx)
            confidences.append(top_prob.sum(dim=-1))
        return {
            "hidden_states": hidden_states,
            "route_probs": route_probs,
            "topk_indices": topk_indices,
            "confidences": confidences,
            "context": context,
        }


def draft_losses(
    draft_out: dict[str, list[torch.Tensor]],
    target_out: dict[str, list[torch.Tensor]],
    align_lambda: float,
) -> dict[str, torch.Tensor]:
    route_loss = torch.zeros((), device=target_out["route_probs"][0].device)
    align_loss = torch.zeros_like(route_loss)
    for pred, target in zip(draft_out["route_probs"], target_out["route_probs"]):
        route_loss = route_loss + F.kl_div(pred.clamp_min(1e-8).log(), target.detach(), reduction="batchmean")
    for pred_h, target_h in zip(draft_out["hidden_states"], target_out["hidden_states"]):
        align_loss = align_loss + F.mse_loss(pred_h, target_h.detach())
    route_loss = route_loss / len(target_out["route_probs"])
    align_loss = align_loss / len(target_out["hidden_states"])
    return {
        "route_loss": route_loss,
        "align_loss": align_loss,
        "loss": route_loss + align_lambda * align_loss,
    }


@torch.no_grad()
def routing_metrics(
    draft_out: dict[str, list[torch.Tensor]],
    target_out: dict[str, list[torch.Tensor]],
    top_k: int,
) -> dict[str, float]:
    total_slots = 0
    hit_slots = 0
    exact_sets = 0
    total_sets = 0
    conf_sum = 0.0
    conf_count = 0
    depth_conf: dict[str, list[float]] = {}
    for layer, (pred, target, conf) in enumerate(
        zip(draft_out["topk_indices"], target_out["topk_indices"], draft_out["confidences"])
    ):
        pred_sets = topk_sets(pred)
        target_sets = topk_sets(target)
        for ps, ts in zip(pred_sets, target_sets):
            hit_slots += len(ps.intersection(ts))
            total_slots += top_k
            exact_sets += int(ps == ts)
            total_sets += 1
        vals = conf.detach().float().cpu().tolist()
        depth_conf[str(layer)] = vals
        conf_sum += float(conf.sum().item())
        conf_count += int(conf.numel())
    return {
        "slot_hit_rate": hit_slots / max(1, total_slots),
        "fallback_slot_rate": 1.0 - hit_slots / max(1, total_slots),
        "exact_set_match_rate": exact_sets / max(1, total_sets),
        "mean_confidence": conf_sum / max(1, conf_count),
    }


def checkpoint_payload(
    cfg: DraftConfig,
    target: FrozenTargetMoE,
    draft: SPICEDraftModel,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "config": asdict(cfg),
        "target_state": target.state_dict(),
        "draft_state": draft.state_dict(),
        "extra": extra or {},
    }


def load_checkpoint(path: str, device: torch.device) -> tuple[DraftConfig, FrozenTargetMoE, SPICEDraftModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = DraftConfig(**payload["config"])
    target = FrozenTargetMoE(cfg).to(device)
    target.load_state_dict(payload["target_state"])
    draft = SPICEDraftModel(cfg, target).to(device)
    draft.load_state_dict(payload["draft_state"])
    return cfg, target, draft, payload.get("extra", {})
