from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


@dataclass
class RealLoREConfig:
    model_id: str
    layers: int
    experts: int
    top_k: int
    router_dim: int
    hidden: int
    rank: int = 64
    route_context: int = 64
    history: str = "gru"
    teacher_force_context: bool = False


class FrozenRouter(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone().float())
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LoRETransition(nn.Module):
    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class RealLoREDraft(nn.Module):
    def __init__(
        self,
        cfg: RealLoREConfig,
        routers: list[FrozenRouter] | None,
    ):
        super().__init__()
        self.cfg = cfg
        use_frozen = routers is not None and len(routers) > 0 and int(routers[0].weight.shape[0]) == cfg.router_dim
        if use_frozen:
            self.routers = nn.ModuleList(routers)
            self.router_mode = "frozen"
        else:
            self.routers = nn.ModuleList([nn.Linear(cfg.hidden, cfg.router_dim, bias=False) for _ in range(cfg.layers)])
            self.router_mode = "trainable"
        self.transitions = nn.ModuleList([LoRETransition(cfg.hidden, cfg.rank) for _ in range(cfg.layers)])
        if cfg.history == "gru":
            self.history_cell = nn.GRUCell(cfg.router_dim, cfg.route_context)
            self.route_in = None
            self.alpha_logit = None
        elif cfg.history == "ema":
            self.history_cell = None
            self.route_in = nn.Linear(cfg.router_dim, cfg.route_context, bias=False)
            self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(f"unknown history mode: {cfg.history}")
        self.context_to_logits = nn.Linear(cfg.route_context, cfg.router_dim, bias=False)

    def initial_context(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.route_context, device=device)

    def update_context(self, probs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.history_cell is not None:
            return self.history_cell(probs, context)
        assert self.route_in is not None and self.alpha_logit is not None
        routed = self.route_in(probs)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * context + (1.0 - alpha) * routed

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        teacher_force_context: bool = False,
    ) -> dict[str, list[torch.Tensor]]:
        h = hidden_states[0]
        context = self.initial_context(h.shape[0], h.device)
        pred_hidden: list[torch.Tensor] = []
        pred_probs: list[torch.Tensor] = []
        steps = min(self.cfg.layers, len(hidden_states) - 1)
        if steps <= 0:
            raise ValueError("RealLoREDraft.forward needs at least two hidden states")
        for layer in range(steps):
            z = h + self.transitions[layer](h)
            logits = self.routers[layer](z) + self.context_to_logits(context)
            probs = F.softmax(logits, dim=-1)
            pred_hidden.append(z)
            pred_probs.append(probs)
            if teacher_force_context:
                context = self.update_context(
                    F.softmax(self.routers[layer](hidden_states[layer]), dim=-1).detach(),
                    context,
                )
            else:
                context = self.update_context(probs, context)
            h = z
        return {"hidden_states": pred_hidden, "route_probs": pred_probs}


def iter_trace_files(trace_dir: str) -> list[Path]:
    paths = sorted(Path(trace_dir).glob("trace_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no trace_*.pt files found in {trace_dir}")
    return paths


def _first_present(obj: dict, keys: list[str]):
    for key in keys:
        if key in obj and obj[key] is not None:
            return key, obj[key]
    return None, None


def _probs_from_topk(topk: torch.Tensor, experts: int) -> torch.Tensor:
    flat = topk.reshape(-1, topk.shape[-1]).long()
    target = torch.zeros(flat.shape[0], experts, dtype=torch.float32)
    target.scatter_(1, flat, 1.0 / float(flat.shape[1]))
    return target


def load_trace(path: Path) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    obj = torch.load(path, map_location="cpu")
    hidden_key, hidden_states = _first_present(obj, ["moe_hidden", "moe_hidden_states", "hidden_states"])
    _, router_probs = _first_present(obj, ["route_probs", "router_probs", "moe_route_probs"])
    _, route_topk = _first_present(obj, ["topk", "route_topk", "router_topk", "topk_idx"])
    router_names = obj.get("router_module_names") or []
    if hidden_states is None:
        raise ValueError(f"{path} does not contain hidden_states/moe_hidden; recollect traces with hidden states enabled")

    def flatten_tokens(t: torch.Tensor) -> torch.Tensor:
        if t.ndim <= 2:
            return t.float()
        return t.reshape(-1, t.shape[-1]).float()

    hs = [flatten_tokens(t) for t in hidden_states]
    if router_probs is not None:
        rp = [flatten_tokens(t) for t in router_probs]
    elif route_topk is not None:
        experts = int(obj.get("num_experts") or obj.get("experts") or 0)
        if experts <= 0:
            experts = int(max(int(t.max().item()) for t in route_topk)) + 1
        rp = [_probs_from_topk(t, experts) for t in route_topk]
    else:
        raise ValueError(f"{path} does not contain route_probs/router_probs or topk targets")

    offset = 0
    if hidden_key in {"moe_hidden", "moe_hidden_states"} or len(hs) == len(rp):
        offset = 0
    elif router_names:
        first = str(router_names[0])
        m = re.search(r"layers\.(\d+)", first)
        if m:
            offset = int(m.group(1))
    return hs, rp, offset


def slice_trace_window(
    hidden_states: list[torch.Tensor],
    target_probs: list[torch.Tensor],
    offset: int,
    max_pairs: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if len(hidden_states) < 2:
        raise ValueError("trace needs at least two hidden tensors for hidden-align training")
    start = max(0, min(int(offset), len(hidden_states) - 2))
    available_pairs = len(hidden_states) - start - 1
    pairs = min(len(target_probs), available_pairs)
    if max_pairs is not None:
        pairs = min(pairs, int(max_pairs))
    if pairs <= 0:
        raise ValueError(
            f"empty trace window: hidden={len(hidden_states)} probs={len(target_probs)} offset={offset}"
        )
    return hidden_states[start : start + pairs + 1], target_probs[:pairs]


def extract_layer_idx(name: str) -> int:
    m = re.search(r"layers\.(\d+)", name)
    if not m:
        raise ValueError(f"cannot extract layer index from {name}")
    return int(m.group(1))


def load_router_weights(model_id: str) -> list[torch.Tensor]:
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - fallback path
        raise RuntimeError("safetensors is required to extract frozen router weights") from exc

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    root = Path(model_id)
    shards = sorted(list(root.rglob("*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no .safetensors shards found under {model_id}")

    candidates: list[tuple[int, str, torch.Tensor]] = []
    key_pattern = re.compile(r"(?:^|\.)(?:mlp\.)?(?:gate|router)\.weight$")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if "shared_expert_gate" in key:
                    continue
                if not key_pattern.search(key):
                    continue
                layer_idx = extract_layer_idx(key)
                candidates.append((layer_idx, key, handle.get_tensor(key)))

    if not candidates:
        raise RuntimeError(f"no router weight tensors matched under {model_id}")

    candidates.sort(key=lambda x: (x[0], x[1]))
    routers = [tensor.float() for _, _, tensor in candidates]
    if len(routers) != int(getattr(config, "num_hidden_layers", len(routers))):
        print(
            {
                "warning": "router layer count mismatch",
                "config_layers": int(getattr(config, "num_hidden_layers", -1)),
                "matched_routers": len(routers),
            }
        )
    return routers


def topk_sets(topk: torch.Tensor) -> list[set[int]]:
    return [set(row.tolist()) for row in topk.detach().cpu()]


def routing_metrics(pred_probs: list[torch.Tensor], target_probs: list[torch.Tensor], top_k: int) -> dict[str, float]:
    total_slots = 0
    hit_slots = 0
    exact_sets = 0
    total_sets = 0
    conf_sum = 0.0
    conf_count = 0
    for pred, target in zip(pred_probs, target_probs):
        pred_top = torch.topk(pred, k=top_k, dim=-1).indices
        target_top = torch.topk(target, k=top_k, dim=-1).indices
        pred_sets = topk_sets(pred_top)
        target_sets = topk_sets(target_top)
        for ps, ts in zip(pred_sets, target_sets):
            hit_slots += len(ps & ts)
            total_slots += top_k
            exact_sets += int(ps == ts)
            total_sets += 1
        conf = pred.topk(k=top_k, dim=-1).values.sum(dim=-1)
        conf_sum += float(conf.sum().item())
        conf_count += int(conf.numel())
    return {
        "slot_hit_rate": hit_slots / max(1, total_slots),
        "fallback_slot_rate": 1.0 - hit_slots / max(1, total_slots),
        "exact_set_match_rate": exact_sets / max(1, total_sets),
        "mean_confidence": conf_sum / max(1, conf_count),
    }


def draft_losses(
    draft_out: dict[str, list[torch.Tensor]],
    target_hidden: list[torch.Tensor],
    target_probs: list[torch.Tensor],
    align_lambda: float,
) -> dict[str, torch.Tensor]:
    route_loss = torch.zeros((), device=target_hidden[0].device)
    align_loss = torch.zeros_like(route_loss)
    for pred, target in zip(draft_out["route_probs"], target_probs):
        route_loss = route_loss + F.kl_div(pred.clamp_min(1e-8).log(), target.detach(), reduction="batchmean")
    for pred_h, target_h in zip(draft_out["hidden_states"], target_hidden[1:]):
        align_loss = align_loss + F.mse_loss(pred_h, target_h.detach())
    route_loss = route_loss / len(target_probs)
    align_loss = align_loss / len(target_hidden[1:])
    return {
        "route_loss": route_loss,
        "align_loss": align_loss,
        "loss": route_loss + align_lambda * align_loss,
    }


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * float(step + 1) / max(1, warmup)
    if total_steps <= warmup:
        return base_lr
    progress = float(step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(
    model: RealLoREDraft,
    trace_files: list[Path],
    align_lambda: float,
    teacher_force_context: bool,
) -> dict[str, float]:
    model.eval()
    route_loss_sum = 0.0
    align_loss_sum = 0.0
    slot_hit = 0.0
    fallback = 0.0
    exact = 0.0
    conf = 0.0
    count = 0
    with torch.no_grad():
        for path in trace_files:
            hidden_states, target_probs, offset = load_trace(path)
            hidden_states = [t.to(next(model.parameters()).device) for t in hidden_states]
            target_probs = [t.to(next(model.parameters()).device) for t in target_probs]
            hidden_slice, probs_slice = slice_trace_window(
                hidden_states, target_probs, offset, max_pairs=model.cfg.layers
            )
            draft_out = model(hidden_slice, teacher_force_context=teacher_force_context)
            losses = draft_losses(draft_out, hidden_slice, probs_slice, align_lambda=align_lambda)
            metrics = routing_metrics(draft_out["route_probs"], probs_slice, model.cfg.top_k)
            route_loss_sum += float(losses["route_loss"].item())
            align_loss_sum += float(losses["align_loss"].item())
            slot_hit += metrics["slot_hit_rate"]
            fallback += metrics["fallback_slot_rate"]
            exact += metrics["exact_set_match_rate"]
            conf += metrics["mean_confidence"]
            count += 1
    denom = max(1, count)
    return {
        "route_kl": route_loss_sum / denom,
        "align_mse": align_loss_sum / denom,
        "slot_hit_rate": slot_hit / denom,
        "fallback_slot_rate": fallback / denom,
        "exact_set_match_rate": exact / denom,
        "mean_confidence": conf / denom,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train real-model SPICE LoRE from hidden/router traces")
    p.add_argument("--model_id", required=True)
    p.add_argument("--trace_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--route_context", type=int, default=64)
    p.add_argument("--history", choices=["gru", "ema"], default="gru")
    p.add_argument("--teacher_force_context", action="store_true")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch_traces", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--align_lambda", type=float, default=0.2)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_name", type=str, default="real_lore.pt")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True, local_files_only=True)
    routers = load_router_weights(args.model_id)
    hidden = int(getattr(config, "hidden_size"))
    layers = int(getattr(config, "num_hidden_layers", len(routers)))
    experts = int(getattr(config, "num_experts", routers[0].shape[0]))
    top_k = int(getattr(config, "num_experts_per_tok", min(4, experts)))
    if len(routers) != layers:
        layers = min(layers, len(routers))
        routers = routers[:layers]

    trace_files = iter_trace_files(args.trace_dir)
    rng = random.Random(args.seed)
    rng.shuffle(trace_files)
    n_val = max(1, int(round(len(trace_files) * args.val_fraction)))
    val_files = trace_files[:n_val]
    train_files = trace_files[n_val:] if len(trace_files) > n_val else trace_files
    if not train_files:
        train_files = val_files

    sample_hidden, sample_probs, sample_offset = load_trace(trace_files[0])
    router_dim = int(sample_probs[0].shape[-1])
    if len(routers) != layers:
        layers = min(layers, len(routers))
        routers = routers[:layers]
    if len(sample_probs) > 0 and int(sample_probs[0].shape[-1]) != int(routers[0].shape[0]):
        routers_for_model: list[FrozenRouter] | None = None
    else:
        routers_for_model = [FrozenRouter(w) for w in routers]

    cfg = RealLoREConfig(
        model_id=args.model_id,
        layers=layers,
        experts=experts,
        top_k=top_k,
        router_dim=router_dim,
        hidden=hidden,
        rank=args.rank,
        route_context=args.route_context,
        history=args.history,
        teacher_force_context=args.teacher_force_context,
    )
    model = RealLoREDraft(cfg, routers_for_model).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial_eval = evaluate(model, val_files, args.align_lambda, args.teacher_force_context)
    logs: list[dict[str, float]] = []
    model.train()
    for step in range(args.steps):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_paths = [rng.choice(train_files) for _ in range(args.batch_traces)]
        total_loss = torch.zeros((), device=device)
        total_route = torch.zeros_like(total_loss)
        total_align = torch.zeros_like(total_loss)
        for path in batch_paths:
            hidden_states, target_probs, offset = load_trace(path)
            hidden_states = [t.to(device) for t in hidden_states]
            target_probs = [t.to(device) for t in target_probs]
            hidden_slice, probs_slice = slice_trace_window(hidden_states, target_probs, offset, max_pairs=model.cfg.layers)
            draft_out = model(hidden_slice, teacher_force_context=args.teacher_force_context)
            losses = draft_losses(draft_out, hidden_slice, probs_slice, args.align_lambda)
            total_loss = total_loss + losses["loss"]
            total_route = total_route + losses["route_loss"]
            total_align = total_align + losses["align_loss"]

        total_loss = total_loss / max(1, len(batch_paths))
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (step + 1) % args.log_every == 0 or step == args.steps - 1:
            model.eval()
            eval_metrics = evaluate(model, val_files, args.align_lambda, args.teacher_force_context)
            logs.append(
                {
                    "step": step + 1,
                    "lr": lr,
                    "loss": float(total_loss.item()),
                    "route_kl": float(total_route.item() / max(1, len(batch_paths))),
                    "align_mse": float(total_align.item() / max(1, len(batch_paths))),
                    **eval_metrics,
                }
            )
            print(logs[-1])
            model.train()

    final_eval = evaluate(model, val_files, args.align_lambda, args.teacher_force_context)

    ckpt_path = out_dir / args.checkpoint_name
    torch.save(
        {
            "format": "spice_real_lore_v1",
            "config": asdict(cfg),
            "router_state": model.routers.state_dict(),
            "draft_state": model.state_dict(),
            "train_files": [str(p) for p in train_files],
            "val_files": [str(p) for p in val_files],
            "extra": {
                "initial_eval": initial_eval,
                "final_eval": final_eval,
                "train_steps": args.steps,
                "align_lambda": args.align_lambda,
                "teacher_force_context": args.teacher_force_context,
            },
        },
        ckpt_path,
    )

    result = {
        "experiment": "real_lore_train",
        "model_id": args.model_id,
        "trace_dir": args.trace_dir,
        "device": str(device),
        "config": asdict(cfg),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "logs": logs,
        "checkpoint": str(ckpt_path),
    }
    (out_dir / "real_lore_train.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
