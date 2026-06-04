from __future__ import annotations

import sys, pathlib  # bootstrap: resolve sibling core modules regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
from pathlib import Path

import torch

from common import build_arg_parser, device_from_arg, ensure_dir, set_seed, write_json
from draft_model import (
    DraftConfig,
    FrozenTargetMoE,
    SPICEDraftModel,
    checkpoint_payload,
    count_total,
    count_trainable,
    draft_losses,
    routing_metrics,
)


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / max(1, warmup)
    progress = float(step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def sample_batch(batch: int, hidden: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch, hidden, device=device)


@torch.no_grad()
def evaluate(target: FrozenTargetMoE, draft: SPICEDraftModel, batches: int, batch: int, hidden: int, device: torch.device) -> dict:
    target.eval()
    draft.eval()
    route_loss_sum = 0.0
    align_loss_sum = 0.0
    slot_hit = 0.0
    fallback = 0.0
    exact = 0.0
    conf = 0.0
    for _ in range(batches):
        x = sample_batch(batch, hidden, device)
        target_out = target(x)
        draft_out = draft(x)
        losses = draft_losses(draft_out, target_out, align_lambda=0.0)
        metrics = routing_metrics(draft_out, target_out, target.cfg.top_k)
        route_loss_sum += float(losses["route_loss"].item())
        align_loss_sum += float(losses["align_loss"].item())
        slot_hit += metrics["slot_hit_rate"]
        fallback += metrics["fallback_slot_rate"]
        exact += metrics["exact_set_match_rate"]
        conf += metrics["mean_confidence"]
    denom = max(1, batches)
    return {
        "route_kl": route_loss_sum / denom,
        "align_mse": align_loss_sum / denom,
        "slot_hit_rate": slot_hit / denom,
        "fallback_slot_rate": fallback / denom,
        "exact_set_match_rate": exact / denom,
        "mean_confidence": conf / denom,
    }


def main() -> None:
    parser = build_arg_parser("Train SPICE LoRE draft routing predictor")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--expert_hidden", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--route_context", type=int, default=64)
    parser.add_argument("--history", choices=["gru", "ema"], default="gru")
    parser.add_argument("--no_shared_down", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--align_lambda", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--checkpoint_name", type=str, default="spice_draft.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = device_from_arg(args.gpu)
    cfg = DraftConfig(
        layers=args.layers,
        experts=args.experts,
        top_k=args.top_k,
        hidden=args.hidden,
        expert_hidden=args.expert_hidden,
        rank=args.rank,
        route_context=args.route_context,
        shared_down=not args.no_shared_down,
        history=args.history,
    )
    target = FrozenTargetMoE(cfg).to(device)
    draft = SPICEDraftModel(cfg, target).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    logs = []
    initial_eval = evaluate(target, draft, args.eval_batches, args.batch, args.hidden, device)
    draft.train()
    for step in range(args.steps):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x = sample_batch(args.batch, args.hidden, device)
        with torch.no_grad():
            target_out = target(x)
        draft_out = draft(x)
        losses = draft_losses(draft_out, target_out, args.align_lambda)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), max_norm=1.0)
        optimizer.step()
        if (step + 1) % args.log_every == 0 or step == args.steps - 1:
            metrics = routing_metrics(draft_out, target_out, args.top_k)
            logs.append(
                {
                    "step": step + 1,
                    "lr": lr,
                    "loss": float(losses["loss"].item()),
                    "route_kl": float(losses["route_loss"].item()),
                    "align_mse": float(losses["align_loss"].item()),
                    **metrics,
                }
            )
            print(logs[-1])
    final_eval = evaluate(target, draft, args.eval_batches, args.batch, args.hidden, device)

    ckpt_path = Path(out_dir) / args.checkpoint_name
    torch.save(
        checkpoint_payload(
            cfg,
            target,
            draft,
            extra={
                "initial_eval": initial_eval,
                "final_eval": final_eval,
                "train_steps": args.steps,
                "align_lambda": args.align_lambda,
            },
        ),
        ckpt_path,
    )
    result = {
        "experiment": "spice_draft_train",
        "device": str(device),
        "config": cfg.__dict__,
        "trainable_draft_params": count_trainable(draft),
        "target_params": count_total(target),
        "draft_total_params": count_total(draft) - count_total(target),
        "param_overhead_vs_target": count_trainable(draft) / max(1, count_total(target)),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "logs": logs,
        "checkpoint": str(ckpt_path),
    }
    write_json(Path(out_dir) / "draft_train.json", result)
    print(result)


if __name__ == "__main__":
    main()
