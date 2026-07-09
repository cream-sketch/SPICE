from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RealLoREConfig:
    model_id: str
    layers: int
    experts: int
    top_k: int
    hidden: int
    router_dim: int = 0
    rank: int = 64
    route_context: int = 64
    history: str = "gru"
    teacher_force_context: bool = False


class LoRETransition(nn.Module):
    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class RealLoREDraft(nn.Module):
    def __init__(self, cfg: RealLoREConfig):
        super().__init__()
        self.cfg = cfg
        self.routers = nn.ModuleList([nn.Linear(cfg.hidden, cfg.router_dim, bias=False) for _ in range(cfg.layers)])
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

    def rollout(self, hidden_states: list[torch.Tensor], offset: int, max_horizon: int) -> torch.Tensor:
        batch = hidden_states[0].shape[0]
        device = hidden_states[0].device
        pred = torch.full((self.cfg.layers, max_horizon, batch, self.cfg.top_k), -1, dtype=torch.long, device=device)
        for anchor in range(self.cfg.layers):
            h = hidden_states[offset + anchor]
            context = self.initial_context(batch, device)
            for d in range(max_horizon):
                layer_idx = anchor + d
                if layer_idx >= self.cfg.layers:
                    break
                z = h + self.transitions[layer_idx](h)
                logits = self.routers[layer_idx](z) + self.context_to_logits(context)
                probs = F.softmax(logits, dim=-1)
                pred[anchor, d] = torch.topk(probs, k=self.cfg.top_k, dim=-1).indices
                context = self.update_context(probs, context)
                h = z
        return pred


def flatten_step_hidden(step_hidden: list[torch.Tensor]) -> list[torch.Tensor]:
    hs = []
    for t in step_hidden:
        if t.ndim == 3:
            hs.append(t[:, -1, :].float())
        elif t.ndim == 2:
            hs.append(t.float())
        else:
            hs.append(t.float())
    return hs


def load_checkpoint(path: str, device: torch.device) -> RealLoREDraft:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(payload["config"])
    if "router_dim" not in cfg_dict or not cfg_dict["router_dim"]:
        draft_state = payload["draft_state"]
        router_keys = sorted(k for k in draft_state.keys() if k.startswith("routers.") and k.endswith(".weight"))
        if not router_keys:
            raise ValueError("could not infer router_dim from checkpoint")
        cfg_dict["router_dim"] = int(draft_state[router_keys[0]].shape[0])
    cfg = RealLoREConfig(**cfg_dict)
    model = RealLoREDraft(cfg).to(device)
    model.load_state_dict(payload["draft_state"], strict=True)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Build forecast dump from decode traces using trained real-LoRE")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_horizon", type=int, default=6)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)

    root = Path(args.decode_dir)
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []

    for name in man["files"]:
        d = torch.load(root / name, map_location="cpu", weights_only=False)
        steps = d["steps"]
        if not steps:
            continue
        L = len(steps[0]["topk"])
        K = len(steps[0]["topk"][0])
        T = len(steps)
        true_top = torch.full((L, T, K), -1, dtype=torch.long)
        fcast = torch.full((L, args.max_horizon, T, K), -1, dtype=torch.long)
        for ti, step in enumerate(steps):
            true_top[:, ti] = torch.tensor(step["topk"], dtype=torch.long)
            if "hidden_states" not in step:
                continue
            hidden_states = flatten_step_hidden(step["hidden_states"])
            offset = max(0, len(hidden_states) - L - 1)
            hidden_states = [hs.to(device) for hs in hidden_states]
            with torch.no_grad():
                pred = model.rollout(hidden_states, offset=offset, max_horizon=args.max_horizon)
            fcast[:, :, ti] = pred.squeeze(2).cpu()
        torch.save(
            {
                "true_top": true_top,
                "fcast": fcast,
                "num_layers": L,
                "top_k": K,
                "max_horizon": args.max_horizon,
            },
            out / f"fc_{Path(name).stem.split('_')[-1]}.pt",
        )
        files.append(f"fc_{Path(name).stem.split('_')[-1]}.pt")

    out_manifest = {
        "files": files,
        "top_k": int(man.get("top_k", getattr(model.cfg, "top_k", 0))),
        "experts": int(man.get("experts", getattr(model.cfg, "experts", 0))),
        "num_layers": int(man.get("num_layers", getattr(model.cfg, "layers", 0))),
    }
    (out / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print({"forecast_files": len(files), "out_dir": str(out)})


if __name__ == "__main__":
    main()
