"""Train a SPICE LoRE draft model for Qwen3-30B-A3B on collected traces.

Faithful port of the paper's draft design to the real target model:
  - frozen target attention + norms + router gates (loaded from the checkpoint
    without materializing expert weights: ~1.5 GB total);
  - trainable LoRE expert surrogates  E~_i(z) = z + B_i (A z)  with a shared
    per-layer down-projection A (paper Eq. 7) and per-expert up-projections;
  - routing-history context: EMA of routing descriptors applied as logit bias;
  - loss = mean-layer KL(target router || draft router) + lambda * hidden MSE.

Training data: traces from collect_hf_moe_traces.py (router logits + hidden
states per layer). Each step anchors the draft at a random layer l0 with the
*true* hidden state, rolls forward `depth` draft layers over the full sequence
(teacher-forced at the anchor only), and supervises every predicted layer.

Output checkpoint is consumed by qwen3_offload_bench.py --policy spice.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
)


# --------------------------------------------------------------------------- #
# Frozen backbone loading (attention + norms + router gates only)
# --------------------------------------------------------------------------- #

NONEXPERT_SUBKEYS = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate.weight",
)


def load_nonexpert_weights(model_dir: str, num_layers: int) -> dict[str, torch.Tensor]:
    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    wanted = {}
    for li in range(num_layers):
        for sub in NONEXPERT_SUBKEYS:
            wanted[f"model.layers.{li}.{sub}"] = weight_map[f"model.layers.{li}.{sub}"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in wanted.items():
        by_shard.setdefault(shard, []).append(name)
    out = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(Path(model_dir) / shard), framework="pt", device="cpu") as f:
            for name in names:
                out[name] = f.get_tensor(name)
    return out


class FrozenDraftLayer(torch.nn.Module):
    """One draft decoder layer: frozen attention/norms/router + LoRE surrogate."""

    def __init__(self, cfg, layer_idx: int, rank: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = Qwen3MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Qwen3MoeAttention(cfg, layer_idx)
        self.router_weight = torch.nn.Parameter(
            torch.empty(cfg.num_experts, cfg.hidden_size), requires_grad=False
        )
        # LoRE: shared down-projection A [rank, hidden]; per-expert up B [E, hidden, rank]
        self.lore_A = torch.nn.Parameter(torch.randn(rank, cfg.hidden_size) * 0.02)
        self.lore_B = torch.nn.Parameter(torch.zeros(cfg.num_experts, cfg.hidden_size, rank))
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts

    def forward(self, h, position_embeddings, attention_mask, route_ctx, Wc):
        residual = h
        z = self.input_layernorm(h)
        attn_out, _ = self.self_attn(
            hidden_states=z, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=None,
        )
        h = residual + attn_out
        residual = h
        z = self.post_attention_layernorm(h)

        logits = F.linear(z, self.router_weight)
        if route_ctx is not None:
            logits = logits + F.linear(route_ctx, Wc.t())  # [*, d_r] -> [*, E] bias
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        topw, topi = torch.topk(probs, self.top_k, dim=-1)
        topw = (topw / topw.sum(dim=-1, keepdim=True)).to(z.dtype)

        # LoRE mixture: y = sum_k w_k * (z + B_k (A z)); trainables kept in fp32
        z32 = z.float()
        Az = F.linear(z32, self.lore_A)                      # [B, T, r]
        Bsel = self.lore_B[topi]                             # [B, T, K, hidden, r]
        y = torch.einsum("btkhr,btr->btkh", Bsel, Az)        # [B, T, K, hidden]
        moe_out = z + (topw.float().unsqueeze(-1) * y).sum(dim=2).to(z.dtype)
        return residual + moe_out, logits, probs


class Qwen3Draft(torch.nn.Module):
    def __init__(self, cfg, rank: int, d_route: int):
        super().__init__()
        self.cfg = cfg
        self.layers = torch.nn.ModuleList(
            FrozenDraftLayer(cfg, li, rank) for li in range(cfg.num_hidden_layers)
        )
        self.rotary = Qwen3MoeRotaryEmbedding(cfg)
        self.Wr = torch.nn.Parameter(torch.randn(d_route, cfg.num_experts) * 0.02)
        self.Wc = torch.nn.Parameter(torch.zeros(cfg.num_experts, d_route).t().contiguous())
        self.alpha_logit = torch.nn.Parameter(torch.tensor(1.5))  # sigmoid -> ~0.82

    def load_frozen(self, weights: dict[str, torch.Tensor], dtype):
        for li, layer in enumerate(self.layers):
            p = f"model.layers.{li}."
            layer.self_attn.q_proj.weight.data = weights[p + "self_attn.q_proj.weight"].to(dtype)
            layer.self_attn.k_proj.weight.data = weights[p + "self_attn.k_proj.weight"].to(dtype)
            layer.self_attn.v_proj.weight.data = weights[p + "self_attn.v_proj.weight"].to(dtype)
            layer.self_attn.o_proj.weight.data = weights[p + "self_attn.o_proj.weight"].to(dtype)
            layer.self_attn.q_norm.weight.data = weights[p + "self_attn.q_norm.weight"].to(dtype)
            layer.self_attn.k_norm.weight.data = weights[p + "self_attn.k_norm.weight"].to(dtype)
            layer.input_layernorm.weight.data = weights[p + "input_layernorm.weight"].to(dtype)
            layer.post_attention_layernorm.weight.data = weights[p + "post_attention_layernorm.weight"].to(dtype)
            layer.router_weight.data = weights[p + "mlp.gate.weight"].to(dtype)
        for name, param in self.named_parameters():
            param.requires_grad = ("lore_" in name) or name in {"Wr", "Wc", "alpha_logit"}

    def rollout(self, h_anchor, anchor_layer: int, depth: int, target_probs, attention_mask):
        """Roll draft layers [anchor+1 .. anchor+depth] from true h_anchor.

        target_probs: list of [B, T, E] float tensors indexed by absolute layer.
        Returns (kl_per_depth, pred_hidden, pred_probs_list).
        """
        B, T, _ = h_anchor.shape
        pos = torch.arange(T, device=h_anchor.device).unsqueeze(0)
        pe = self.rotary(h_anchor, pos)
        if attention_mask is None:
            attention_mask = torch.full(
                (T, T), torch.finfo(h_anchor.dtype).min, device=h_anchor.device,
                dtype=h_anchor.dtype,
            ).triu(diagonal=1)[None, None]  # causal [1, 1, T, T]
        alpha = torch.sigmoid(self.alpha_logit)
        ctx = None
        h = h_anchor
        kls, preds = [], []
        for d in range(1, depth + 1):
            layer = self.layers[anchor_layer + d]
            h, logits, probs = layer(h, pe, attention_mask, ctx, self.Wc)
            tgt = target_probs[anchor_layer + d]  # [B, T, E]
            kl = (tgt * (tgt.clamp_min(1e-9).log() - probs.clamp_min(1e-9).log())).sum(-1).mean()
            kls.append(kl)
            preds.append(probs)
            desc = F.linear(probs.to(self.Wr.dtype), self.Wr)  # [B, T, d_r]
            ctx = desc if ctx is None else alpha * ctx + (1 - alpha) * desc
        return kls, h, preds


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--d_route", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--align_lambda", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--holdout", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    cfg._attn_implementation = "eager"
    print("[draft] loading frozen non-expert weights", flush=True)
    weights = load_nonexpert_weights(args.model, cfg.num_hidden_layers)
    draft = Qwen3Draft(cfg, args.rank, args.d_route)
    draft.load_frozen(weights, torch.bfloat16)
    draft = draft.to(device=device, dtype=torch.bfloat16)
    # keep trainables in fp32 for stable optimization
    for name, p in draft.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    del weights

    files = sorted(glob.glob(str(Path(args.trace_dir) / "trace_*.pt")))
    assert len(files) > args.holdout, "not enough traces"
    train_files, eval_files = files[args.holdout:], files[: args.holdout]
    print(f"[draft] {len(train_files)} train / {len(eval_files)} eval traces", flush=True)

    def load_trace(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        hs = [t.to(device=device, dtype=torch.bfloat16) for t in d["hidden_states"]]
        probs = [t.to(device=device, dtype=torch.float) for t in d["router_probs"]]
        # router capture is [T, E]; make [1, T, E]
        probs = [p.unsqueeze(0) if p.dim() == 2 else p for p in probs]
        return hs, probs

    trainables = [p for p in draft.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainables)
    print(f"[draft] trainable params: {n_train/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (
            1 + torch.cos(torch.tensor(min(1.0, s / args.steps) * 3.14159)).item()
        ),
    )

    cache: dict[str, tuple] = {}

    def get_trace(path):
        if path not in cache:
            if len(cache) >= 6:  # keep GPU memory bounded
                cache.pop(next(iter(cache)))
            cache[path] = load_trace(path)
        return cache[path]

    @torch.no_grad()
    def evaluate():
        draft.eval()
        agg = {"kl": 0.0, "slot_hit": 0.0, "n": 0}
        for path in eval_files:
            hs, probs = get_trace(path)
            L = cfg.num_hidden_layers
            for anchor in range(0, L - args.depth, max(1, (L - args.depth) // 6)):
                kls, _, preds = draft.rollout(hs[anchor], anchor, args.depth, probs, None)
                agg["kl"] += sum(k.item() for k in kls) / len(kls)
                # slot hit: draft top-k vs target top-k at depth 1
                tgt_top = probs[anchor + 1].topk(cfg.num_experts_per_tok, dim=-1).indices
                pred_top = preds[0].topk(cfg.num_experts_per_tok, dim=-1).indices
                hit = 0.0
                for b in range(tgt_top.shape[0]):
                    for t in range(tgt_top.shape[1]):
                        s1 = set(tgt_top[b, t].tolist()); s2 = set(pred_top[b, t].tolist())
                        hit += len(s1 & s2) / len(s1)
                agg["slot_hit"] += hit / (tgt_top.shape[0] * tgt_top.shape[1])
                agg["n"] += 1
        draft.train()
        return {"eval_kl": agg["kl"] / agg["n"], "eval_slot_hit_d1": agg["slot_hit"] / agg["n"]}

    print("[draft] training", flush=True)
    logs = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        path = random.choice(train_files)
        hs, probs = get_trace(path)
        L = cfg.num_hidden_layers
        anchor = random.randint(0, L - args.depth - 1)
        kls, h_pred, _ = draft.rollout(hs[anchor], anchor, args.depth, probs, None)
        kl_loss = torch.stack(kls).mean()
        align = F.mse_loss(h_pred.float(), hs[anchor + args.depth].float())
        loss = kl_loss + args.align_lambda * align
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate()
            rec = {"step": step, "loss": loss.item(), "route_kl": kl_loss.item(),
                   "align_mse": align.item(), **ev, "elapsed_s": time.time() - t0}
            logs.append(rec)
            print(rec, flush=True)

    ckpt = {
        "config": {"rank": args.rank, "d_route": args.d_route, "depth": args.depth,
                   "model": args.model, "seed": args.seed},
        "state_dict": {k: v for k, v in draft.state_dict().items()
                       if "lore_" in k or k in {"Wr", "Wc", "alpha_logit"}},
        "logs": logs,
    }
    torch.save(ckpt, out_dir / "qwen3_spice_draft.pt")
    (out_dir / "train_summary.json").write_text(json.dumps(logs, indent=2))
    print(f"[draft] saved {out_dir / 'qwen3_spice_draft.pt'}", flush=True)


if __name__ == "__main__":
    main()
