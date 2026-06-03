"""Real-Qwen SPICE draft (training-free) + multi-horizon routing-prediction quality.

真实 Qwen 的 SPICE draft (免训练版) + 多 horizon 路由预测质量度量.

This is the repo's missing real-model draft path. SPICE's draft reuses the
target's FROZEN attention + FROZEN router and only predicts ROUTING (expert ids),
not expert outputs. The training-free version replaces each routed-expert MLP
with a SHARED-EXPERT-ONLY surrogate for hidden-state propagation, while still
reading the frozen router to predict the top-K experts. We anchor at each true
layer state and roll forward up to K layers (SPICE anchor re-initialization),
measuring how well the draft predicts the TRUE top-K experts at horizon d.
免训练 draft: 用 shared-expert-only 代理传播 hidden, 仍读冻结 router 预测 top-K;
从每层真实状态锚定, 向前滚动至多 K 层, 度量对真实 top-K 的预测质量.

Codex gate to justify training a LoRE draft (per notes/codex_plan_eviction_feedback.md):
  slot_recall@K >= 0.50 AND resident_next_use_AUC >= 0.70 AND >= +0.10 over
  layer_prior / anchor_repeat. exact-set match is NOT the gate (too harsh).
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def shared_only_mlp_forward(mlp, hidden_states: torch.Tensor):
    """Training-free draft MLP: frozen router (for prediction) + shared expert only.

    免训练 draft MLP: 仍算冻结 router 用于预测, 但 hidden 传播只用 shared expert (跳 routed).
    Returns (hidden_out, router_logits) like the real Qwen2MoeSparseMoeBlock.forward.
    """
    b, s, d = hidden_states.shape
    h = hidden_states.view(-1, d)
    router_logits = mlp.gate(h)
    shared = mlp.shared_expert(h)
    shared = F.sigmoid(mlp.shared_expert_gate(h)) * shared
    return shared.view(b, s, d), router_logits


def topk_sets_from_logits(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """[N, E] router logits -> [N, top_k] selected expert ids (top-K of softmax).

    由 router logits 取 top-K 专家 id.
    """
    probs = F.softmax(router_logits.float(), dim=-1)
    return torch.topk(probs, k=top_k, dim=-1).indices


@torch.no_grad()
def true_forward(model, input_ids, attention_mask, top_k):
    """Run the true model; return true per-layer top-K [L][N,top_k] and hidden states.

    真实前向: 返回每层真实 top-K 与各层 hidden state.
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_router_logits=True,
        use_cache=False,
        return_dict=True,
    )
    true_topk = [topk_sets_from_logits(rl, top_k) for rl in out.router_logits]  # per layer [N,top_k]
    return true_topk, out.hidden_states  # hidden_states: tuple len L+1, hs[j]=input to layer j


@torch.no_grad()
def draft_rollout_predict(model, hidden_states_tuple, attention_mask, top_k, max_horizon):
    """Anchored training-free draft: for each anchor layer a, roll shared-only layers
    a..a+d-1 from true input hs[a], predict top-K at each visited layer.

    锚定免训练 draft: 对每个锚 a, 从真实 hs[a] 用 shared-only 滚动, 预测 a..a+d-1 层 top-K.
    Returns pred[(anchor, target_layer)] = [N, top_k] predicted ids, where
    horizon d = target_layer - anchor + 1 (d=1 is exact: attn+router on true state).
    """
    base = model.model  # Qwen2MoeModel
    layers = base.layers
    num_layers = len(layers)
    device = hidden_states_tuple[0].device

    b, s, d = hidden_states_tuple[0].shape
    position_ids = torch.arange(s, device=device).unsqueeze(0)
    # rotary position embeddings (cos, sin)
    cos, sin = base.rotary_emb(hidden_states_tuple[0], position_ids)
    pos_emb = (cos, sin)
    # 4D additive causal mask (1,1,S,S): 0 allowed, -inf future
    causal = torch.full((s, s), float("-inf"), device=device, dtype=hidden_states_tuple[0].dtype)
    causal = torch.triu(causal, diagonal=1)[None, None, :, :]

    # patch all MLPs to shared-only for the draft rollout
    originals = []
    for layer in layers:
        originals.append(layer.mlp.forward)
        layer.mlp.forward = types.MethodType(shared_only_mlp_forward, layer.mlp)

    preds: dict[tuple[int, int], torch.Tensor] = {}
    try:
        for anchor in range(num_layers):
            h = hidden_states_tuple[anchor]  # true input to layer `anchor`
            for d_off in range(max_horizon):
                target = anchor + d_off
                if target >= num_layers:
                    break
                layer = layers[target]
                layer_out = layer(
                    hidden_states=h,
                    attention_mask=causal,
                    position_ids=position_ids,
                    position_embeddings=pos_emb,
                    output_router_logits=True,
                    use_cache=False,
                )
                # DecoderLayer returns (hidden, [attn], [router_logits]); router_logits last when requested
                h_next = layer_out[0]
                router_logits = layer_out[-1]
                preds[(anchor, target)] = topk_sets_from_logits(router_logits, top_k)
                h = h_next
    finally:
        for layer, orig in zip(layers, originals):
            layer.mlp.forward = orig
    return preds, num_layers


def recall_at_k(pred_ids: torch.Tensor, true_ids: torch.Tensor, top_k: int) -> float:
    """Mean over tokens of |pred ∩ true| / top_k. 每 token 取交集比例的均值."""
    n = pred_ids.shape[0]
    total = 0.0
    for i in range(n):
        ps = set(pred_ids[i].tolist())
        ts = set(true_ids[i].tolist())
        total += len(ps & ts) / top_k
    return total / max(1, n)


def baseline_anchor_repeat(true_topk, target_layer):
    """anchor_repeat: predict target layer's experts as the PREVIOUS layer's experts.

    用上一层的真实专家预测本层 (anchor_repeat 基线).
    """
    src = max(0, target_layer - 1)
    return true_topk[src]


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Qwen training-free SPICE draft prediction quality")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--text_file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--max_horizon", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dump_forecast", default=None,
                        help="if set, dump per-text true routing + draft forecast tensors to this dir")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()

    texts = [l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][: args.max_samples]

    # accumulate recall by horizon for draft + baselines
    horizon_recall_draft: dict[int, list[float]] = {}
    horizon_recall_anchor: dict[int, list[float]] = {}
    horizon_recall_prior: dict[int, list[float]] = {}

    dump_dir = Path(args.dump_forecast) if args.dump_forecast else None
    dump_files: list[str] = []
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    for ti, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        true_topk, hs = true_forward(model, enc["input_ids"], enc.get("attention_mask"), args.top_k)
        preds, num_layers = draft_rollout_predict(model, hs, enc.get("attention_mask"), args.top_k, args.max_horizon)

        if dump_dir:
            # true_top [L, S, top_k]; fcast [L_anchor, H, S, top_k] (target=anchor+h; -1 padded)
            s = true_topk[0].shape[0]
            true_top = torch.stack([t.cpu() for t in true_topk], dim=0)  # [L,S,k]
            fcast = torch.full((num_layers, args.max_horizon, s, args.top_k), -1, dtype=torch.long)
            for (anchor, target), pred_ids in preds.items():
                h = target - anchor
                if h < args.max_horizon:
                    fcast[anchor, h] = pred_ids.cpu()
            fname = f"fc_{ti:05d}.pt"
            torch.save({"true_top": true_top, "fcast": fcast, "num_layers": num_layers,
                        "top_k": args.top_k, "max_horizon": args.max_horizon}, dump_dir / fname)
            dump_files.append(fname)

        # running layer prior (frequency of experts per layer) for layer_prior baseline
        n_experts = int(model.config.num_experts)
        layer_freq = [torch.zeros(n_experts, device=device) for _ in range(num_layers)]
        # precompute layer_prior over the whole sequence's true routing (counts)
        for l in range(num_layers):
            flat = true_topk[l].reshape(-1)
            layer_freq[l].scatter_add_(0, flat, torch.ones_like(flat, dtype=layer_freq[l].dtype))
        prior_topk = [torch.topk(layer_freq[l], k=args.top_k).indices for l in range(num_layers)]

        for (anchor, target), pred_ids in preds.items():
            d = target - anchor + 1
            true_ids = true_topk[target]
            horizon_recall_draft.setdefault(d, []).append(recall_at_k(pred_ids, true_ids, args.top_k))
            # baselines (per target layer, independent of anchor)
            ar = baseline_anchor_repeat(true_topk, target)
            horizon_recall_anchor.setdefault(d, []).append(recall_at_k(ar, true_ids, args.top_k))
            lp = prior_topk[target].unsqueeze(0).expand(true_ids.shape[0], -1)
            horizon_recall_prior.setdefault(d, []).append(recall_at_k(lp, true_ids, args.top_k))

    def summarize(rec: dict[int, list[float]]):
        return {str(d): sum(v) / max(1, len(v)) for d, v in sorted(rec.items())}

    result = {
        "experiment": "qwen_trainingfree_spice_draft_prediction",
        "model_dir": args.model_dir,
        "top_k": args.top_k,
        "max_horizon": args.max_horizon,
        "num_texts": len(texts),
        "recall_at_k_by_horizon": {
            "draft_trainingfree": summarize(horizon_recall_draft),
            "anchor_repeat": summarize(horizon_recall_anchor),
            "layer_prior": summarize(horizon_recall_prior),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["recall_at_k_by_horizon"], indent=2))
    print(f"[done] wrote {args.out}")

    if dump_dir:
        (dump_dir / "manifest.json").write_text(
            json.dumps({"files": dump_files, "top_k": args.top_k, "max_horizon": args.max_horizon,
                        "model_dir": args.model_dir}, indent=2), encoding="utf-8")
        print(f"[dump] {len(dump_files)} forecast files -> {dump_dir}")


if __name__ == "__main__":
    main()
