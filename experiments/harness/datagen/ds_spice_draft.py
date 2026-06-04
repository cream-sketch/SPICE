"""Real DeepSeek-V2-Lite SPICE draft (training-free) + routing-prediction quality.

真实 DeepSeek-V2-Lite 的 SPICE draft (免训练版) + 多 horizon 路由预测质量度量.

DeepSeek port of qwen_spice_draft.py. SPICE's draft reuses the target's FROZEN
attention + FROZEN router and predicts only ROUTING (expert ids). The training-free
draft propagates hidden states with a SHARED-EXPERT-ONLY surrogate (skip routed
experts) while still reading the frozen gate to predict top-K. We anchor at each
true layer state and roll forward up to H layers, measuring recall of the TRUE
top-K experts at each horizon.

DeepSeek-V2-Lite specifics handled here (verified from config + modeling_deepseek.py):
  - 27 hidden layers; first_k_dense_replace=1 -> layer 0 is a DENSE MLP, layers
    1..26 are MoE (DeepseekV2MoE). Only MoE layers have a router/gate.
  - Gating: n_group=1, topk_group=1, topk_method="greedy", scoring_func="softmax",
    no correction bias -> plain softmax top-6. So top-k of softmax(gate logits)
    reproduces the true selection.
  - MoEGate.forward returns (topk_idx, topk_weight, aux_loss), NOT logits; the gate
    holds the projection as `gate.weight` ([n_routed_experts, hidden]). We compute
    logits = F.linear(h, gate.weight) ourselves and capture them via a stash.
  - DeepseekV2MoE.forward returns only the hidden tensor and adds shared_experts;
    DeepseekV2DecoderLayer.forward computes RoPE internally from position_ids (no
    position_embeddings arg) and does not support output_router_logits.
  - MLA attention with eager impl accepts a 4D additive causal mask (b,1,q,k).
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_moe_block(mlp) -> bool:
    """A DeepseekV2MoE block has a `.gate` MoEGate with a `.weight`; the dense
    layer-0 mlp (DeepseekV2MLP) does not. 判断是否为 MoE 层 (dense layer0 无 gate)."""
    return hasattr(mlp, "gate") and hasattr(getattr(mlp, "gate"), "weight") and hasattr(mlp, "shared_experts")


def gate_logits(mlp, hidden_states: torch.Tensor) -> torch.Tensor:
    """Compute router logits the same way MoEGate does: F.linear(h_fp32, gate.weight).
    返回 [N, n_routed_experts] 的 router logits."""
    h = hidden_states.reshape(-1, hidden_states.shape[-1])
    # Match MoEGate's fp32 logit computation; cast BOTH operands to avoid bf16/fp32 mismatch.
    return F.linear(h.float(), mlp.gate.weight.float())


def topk_sets_from_logits(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """[N, E] logits -> [N, top_k] selected expert ids (top-K of softmax). 取 top-K 专家 id."""
    probs = F.softmax(router_logits.float(), dim=-1)
    return torch.topk(probs, k=top_k, dim=-1).indices


def ds_shared_only_forward(self, hidden_states):
    """Training-free draft MoE forward: stash router logits for prediction, propagate
    hidden with SHARED experts only (skip routed experts).
    免训练 draft: 捕获 gate logits 用于预测, hidden 只走 shared experts."""
    self._captured_logits = gate_logits(self, hidden_states)  # [N, E], stashed for the rollout
    return self.shared_experts(hidden_states)


def true_forward(model, input_ids, attention_mask, top_k):
    """Run the true model; capture per-MoE-layer true top-K via gate forward hooks,
    and return hidden states. 真实前向: 用 gate hook 捕获每个 MoE 层真实 top-K, 返回 hidden.

    Returns (true_topk_by_layer, hidden_states) where true_topk_by_layer is a dict
    {layer_index: [S, top_k]} for MoE layers only, and hidden_states is the tuple
    (len num_layers+1) from output_hidden_states (hs[j] = input to layer j)."""
    base = model.model
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            # MoEGate.forward returns (topk_idx, topk_weight, aux_loss); topk_idx is [N, top_k]
            topk_idx = output[0]
            captured[layer_idx] = topk_idx[:, :top_k].detach()
        return hook

    for li, layer in enumerate(base.layers):
        if is_moe_block(layer.mlp):
            handles.append(layer.mlp.gate.register_forward_hook(make_hook(li)))
    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()
    return captured, out.hidden_states


def build_causal_mask(s: int, device, dtype):
    """4D additive causal mask (1,1,S,S): 0 allowed, -inf future. 4D 因果掩码."""
    causal = torch.full((s, s), float("-inf"), device=device, dtype=dtype)
    return torch.triu(causal, diagonal=1)[None, None, :, :]


def draft_rollout_predict(model, hidden_states_tuple, top_k, max_horizon):
    """Anchored training-free draft: from each MoE anchor layer a, roll shared-only
    layers a..a+d-1 from true hs[a], predict top-K at each visited MoE layer.
    锚定免训练 draft, 返回 pred[(anchor, target)] = [N, top_k]."""
    base = model.model
    layers = base.layers
    num_layers = len(layers)
    device = hidden_states_tuple[0].device
    dtype = hidden_states_tuple[0].dtype
    b, s, d = hidden_states_tuple[0].shape
    position_ids = torch.arange(s, device=device).unsqueeze(0)
    causal = build_causal_mask(s, device, dtype)

    # patch only MoE layers' mlp.forward to shared-only (dense layer-0 keeps its real MLP)
    originals = {}
    for li, layer in enumerate(layers):
        if is_moe_block(layer.mlp):
            originals[li] = layer.mlp.forward
            layer.mlp.forward = types.MethodType(ds_shared_only_forward, layer.mlp)

    preds: dict[tuple[int, int], torch.Tensor] = {}
    try:
        for anchor in range(num_layers):
            if anchor not in originals:  # only anchor at MoE layers (dense layer has no routing)
                continue
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
                    use_cache=False,
                )
                h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                if target in originals:  # MoE layer -> a prediction is available
                    logits = layer.mlp._captured_logits
                    preds[(anchor, target)] = topk_sets_from_logits(logits, top_k)
    finally:
        for li, orig in originals.items():
            layers[li].mlp.forward = orig
    return preds, num_layers


def recall_at_k(pred_ids: torch.Tensor, true_ids: torch.Tensor, top_k: int) -> float:
    """Mean over tokens of |pred ∩ true| / top_k. 每 token 交集比例均值."""
    n = pred_ids.shape[0]
    total = 0.0
    for i in range(n):
        total += len(set(pred_ids[i].tolist()) & set(true_ids[i].tolist())) / top_k
    return total / max(1, n)


def main() -> None:
    p = argparse.ArgumentParser(description="Real DeepSeek-V2-Lite training-free SPICE draft prediction quality")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--text_file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--top_k", type=int, default=6)
    p.add_argument("--max_horizon", type=int, default=6)
    p.add_argument("--max_samples", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--dump_forecast", default=None,
                   help="if set, dump per-text true routing + draft forecast tensors to this dir")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager",
    ).to(device).eval()

    texts = [l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][: args.max_samples]

    horizon_recall_draft: dict[int, list[float]] = {}
    horizon_recall_anchor: dict[int, list[float]] = {}

    dump_dir = Path(args.dump_forecast) if args.dump_forecast else None
    dump_files: list[str] = []
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for ti, text in enumerate(texts):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            true_topk, hs = true_forward(model, enc["input_ids"], enc.get("attention_mask"), args.top_k)
            preds, num_layers = draft_rollout_predict(model, hs, args.top_k, args.max_horizon)
            moe_layers = sorted(true_topk.keys())
            s = true_topk[moe_layers[0]].shape[0]

            for (anchor, target), pred_ids in preds.items():
                if target not in true_topk:
                    continue
                d = target - anchor + 1  # horizon (d=1 exact: attn+gate on true state)
                horizon_recall_draft.setdefault(d, []).append(recall_at_k(pred_ids, true_topk[target], args.top_k))
                # anchor_repeat baseline: predict target from the previous MoE layer's true routes
                prev = target - 1
                if prev in true_topk:
                    horizon_recall_anchor.setdefault(d, []).append(recall_at_k(true_topk[prev], true_topk[target], args.top_k))

            if dump_dir:
                # GOS dump is indexed over MoE LAYERS ONLY (drop the dense layer 0), matching the
                # existing DeepSeek oracle dump (make_forecast_from_dec) and avoiding -1 expert ids
                # leaking into popularity/routing. MoE model layers are contiguous (1..L), so a dump
                # index i maps to model layer moe_layers[i] and a horizon h target index is i+h.
                # GOS dump 仅索引 MoE 层(丢 dense layer0),与 oracle 一致,避免 -1 专家 id 污染。
                idx_of = {ml: i for i, ml in enumerate(moe_layers)}
                ndump = len(moe_layers)
                true_top = torch.full((ndump, s, args.top_k), -1, dtype=torch.long)
                for i, ml in enumerate(moe_layers):
                    true_top[i] = true_topk[ml].cpu()
                fcast = torch.full((ndump, args.max_horizon, s, args.top_k), -1, dtype=torch.long)
                for (anchor, target), pred_ids in preds.items():
                    if anchor not in idx_of or target not in idx_of:
                        continue
                    h = target - anchor  # contiguous MoE layers -> dump target index = idx_of[anchor]+h
                    if 0 <= h < args.max_horizon:
                        fcast[idx_of[anchor], h] = pred_ids.cpu()
                fname = f"fc_{ti:05d}.pt"
                torch.save({"true_top": true_top, "fcast": fcast, "num_layers": ndump,
                            "top_k": args.top_k, "max_horizon": args.max_horizon}, dump_dir / fname)
                dump_files.append(fname)

    def summarize(rec: dict[int, list[float]]) -> dict[str, float]:
        return {str(d): sum(v) / len(v) for d, v in sorted(rec.items()) if v}

    report = {
        "experiment": "deepseek_v2_lite_trainingfree_spice_draft_prediction",
        "model_dir": args.model_dir,
        "top_k": args.top_k,
        "max_horizon": args.max_horizon,
        "num_texts": len(texts),
        "recall_at_k_by_horizon": {
            "draft_trainingfree": summarize(horizon_recall_draft),
            "anchor_repeat": summarize(horizon_recall_anchor),
        },
        "dump_dir": str(dump_dir) if dump_dir else None,
        "dump_files": dump_files,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    if dump_dir:
        # manifest.json is REQUIRED by the GOS loader (load_forecast_sequences reads it unconditionally).
        (dump_dir / "manifest.json").write_text(
            json.dumps({"files": dump_files, "top_k": args.top_k, "max_horizon": args.max_horizon,
                        "model_dir": args.model_dir, "moe_layers_only": True}, indent=2))
        print(f"[dump] {len(dump_files)} forecast files + manifest -> {dump_dir}")
    print(json.dumps(report["recall_at_k_by_horizon"], indent=2))


if __name__ == "__main__":
    main()
