"""Model-agnostic anchored training-free SPICE draft rollout.

锚定免训练 SPICE draft 滚动(模型无关,差异由 adapter 封装).

For each MoE anchor layer a, from the TRUE hidden hs[a], roll the shared-expert-only
surrogate forward through layers a..a+H-1 (frozen attention + frozen router), predicting
top-K at each visited MoE layer. d=1 (attn+gate on the true state) is exact -> recall@1
is the adapter-correctness canary.
"""
from __future__ import annotations

import torch

from .adapters.base import topk_sets_from_logits


def build_causal_mask(s: int, device, dtype):
    causal = torch.full((s, s), float("-inf"), device=device, dtype=dtype)
    return torch.triu(causal, diagonal=1)[None, None, :, :]


def draft_rollout_predict(adapter, model, hidden_states, moe_layers, top_k, max_horizon):
    """Returns preds {(anchor_layer, target_layer): LongTensor[S, top_k]} for MoE targets."""
    num_layers = len(model.model.layers)
    device = hidden_states[0].device
    dtype = hidden_states[0].dtype
    _, s, _ = hidden_states[0].shape
    position_ids = torch.arange(s, device=device).unsqueeze(0)
    causal = build_causal_mask(s, device, dtype)
    moe_set = set(moe_layers)

    state = adapter.begin_rollout(model)
    preds: dict[tuple[int, int], torch.Tensor] = {}
    try:
        for anchor in moe_layers:
            h = hidden_states[anchor]  # true input to layer `anchor`
            for d_off in range(max_horizon):
                target = anchor + d_off
                if target >= num_layers:
                    break
                h, router_logits = adapter.rollout_layer(model, target, h, position_ids, causal)
                if target in moe_set:
                    preds[(anchor, target)] = topk_sets_from_logits(router_logits, top_k)
    finally:
        adapter.end_rollout(model, state)
    return preds, num_layers
