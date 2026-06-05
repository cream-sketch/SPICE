"""Qwen1.5-MoE (Qwen2MoE) adapter for the SPICE training-free draft.

Qwen2Moe specifics:
  - Every decoder layer is MoE (no dense prefix).
  - Qwen2MoeSparseMoeBlock.gate(h) returns router_logits directly; true routes via
    output_router_logits=True; the shared-only surrogate uses shared_expert +
    sigmoid(shared_expert_gate).
  - DecoderLayer takes position_embeddings=(cos,sin) from model.model.rotary_emb and
    supports output_router_logits (router_logits is the last element of layer_out).
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .base import DraftAdapter, topk_sets_from_logits


def _shared_only_mlp_forward(mlp, hidden_states: torch.Tensor):
    """Frozen gate (for prediction) + shared-expert-only hidden propagation.
    Returns (hidden_out, router_logits) like Qwen2MoeSparseMoeBlock.forward."""
    b, s, d = hidden_states.shape
    h = hidden_states.view(-1, d)
    router_logits = mlp.gate(h)
    shared = mlp.shared_expert(h)
    shared = F.sigmoid(mlp.shared_expert_gate(h)) * shared
    return shared.view(b, s, d), router_logits


class Qwen2MoEAdapter(DraftAdapter):
    top_k_default = 4

    def load_model(self, model_dir, device, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True,
        ).to(device).eval()
        return model

    def moe_layer_indices(self, model) -> list[int]:
        return list(range(len(model.model.layers)))  # all layers are MoE

    def true_forward(self, model, input_ids, attention_mask, top_k):
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, output_router_logits=True, use_cache=False, return_dict=True)
        true_topk = {l: topk_sets_from_logits(rl, top_k) for l, rl in enumerate(out.router_logits)}
        return true_topk, out.hidden_states

    def begin_rollout(self, model):
        originals = {}
        for li, layer in enumerate(model.model.layers):
            originals[li] = layer.mlp.forward
            layer.mlp.forward = types.MethodType(_shared_only_mlp_forward, layer.mlp)
        return originals

    def end_rollout(self, model, originals) -> None:
        for li, orig in originals.items():
            model.model.layers[li].mlp.forward = orig

    def rollout_layer(self, model, layer_idx, h, position_ids, causal):
        base = model.model
        cos, sin = base.rotary_emb(h, position_ids)
        layer = base.layers[layer_idx]
        out = layer(hidden_states=h, attention_mask=causal, position_ids=position_ids,
                    position_embeddings=(cos, sin), output_router_logits=True, use_cache=False)
        return out[0], out[-1]
