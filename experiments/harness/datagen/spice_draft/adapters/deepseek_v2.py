"""DeepSeek-V2-Lite adapter for the SPICE training-free draft.

DeepSeek-V2-Lite specifics (verified from config + modeling_deepseek.py):
  - first_k_dense_replace=1 -> layer 0 DENSE, layers 1.. MoE; only MoE layers have a gate.
  - Gating n_group=1/topk_group=1/greedy/softmax/no-bias -> plain softmax top-k; top-k of
    softmax(F.linear(h.float(), gate.weight.float())) reproduces the true selection.
  - MoEGate.forward returns (topk_idx, topk_weight, aux); logits computed internally, so we
    recompute them; true routes captured via gate forward hooks.
  - DeepseekV2DecoderLayer computes RoPE internally from position_ids (no position_embeddings),
    no output_router_logits; MLA eager attention accepts a 4D additive mask.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .base import DraftAdapter


def _is_moe_block(mlp) -> bool:
    return hasattr(mlp, "gate") and hasattr(getattr(mlp, "gate"), "weight") and hasattr(mlp, "shared_experts")


def _gate_logits(mlp, hidden_states: torch.Tensor) -> torch.Tensor:
    h = hidden_states.reshape(-1, hidden_states.shape[-1])
    # Match MoEGate's fp32 logit computation; cast BOTH operands to avoid bf16/fp32 mismatch.
    return F.linear(h.float(), mlp.gate.weight.float())


def _ds_shared_only_forward(self, hidden_states):
    """Training-free draft MoE forward: stash router logits, propagate shared experts only."""
    self._captured_logits = _gate_logits(self, hidden_states)
    return self.shared_experts(hidden_states)


class DeepSeekV2Adapter(DraftAdapter):
    top_k_default = 6

    def load_model(self, model_dir, device):
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, local_files_only=True,
            low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager",
        ).to(device).eval()
        return model

    def moe_layer_indices(self, model) -> list[int]:
        return [li for li, layer in enumerate(model.model.layers) if _is_moe_block(layer.mlp)]

    def true_forward(self, model, input_ids, attention_mask, top_k):
        base = model.model
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                captured[layer_idx] = output[0][:, :top_k].detach()  # output[0] = topk_idx [N, top_k]
            return hook

        for li, layer in enumerate(base.layers):
            if _is_moe_block(layer.mlp):
                handles.append(layer.mlp.gate.register_forward_hook(make_hook(li)))
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False, return_dict=True)
        finally:
            for h in handles:
                h.remove()
        return captured, out.hidden_states

    def begin_rollout(self, model):
        originals = {}
        for li, layer in enumerate(model.model.layers):
            if _is_moe_block(layer.mlp):
                originals[li] = layer.mlp.forward
                layer.mlp.forward = types.MethodType(_ds_shared_only_forward, layer.mlp)
        return originals

    def end_rollout(self, model, originals) -> None:
        for li, orig in originals.items():
            model.model.layers[li].mlp.forward = orig

    def rollout_layer(self, model, layer_idx, h, position_ids, causal):
        layer = model.model.layers[layer_idx]
        out = layer(hidden_states=h, attention_mask=causal, position_ids=position_ids, use_cache=False)
        h_next = out[0] if isinstance(out, tuple) else out
        return h_next, layer.mlp._captured_logits
