"""Adapter protocol for the model-agnostic SPICE training-free draft.

模型无关 SPICE 训练自由 draft 的 adapter 协议.

Each adapter encapsulates ONLY the per-model differences of the training-free draft:
  - loading the target model,
  - identifying MoE layers,
  - the true forward (true per-MoE-layer top-K + hidden states),
  - the shared-expert-only rollout (patch + per-layer step capturing router logits).
The frozen attention + frozen router + shared-expert-only-hidden-surrogate recipe is
shared; the rollout/recall/dump live in rollout.py + forecast_io.py.
"""
from __future__ import annotations

import torch


class DraftAdapter:
    top_k_default: int = 4

    def load_model(self, model_dir: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        """Return the loaded, eval-mode target model on `device`."""
        raise NotImplementedError

    def moe_layer_indices(self, model) -> list[int]:
        """Sorted list of decoder-layer indices that are MoE (have a router/gate)."""
        raise NotImplementedError

    def true_forward(self, model, input_ids, attention_mask, top_k: int):
        """Return (true_topk, hidden_states):
          true_topk: dict {moe_layer_idx: LongTensor[S, top_k]} gate-descending true routes,
          hidden_states: tuple len num_layers+1 (hs[j] = input to decoder layer j).
        """
        raise NotImplementedError

    def begin_rollout(self, model):
        """Patch MoE blocks to shared-expert-only hidden propagation; return restore state."""
        raise NotImplementedError

    def end_rollout(self, model, state) -> None:
        """Restore the original MoE forwards."""
        raise NotImplementedError

    def rollout_layer(self, model, layer_idx: int, h, position_ids, causal):
        """Run one decoder layer (frozen attn + shared-only mlp) on `h`; return
        (h_next, router_logits[N, n_experts]) for this MoE layer."""
        raise NotImplementedError


def topk_sets_from_logits(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """[N, E] logits -> [N, top_k] selected expert ids (top-K of softmax)."""
    import torch.nn.functional as F
    probs = F.softmax(router_logits.float(), dim=-1)
    return torch.topk(probs, k=top_k, dim=-1).indices
