"""Forecast dump schema shared by all SPICE training-free drafts.

所有 SPICE 训练自由 draft 共享的 forecast dump schema.

The GOS runtime loader (spice_shallow_issuer_runtime.load_forecast_sequences) reads,
per text, a `fc_*.pt` with:
  true_top [L, S, K]  -- gate-descending true top-K expert ids per (layer, token)
  fcast    [L, H, S, K] -- predicted top-K per (layer, lead-horizon, token); -1 padded
  num_layers, top_k, max_horizon
and a dir-level `manifest.json` {files, top_k, max_horizon, model_dir, ...}.

Layers are indexed over the MoE layers ONLY (dense layers dropped) when the model
has dense prefix layers (e.g. DeepSeek first_k_dense_replace); for all-MoE models
(e.g. Qwen) every layer is included. The mapping dump-index -> model-layer is the
sorted list of MoE model-layer indices, which is contiguous for current models so
fcast horizon h maps target dump-index = anchor_dump + h.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def build_dump_tensors(true_topk: dict, preds: dict, moe_layers: list, seq_len: int,
                       top_k: int, max_horizon: int, oracle_fcast: bool):
    """Assemble (true_top, fcast) over MoE-layer-only dump indices.

    true_topk: {model_layer_idx: LongTensor[S, top_k]} true routes (MoE layers only)
    preds:     {(anchor_model_layer, target_model_layer): LongTensor[S, top_k]} draft predictions
    moe_layers: sorted list of MoE model-layer indices (dump index i <-> moe_layers[i])
    oracle_fcast: if True, fcast = true future routes (perfect-prediction upper bound)
    """
    idx_of = {ml: i for i, ml in enumerate(moe_layers)}
    ndump = len(moe_layers)
    true_top = torch.full((ndump, seq_len, top_k), -1, dtype=torch.long)
    for i, ml in enumerate(moe_layers):
        true_top[i] = true_topk[ml].cpu()
    fcast = torch.full((ndump, max_horizon, seq_len, top_k), -1, dtype=torch.long)
    if oracle_fcast:
        for i in range(ndump):
            for h in range(max_horizon):
                if i + h < ndump:
                    fcast[i, h] = true_top[i + h]
    else:
        for (anchor, target), pred_ids in preds.items():
            if anchor not in idx_of or target not in idx_of:
                continue
            h = target - anchor  # contiguous MoE layers -> dump target index = idx_of[anchor] + h
            if 0 <= h < max_horizon:
                fcast[idx_of[anchor], h] = pred_ids.cpu()
    return true_top, fcast, ndump


def save_forecast(dump_dir: Path, fname: str, true_top, fcast, ndump, top_k, max_horizon):
    torch.save({"true_top": true_top, "fcast": fcast, "num_layers": ndump,
                "top_k": top_k, "max_horizon": max_horizon}, dump_dir / fname)


def write_manifest(dump_dir: Path, dump_files: list, top_k: int, max_horizon: int,
                   model_dir: str, extra: dict | None = None):
    """manifest.json is REQUIRED by the GOS loader (read unconditionally)."""
    man = {"files": dump_files, "top_k": top_k, "max_horizon": max_horizon, "model_dir": model_dir}
    if extra:
        man.update(extra)
    (dump_dir / "manifest.json").write_text(json.dumps(man, indent=2))


def recall_at_k(pred_ids: torch.Tensor, true_ids: torch.Tensor, top_k: int) -> float:
    """Mean over tokens of |pred ∩ true| / top_k. 每 token 交集比例均值."""
    n = pred_ids.shape[0]
    total = 0.0
    for i in range(n):
        total += len(set(pred_ids[i].tolist()) & set(true_ids[i].tolist())) / top_k
    return total / max(1, n)
