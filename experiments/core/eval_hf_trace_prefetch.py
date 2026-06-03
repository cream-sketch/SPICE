from __future__ import annotations

import sys, pathlib  # bootstrap: resolve sibling core modules regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from common import write_json


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.store: OrderedDict[tuple[int, int], None] = OrderedDict()

    def contains(self, key: tuple[int, int]) -> bool:
        ok = key in self.store
        if ok:
            self.store.move_to_end(key)
        return ok

    def add(self, key: tuple[int, int]) -> bool:
        if key in self.store:
            self.store.move_to_end(key)
            return False
        self.store[key] = None
        while len(self.store) > self.capacity:
            self.store.popitem(last=False)
        return True


def load_manifest(trace_dir: Path) -> dict:
    with (trace_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_valid(topk: torch.Tensor, attention_mask: torch.Tensor | None) -> list[list[int]]:
    if topk.ndim == 3:
        batch, seq, k = topk.shape
        flat = topk.reshape(batch * seq, k)
        if attention_mask is None:
            keep = torch.ones(batch * seq, dtype=torch.bool)
        else:
            keep = attention_mask.reshape(batch * seq).bool()
        return flat[keep].cpu().tolist()
    if topk.ndim == 2:
        if attention_mask is not None and attention_mask.numel() == topk.shape[0]:
            topk = topk[attention_mask.reshape(-1).bool()]
        return topk.cpu().tolist()
    raise ValueError(f"unsupported topk shape: {tuple(topk.shape)}")


def load_trace_layers(trace_path: Path, top_k: int) -> tuple[list[list[list[int]]], int]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    attention_mask = payload.get("attention_mask")
    layers: list[list[list[int]]] = []
    for probs in payload["router_probs"]:
        topk = torch.topk(probs.float(), k=top_k, dim=-1).indices
        layers.append(flatten_valid(topk, attention_mask))
    if not layers:
        return [], 0
    token_count = min(len(layer) for layer in layers)
    layers = [layer[:token_count] for layer in layers]
    return layers, token_count


def predict_set(
    mode: str,
    target_layers: list[list[list[int]]],
    token: int,
    anchor: int,
    future: int,
    top_k: int,
    layer_priors: list[torch.Tensor],
) -> set[int]:
    if mode == "oracle":
        return set(target_layers[future][token])
    if mode == "anchor_repeat":
        if anchor > 0:
            return set(target_layers[anchor - 1][token])
        counts = layer_priors[future]
        if counts.sum() == 0:
            return set()
        return set(torch.topk(counts, k=top_k).indices.cpu().tolist())
    if mode == "layer_prior":
        counts = layer_priors[future]
        if counts.sum() == 0:
            return set(target_layers[anchor][token])
        return set(torch.topk(counts, k=top_k).indices.cpu().tolist())
    raise ValueError(f"unknown predictor mode: {mode}")


def update_priors(layer_priors: list[torch.Tensor], target_layers: list[list[list[int]]], token: int) -> None:
    for layer, topk_rows in enumerate(target_layers):
        for expert in topk_rows[token]:
            layer_priors[layer][expert] += 1


def simulate_trace_dir(
    trace_dir: Path,
    top_k: int,
    predictor: str,
    cache_capacity: int,
    l_min: int,
    l_max: int,
) -> dict:
    manifest = load_manifest(trace_dir)
    cache = LRUCache(cache_capacity)
    total_slots = 0
    hit_slots = 0
    fallback_slots = 0
    predicted_slot_attempts = 0
    issued_prefetches = 0
    wrong_prefetches = 0
    depth_sum = 0
    depth_count = 0
    trace_files = manifest.get("trace_files", [])

    layer_priors: list[torch.Tensor] | None = None
    expert_count = 0
    total_tokens = 0
    total_layers = 0

    for trace_name in trace_files:
        target_layers, token_count = load_trace_layers(trace_dir / trace_name, top_k)
        if not target_layers:
            continue
        total_tokens += token_count
        total_layers = max(total_layers, len(target_layers))
        inferred_experts = max(max(row) for layer in target_layers for row in layer) + 1
        expert_count = max(expert_count, inferred_experts)
        if layer_priors is None:
            layer_priors = [torch.zeros(expert_count, dtype=torch.long) for _ in target_layers]
        elif expert_count > layer_priors[0].numel():
            layer_priors = [
                torch.nn.functional.pad(counts, (0, expert_count - counts.numel()))
                for counts in layer_priors
            ]

        for token in range(token_count):
            assert layer_priors is not None
            for anchor in range(len(target_layers)):
                depth = 0
                for future in range(anchor, min(len(target_layers), anchor + l_max)):
                    depth += 1
                    pred = predict_set(predictor, target_layers, token, anchor, future, top_k, layer_priors)
                    target = set(target_layers[future][token])
                    for expert in pred:
                        predicted_slot_attempts += 1
                        issued_prefetches += int(cache.add((future, expert)))
                    wrong_prefetches += len(pred.difference(target))
                    if depth >= l_min and predictor == "layer_prior" and layer_priors[future].sum() < top_k:
                        break
                depth_sum += depth
                depth_count += 1

                for expert in target_layers[anchor][token]:
                    total_slots += 1
                    if cache.contains((anchor, expert)):
                        hit_slots += 1
                    else:
                        fallback_slots += 1
                        cache.add((anchor, expert))
            update_priors(layer_priors, target_layers, token)

    return {
        "experiment": "hf_trace_prefetch_eval",
        "trace_dir": str(trace_dir),
        "model": manifest.get("model"),
        "model_config": manifest.get("model_config", {}),
        "predictor": predictor,
        "top_k": top_k,
        "cache_capacity": cache_capacity,
        "l_min": l_min,
        "l_max": l_max,
        "trace_files": len(trace_files),
        "tokens": total_tokens,
        "layers": total_layers,
        "experts_observed": expert_count,
        "total_slots": total_slots,
        "prefetch_slot_hit_rate": hit_slots / max(1, total_slots),
        "fallback_slot_rate": fallback_slots / max(1, total_slots),
        "fallback_slots": fallback_slots,
        "predicted_slot_attempts": predicted_slot_attempts,
        "issued_prefetches": issued_prefetches,
        "wrong_prefetches": wrong_prefetches,
        "wrong_prefetch_rate": wrong_prefetches / max(1, predicted_slot_attempts),
        "avg_lookahead_depth": depth_sum / max(1, depth_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate verified prefetching on saved HF MoE router traces")
    parser.add_argument("--trace_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--predictor", choices=["oracle", "anchor_repeat", "layer_prior"], default="anchor_repeat")
    parser.add_argument("--cache_capacity", type=int, default=256)
    parser.add_argument("--l_min", type=int, default=2)
    parser.add_argument("--l_max", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = simulate_trace_dir(
        trace_dir=Path(args.trace_dir),
        top_k=args.top_k,
        predictor=args.predictor,
        cache_capacity=args.cache_capacity,
        l_min=args.l_min,
        l_max=args.l_max,
    )
    write_json(out_dir / f"hf_trace_prefetch_{args.predictor}.json", result)
    print(result)


if __name__ == "__main__":
    main()
