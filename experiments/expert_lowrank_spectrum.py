"""Experiment 0a: low-rank spectrum of real MoE expert weights.

实验0a：真实 MoE 专家权重的低秩谱分析。

Goal / 目标:
  Test hypothesis H0 (low-rank approximability) for the SPICE
  "verified low-rank expert approximation as miss recovery" thesis.
  测试 SPICE "用低秩代理做 miss recovery" 主张的地基假设 H0:
  真实专家权重是否在 rank r << full 下保留绝大部分能量.

It reads expert weight matrices directly from the safetensors shards
(no full model load), computes singular-value energy spectra, and reports
the rank needed for 90/95/99% Frobenius energy plus the energy fraction at
fixed ranks. Read-only analysis; writes a JSON summary.
直接从 safetensors 分片读取专家权重(不加载整模型), 计算奇异值能量谱,
报告达到 90/95/99% 能量所需 rank 以及固定 rank 下的能量占比. 只读分析.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors import safe_open


def build_key_index(model_dir: Path) -> dict[str, str]:
    """Map every tensor name to its shard file via the safetensors index.

    通过 safetensors index 把每个张量名映射到所在分片文件.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing index: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    return index["weight_map"]


def discover_expert_keys(weight_map: dict[str, str]) -> dict[int, dict[int, dict[str, str]]]:
    """Group routed-expert projection keys as layer -> expert -> proj -> name.

    把 routed-expert 投影权重按 层->专家->投影 分组.
    Expects names like model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight.
    """
    layers: dict[int, dict[int, dict[str, str]]] = {}
    for name in weight_map:
        if ".mlp.experts." not in name or not name.endswith("_proj.weight"):
            continue
        parts = name.split(".")
        # ... layers, {L}, mlp, experts, {E}, {proj}_proj, weight
        layer_idx = int(parts[parts.index("layers") + 1])
        expert_idx = int(parts[parts.index("experts") + 1])
        proj = parts[-2].replace("_proj", "")  # gate / up / down
        layers.setdefault(layer_idx, {}).setdefault(expert_idx, {})[proj] = name
    if not layers:
        raise ValueError("no routed-expert keys found; check model layout")
    return layers


def load_matrix(model_dir: Path, weight_map: dict[str, str], name: str, device: torch.device) -> torch.Tensor:
    """Load a single weight tensor from its shard as float32 on device.

    从对应分片加载单个权重张量, 转 float32 放到 device.
    """
    shard = model_dir / weight_map[name]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
    return tensor.to(device=device, dtype=torch.float32)


def spectrum_stats(singular_values: torch.Tensor, fixed_ranks: list[int], energy_targets: list[float]) -> dict:
    """Compute energy-based low-rank stats from singular values.

    由奇异值计算基于能量(奇异值平方和)的低秩统计.
    """
    sv = singular_values.detach().float()
    energy = sv * sv
    total = float(energy.sum().item())
    full_rank = int(sv.numel())
    cumulative = torch.cumsum(energy, dim=0) / max(total, 1e-12)

    rank_for_energy: dict[str, int] = {}
    for target in energy_targets:
        # 第一个使累计能量 >= target 的 rank (1-indexed)
        idx = int(torch.searchsorted(cumulative, torch.tensor(target)).item()) + 1
        rank_for_energy[f"{target:.2f}"] = min(idx, full_rank)

    energy_at_rank: dict[str, float] = {}
    for r in fixed_ranks:
        if r >= full_rank:
            energy_at_rank[str(r)] = 1.0
        else:
            energy_at_rank[str(r)] = float(cumulative[r - 1].item())

    # 有效秩 (谱熵指数): exp(H) where H = -sum p log p, p = energy normalized
    p = energy / max(total, 1e-12)
    p = p[p > 0]
    spectral_entropy = float((-(p * p.log()).sum()).item())
    effective_rank = float(torch.exp(torch.tensor(spectral_entropy)).item())

    return {
        "full_rank": full_rank,
        "rank_for_energy": rank_for_energy,
        "energy_at_rank": energy_at_rank,
        "effective_rank": effective_rank,
    }


def aggregate(records: list[dict], fixed_ranks: list[int], energy_targets: list[float]) -> dict:
    """Mean over records for each energy/rank metric, grouped by projection.

    按投影类型对各指标求均值.
    """
    by_proj: dict[str, list[dict]] = {}
    for rec in records:
        by_proj.setdefault(rec["proj"], []).append(rec["stats"])
    summary: dict[str, dict] = {}
    for proj, stats_list in by_proj.items():
        n = len(stats_list)
        full_rank = stats_list[0]["full_rank"]
        mean_rank_for_energy = {
            t: sum(s["rank_for_energy"][t] for s in stats_list) / n for t in stats_list[0]["rank_for_energy"]
        }
        mean_energy_at_rank = {
            str(r): sum(s["energy_at_rank"][str(r)] for s in stats_list) / n for r in fixed_ranks
        }
        mean_eff_rank = sum(s["effective_rank"] for s in stats_list) / n
        summary[proj] = {
            "count": n,
            "full_rank": full_rank,
            "mean_rank_for_energy": mean_rank_for_energy,
            "mean_energy_at_rank": mean_energy_at_rank,
            "mean_effective_rank": mean_eff_rank,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 0a: real MoE expert low-rank spectrum")
    parser.add_argument("--model_dir", required=True, help="HF model dir with safetensors + index")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layers", type=int, default=6, help="number of layers to sample")
    parser.add_argument("--experts_per_layer", type=int, default=8, help="experts sampled per layer")
    parser.add_argument("--fixed_ranks", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--energy_targets", type=str, default="0.90,0.95,0.99")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fixed_ranks = [int(x) for x in args.fixed_ranks.split(",") if x.strip()]
    energy_targets = [float(x) for x in args.energy_targets.split(",") if x.strip()]

    model_dir = Path(args.model_dir)
    weight_map = build_key_index(model_dir)
    expert_keys = discover_expert_keys(weight_map)

    all_layers = sorted(expert_keys.keys())
    # 均匀抽样层 (含首/中/尾)
    if args.layers >= len(all_layers):
        sampled_layers = all_layers
    else:
        step = len(all_layers) / args.layers
        sampled_layers = sorted({all_layers[min(len(all_layers) - 1, int(i * step))] for i in range(args.layers)})

    records: list[dict] = []
    for layer in sampled_layers:
        experts = sorted(expert_keys[layer].keys())
        chosen = experts if args.experts_per_layer >= len(experts) else random.sample(experts, args.experts_per_layer)
        for expert in chosen:
            for proj, name in expert_keys[layer][expert].items():
                matrix = load_matrix(model_dir, weight_map, name, device)
                sv = torch.linalg.svdvals(matrix)
                stats = spectrum_stats(sv, fixed_ranks, energy_targets)
                records.append({"layer": layer, "expert": expert, "proj": proj,
                                "shape": list(matrix.shape), "stats": stats})
                del matrix, sv
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print(f"[layer {layer}] processed {len(chosen)} experts")

    summary = aggregate(records, fixed_ranks, energy_targets)
    out = {
        "experiment": "expert_lowrank_spectrum_0a",
        "model_dir": str(model_dir),
        "device": str(device),
        "sampled_layers": sampled_layers,
        "experts_per_layer": args.experts_per_layer,
        "fixed_ranks": fixed_ranks,
        "energy_targets": energy_targets,
        "num_matrices": len(records),
        "summary_by_proj": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
