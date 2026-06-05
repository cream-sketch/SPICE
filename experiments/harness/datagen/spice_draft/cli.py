"""Real-model training-free SPICE draft: routing-prediction quality + forecast dump.

真实模型训练自由 SPICE draft: 多 horizon 路由预测质量 + forecast dump (模型无关).

Unifies the former qwen_spice_draft.py + ds_spice_draft.py into one adapter-based tool.
SPICE's draft reuses the target's FROZEN attention + FROZEN router and predicts only
ROUTING (expert ids); hidden propagation uses a shared-expert-only surrogate (no training).

Usage:
  python -m spice_draft.cli --model_type {qwen|deepseek} --model_dir DIR --text_file F \
      --out metrics.json [--dump_forecast DIR] [--oracle_fcast] [--top_k K] [--max_horizon H]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .adapters.deepseek_v2 import DeepSeekV2Adapter
from .adapters.qwen2_moe import Qwen2MoEAdapter
from .forecast_io import build_dump_tensors, recall_at_k, save_forecast, write_manifest
from .rollout import draft_rollout_predict

ADAPTERS = {"qwen": Qwen2MoEAdapter, "deepseek": DeepSeekV2Adapter}


def parse_args():
    p = argparse.ArgumentParser(description="Real-model training-free SPICE draft prediction quality")
    p.add_argument("--model_type", required=True, choices=sorted(ADAPTERS.keys()))
    p.add_argument("--model_dir", required=True)
    p.add_argument("--text_file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=None,
                   help="CUDA device index. Required when CUDA is available; no implicit GPU0 default.")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                   help="dtype used for target forward and draft rollout")
    p.add_argument("--top_k", type=int, default=0, help="0 -> use the model's default (qwen 4 / deepseek 6)")
    p.add_argument("--max_horizon", type=int, default=6)
    p.add_argument("--max_samples", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--dump_forecast", default=None, help="if set, dump per-text true_top + fcast tensors here")
    p.add_argument("--oracle_fcast", action="store_true",
                   help="fill fcast with TRUE future routes (perfect-prediction upper bound) for "
                        "an apples-to-apples real-vs-oracle GOS comparison")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter = ADAPTERS[args.model_type]()
    top_k = args.top_k or adapter.top_k_default
    if torch.cuda.is_available():
        if args.gpu is None:
            raise ValueError("CUDA is available; pass --gpu explicitly (no implicit GPU0 default)")
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = adapter.load_model(args.model_dir, device, dtype)

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
            true_topk, hs = adapter.true_forward(model, enc["input_ids"], enc.get("attention_mask"), top_k)
            moe_layers = sorted(true_topk.keys())
            preds, _num_layers = draft_rollout_predict(adapter, model, hs, moe_layers, top_k, args.max_horizon)
            seq_len = true_topk[moe_layers[0]].shape[0]

            for (anchor, target), pred_ids in preds.items():
                if target not in true_topk:
                    continue
                d = target - anchor + 1  # d=1 is exact (attn+gate on true state)
                horizon_recall_draft.setdefault(d, []).append(recall_at_k(pred_ids, true_topk[target], top_k))
                prev = target - 1
                if prev in true_topk:
                    horizon_recall_anchor.setdefault(d, []).append(recall_at_k(true_topk[prev], true_topk[target], top_k))

            if dump_dir:
                true_top, fcast, ndump = build_dump_tensors(
                    true_topk, preds, moe_layers, seq_len, top_k, args.max_horizon, args.oracle_fcast)
                fname = f"fc_{ti:05d}.pt"
                save_forecast(dump_dir, fname, true_top, fcast, ndump, top_k, args.max_horizon)
                dump_files.append(fname)

    def summarize(rec):
        return {str(d): sum(v) / len(v) for d, v in sorted(rec.items()) if v}

    report = {
        "experiment": f"{args.model_type}_trainingfree_spice_draft_prediction",
        "model_dir": args.model_dir,
        "model_type": args.model_type,
        "top_k": top_k,
        "dtype": args.dtype,
        "max_horizon": args.max_horizon,
        "num_texts": len(texts),
        "recall_at_k_by_horizon": {
            "draft_trainingfree": summarize(horizon_recall_draft),
            "anchor_repeat": summarize(horizon_recall_anchor),
        },
        "dump_dir": str(dump_dir) if dump_dir else None,
        "dump_files": dump_files,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    if dump_dir:
        write_manifest(dump_dir, dump_files, top_k, args.max_horizon, args.model_dir,
                       extra={"oracle_fcast": bool(args.oracle_fcast), "moe_layers_only": True,
                              "dtype": args.dtype})
        print(f"[dump] {len(dump_files)} forecast files + manifest -> {dump_dir}")
    print(json.dumps(report["recall_at_k_by_horizon"], indent=2))


if __name__ == "__main__":
    main()
