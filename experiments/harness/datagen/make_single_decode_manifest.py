from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a manifest for a partially collected decode trace directory")
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--files", default="", help="comma-separated dec_*.pt files; default: all")
    ap.add_argument("--prompt_len", type=int, default=0)
    ap.add_argument("--gen", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.decode_dir)
    files = [x for x in args.files.split(",") if x] or sorted(p.name for p in root.glob("dec_*.pt"))
    if not files:
        raise FileNotFoundError(f"no dec_*.pt files found in {root}")

    obj = torch.load(root / files[0], map_location="cpu", weights_only=False)
    steps = obj["steps"]
    if not steps:
        raise ValueError(f"{files[0]} contains no steps")
    layers = int(obj.get("num_layers", len(steps[0]["topk"])))
    top_k = int(len(steps[0]["topk"][0]))
    experts = int(max(x for name in files
                      for st in torch.load(root / name, map_location="cpu", weights_only=False)["steps"]
                      for layer in st["topk"] for x in layer) + 1)
    timing = []
    for name in files:
        d = torch.load(root / name, map_location="cpu", weights_only=False)
        timing.append({"file": name, "decode_tokens": len(d["steps"])})
    man = {
        "files": files,
        "top_k": top_k,
        "experts": experts,
        "num_layers": layers,
        "model_dir": args.model_dir,
        "save_hidden_states": True,
        "prompt_len": args.prompt_len,
        "gen": args.gen,
        "timing": timing,
    }
    (root / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps({"decode_dir": str(root), "files": len(files), "layers": layers,
                      "top_k": top_k, "experts": experts}, indent=2))


if __name__ == "__main__":
    main()
