"""Convert decode traces into per-step LoRE training traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert decode traces to train_real_lore trace_*.pt files")
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.decode_dir)
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    files = []

    for name in man["files"]:
        obj = torch.load(root / name, map_location="cpu", weights_only=False)
        for si, step in enumerate(obj.get("steps", [])):
            if args.max_steps and count >= args.max_steps:
                break
            if "hidden_states" not in step or "router_probs" not in step:
                continue
            router_probs = step["router_probs"]
            if any(x is None for x in router_probs):
                continue
            fname = f"trace_{count:05d}.pt"
            torch.save(
                {
                    "hidden_states": step["hidden_states"],
                    "router_probs": [
                        (x.float().unsqueeze(0) if x.ndim == 1 else x.float())
                        for x in router_probs
                    ],
                    "router_module_names": [
                        f"model.layers.{i}.mlp.gate" for i in range(len(router_probs))
                    ],
                    "source_decode": name,
                    "source_step": si,
                },
                out / fname,
            )
            files.append(fname)
            count += 1
        if args.max_steps and count >= args.max_steps:
            break

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "files": files,
                "num_files": len(files),
                "source_decode_dir": str(root),
                "model_dir": man.get("model_dir"),
                "top_k": man.get("top_k"),
                "experts": man.get("experts"),
                "num_layers": man.get("num_layers"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
