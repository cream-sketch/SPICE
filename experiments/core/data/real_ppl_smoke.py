from __future__ import annotations

import sys, pathlib  # bootstrap: resolve sibling core modules regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import build_arg_parser, device_from_arg, ensure_dir, set_seed, write_json


def main() -> None:
    parser = build_arg_parser("Real-model perplexity smoke test")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="builtin", choices=["builtin", "wikitext"])
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--local_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = device_from_arg(args.gpu)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_only, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    if args.dataset == "builtin":
        base_texts = [
            "Mixture-of-Experts inference can become dominated by host to device expert transfers when the expert weights exceed GPU memory.",
            "A verified prefetching runtime changes when parameters are moved, but it must not change which target experts execute.",
            "Software and hardware co-design is necessary when accelerator memory capacity is smaller than the working set of a sparse model.",
            "Speculative expert prefetching can hide latency when routing predictions are confident and the PCIe transfer window is large enough.",
            "When denser routing increases the number of selected experts per layer, the PCIe link can become the limiting resource again.",
        ]
        texts = (base_texts * ((args.max_samples + len(base_texts) - 1) // len(base_texts)))[: args.max_samples]
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, args.subset, split=args.split)
        texts = [x["text"] for x in ds if isinstance(x.get("text"), str) and len(x["text"].strip()) > 80]
        texts = texts[: args.max_samples]
    losses = []
    tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 8:
                continue
            out = model(input_ids=input_ids)
            logits = out.logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
            losses.append(float(loss.item()))
            tokens += int(labels.numel())

    nll = sum(losses) / max(1, tokens)
    result = {
        "experiment": "real_ppl_smoke",
        "model": args.model,
        "dataset": f"{args.dataset}/{args.subset}/{args.split}",
        "samples": len(texts),
        "tokens": tokens,
        "nll": nll,
        "ppl": math.exp(nll) if nll < 50 else float("inf"),
        "device": str(device),
        "interpretation": "Reference target-model perplexity smoke test; SPICE verified prefetch should preserve this value if target execution is unchanged.",
    }
    write_json(out_dir / "real_ppl_smoke.json", result)
    print(result)


if __name__ == "__main__":
    main()
