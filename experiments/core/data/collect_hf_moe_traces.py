from __future__ import annotations

import sys, pathlib  # bootstrap: resolve sibling core modules regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ensure_dir, set_seed, write_json


def iter_tensors(obj: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from iter_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from iter_tensors(item)


def load_texts(path: str | None, max_samples: int) -> list[str]:
    if path is None:
        base = [
            "Mixture-of-experts models route each token to a sparse set of expert feed-forward networks.",
            "Speculative prefetching can reduce expert transfer stalls when predictions are verified before use.",
            "The draft model observes previous routing decisions and predicts future expert demand.",
            "Low-rank expert surrogates approximate hidden-state evolution without executing full experts.",
        ]
        repeats = max(1, (max_samples + len(base) - 1) // len(base))
        return (base * repeats)[:max_samples]
    texts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                texts.append(text)
            if len(texts) >= max_samples:
                break
    if not texts:
        raise ValueError(f"no non-empty text lines found in {path}")
    return texts


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def should_capture_router_tensor(tensor: torch.Tensor, min_experts: int, max_experts: int) -> bool:
    if tensor.ndim < 2:
        return False
    experts = int(tensor.shape[-1])
    return min_experts <= experts <= max_experts


def parse_max_memory(spec: str | None) -> dict[int | str, str] | None:
    if not spec:
        return None
    result: dict[int | str, str] = {}
    for item in spec.split(","):
        key, value = item.split("=", maxsplit=1)
        key = key.strip()
        result["cpu" if key == "cpu" else int(key)] = value.strip()
    return result


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hidden/router traces from a local HuggingFace MoE model")
    parser.add_argument("--model", required=True, help="HF model id or local model path")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--text_file", default=None, help="UTF-8 file with one prompt/document per line")
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--router_name_regex", default=r"(router|gate|moe.*gate|mlp.*gate)")
    parser.add_argument("--exclude_name_regex", default=r"(gate_proj|up_proj|down_proj)")
    parser.add_argument("--min_experts", type=int, default=2)
    parser.add_argument("--max_experts", type=int, default=1024)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device_map", choices=["single", "auto", "cpu"], default="single")
    parser.add_argument("--max_memory", default=None, help="Comma-separated device=max spec, e.g. 0=20GiB,1=20GiB,cpu=96GiB")
    parser.add_argument("--offload_folder", default=None)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_hidden_states", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {
        "torch_dtype": dtype_map[args.dtype],
        "local_files_only": not args.allow_download,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    max_memory = parse_max_memory(args.max_memory)
    if args.device_map == "auto":
        load_kwargs["device_map"] = "auto"
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        if args.offload_folder is not None:
            load_kwargs["offload_folder"] = args.offload_folder
    elif args.device_map == "cpu":
        load_kwargs["device_map"] = {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.device_map == "single":
        model = model.to(device)
    model.eval()
    input_device = first_parameter_device(model)

    name_re = re.compile(args.router_name_regex, flags=re.IGNORECASE)
    exclude_re = re.compile(args.exclude_name_regex, flags=re.IGNORECASE) if args.exclude_name_regex else None
    active_trace: dict[str, list[Any]] = {"router_logits": [], "router_module_names": [], "moe_hidden": []}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            for tensor in iter_tensors(output):
                if should_capture_router_tensor(tensor, args.min_experts, args.max_experts):
                    active_trace["router_logits"].append(tensor.detach().float().cpu())
                    active_trace["router_module_names"].append(name)
                    # ALIGNED hidden: the router's own INPUT is exactly what it consumes (post-attention
                    # residual), per MoE layer -> what the LoRE forecaster must predict routing FROM.
                    rin = _inputs[0] if isinstance(_inputs, tuple) else _inputs
                    active_trace["moe_hidden"].append(rin.detach().float().cpu())
                    break

        return hook

    matched_modules = []
    for name, module in model.named_modules():
        if name and name_re.search(name) and not (exclude_re and exclude_re.search(name)):
            handles.append(module.register_forward_hook(make_hook(name)))
            matched_modules.append(name)
    if not matched_modules:
        raise RuntimeError(f"no modules matched router_name_regex={args.router_name_regex!r}")

    texts = load_texts(args.text_file, args.max_samples)
    manifest = {
        "schema": "spice_hf_moe_trace_v1",
        "model": args.model,
        "model_config": {
            key: getattr(model.config, key, None)
            for key in [
                "model_type",
                "num_hidden_layers",
                "hidden_size",
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
            ]
        },
        "num_texts": len(texts),
        "batch": args.batch,
        "max_length": args.max_length,
        "router_name_regex": args.router_name_regex,
        "exclude_name_regex": args.exclude_name_regex,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_memory": args.max_memory,
        "matched_modules": matched_modules,
        "trace_files": [],
        "note": (
            "Each trace stores tokenized inputs, optional hidden_states, and router logits/probs captured "
            "from modules matching router_name_regex. Use local model paths or --allow_download explicitly."
        ),
    }

    with torch.no_grad():
        for batch_id, text_batch in enumerate(batched(texts, args.batch)):
            active_trace = {"router_logits": [], "router_module_names": [], "moe_hidden": []}
            encoded = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(input_device)
            output = model(**encoded, output_hidden_states=not args.no_hidden_states, return_dict=True)
            router_probs = [torch.softmax(logits, dim=-1) for logits in active_trace["router_logits"]]
            payload = {
                "schema": "spice_hf_moe_trace_v1",
                "batch_id": batch_id,
                "texts": text_batch,
                "input_ids": encoded["input_ids"].detach().cpu(),
                "attention_mask": encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).detach().cpu(),
                "router_module_names": active_trace["router_module_names"],
                "router_logits": active_trace["router_logits"],
                "router_probs": router_probs,
                "moe_hidden": active_trace["moe_hidden"],   # ALIGNED per-MoE-layer router-input hidden
                "hidden_states": None
                if args.no_hidden_states
                else [state.detach().float().cpu() for state in output.hidden_states],
            }
            trace_path = Path(out_dir) / f"trace_{batch_id:05d}.pt"
            torch.save(payload, trace_path)
            manifest["trace_files"].append(trace_path.name)
            print({"batch": batch_id, "router_tensors": len(router_probs), "path": str(trace_path)})

    for handle in handles:
        handle.remove()
    write_json(Path(out_dir) / "manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
