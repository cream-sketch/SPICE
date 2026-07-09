"""Generate Qwen2-MoE decode traces with HF Accelerate CPU offload.

This path is for memory-limited trace collection on a single GPU.  It does not
measure SPICE runtime.  It only records true router choices, generated tokens,
and optionally last-token hidden states for LoRE training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2_moe.modeling_qwen2_moe as Mq


CAP: list[tuple[int, list[int], torch.Tensor]] = []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate decode traces with CPU-offloaded Qwen2-MoE")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--text_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_prompts", type=int, default=16)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--prompt_len", type=int, default=128)
    p.add_argument("--save_hidden_states", action="store_true")
    p.add_argument("--gpu_mem", default="70GiB")
    p.add_argument("--cpu_mem", default="600GiB")
    p.add_argument("--offload_folder", required=True)
    p.add_argument("--resume", action="store_true", help="skip existing dec_*.pt files and write a full manifest")
    return p.parse_args()


def input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_gate_hook(layer_idx: int, top_k: int):
    def hook(_module, _inputs, output):
        probs = F.softmax(output.float(), dim=-1)
        sel = torch.topk(probs, top_k, dim=-1).indices
        CAP.append((
            layer_idx,
            [int(x) for x in sel[-1].detach().cpu().tolist()],
            probs[-1].detach().to(torch.float16).cpu(),
        ))

    return hook


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    offload = Path(args.offload_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    offload.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={args.gpu: args.gpu_mem, "cpu": args.cpu_mem},
        offload_folder=str(offload),
    ).eval()

    layers = model.model.layers
    top_k = int(model.config.num_experts_per_tok)
    handles = []
    for li, layer in enumerate(layers):
        if isinstance(layer.mlp, Mq.Qwen2MoeSparseMoeBlock):
            handles.append(layer.mlp.gate.register_forward_hook(make_gate_hook(li, top_k)))

    texts = [l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = texts[: args.n_prompts]
    files: list[str] = []
    timing_rows = []
    in_dev = input_device(model)

    try:
        for pi, text in enumerate(texts):
            fname = f"dec_{pi:05d}.pt"
            existing = out_dir / fname
            if args.resume and existing.exists():
                d = torch.load(existing, map_location="cpu", weights_only=False)
                files.append(fname)
                timing_rows.append({
                    "file": fname,
                    "prompt_tokens": int(torch.tensor(d.get("prompt_ids", [[0]])).numel()),
                    "decode_tokens": len(d.get("steps", [])),
                    "prefill_s": None,
                    "decode_s": None,
                    "ttft_s": None,
                    "tpot_s": None,
                    "resumed": True,
                })
                print(f"prompt {pi}: reuse existing {fname}", flush=True)
                continue
            enc = tok(text, return_tensors="pt", truncation=True, max_length=args.prompt_len)
            cur = enc["input_ids"].to(in_dev)
            past = None
            steps = []
            prefill_s = 0.0
            decode_s = 0.0
            for gi in range(args.gen):
                CAP.clear()
                t0 = time.perf_counter()
                out = model(
                    input_ids=cur if past is None else cur[:, -1:],
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=args.save_hidden_states,
                )
                dt = time.perf_counter() - t0
                if past is None:
                    prefill_s += dt
                else:
                    decode_s += dt
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax().item())
                per_layer = [None] * len(layers)
                router_probs = [None] * len(layers)
                for li, sel, probs in CAP[-len(layers):]:
                    per_layer[li] = sel
                    router_probs[li] = probs
                step = {"token_id": nxt, "topk": per_layer}
                if args.save_hidden_states:
                    step["hidden_states"] = [
                        hs[:, -1, :].detach().to(torch.bfloat16).cpu()
                        for hs in out.hidden_states
                    ]
                    step["router_probs"] = router_probs
                steps.append(step)
                cur = torch.tensor([[nxt]], device=in_dev)
                if nxt == tok.eos_token_id:
                    break

            torch.save(
                {
                    "steps": steps,
                    "prompt": text,
                    "prompt_ids": enc["input_ids"].cpu().tolist(),
                    "num_layers": len(layers),
                },
                out_dir / fname,
            )
            files.append(fname)
            row = {
                "file": fname,
                "prompt_tokens": int(enc["input_ids"].numel()),
                "decode_tokens": len(steps),
                "prefill_s": prefill_s,
                "decode_s": decode_s,
                "ttft_s": prefill_s,
                "tpot_s": decode_s / max(1, len(steps) - 1),
            }
            timing_rows.append(row)
            print(
                f"prompt {pi}: decode={len(steps)} ttft={row['ttft_s']:.3f}s "
                f"tpot={1000.0 * row['tpot_s']:.2f}ms",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    manifest = {
        "files": files,
        "top_k": top_k,
        "experts": int(model.config.num_experts),
        "num_layers": len(layers),
        "model_dir": str(model_dir),
        "save_hidden_states": bool(args.save_hidden_states),
        "prompt_len": args.prompt_len,
        "gen": args.gen,
        "device_map": getattr(model, "hf_device_map", None),
        "timing": timing_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
