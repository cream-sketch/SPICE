"""Decisive test for VERIFIED low-bit expert movement: if routed experts are fetched in
int4/int8 (2-4x less PCIe), how often does the next-token argmax flip vs full bf16, and can a
CHEAP certificate (the low-bit logit top1-top2 margin) separate the flips? If yes -> fetch
low-bit + correct (full-precision re-fetch) only on uncertain tokens -> EXACT output at ~2-4x
effective PCIe. If no -> low-bit-with-verify cannot be made exact cheaply.

Not a simulator: real model forward, real per-expert weight quantization, real decode logits.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def quant_dequant(W, bits, group=0):
    """Symmetric int-N quant-dequant (the lossy transport). group=0 -> per-output-channel
    (whole input row); group=g -> group-wise along the input dim (AWQ/HOBBIT-style, finer scales)."""
    qmax = (1 << (bits - 1)) - 1            # int4: 7, int8: 127
    if group and W.shape[1] % group == 0:
        out, inp = W.shape
        Wg = W.view(out, inp // group, group)
        scale = Wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / qmax
        Wq = torch.round(Wg / scale).clamp(-qmax - 1, qmax)
        return (Wq * scale).view(out, inp).to(W.dtype)
    scale = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    Wq = torch.round(W / scale).clamp(-qmax - 1, qmax)
    return (Wq * scale).to(W.dtype)


def is_moe(mlp):
    return hasattr(mlp, "experts") and len(getattr(mlp, "experts")) > 0


def quantize_experts(model, bits, group=0):
    n = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        if not is_moe(mlp):
            continue
        for exp in mlp.experts:
            for lin in (exp.gate_proj, exp.up_proj, exp.down_proj):
                lin.weight.data = quant_dequant(lin.weight.data, bits, group)
            n += 1
    return n


@torch.inference_mode()
def decode_capture(model, input_ids, n_tokens, token_ids=None):
    """If token_ids is None: greedy decode n_tokens, return (ids, logits[list of [V] cpu fp32]).
    Else: teacher-force the given token_ids, return logits aligned to each step."""
    out = model(input_ids=input_ids, use_cache=True); kv = out.past_key_values
    logits = [out.logits[:, -1, :].float().cpu()[0]]
    if token_ids is None:
        ids = [int(logits[-1].argmax())]
        for _ in range(n_tokens - 1):
            cur = torch.tensor([[ids[-1]]], device=input_ids.device)
            out = model(input_ids=cur, past_key_values=kv, use_cache=True); kv = out.past_key_values
            logits.append(out.logits[:, -1, :].float().cpu()[0]); ids.append(int(logits[-1].argmax()))
        return ids, logits
    for tid in token_ids[:-1]:
        cur = torch.tensor([[tid]], device=input_ids.device)
        out = model(input_ids=cur, past_key_values=kv, use_cache=True); kv = out.past_key_values
        logits.append(out.logits[:, -1, :].float().cpu()[0])
    return token_ids, logits


def margin(lv):
    top2 = torch.topk(lv, 2).values
    return float(top2[0] - top2[1])


def main():
    p = argparse.ArgumentParser(description="Verified low-bit expert movement: flip rate + certificate")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--bits", type=int, required=True, choices=[4, 8])
    p.add_argument("--group_size", type=int, default=0, help="0=per-channel; e.g. 128=group-wise (AWQ/HOBBIT-style)")
    p.add_argument("--decode_tokens", type=int, default=128)
    p.add_argument("--prompt", default="The history of mixture-of-experts models in large language modeling is")
    p.add_argument("--trust_remote_code", action="store_true")
    args = p.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code).to(dev).eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)

    ref_ids, ref_logits = decode_capture(model, ids, args.decode_tokens)
    nq = quantize_experts(model, args.bits, args.group_size)
    _, lb_logits = decode_capture(model, ids, args.decode_tokens, token_ids=ref_ids)

    flips = [int(rl.argmax()) != int(ll.argmax()) for rl, ll in zip(ref_logits, lb_logits)]
    lb_margins = [margin(ll) for ll in lb_logits]      # certificate uses the low-bit margin (runtime-available)
    n = len(flips); nflip = sum(flips)
    print(f"[lowbit] bits={args.bits} group={args.group_size} experts_quantized={nq} tokens={n} "
          f"flip_rate={nflip/n:.4f} ({nflip}/{n})")
    # certificate sweep: among tokens with low-bit margin > tau, how many flipped, and frac kept.
    # HONEST eff_PCIe for exactness-preserving "low-bit-first, then full re-fetch on uncertain":
    # the margin is only known AFTER fetching low-bit, so EVERY token pays low-bit (1/ratio); the
    # (1-frac_safe) uncertain tokens additionally re-fetch full (+1). avg_bytes = 1/ratio + (1-frac_safe).
    # (The earlier 1/(frac_safe/ratio + (1-frac_safe)) was optimistic: it assumed safe tokens skip the
    #  full fetch WITHOUT first paying low-bit, which is not realizable. Caveat: a re-fetched token also
    #  needs its KV/state recomputed at full precision to be truly exact -- not modeled here.)
    ratio = 16 / args.bits
    for tau in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        safe = [i for i in range(n) if lb_margins[i] > tau]
        flip_in_safe = sum(flips[i] for i in safe)
        frac_safe = len(safe) / n
        eff = 1.0 / (1.0 / ratio + (1 - frac_safe))
        print(f"  tau={tau:4.1f}  frac_safe={frac_safe:.3f}  flips_in_safe={flip_in_safe}  eff_PCIe={eff:.2f}x")


if __name__ == "__main__":
    main()
