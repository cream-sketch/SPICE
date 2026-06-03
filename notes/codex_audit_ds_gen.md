AUDIT this DeepSeek-V2-Lite decode trace generator for correctness. It must produce the SAME format/semantics as the Qwen gen_decode_traces.py so spice_x_eviction_value.py (which shifts input_tid_g = generated_tid_{g-1}) works identically cross-model. Be harsh.

Checks:
1. MoEGate.forward wrap: does it capture the LAST position's top-k per MoE layer, once per forward, in layer order? STASH[-L_moe:] correctness across prefill (multi-token) vs decode (1 token)? Could shared experts or aux paths pollute STASH?
2. Alignment: each saved step = (generated_tid_g = argmax after forward, per_layer_g = routing of INPUT token x_g). Matches Qwen so the downstream input_tid shift is valid? Layer 0 is dense (no MoEGate) -> L_moe=26 captured; is mixing "MoE-layer index" as the cache layer key consistent (the value script treats layers 0..25 uniformly)?
3. attention_mask grown by 1 each step -- correct for KV-cache decode with this custom model?
4. Any bug causing per_layer to be misaligned to the wrong token, or to drop/duplicate layers? EOS early-stop (59 steps seen) OK?
5. Verdict: will the produced traces be a VALID cross-model replication for the eviction-value experiment, or is there a semantic mismatch with Qwen traces?
Full script attached.
"""DeepSeek-V2-Lite autoregressive decode trace generator (cross-model fairness for SPICE-X).

Same output format as gen_decode_traces.py so spice_x_eviction_value.py works unchanged:
  dec_*.pt: {"steps": [(generated_tid, [L_moe][top_k experts]), ...], "prompt_ids": [[...]], "num_layers": L_moe}
  manifest.json: {"files":[...], "top_k":6, "experts":64, "num_layers":L_moe}

DeepSeek-V2-Lite: layer 0 dense, layers 1.. are MoE (MoEGate, 64 routed top-6 + 2 shared). We capture
per-MoE-layer top-k of the LAST position each forward by wrapping MoEGate.forward. L_moe = number of
MoE layers (gates). All printed content English.
"""
import argparse, json, types
from pathlib import Path
import torch
import torch.nn.functional as F

STASH = []  # per forward: list over MoE layers of topk_idx[last_token] (list of k ints)


def make_wrap(gate):
    orig = gate.forward
    def f(hidden_states):
        out = orig(hidden_states)            # (topk_idx, topk_weight, aux)
        idx = out[0]                          # [N, k]
        STASH.append([int(x) for x in idx[-1].detach().cpu().tolist()])  # last position's experts
        return out
    return f


def parse_args():
    ap = argparse.ArgumentParser(description="DeepSeek-V2-Lite decode trace generator")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--n_prompts", type=int, required=True)
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--prompt_len", type=int, required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, low_cpu_mem_usage=True).to(dev).eval()
    gates = [m for m in model.modules() if type(m).__name__ == "MoEGate"]
    for g in gates:
        g.forward = make_wrap(g)  # wrap bound forward; captures last-token top-k per MoE layer
    L_moe = len(gates)
    n_experts = int(model.config.n_routed_experts)
    top_k = int(model.config.num_experts_per_tok)
    print(f"[cfg] MoE layers={L_moe} routed_experts={n_experts} top_k={top_k}", flush=True)

    texts = [l.strip() for l in Path(a.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:a.n_prompts]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); files = []
    for pi, t in enumerate(texts):
        enc = tok(t, return_tensors="pt", truncation=True, max_length=a.prompt_len).to(dev)
        ids = enc["input_ids"]; past = None; cur = ids; steps = []
        attn = enc["attention_mask"]  # DeepSeek custom code asserts attention_mask is not None
        with torch.no_grad():
            for g in range(a.gen):
                STASH.clear()
                out_m = model(input_ids=cur if past is None else cur[:, -1:],
                              attention_mask=attn, past_key_values=past, use_cache=True, return_dict=True)
                past = out_m.past_key_values
                nxt = int(out_m.logits[0, -1].argmax().item())
                per_layer = list(STASH[-L_moe:])  # this forward's MoE-layer top-k (last token)
                if len(per_layer) == L_moe:
                    steps.append((nxt, per_layer))
                cur = torch.tensor([[nxt]], device=dev)
                attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device=dev)], dim=1)  # grow mask
                if nxt == tok.eos_token_id:
                    break
        torch.save({"steps": steps, "prompt_ids": ids.cpu().tolist(), "num_layers": L_moe},
                   out / f"dec_{pi:05d}.pt")
        files.append(f"dec_{pi:05d}.pt")
        if pi % 10 == 0:
            print(f"prompt {pi}: {len(steps)} decode steps", flush=True)
    (out / "manifest.json").write_text(json.dumps({"files": files, "top_k": top_k,
                                                   "experts": n_experts, "num_layers": L_moe}))
    print(f"[done] {len(files)} traces -> {out}", flush=True)


if __name__ == "__main__":
    main()
