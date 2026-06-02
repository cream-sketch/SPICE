"""Gating #2 (DeepSeek): quality cost (PPL) of importance-based expert dropping.

干净做法: wrap DeepseekV2 的 MoEGate.forward, 把每 token 最低 drop_n 个 routed 权重置零
(等价于 drop 这些低重要专家), 不重写复杂 MoE forward. 测 WikiText PPL vs 不 drop.
DeepSeek-V2-Lite: 64 routed + 2 shared, top-6; 若路由偏斜, drop 低 rank 近乎免费.
"""
from __future__ import annotations
import argparse, json, types, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DROP_N = 0

def make_wrapper(mod):
    orig = mod.forward
    def f(hidden_states):
        idx, w, aux = orig(hidden_states)
        if DROP_N > 0:
            wf = w.float()
            order = torch.argsort(wf, dim=-1)          # ascending; lowest first
            drop_pos = order[:, :DROP_N]
            w = w.scatter(1, drop_pos, torch.zeros_like(w))
        return idx, w, aux
    return f

@torch.no_grad()
def ppl(model, tok, texts, device, max_len):
    nll, ntok = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        ids = enc["input_ids"]
        if ids.shape[1] < 2: continue
        out = model(input_ids=ids, labels=ids)
        n = ids.shape[1] - 1
        nll += out.loss.item() * n; ntok += n
    return math.exp(nll / max(1, ntok))

def main():
    global DROP_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--drops", type=str, default="0,1,2,3,4,5")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,
                                                 local_files_only=True, trust_remote_code=True,
                                                 low_cpu_mem_usage=True).to(device).eval()
    # wrap every MoEGate
    n_wrap = 0
    for mod in model.modules():
        if type(mod).__name__ == "MoEGate":
            mod.forward = make_wrapper(mod); n_wrap += 1
    print(f"wrapped {n_wrap} MoEGate modules")
    texts = [l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:args.max_samples]
    res = {}
    for dn in [int(x) for x in args.drops.split(",")]:
        DROP_N = dn
        res[f"drop_{dn}"] = ppl(model, tok, texts, device, args.max_len)
        print(f"DROP_N={dn} (drop lowest {dn} of top-6) PPL={res[f'drop_{dn}']:.4f}", flush=True)
    base = res.get("drop_0", 1.0)
    out = {"model": args.model_dir, "ppl": res,
           "ppl_rel_pct_vs_full": {k: (v/base-1.0)*100 for k,v in res.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
