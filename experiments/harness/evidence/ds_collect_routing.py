"""Collect REAL DeepSeek routing (expert IDs + full 64-dim scores) for structure analysis.
之前 ds_traces 因 collect 脚本双 softmax 丢了真实专家ID; 这里 wrap MoEGate 直接存 topk_idx 与 full scores.
"""
from __future__ import annotations
import argparse, json, types
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

STASH=[]  # per forward: list over layers of (topk_idx [N,k], full_scores [N,E])
def make_wrap(gate):
    orig=gate.forward
    def f(hidden_states):
        out=orig(hidden_states)
        # MoEGate returns (topk_idx, topk_weight, aux); also has self.weight -> recompute full scores
        logits=F.linear(hidden_states, gate.weight)
        scores=logits.softmax(dim=-1)
        STASH.append((out[0].detach().cpu(), scores.detach().float().cpu()))
        return out
    return f

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir",required=True);ap.add_argument("--text_file",required=True)
    ap.add_argument("--out_dir",required=True);ap.add_argument("--gpu",type=int,default=0)
    ap.add_argument("--max_samples",type=int,default=60);ap.add_argument("--max_len",type=int,default=256)
    args=ap.parse_args()
    dev=torch.device(f"cuda:{args.gpu}")
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,torch_dtype=torch.bfloat16,
            local_files_only=True,trust_remote_code=True,low_cpu_mem_usage=True).to(dev).eval()
    gates=[m for m in model.modules() if type(m).__name__=="MoEGate"]
    for g in gates: g.forward=make_wrap(g)
    texts=[l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:args.max_samples]
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); files=[]
    for i,t in enumerate(texts):
        STASH.clear()
        enc=tok(t,return_tensors="pt",truncation=True,max_length=args.max_len).to(dev)
        model(**enc)
        # STASH: L entries, each (idx[N,k], scores[N,E]); N=seq
        idxs=[a for a,_ in STASH]; scs=[b for _,b in STASH]
        torch.save({"topk_idx":idxs,"scores":scs,"input_ids":enc["input_ids"].cpu()}, out/f"r_{i:05d}.pt")
        files.append(f"r_{i:05d}.pt")
        if i%20==0: print(f"{i} layers={len(idxs)} E={scs[0].shape[-1] if scs else 0}",flush=True)
    (out/"manifest.json").write_text(json.dumps({"files":files}))
    print(f"[done] {len(files)} -> {out}")

if __name__=="__main__":
    main()
