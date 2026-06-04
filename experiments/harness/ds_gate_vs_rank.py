"""DeepSeek gate-vs-rank QUALITY ablation (full-sequence teacher-forced, on-policy).

Experiment B (generalization): does verified gate-weight drop beat fixed rank
drop at matched drop rate on DeepSeek-V2-Lite? Full-sequence teacher-forced
(drops perturb downstream via causal attention within the forward). No cache sim
needed for the quality axis. Mode gate: zero routed weights < threshold. Mode
rank: zero lowest rank_drop of top-k. Reports actual drop rate + PPL.
"""
from __future__ import annotations
import argparse, json, types, math
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODE="gate"; THR=0.0; RANK_DROP=0; MASS_P=1.0
STAT={"total":0,"dropped":0}

def make_wrapper(mod):
    orig=mod.forward
    def f(hidden_states):
        idx,w,aux=orig(hidden_states)
        wf=w.float()
        STAT["total"]+=wf.numel()
        if MODE=="gate":
            mask=wf<THR
        elif MODE=="rank":
            order=torch.argsort(wf,dim=-1)  # ascending
            mask=torch.zeros_like(wf,dtype=torch.bool)
            if RANK_DROP>0: mask.scatter_(1, order[:,:RANK_DROP], True)
        else:  # mass / top-p: keep smallest set with cumsum(desc weight)>=MASS_P, drop tail
            sw,si=torch.sort(wf,dim=-1,descending=True)
            denom=sw.sum(-1,keepdim=True).clamp_min(1e-9)
            cum=torch.cumsum(sw,dim=-1)/denom
            keep=cum<MASS_P                      # positions strictly before reaching p
            keep[:,0]=True                       # always keep top-1
            keep_shift=torch.zeros_like(keep); keep_shift[:,1:]=cum[:,:-1]<MASS_P
            keep=keep|keep_shift|(torch.arange(sw.shape[1],device=wf.device)[None,:]==0)
            drop_sorted=~keep
            mask=torch.zeros_like(wf,dtype=torch.bool)
            mask.scatter_(1, si, drop_sorted)
        STAT["dropped"]+=int(mask.sum().item())
        w=w.masked_fill(mask, 0.0)
        return idx,w,aux
    return f

@torch.no_grad()
def ppl(model,tok,texts,device,max_len):
    nll=0.0;ntok=0
    for t in texts:
        enc=tok(t,return_tensors="pt",truncation=True,max_length=max_len).to(device)
        ids=enc["input_ids"]
        if ids.shape[1]<2: continue
        out=model(input_ids=ids,labels=ids); n=ids.shape[1]-1
        nll+=out.loss.item()*n; ntok+=n
    return math.exp(nll/max(1,ntok))

def main():
    global MODE,THR,RANK_DROP,MASS_P,STAT
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir",required=True);ap.add_argument("--text_file",required=True)
    ap.add_argument("--out",required=True);ap.add_argument("--gpu",type=int,default=0)
    ap.add_argument("--max_samples",type=int,default=24);ap.add_argument("--max_len",type=int,default=256)
    ap.add_argument("--gate_thresholds",type=str,default="0,0.02,0.05,0.1,0.2")
    ap.add_argument("--rank_drops",type=str,default="0,1,2,3,4")
    ap.add_argument("--mass_ps",type=str,default="1.0,0.99,0.95,0.9,0.8")
    args=ap.parse_args()
    device=torch.device(f"cuda:{args.gpu}")
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,torch_dtype=torch.bfloat16,
            local_files_only=True,trust_remote_code=True,low_cpu_mem_usage=True).to(device).eval()
    for mod in model.modules():
        if type(mod).__name__=="MoEGate": mod.forward=make_wrapper(mod)
    texts=[l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:args.max_samples]
    rows=[]
    for th in [float(x) for x in args.gate_thresholds.split(",")]:
        MODE="gate";THR=th;STAT={"total":0,"dropped":0}
        p=ppl(model,tok,texts,device,args.max_len); dr=STAT["dropped"]/max(1,STAT["total"])
        rows.append({"policy":"gate","knob":th,"drop_rate":dr,"ppl":p})
        print(f"gate thr={th} drop={dr:.3f} ppl={p:.4f}",flush=True)
    for rd in [int(x) for x in args.rank_drops.split(",")]:
        MODE="rank";RANK_DROP=rd;STAT={"total":0,"dropped":0}
        p=ppl(model,tok,texts,device,args.max_len); dr=STAT["dropped"]/max(1,STAT["total"])
        rows.append({"policy":"rank","knob":rd,"drop_rate":dr,"ppl":p})
        print(f"rank drop={rd} drop={dr:.3f} ppl={p:.4f}",flush=True)
    for mp in [float(x) for x in args.mass_ps.split(",")]:
        MODE="mass";MASS_P=mp;STAT={"total":0,"dropped":0}
        p=ppl(model,tok,texts,device,args.max_len); dr=STAT["dropped"]/max(1,STAT["total"])
        rows.append({"policy":"mass","knob":mp,"drop_rate":dr,"ppl":p})
        print(f"mass p={mp} drop={dr:.3f} ppl={p:.4f}",flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps({"model":args.model_dir,"rows":rows},indent=2))
    print(f"[done] {args.out}")

if __name__=="__main__":
    main()
