"""Qwen gate/rank/mass drop QUALITY ablation (full-seq teacher-forced, on-policy).

三模式 miss-drop: gate(weight<thr) / rank(lowest n) / mass(top-p cumulative gate mass).
全序列 teacher-forced PPL (drop 经因果 attention 影响下游 = on-policy). 报实际 drop 率.
"""
from __future__ import annotations
import argparse, json, types, math
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2_moe.modeling_qwen2_moe as Mq

MODE="gate"; THR=0.0; RANK_DROP=0; MASS_P=1.0
STAT={"total":0,"dropped":0}

def patched_forward(mlp, hidden_states):
    b,s,d=hidden_states.shape
    h=hidden_states.view(-1,d)
    router_logits=mlp.gate(h)
    rw=F.softmax(router_logits,dim=1,dtype=torch.float)
    rw,sel=torch.topk(rw,mlp.top_k,dim=-1)         # descending
    if mlp.norm_topk_prob: rw=rw/rw.sum(dim=-1,keepdim=True)
    rwf=rw.float()
    STAT["total"]+=rwf.numel()
    if MODE=="gate":
        dmask=rwf<THR
    elif MODE=="rank":
        dmask=torch.zeros_like(rwf,dtype=torch.bool)
        if RANK_DROP>0: dmask[:, mlp.top_k-RANK_DROP:]=True   # lowest ranks (rw is descending)
    else:  # mass top-p
        denom=rwf.sum(-1,keepdim=True).clamp_min(1e-9)
        cum=torch.cumsum(rwf,dim=-1)/denom
        prev=torch.zeros_like(cum); prev[:,1:]=cum[:,:-1]
        keep=prev<MASS_P                                       # keep until cumulative(before)>=p
        dmask=~keep
    STAT["dropped"]+=int(dmask.sum().item())
    rw=rw.to(h.dtype).clone(); rw[dmask]=0.0
    final=torch.zeros((b*s,d),dtype=h.dtype,device=h.device)
    mask=F.one_hot(sel,num_classes=mlp.num_experts).permute(2,1,0)
    for ei in range(mlp.num_experts):
        idx,topx=torch.where(mask[ei])
        if topx.numel()==0: continue
        cur=h[None,topx].reshape(-1,d)
        out=mlp.experts[ei](cur)*rw[topx,idx,None]
        final.index_add_(0,topx,out.to(h.dtype))
    shared=mlp.shared_expert(h); shared=F.sigmoid(mlp.shared_expert_gate(h))*shared
    final=final+shared
    return final.view(b,s,d), router_logits

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
    ap.add_argument("--gate_thresholds",type=str,default="0,0.02,0.05,0.08,0.12")
    ap.add_argument("--rank_drops",type=str,default="0,1,2,3")
    ap.add_argument("--mass_ps",type=str,default="1.0,0.95,0.9,0.8,0.7")
    args=ap.parse_args()
    device=torch.device(f"cuda:{args.gpu}")
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,torch_dtype=torch.bfloat16,
            local_files_only=True,low_cpu_mem_usage=True).to(device).eval()
    for lyr in model.model.layers:
        if isinstance(lyr.mlp,Mq.Qwen2MoeSparseMoeBlock):
            lyr.mlp.forward=types.MethodType(patched_forward,lyr.mlp)
    texts=[l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:args.max_samples]
    rows=[]
    for th in [float(x) for x in args.gate_thresholds.split(",")]:
        MODE="gate";THR=th;STAT={"total":0,"dropped":0}
        p=ppl(model,tok,texts,device,args.max_len);dr=STAT["dropped"]/max(1,STAT["total"])
        rows.append({"policy":"gate","knob":th,"drop_rate":dr,"ppl":p});print(f"gate thr={th} drop={dr:.3f} ppl={p:.4f}",flush=True)
    for rd in [int(x) for x in args.rank_drops.split(",")]:
        MODE="rank";RANK_DROP=rd;STAT={"total":0,"dropped":0}
        p=ppl(model,tok,texts,device,args.max_len);dr=STAT["dropped"]/max(1,STAT["total"])
        rows.append({"policy":"rank","knob":rd,"drop_rate":dr,"ppl":p});print(f"rank drop={rd} drop={dr:.3f} ppl={p:.4f}",flush=True)
    for mp in [float(x) for x in args.mass_ps.split(",")]:
        MODE="mass";MASS_P=mp;STAT={"total":0,"dropped":0}
        p=ppl(model,tok,texts,device,args.max_len);dr=STAT["dropped"]/max(1,STAT["total"])
        rows.append({"policy":"mass","knob":mp,"drop_rate":dr,"ppl":p});print(f"mass p={mp} drop={dr:.3f} ppl={p:.4f}",flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps({"model":args.model_dir,"rows":rows},indent=2))
    print(f"[done] {args.out}")

if __name__=="__main__":
    main()
