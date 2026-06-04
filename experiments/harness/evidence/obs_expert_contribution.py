"""New viewpoint: expert OUTPUT contribution ||gate*E(x)|| vs router gate weight.

观察 MoE 计算侧信息(非路由): 每个选中专家对层输出的真实贡献范数, shared vs routed 扰动,
贡献的偏斜, 以及 gate 权重是否=贡献序. 若 routed 扰动小/贡献极偏斜/gate!=贡献 -> 新可丢弃信号.
"""
from __future__ import annotations
import argparse, json, types, math
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2_moe.modeling_qwen2_moe as Mq

REC=[]  # rows: dict per (token,layer)
def patched(mlp, hidden_states):
    b,s,d=hidden_states.shape
    h=hidden_states.view(-1,d)
    rl=mlp.gate(h); rw=F.softmax(rl,dim=1,dtype=torch.float)
    rw,sel=torch.topk(rw,mlp.top_k,dim=-1)
    if mlp.norm_topk_prob: rw=rw/rw.sum(dim=-1,keepdim=True)
    rw=rw.to(h.dtype)
    N=h.shape[0]
    routed=torch.zeros_like(h)
    contrib=torch.zeros(N,mlp.top_k,device=h.device)  # ||gate_i*E_i(h)|| per slot
    mask=F.one_hot(sel,num_classes=mlp.num_experts).permute(2,1,0)
    for ei in range(mlp.num_experts):
        idx,topx=torch.where(mask[ei])
        if topx.numel()==0: continue
        eo=mlp.experts[ei](h[topx])               # [n,d]
        w=rw[topx,idx,None]
        cw=(eo*w)
        routed.index_add_(0,topx,cw.to(h.dtype))
        contrib[topx,idx]=cw.norm(dim=-1).float()
    sh=mlp.shared_expert(h); sh=F.sigmoid(mlp.shared_expert_gate(h))*sh
    hn=h.norm(dim=-1).float(); rn=routed.norm(dim=-1).float(); shn=sh.norm(dim=-1).float()
    cs,_=torch.sort(contrib,dim=-1,descending=True)  # per-token sorted contribution
    for t in range(N):
        REC.append((float(hn[t]),float(rn[t]),float(shn[t]),cs[t].tolist(),rw[t].float().tolist()))
    return (routed+sh).view(b,s,d), rl

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir",required=True);ap.add_argument("--text_file",required=True)
    ap.add_argument("--out",required=True);ap.add_argument("--gpu",type=int,default=0)
    ap.add_argument("--max_samples",type=int,default=12);ap.add_argument("--max_len",type=int,default=128)
    args=ap.parse_args()
    dev=torch.device(f"cuda:{args.gpu}")
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,torch_dtype=torch.bfloat16,
            local_files_only=True,low_cpu_mem_usage=True).to(dev).eval()
    for lyr in model.model.layers:
        if isinstance(lyr.mlp,Mq.Qwen2MoeSparseMoeBlock): lyr.mlp.forward=types.MethodType(patched,lyr.mlp)
    texts=[l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:args.max_samples]
    for t in texts:
        enc=tok(t,return_tensors="pt",truncation=True,max_length=args.max_len).to(dev); model(**enc)
    import statistics as st
    def gini(x):
        x=sorted(x); n=len(x); s=sum(x)
        return 0.0 if s==0 else (2*sum((i+1)*v for i,v in enumerate(x)))/(n*s)-(n+1)/n
    hn=[r[0] for r in REC]; rn=[r[1] for r in REC]; shn=[r[2] for r in REC]
    routed_pert=[r[1]/r[0] for r in REC if r[0]>0]     # ||routed||/||h_in||
    shared_vs_routed=[r[2]/r[1] for r in REC if r[1]>0]
    contrib_gini=[gini(r[3]) for r in REC]
    # gate-weight rank vs contribution rank agreement: is argmax gate == argmax contribution?
    agree_top=sum(1 for r in REC if max(range(len(r[4])),key=lambda i:r[4][i])==max(range(len(r[3])),key=lambda i:r[3][i]) ) / max(1,len(REC))
    # is the LOWEST-gate expert also the lowest-contribution? (drop-decision relevance)
    agree_bot=sum(1 for r in REC if min(range(len(r[4])),key=lambda i:r[4][i])==min(range(len(r[3])),key=lambda i:r[3][i]) ) / max(1,len(REC))
    # mean per-rank contribution (sorted desc) normalized
    K=len(REC[0][3]); rank_contrib=[st.mean(r[3][k] for r in REC) for k in range(K)]
    res={"n":len(REC),"routed_pert_to_hidden_mean":st.mean(routed_pert),
         "shared_to_routed_mean":st.mean(shared_vs_routed),
         "contribution_gini_mean":st.mean(contrib_gini),
         "gate_top==contrib_top":agree_top,"gate_bot==contrib_bot":agree_bot,
         "mean_contrib_by_rank":rank_contrib}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))

if __name__=="__main__":
    main()
