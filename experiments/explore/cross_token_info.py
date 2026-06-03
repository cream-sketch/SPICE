"""Information-content probe: does the FULL router distribution carry cross-token
expert-reuse signal that selected-set / recency policies throw away?

用户洞察: 我们一直用低信息信号(top-k ID / 标量权重 / recency). 完整 router 概率分布
(含非选中专家)可能携带 cross-token 复用信息 -> 抓 oracle eviction 头room.
度量: 对每层, 用 prob[t][l][e] 预测 e 是否在 t+1 同层 top-k 的 AUC; 对比 selected[t]
(recency 类) 与 running frequency. 若 full-prob AUC >> selected -> 有被丢弃的高信息.
读 Qwen 完整 router_probs (60维).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

def auc(scores, labels):
    # scores, labels: 1D tensors; labels in {0,1}. AUC via rank statistic.
    order=torch.argsort(scores)
    ranks=torch.empty_like(order,dtype=torch.float); ranks[order]=torch.arange(1,len(scores)+1,dtype=torch.float)
    n_pos=labels.sum().item(); n_neg=len(labels)-n_pos
    if n_pos==0 or n_neg==0: return float('nan')
    sum_pos=ranks[labels==1].sum().item()
    return (sum_pos - n_pos*(n_pos+1)/2)/(n_pos*n_neg)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace_dir",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--top_k",type=int,default=4); ap.add_argument("--horizon",type=int,default=1)
    ap.add_argument("--max_traces",type=int,default=60)
    args=ap.parse_args()
    man=json.loads((Path(args.trace_dir)/"manifest.json").read_text())
    # accumulate per-(layer) score/label arrays for 3 predictors
    SP={"prob":[],"selected":[],"freq":[]}; LAB=[]
    files=man["trace_files"][:args.max_traces]
    for f in files:
        d=torch.load(Path(args.trace_dir)/f,map_location="cpu",weights_only=False)
        probs=d["router_probs"]
        if not probs: continue
        L=len(probs)
        # per layer: P[l] = [T, E]
        P=[ (p.float().reshape(-1,p.shape[-1]) if p.ndim==3 else p.float()) for p in probs]
        T=min(x.shape[0] for x in P); E=P[0].shape[-1]
        # top-k selection per layer per token
        SEL=[torch.zeros(T,E) for _ in range(L)]
        for l in range(L):
            idx=torch.topk(P[l][:T],k=args.top_k,dim=-1).indices
            SEL[l].scatter_(1,idx,1.0)
        freq=[torch.zeros(E) for _ in range(L)]
        for t in range(T-args.horizon):
            for l in range(L):
                win=SEL[l][t+1:t+1+args.horizon]               # next H tokens
                lab=(win.sum(0)>0).float()                     # selected WITHIN next H (any)
                SP["prob"].append(P[l][t]); SP["selected"].append(SEL[l][t]); SP["freq"].append(freq[l].clone())
                LAB.append(lab)
                freq[l]+=SEL[l][t]
    prob=torch.cat(SP["prob"]); sel=torch.cat(SP["selected"]); fr=torch.cat(SP["freq"]); lab=torch.cat(LAB)
    res={"horizon":args.horizon,"top_k":args.top_k,"n":int(lab.numel()),"pos_rate":float(lab.mean()),
         "AUC_full_prob":auc(prob,lab),"AUC_selected_recency":auc(sel,lab),"AUC_frequency":auc(fr,lab)}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))
    print(f"\n[INFO-CONTENT] full-prob AUC={res['AUC_full_prob']:.3f} vs selected/recency={res['AUC_selected_recency']:.3f} vs freq={res['AUC_frequency']:.3f}")
    print("若 full-prob 显著高 -> 完整分布有被丢弃的 cross-token 信息 -> 重开高信息 eviction")

if __name__=="__main__":
    main()
