"""First-principles OBSERVATION battery on real MoE routing structure.

观察实验(非策略): 测 MoE 专家访问的内在结构, 判断 caching 的可行性上限与可利用结构.
A. popularity skew + static hot-set hit-rate curve (固定缓存最热N个/层 -> 命中率 vs N)
B. routing autocorrelation vs lag (top-k Jaccard at same layer, lag 1/2/4/8/16)
C. token-conditional determinism (路由由 token-id 决定 还是 context? -> 高信息信号?)
D. working-set size per layer over window W (thrashing 本质?)
读 trace 的 router_probs + input_ids.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import torch

def jaccard(a,b):
    a=set(a); b=set(b); u=len(a|b); return len(a&b)/u if u else 1.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace_dir",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--top_k",type=int,default=4); ap.add_argument("--max_traces",type=int,default=60)
    args=ap.parse_args()
    man=json.loads((Path(args.trace_dir)/"manifest.json").read_text())
    files=man["trace_files"][:args.max_traces]
    # gather per-layer topk lists per trace, + token ids
    L=0; E=0
    layer_freq=None; total_demands=0
    autocorr={lag:[] for lag in [1,2,4,8,16]}
    ws_window=8; working_set=[]   # distinct experts per layer over window
    tok_expert=defaultdict(lambda: defaultdict(int))  # (layer,tokid)->expert->count
    tok_count=defaultdict(int)                          # (layer,tokid)->n
    hotset_hits=None; hotset_total=0
    for f in files:
        d=torch.load(Path(args.trace_dir)/f,map_location="cpu",weights_only=False)
        probs=d["router_probs"];
        if not probs: continue
        ids=d.get("input_ids")
        P=[(p.float().reshape(-1,p.shape[-1]) if p.ndim==3 else p.float()) for p in probs]
        L=len(P); T=min(x.shape[0] for x in P); E=P[0].shape[-1]
        TK=[torch.topk(P[l][:T],k=args.top_k,dim=-1).indices for l in range(L)]  # [T,k]
        if layer_freq is None: layer_freq=[torch.zeros(E) for _ in range(L)]
        for l in range(L):
            for t in range(T):
                for e in TK[l][t].tolist(): layer_freq[l][e]+=1; total_demands+=1
        # autocorr
        for lag in autocorr:
            for l in range(L):
                for t in range(T-lag):
                    autocorr[lag].append(jaccard(TK[l][t].tolist(),TK[l][t+lag].tolist()))
        # working set over window
        for l in range(L):
            for t in range(0,T-ws_window,ws_window):
                s=set()
                for tt in range(t,t+ws_window):
                    s.update(TK[l][tt].tolist())
                working_set.append(len(s))
        # token-conditional
        if ids is not None:
            idl=ids.reshape(-1).tolist()[:T]
            for l in range(L):
                for t in range(T):
                    key=(l,idl[t]); tok_count[key]+=1
                    for e in TK[l][t].tolist(): tok_expert[key][e]+=1
    # A. popularity skew + static hot-set hit rate
    import math
    def gini(x):
        x=sorted(x); n=len(x); s=sum(x)
        if s==0: return 0.0
        return (2*sum((i+1)*v for i,v in enumerate(x)))/(n*s)-(n+1)/n
    mean_gini=sum(gini(layer_freq[l].tolist()) for l in range(L))/L
    hot_curve={}
    for N in [4,8,16,32]:
        hit=0; tot=0
        for l in range(L):
            hot=set(torch.topk(layer_freq[l],k=min(N,E)).indices.tolist())
            # fraction of demands served by hot set = sum freq of hot / total freq
            tot_l=layer_freq[l].sum().item(); hit+=sum(layer_freq[l][e].item() for e in hot); tot+=tot_l
        hot_curve[N]=hit/max(1,tot)
    # B autocorr
    ac={lag:sum(v)/len(v) for lag,v in autocorr.items() if v}
    # C token-conditional: avg top-expert consistency P(mode expert | token) and conditional vs marginal
    cons=[]; nkeys=0
    for key,exp in tok_expert.items():
        n=tok_count[key]
        if n<3: continue   # need repeats
        top=max(exp.values()); cons.append(top/(n*args.top_k))  # fraction of slots on the single most-common expert
        nkeys+=1
    tok_consistency=sum(cons)/max(1,len(cons))
    # D working set
    ws_mean=sum(working_set)/max(1,len(working_set))
    res={"layers":L,"experts":E,"top_k":args.top_k,
         "popularity_gini_mean":mean_gini,
         "static_hotset_hit_rate":hot_curve,    # cache N hottest/layer -> hit fraction
         "autocorr_jaccard_by_lag":ac,
         "token_conditional_consistency":tok_consistency,  # 0=random, 1=token fully determines expert
         "token_keys_with_repeats":nkeys,
         "working_set_mean_per_layer_window8":ws_mean,
         "working_set_vs_experts":ws_mean/E}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))

if __name__=="__main__":
    main()
