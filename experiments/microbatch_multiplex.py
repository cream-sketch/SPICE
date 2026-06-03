"""Kill experiment for codex idea #1: uniformity -> statistical multiplexing.

近乎均匀的专家访问对 caching 是诅咒, 但对跨流复用是祝福: B 个独立 token 一起处理,
每层每个唯一专家只加载一次 -> bytes/token = unique_experts_per_layer / (B*top_k).
测真实路由: 随机抽 B 个 token(跨序列/位置)组成 microbatch, 每层取 top-k 并集的唯一数,
多次平均. 对比理论 60*(1-(1-k/E)^B)/(k*B). B={1,2,4,8,16,32,64}.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch

def load_pool(trace_dir, top_k, max_traces):
    man=json.loads((Path(trace_dir)/"manifest.json").read_text())
    files=man.get("trace_files",man.get("files",[]))[:max_traces]
    # pool[layer] = list of frozensets (top-k expert ids) over all tokens
    pool=None; E=0
    for f in files:
        d=torch.load(Path(trace_dir)/f,map_location="cpu",weights_only=False)
        if "topk_idx" in d:
            II=d["topk_idx"]; L=len(II)
            if not L: continue
            II=[(x.reshape(-1,x.shape[-1]) if x.ndim==3 else x) for x in II]
            T=min(x.shape[0] for x in II)
            if d.get("scores"): E=max(E,d["scores"][0].shape[-1])
            TK=[II[l][:T,:top_k].long().tolist() for l in range(L)]
        else:
            probs=d["router_probs"]
            if not probs: continue
            P=[(p.float().reshape(-1,p.shape[-1]) if p.ndim==3 else p.float()) for p in probs]
            L=len(P); T=min(x.shape[0] for x in P); E=max(E,P[0].shape[-1])
            TK=[torch.topk(P[l][:T],k=top_k,dim=-1).indices.tolist() for l in range(L)]
        if pool is None: pool=[[] for _ in range(L)]
        for l in range(L):
            for t in range(T): pool[l].append(TK[l][t])
    return pool, E

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace_dir",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--top_k",type=int,default=4); ap.add_argument("--max_traces",type=int,default=80)
    ap.add_argument("--Bs",type=str,default="1,2,4,8,16,32,64,128"); ap.add_argument("--samples",type=int,default=2000)
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()
    random.seed(args.seed)
    pool,E=load_pool(args.trace_dir,args.top_k,args.max_traces)
    L=len(pool)
    rows=[]
    for B in [int(x) for x in args.Bs.split(",")]:
        uniq_acc=0.0; n=0
        for _ in range(args.samples):
            l=random.randrange(L)
            idxs=random.choices(pool[l],k=B)         # B tokens at this layer (independent streams)
            u=set()
            for s in idxs: u.update(s)
            uniq_acc+=len(u); n+=1
        uniq=uniq_acc/n
        byte_ratio=uniq/(B*args.top_k)               # loaded experts / demanded slots
        theo=E*(1-(1-args.top_k/E)**B)/(args.top_k*B)
        rows.append({"B":B,"unique_experts_per_layer":uniq,"byte_ratio_vs_single":byte_ratio,
                     "byte_reduction_x":1.0/byte_ratio,"theory_byte_ratio":theo})
        print(f"B={B:>4} unique/layer={uniq:6.1f}/{E} byte_ratio={byte_ratio:.3f} ({1/byte_ratio:.2f}x reduction) theory={theo:.3f}",flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps({"experts":E,"top_k":args.top_k,"layers":L,"rows":rows},indent=2))

if __name__=="__main__":
    main()
