"""Utility experiment: does a token->expert table beat LFU/LS for offloaded-MoE prefetch?

codex go/no-go: token-table+admission must beat LFU/static AND SpecMD-LS by >=10-15%
exposed stall at <= H2D, and close >=25% of LS->oracle gap. Else it's LFU-with-token-keys.
REAL decode traces; train/eval split by prompt; table built on TRAIN only. deadline-aware DMA.
"""
from __future__ import annotations
import argparse, json
from collections import OrderedDict, Counter, defaultdict
from pathlib import Path
import torch

def load(dir, frac_train=0.5):
    man=json.loads((Path(dir)/"manifest.json").read_text())
    files=man["files"]; L=man["num_layers"]; K=man["top_k"]; E=man["experts"]
    traces=[]
    for f in files:
        d=torch.load(Path(dir)/f,map_location="cpu",weights_only=False)
        steps=[(tid,per) for tid,per in d["steps"] if all(x is not None for x in per)]
        if steps: traces.append(steps)
    n=len(traces); ntr=max(1,int(n*frac_train))
    return traces[:ntr], traces[ntr:], L, K, E

def build_table(train, L, K):
    cnt=defaultdict(Counter)
    for steps in train:
        for tid,per in steps:
            for l in range(L):
                for e in per[l]: cnt[(l,tid)][e]+=1
    table={k:[e for e,_ in c.most_common(K)] for k,c in cnt.items()}
    return table

def simulate(eval_traces, L, K, table, policy, capacity, expert_bytes, bw_gbps, t_layer_ms, hot=None):
    Bpm=bw_gbps*(1024**3)/1000.0; fetch=expert_bytes/Bpm
    cache=OrderedDict()  # key->{ready,last,layer}
    now=0.0; dma=0.0; step=0
    hits=0; miss=0; tot=0; h2d_pf=0; h2d_dm=0; stall=0.0; covered=0; covtot=0
    occ=None; occ_ptr=None
    if policy=="oracle":
        # precompute global next-use over eval stream (flatten)
        flat=[]
        for steps in eval_traces:
            for tid,per in steps:
                for l in range(L):
                    for e in per[l]: flat.append((l,e))
        occ=defaultdict(list)
        for i,k in enumerate(flat): occ[k].append(i)
        occ_ptr={k:0 for k in occ}
        gpos=0
    def evict(protect,cur_layer,gp):
        cand=[k for k in cache if k not in protect] or list(cache.keys())
        if policy=="oracle":
            def nu(k):
                ps=occ.get(k,()); p=occ_ptr.get(k,0)
                while p<len(ps) and ps[p]<=gp: p+=1
                occ_ptr[k]=p; return ps[p] if p<len(ps) else 10**9
            v=max(cand,key=nu)
        elif policy in ("lfu","static"):
            v=min(cand,key=lambda k:(freq.get(k,0),cache[k]["last"]))
        else:  # ls + token/none
            v=max(cand,key=lambda k:(((cache[k]["layer"]-cur_layer)%L) or L,-cache[k]["last"]))
        cache.pop(v)
    freq=defaultdict(int)
    for steps in eval_traces:
        cache.clear(); now=0.0; dma=0.0
        for tid,per in steps:
            # token-table prefetch at token start (next-token-id known)
            if policy=="token":
                covtot+=1
                if any((l,tid) in table for l in range(L)): covered+=1
                for l in range(L):
                    for e in table.get((l,tid),[]):
                        key=(l,e)
                        if key in cache: continue
                        start=max(now,dma); arr=start+fetch; dma=arr
                        while len(cache)>=capacity and cache: evict(set(),0,0)
                        cache[key]={"ready":arr,"last":step,"layer":l}
                        h2d_pf+=expert_bytes
            for l in range(L):
                step+=1
                # demand
                for e in per[l]:
                    tot+=1; freq[(l,e)]+=1; key=(l,e)
                    if policy=="oracle": gpos+=1
                    ent=cache.get(key)
                    if ent is not None and ent["ready"]<=now:
                        hits+=1; ent["last"]=step; cache.move_to_end(key)
                    elif ent is not None:
                        stall+=ent["ready"]-now; now=ent["ready"]; hits+=1; ent["last"]=step
                    else:
                        miss+=1; h2d_dm+=expert_bytes; arr=now+fetch; stall+=fetch
                        dma=max(dma,now)+fetch; now=arr
                        prot={(l,x) for x in per[l]}
                        while len(cache)>=capacity and cache: evict(prot,l,gpos if policy=='oracle' else 0)
                        cache[key]={"ready":arr,"last":step,"layer":l}
                now+=t_layer_ms
    toks=sum(len(s) for s in eval_traces)
    return {"policy":policy,"capacity":capacity,"hit_rate":hits/max(1,tot),
            "exposed_stall_ms_per_token":stall/max(1,toks),
            "h2d_gb":(h2d_pf+h2d_dm)/(1024**3),"coverage":covered/max(1,covtot) if covtot else None,
            "tokens":toks}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--decode_dir",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--expert_mb",type=float,default=17.0); ap.add_argument("--bw_gbps",type=float,default=12.0)
    ap.add_argument("--t_layer_ms",type=float,default=0.4); ap.add_argument("--caps",type=str,default="72,144,288")
    args=ap.parse_args()
    train,ev,L,K,E=load(args.decode_dir)
    table=build_table(train,L,K); eb=int(args.expert_mb*1024*1024)
    print(f"L={L} K={K} E={E} train={len(train)} eval={len(ev)} table_entries={len(table)}")
    rows=[]
    for cap in [int(x) for x in args.caps.split(",")]:
        for pol in ["lru","lfu","ls","token","oracle"]:
            r=simulate(ev,L,K,table,pol,cap,eb,args.bw_gbps,args.t_layer_ms)
            rows.append(r)
            print(f"cap={cap:>4} {pol:>7} hit={r['hit_rate']:.3f} stall/tok={r['exposed_stall_ms_per_token']:7.2f} h2d_gb={r['h2d_gb']:.1f} cov={r['coverage']}",flush=True)
        # go/no-go vs LFU and LS, gap closed vs oracle
        d={r['policy']:r for r in rows if r['capacity']==cap}
        base=min(d['lfu']['exposed_stall_ms_per_token'],d['ls']['exposed_stall_ms_per_token'])
        tok=d['token']['exposed_stall_ms_per_token']; orc=d['oracle']['exposed_stall_ms_per_token']
        gain=(base-tok)/max(1e-9,base); gap=(d['ls']['exposed_stall_ms_per_token']-tok)/max(1e-9,d['ls']['exposed_stall_ms_per_token']-orc)
        print(f"  [cap={cap}] token vs best(LFU/LS) stall gain={gain*100:.1f}% ; LS->oracle gap closed={gap*100:.1f}%")
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps({"rows":rows},indent=2))

if __name__=="__main__":
    main()
