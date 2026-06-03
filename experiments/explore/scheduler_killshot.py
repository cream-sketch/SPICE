"""A kill-shot: is the prefetch-depth scheduler roofline-derivable? (bandit decoration test)

codex A 裁决: 若 analytic-roofline 深度调度器 ~= per-regime-oracle, 则无可学 -> bandit 装饰.
延迟模拟器(无需模型), oracle 预取(隔离调度 vs 预测). per (token,layer): demand miss -> DMA
stall + LS evict; prefetch next-`depth` layers' ORACLE experts, deadline-aware DMA (multi-layer
in-flight, demand-priority). 扫 regime (bw x cache), 每 regime 比较三种 depth 策略:
  static(tuned@ref regime) / roofline(l_min from bw) / oracle(grid-best depth).
若 roofline 的 stall ~= oracle 的 stall 跨所有 regime -> bandit 无价值 -> 杀 A.
"""
from __future__ import annotations
import argparse, json, math
from collections import OrderedDict
from pathlib import Path
import torch

def load_routes(trace_dir, top_k):
    man=json.loads((Path(trace_dir)/"manifest.json").read_text())
    seqs=[]; L=0
    for f in man["trace_files"]:
        d=torch.load(Path(trace_dir)/f, map_location="cpu", weights_only=False)
        probs=d["router_probs"]
        if not probs: continue
        tk=[torch.topk(p.float().reshape(-1,p.shape[-1]) if p.ndim==3 else p.float(),k=top_k,dim=-1).indices for p in probs]
        L=max(L,len(tk)); tc=min(t.shape[0] for t in tk)
        seqs.append([[tk[l][tok].tolist() for l in range(len(tk))] for tok in range(tc)])
    return seqs, L

def sim(seqs, num_layers, depth, capacity, expert_bytes, bw_gbps, t_layer_ms, max_tokens):
    """oracle-prefetch next `depth` layers; deadline-aware DMA; LS evict; return stall_ms/token."""
    Bpm=bw_gbps*(1024**3)/1000.0; fetch_ms=expert_bytes/Bpm
    cache=OrderedDict(); now=0.0; dma_free=0.0; stall=0.0; step=0; toks=0
    for seq in seqs:
        cache.clear(); now=0.0; dma_free=0.0
        for per_layer in seq:
            if toks>=max_tokens: break
            toks+=1
            for l in range(num_layers):
                step+=1
                # demand
                for e in per_layer[l]:
                    key=(l,e)
                    ent=cache.get(key)
                    if ent is not None and ent["ready"]<=now:
                        ent["last"]=step; cache.move_to_end(key)
                    elif ent is not None:
                        stall+=ent["ready"]-now; now=ent["ready"]; ent["last"]=step; cache.move_to_end(key)
                    else:
                        arrive=now+fetch_ms; stall+=fetch_ms; dma_free=max(dma_free,now)+fetch_ms; now=arrive
                        protect={(l,x) for x in per_layer[l]}
                        while len(cache)>=capacity and cache:
                            cand=[k for k in cache if k not in protect] or list(cache.keys())
                            v=max(cand,key=lambda k:(((cache[k]["layer"]-l)%num_layers) or num_layers,-cache[k]["last"]))
                            cache.pop(v)
                        cache[key]={"ready":arrive,"last":step,"layer":l}; cache.move_to_end(key)
                now+=t_layer_ms
                # oracle prefetch next `depth` layers
                for d in range(1,depth+1):
                    tl=l+d
                    if tl>=num_layers: break
                    for e in per_layer[tl]:
                        key=(tl,e)
                        if key in cache: continue
                        start=max(now,dma_free); arrive=start+fetch_ms; dma_free=arrive
                        protect={(l,x) for x in per_layer[l]}|{key}
                        while len(cache)>=capacity and cache:
                            cand=[k for k in cache if k not in protect] or list(cache.keys())
                            v=max(cand,key=lambda k:(((cache[k]["layer"]-l)%num_layers) or num_layers,-cache[k]["last"]))
                            cache.pop(v)
                        cache[key]={"ready":arrive,"last":step,"layer":tl}; cache.move_to_end(key)
        if toks>=max_tokens: break
    return stall/max(1,toks)

def l_min(top_k, expert_bytes, bw_gbps, t_layer_ms):
    Bpm=bw_gbps*(1024**3)/1000.0
    return max(1, math.ceil(top_k*expert_bytes/(Bpm*t_layer_ms)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace_dir",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--top_k",type=int,default=4); ap.add_argument("--expert_mb",type=float,default=17.0)
    ap.add_argument("--t_layer_ms",type=float,default=0.4); ap.add_argument("--max_tokens",type=int,default=3000)
    ap.add_argument("--bws",type=str,default="5,12,24"); ap.add_argument("--caps",type=str,default="72,144,288")
    ap.add_argument("--depths",type=str,default="1,2,3,4,6,8,10,12")
    ap.add_argument("--ref_bw",type=float,default=12); ap.add_argument("--ref_cap",type=int,default=144)
    args=ap.parse_args()
    seqs,L=load_routes(args.trace_dir,args.top_k); eb=int(args.expert_mb*1024*1024)
    bws=[float(x) for x in args.bws.split(",")]; caps=[int(x) for x in args.caps.split(",")]
    depths=[int(x) for x in args.depths.split(",")]
    # tuned-static depth = oracle-best depth at ref regime
    ref={d:sim(seqs,L,d,args.ref_cap,eb,args.ref_bw,args.t_layer_ms,args.max_tokens) for d in depths}
    static_depth=min(ref,key=ref.get)
    rows=[]
    for bw in bws:
        for cap in caps:
            per_depth={d:sim(seqs,L,d,cap,eb,bw,args.t_layer_ms,args.max_tokens) for d in depths}
            oracle_d=min(per_depth,key=per_depth.get); oracle_stall=per_depth[oracle_d]
            rf_d=min(max(depths[0],l_min(args.top_k,eb,bw,args.t_layer_ms)),depths[-1])
            rf_d=min(depths,key=lambda d:abs(d-rf_d))  # nearest available depth
            roofline_stall=per_depth[rf_d]; static_stall=per_depth[static_depth]
            rows.append({"bw":bw,"cap":cap,"oracle_depth":oracle_d,"oracle_stall":oracle_stall,
                "roofline_depth":rf_d,"roofline_stall":roofline_stall,"static_depth":static_depth,"static_stall":static_stall,
                "roofline_regret":(roofline_stall-oracle_stall)/max(1e-9,oracle_stall),
                "static_regret":(static_stall-oracle_stall)/max(1e-9,oracle_stall)})
            print(f"bw={bw:>4} cap={cap:>4} oracle(d={oracle_d},{oracle_stall:.2f}) roofline(d={rf_d},{roofline_stall:.2f},reg={rows[-1]['roofline_regret']*100:.1f}%) static(d={static_depth},reg={rows[-1]['static_regret']*100:.1f}%)",flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps({"static_depth":static_depth,"rows":rows},indent=2))
    import statistics as st
    print(f"\nMEAN roofline_regret={st.mean(r['roofline_regret'] for r in rows)*100:.1f}%  static_regret={st.mean(r['static_regret'] for r in rows)*100:.1f}%")
    print("[KILL-SHOT] if roofline_regret small across regimes -> bandit decoration -> kill A")

if __name__=="__main__":
    main()
