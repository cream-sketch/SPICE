"""CORRECTED token-conditional routing analysis (audit fix for the mis-scaled metric).

之前 bug: top/(n*top_k) 最大只能到 1/top_k -> 误判"非 token 决定". 这里正确度量:
对每个 (layer, token_id) 用 LEAVE-ONE-OUT: 用其它出现构建该 token 的历史 modal top-K 专家,
预测留出那次的真实 top-k, 报 recall (覆盖率). 对比随机基线 K/E. 分层(早/中/晚)报告.
若 token->expert 预测 recall 远高于随机 -> token 身份是强可利用信号 -> 重开方向.
读 Qwen(router_probs+input_ids) 或 ds_routing(topk_idx+input_ids).
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import torch

def load(trace_dir, top_k, max_traces):
    man=json.loads((Path(trace_dir)/"manifest.json").read_text())
    files=man.get("trace_files",man.get("files",[]))[:max_traces]
    seqs=[]  # each: (TK[L][T] list, ids[T])
    for f in files:
        d=torch.load(Path(trace_dir)/f,map_location="cpu",weights_only=False)
        ids=d.get("input_ids")
        if ids is None: continue
        if "topk_idx" in d:
            II=d["topk_idx"];
            if not II: continue
            II=[(x.reshape(-1,x.shape[-1]) if x.ndim==3 else x) for x in II]
            T=min(x.shape[0] for x in II); L=len(II)
            TK=[II[l][:T,:top_k].long().tolist() for l in range(L)]
        else:
            probs=d["router_probs"]
            if not probs: continue
            P=[(p.float().reshape(-1,p.shape[-1]) if p.ndim==3 else p.float()) for p in probs]
            L=len(P); T=min(x.shape[0] for x in P)
            TK=[torch.topk(P[l][:T],k=top_k,dim=-1).indices.tolist() for l in range(L)]
        idl=ids.reshape(-1).tolist()[:T]
        seqs.append((TK,idl,L))
    return seqs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace_dir",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--top_k",type=int,default=4); ap.add_argument("--experts",type=int,default=60)
    ap.add_argument("--max_traces",type=int,default=80); ap.add_argument("--min_occ",type=int,default=5)
    args=ap.parse_args()
    seqs=load(args.trace_dir,args.top_k,args.max_traces)
    L=max(s[2] for s in seqs)
    # gather per (layer, token_id) -> list of top-k sets (across all positions/seqs)
    occ=defaultdict(list)
    for TK,idl,Ls in seqs:
        for l in range(Ls):
            for t in range(len(idl)):
                occ[(l,idl[t])].append(tuple(TK[l][t]))
    # leave-one-out: for each occurrence, predict from the OTHER occurrences' modal top-K
    import random; random.seed(0)
    per_layer_recall=defaultdict(list); per_layer_n=defaultdict(int)
    overall=[]
    for (l,tid),lst in occ.items():
        if len(lst)<args.min_occ: continue
        # build frequency over experts from all occ
        from collections import Counter
        for i in range(len(lst)):
            train=lst[:i]+lst[i+1:]
            cnt=Counter()
            for s in train:
                for e in s: cnt[e]+=1
            pred=set([e for e,_ in cnt.most_common(args.top_k)])
            actual=set(lst[i])
            rec=len(pred&actual)/args.top_k
            per_layer_recall[l].append(rec); overall.append(rec)
        per_layer_n[l]+=1
    rnd=args.top_k/args.experts
    layer_means={l:sum(v)/len(v) for l,v in sorted(per_layer_recall.items()) if v}
    res={"top_k":args.top_k,"experts":args.experts,"random_recall":rnd,
         "overall_token_pred_recall":sum(overall)/max(1,len(overall)),
         "n_predictions":len(overall),
         "recall_by_layer":layer_means,
         "early_layers_mean":sum(list(layer_means.values())[:max(1,L//3)])/max(1,len(list(layer_means.values())[:max(1,L//3)])),
         "late_layers_mean":sum(list(layer_means.values())[-max(1,L//3):])/max(1,len(list(layer_means.values())[-max(1,L//3):]))}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(res,indent=2))
    print(json.dumps({k:res[k] for k in ["random_recall","overall_token_pred_recall","early_layers_mean","late_layers_mean","n_predictions"]},indent=2))
    print(f"[INTERP] token->expert leave-one-out recall {res['overall_token_pred_recall']:.3f} vs random {rnd:.3f} "
          f"(ratio {res['overall_token_pred_recall']/rnd:.1f}x). High -> token identity is a STRONG exploitable signal -> REOPENS direction.")

if __name__=="__main__":
    main()
