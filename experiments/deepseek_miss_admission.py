"""DeepSeek-V2-Lite online teacher-forced verified miss-admission (generalization).

泛化实验 B: 在 DeepSeek (64 routed + 2 shared, top-6) 上重复 Qwen 的 verified
importance-aware miss admission + gate-vs-rank, 验证主贡献跨模型成立.
token-by-token, live KV; wrap MoEGate 施加 controller; drop 改变下游 (on-policy).
"""
from __future__ import annotations
import argparse, json, math, types
from collections import OrderedDict
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class Controller:
    def __init__(self, capacity, fetch_ms, threshold, num_layers, policy='gate', rank_keep=6):
        self.capacity=capacity; self.fetch_ms=fetch_ms; self.threshold=threshold
        self.num_layers=num_layers; self.policy=policy; self.rank_keep=rank_keep; self.reset()
    def reset(self):
        self.reset_cache(); self.stall_ms=0.0; self.total=0; self.hits=0
        self.fetched=0; self.dropped=0; self.dropped_mass=0.0
    def reset_cache(self):
        self.cache=OrderedDict(); self.step=0
    def _evict(self, cur_layer, protect):
        cand=[k for k in self.cache if k not in protect] or list(self.cache.keys())
        victim=max(cand, key=lambda k:(((self.cache[k]["layer"]-cur_layer)%self.num_layers) or self.num_layers, -self.cache[k]["last"]))
        self.cache.pop(victim)
    def access(self, layer, experts, weights):
        self.step+=1; protect={(layer,e) for e in experts}; keep=set()
        for rank,(e,w) in enumerate(zip(experts,weights)):
            self.total+=1; key=(layer,e)
            admit=(w>=self.threshold) if self.policy=='gate' else (rank<self.rank_keep)
            if key in self.cache:
                self.hits+=1; self.cache[key]["last"]=self.step; self.cache.move_to_end(key); keep.add(e)
            elif admit:
                self.fetched+=1; self.stall_ms+=self.fetch_ms
                while len(self.cache)>=self.capacity and self.cache: self._evict(layer,protect)
                self.cache[key]={"last":self.step,"layer":layer}; self.cache.move_to_end(key); keep.add(e)
            else:
                self.dropped+=1; self.dropped_mass+=float(w)
        return keep

CTRL=None

def make_gate_wrapper(gate, layer_idx):
    orig=gate.forward
    def f(hidden_states):
        idx,w,aux=orig(hidden_states)  # idx,w: [N, top_k]
        N=idx.shape[0]
        w2=w.clone()
        for r in range(N):
            experts=idx[r].tolist(); weights=w[r].float().tolist()
            keep=CTRL.access(layer_idx, experts, weights)
            for j,e in enumerate(experts):
                if e not in keep: w2[r,j]=0.0
        return idx, w2, aux
    return f

@torch.no_grad()
def run_text(model, tok, text, device, max_tokens):
    enc=tok(text, return_tensors="pt", truncation=True, max_length=max_tokens+1).to(device)
    ids=enc["input_ids"]
    if ids.shape[1]<2: return 0.0,0
    past=None; nll=0.0; n=0
    for t in range(ids.shape[1]-1):
        out=model(input_ids=ids[:,t:t+1], past_key_values=past, use_cache=True, return_dict=True)
        past=out.past_key_values
        logp=F.log_softmax(out.logits[:,-1].float(),dim=-1)
        nll+=-logp[0,ids[0,t+1]].item(); n+=1
    return nll,n

def main():
    global CTRL
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir",required=True); ap.add_argument("--text_file",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--gpu",type=int,default=0)
    ap.add_argument("--max_samples",type=int,default=8); ap.add_argument("--max_tokens",type=int,default=96)
    ap.add_argument("--capacity",type=int,default=160); ap.add_argument("--expert_mb",type=float,default=11.0)
    ap.add_argument("--bandwidth_gbps",type=float,default=12.0)
    ap.add_argument("--policy",choices=["gate","rank"],default="gate")
    ap.add_argument("--thresholds",type=str,default="0,0.02,0.05,0.1,0.2")
    ap.add_argument("--rank_keeps",type=str,default="6,4,2,1")
    args=ap.parse_args()
    device=torch.device(f"cuda:{args.gpu}")
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,torch_dtype=torch.bfloat16,
            local_files_only=True,trust_remote_code=True,low_cpu_mem_usage=True).to(device).eval()
    layers=model.model.layers; num_layers=len(layers); nw=0
    for li,lyr in enumerate(layers):
        g=getattr(lyr.mlp,"gate",None)
        if g is not None and type(g).__name__=="MoEGate":
            lyr.mlp.gate.forward=make_gate_wrapper(lyr.mlp.gate,li); nw+=1
    print(f"wrapped {nw} MoEGate")
    fetch_ms=args.expert_mb/(args.bandwidth_gbps*1024)*1000.0
    texts=[l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:args.max_samples]
    knobs=[float(x) for x in args.thresholds.split(",")] if args.policy=='gate' else [int(x) for x in args.rank_keeps.split(",")]
    rows=[]
    for kn in knobs:
        if args.policy=='gate': CTRL=Controller(args.capacity,fetch_ms,kn,num_layers,'gate')
        else: CTRL=Controller(args.capacity,fetch_ms,-1.0,num_layers,'rank',rank_keep=kn)
        nll=0.0; ntok=0
        for text in texts:
            CTRL.reset_cache(); a,b=run_text(model,tok,text,device,args.max_tokens); nll+=a; ntok+=b
        ppl=math.exp(nll/max(1,ntok))
        rows.append({"knob":kn,"policy":args.policy,"ppl":ppl,
                     "stall_ms_per_token":CTRL.stall_ms/max(1,ntok),
                     "hit_rate":CTRL.hits/max(1,CTRL.total),"drop_rate":CTRL.dropped/max(1,CTRL.total),
                     "decode_tokens":ntok})
        print(f"{args.policy} knob={kn} ppl={ppl:8.3f} stall/tok={rows[-1]['stall_ms_per_token']:7.2f} "
              f"hit={rows[-1]['hit_rate']:.3f} drop={rows[-1]['drop_rate']:.3f}",flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps({"model":args.model_dir,"config":{"capacity":args.capacity,
        "expert_mb":args.expert_mb,"bandwidth_gbps":args.bandwidth_gbps,"fetch_ms":fetch_ms},"rows":rows},indent=2))
    print(f"[done] {args.out}")

if __name__=="__main__":
    main()
