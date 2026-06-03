"""Candidate B premise: fraction of next-token argmax UNCHANGED when ALL routed experts are dropped
(shared expert + attention only). Teacher-forced on real text, per-position. High -> most tokens
don't need routed fetch -> selective routed execution (fetch routed only for 'routed-critical' tokens).
Also reports top-1 logit-margin separation between flip vs no-flip (can we PREDICT which need routed?).
"""
from __future__ import annotations
import argparse, json, types
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2_moe.modeling_qwen2_moe as Mq
DROP=False
def patched(mlp,h_in):
    b,s,d=h_in.shape; h=h_in.view(-1,d); rl=mlp.gate(h)
    final=torch.zeros_like(h)
    if not DROP:
        rw=F.softmax(rl,dim=1,dtype=torch.float); rw,sel=torch.topk(rw,mlp.top_k,dim=-1)
        if mlp.norm_topk_prob: rw=rw/rw.sum(-1,keepdim=True)
        rw=rw.to(h.dtype); mask=F.one_hot(sel,num_classes=mlp.num_experts).permute(2,1,0)
        for ei in range(mlp.num_experts):
            idx,topx=torch.where(mask[ei])
            if topx.numel()==0: continue
            final.index_add_(0,topx,(mlp.experts[ei](h[topx])*rw[topx,idx,None]).to(h.dtype))
    sh=mlp.shared_expert(h); sh=F.sigmoid(mlp.shared_expert_gate(h))*sh
    return (final+sh).view(b,s,d), rl
@torch.no_grad()
def main():
    global DROP
    ap=argparse.ArgumentParser(); ap.add_argument("--model_dir",required=True);ap.add_argument("--text_file",required=True)
    ap.add_argument("--out",required=True);ap.add_argument("--gpu",type=int,default=0);ap.add_argument("--n",type=int,default=30);ap.add_argument("--maxlen",type=int,default=256)
    a=ap.parse_args(); dev=torch.device(f"cuda:{a.gpu}")
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_dir,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).to(dev).eval()
    for lyr in model.model.layers:
        if isinstance(lyr.mlp,Mq.Qwen2MoeSparseMoeBlock): lyr.mlp.forward=types.MethodType(patched,lyr.mlp)
    texts=[l.strip() for l in Path(a.text_file).read_text().splitlines() if l.strip()][:a.n]
    agree=0;tot=0; margin_flip=[]; margin_keep=[]
    for t in texts:
        enc=tok(t,return_tensors="pt",truncation=True,max_length=a.maxlen).to(dev)
        DROP=False; full=model(**enc,use_cache=False,return_dict=True).logits[0].float()
        DROP=True;  sho =model(**enc,use_cache=False,return_dict=True).logits[0].float()
        fa=full.argmax(-1); sa=sho.argmax(-1)
        # shared-only top1-top2 margin as a predictor of whether routed flips it
        st=torch.topk(F.softmax(sho,-1),2,dim=-1).values; marg=(st[:,0]-st[:,1])
        for i in range(fa.shape[0]):
            tot+=1
            if int(fa[i])==int(sa[i]): agree+=1; margin_keep.append(float(marg[i]))
            else: margin_flip.append(float(marg[i]))
    import statistics as st
    res={"model":a.model_dir,"positions":tot,"shared_only_argmax_agree":agree/max(1,tot),
         "mean_margin_keep":st.mean(margin_keep) if margin_keep else None,
         "mean_margin_flip":st.mean(margin_flip) if margin_flip else None,
         "flip_rate":1-agree/max(1,tot)}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))
    print(f"[B-premise] {res['shared_only_argmax_agree']*100:.1f}% of tokens have SAME argmax without ANY routed expert.")
    print(f"  margin keep={res['mean_margin_keep']} vs flip={res['mean_margin_flip']} (separable -> can predict which need routed)")
if __name__=="__main__": main()
