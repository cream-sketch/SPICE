"""Generate REAL autoregressive DECODE traces (codex audit fix: prior traces were prefill).

逐 token 自回归生成, 每步捕获每层 top-k 专家 + 生成的 token-id. 用于 token-table 利用实验.
Qwen token-by-token + KV cache. 存: 每序列 [G步][L层 top-k] + token_ids(prompt+generated) + 标记 decode 起点.
"""
from __future__ import annotations
import argparse, json, types
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2_moe.modeling_qwen2_moe as Mq

CAP=[]  # per layer-call: top-k list for the current step (cleared each step)
def make_fwd(li):
    def f(mlp,hidden_states):
        b,s,d=hidden_states.shape; h=hidden_states.view(-1,d)
        rl=mlp.gate(h); rw=F.softmax(rl,dim=1,dtype=torch.float)
        rwv,sel=torch.topk(rw,mlp.top_k,dim=-1)
        # capture last token's selection (decode step = last position)
        CAP.append((li, sel[-1].tolist()))
        # normal compute
        if mlp.norm_topk_prob: rwv=rwv/rwv.sum(-1,keepdim=True)
        rwv=rwv.to(h.dtype)
        final=torch.zeros_like(h); mask=F.one_hot(sel,num_classes=mlp.num_experts).permute(2,1,0)
        for ei in range(mlp.num_experts):
            idx,topx=torch.where(mask[ei])
            if topx.numel()==0: continue
            final.index_add_(0,topx,(mlp.experts[ei](h[topx])*rwv[topx,idx,None]).to(h.dtype))
        sh=mlp.shared_expert(h); sh=F.sigmoid(mlp.shared_expert_gate(h))*sh
        return (final+sh).view(b,s,d), rl
    return f

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir",required=True);ap.add_argument("--text_file",required=True)
    ap.add_argument("--out_dir",required=True);ap.add_argument("--gpu",type=int,default=0)
    ap.add_argument("--n_prompts",type=int,default=40);ap.add_argument("--gen",type=int,default=128)
    ap.add_argument("--prompt_len",type=int,default=16)
    args=ap.parse_args()
    dev=torch.device(f"cuda:{args.gpu}")
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,torch_dtype=torch.bfloat16,
            local_files_only=True,low_cpu_mem_usage=True).to(dev).eval()
    layers=model.model.layers
    for li,lyr in enumerate(layers):
        if isinstance(lyr.mlp,Mq.Qwen2MoeSparseMoeBlock): lyr.mlp.forward=types.MethodType(make_fwd(li),lyr.mlp)
    texts=[l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:args.n_prompts]
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); files=[]
    L=len(layers)
    for pi,t in enumerate(texts):
        enc=tok(t,return_tensors="pt",truncation=True,max_length=args.prompt_len).to(dev)
        ids=enc["input_ids"]; past=None
        steps=[]  # per decode step: (token_id, [L][top_k])
        cur=ids
        for g in range(args.gen):
            CAP.clear()
            out_m=model(input_ids=cur if past is None else cur[:,-1:], past_key_values=past, use_cache=True, return_dict=True)
            past=out_m.past_key_values
            nxt=int(out_m.logits[0,-1].argmax().item())
            # CAP has entries for this forward's layers; for prefill first step it's all positions' last token -> fine
            per_layer=[None]*L
            for (li,sl) in CAP[-L:]: per_layer[li]=sl
            steps.append((nxt, per_layer))
            cur=torch.tensor([[nxt]],device=dev)
            if nxt==tok.eos_token_id: break
        torch.save({"steps":steps,"prompt_ids":ids.cpu().tolist(),"num_layers":L}, out/f"dec_{pi:05d}.pt")
        files.append(f"dec_{pi:05d}.pt")
        if pi%10==0: print(f"prompt {pi}: {len(steps)} decode steps",flush=True)
    (out/"manifest.json").write_text(json.dumps({"files":files,"top_k":model.config.num_experts_per_tok,"experts":model.config.num_experts,"num_layers":L}))
    print(f"[done] {len(files)} decode traces -> {out}")

if __name__=="__main__":
    main()
