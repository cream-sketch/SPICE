"""Candidate A premise: CPU expert-compute latency vs PCIe weight-fetch latency (batch=1 decode).
If CPU-compute one routed expert (reading its weights from CPU DRAM) is faster than fetching the
17MB weight over PCIe, then 'compute on CPU, move 8KB activation' beats 'fetch weight'. EXACT.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch, torch.nn.functional as F
from safetensors import safe_open

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model_dir",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--n_experts",type=int,default=16); ap.add_argument("--batches",type=str,default="1,4,8,16")
    ap.add_argument("--iters",type=int,default=50); ap.add_argument("--threads",type=int,default=0)
    a=ap.parse_args()
    if a.threads>0: torch.set_num_threads(a.threads)
    md=Path(a.model_dir); idx=json.loads((md/"model.safetensors.index.json").read_text())["weight_map"]
    def load(name): 
        with safe_open(str(md/idx[name]),framework="pt",device="cpu") as h: return h.get_tensor(name).to(torch.float32)
    # load n experts from layer 0
    L=0; experts=[]
    for e in range(a.n_experts):
        g=load(f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight")
        u=load(f"model.layers.{L}.mlp.experts.{e}.up_proj.weight")
        d=load(f"model.layers.{L}.mlp.experts.{e}.down_proj.weight")
        experts.append((g,u,d))
    d_model=experts[0][0].shape[1]; inter=experts[0][0].shape[0]
    bytes_per_expert=sum(x.numel()*2 for x in experts[0])  # bf16 bytes
    def compute(x,g,u,d): return F.linear(F.silu(F.linear(x,g))*F.linear(x,u),d)
    res={"d_model":d_model,"inter":inter,"bytes_per_expert_MB":bytes_per_expert/1e6,
         "threads":torch.get_num_threads(),"rows":[]}
    for B in [int(x) for x in a.batches.split(",")]:
        x=torch.randn(B,d_model,dtype=torch.float32)
        # warmup
        for _ in range(5):
            for (g,u,d) in experts: compute(x,g,u,d)
        t0=time.perf_counter()
        for _ in range(a.iters):
            for (g,u,d) in experts: compute(x,g,u,d)
        dt=(time.perf_counter()-t0)/(a.iters*len(experts))*1000  # ms/expert
        row={"batch":B,"cpu_ms_per_expert":dt}
        for bw in [5,12,24]:
            row[f"fetch_ms@{bw}gbps"]=bytes_per_expert/(bw*1024**3)*1000
        res["rows"].append(row)
        print(f"B={B:>3} CPU={dt:.3f} ms/expert | fetch@5={row['fetch_ms@5gbps']:.2f} @12={row['fetch_ms@12gbps']:.2f} @24={row['fetch_ms@24gbps']:.2f} ms",flush=True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(res,indent=2))
    print(f"[A-premise] expert={res['bytes_per_expert_MB']:.1f}MB. If CPU_ms < fetch_ms -> compute-on-CPU beats weight-fetch (exact).")

if __name__=="__main__": main()
