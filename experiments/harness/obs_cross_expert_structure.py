"""New info source: CROSS-expert weight structure (do the experts share a basis?).

exp0a tested PER-expert rank (full). Untested: do the N experts at a layer share a
low-dim structure? experts = shared_mean + low-rank residual? If yes -> transfer
shared basis once (resident) + small per-expert coeffs -> byte reduction, batch-1
compatible, not HOBBIT (cross-expert structure). Read real Qwen expert weights.
Per layer, per projection: stack flattened experts [N, P]; measure (a) mean-energy
fraction, (b) SVD across experts -> #components for 90% variance / N.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from safetensors import safe_open

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--gpu",type=int,default=0); ap.add_argument("--layers",type=int,default=4)
    ap.add_argument("--proj",type=str,default="down,gate,up")
    args=ap.parse_args()
    dev=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    md=Path(args.model_dir); idx=json.loads((md/"model.safetensors.index.json").read_text())["weight_map"]
    # discover layers/experts
    import re
    lay_exp={}
    for name in idx:
        if ".mlp.experts." in name and name.endswith("down_proj.weight"):
            p=name.split("."); L=int(p[p.index("layers")+1]); E=int(p[p.index("experts")+1])
            lay_exp.setdefault(L,set()).add(E)
    all_layers=sorted(lay_exp); n_exp=len(lay_exp[all_layers[0]])
    step=max(1,len(all_layers)//args.layers); sample_layers=all_layers[::step][:args.layers]
    def load(name):
        with safe_open(str(md/idx[name]),framework="pt",device="cpu") as h: return h.get_tensor(name).to(dev,torch.float32)
    res={}
    for proj in args.proj.split(","):
        rows=[]
        for L in sample_layers:
            mats=[load(f"model.layers.{L}.mlp.experts.{e}.{proj}_proj.weight").flatten() for e in range(n_exp)]
            X=torch.stack(mats)             # [N, P]
            mean=X.mean(0,keepdim=True)
            mean_energy=float((mean.norm()**2*X.shape[0])/ (X.norm()**2))  # frac of total energy in the shared mean
            Xc=X-mean
            # SVD across experts (N x P, N=60 small) -> singular values over experts
            sv=torch.linalg.svdvals(Xc)     # length N
            en=sv**2; cum=torch.cumsum(en,0)/en.sum()
            r90=int(torch.searchsorted(cum,torch.tensor(0.90,device=dev)).item())+1
            r50=int(torch.searchsorted(cum,torch.tensor(0.50,device=dev)).item())+1
            rows.append({"layer":L,"mean_energy_frac":mean_energy,"N":X.shape[0],
                         "rank90_over_N":r90/X.shape[0],"rank50_over_N":r50/X.shape[0]})
            del mats,X,Xc; torch.cuda.empty_cache() if dev.type=="cuda" else None
        res[proj]={"per_layer":rows,
                   "mean_energy_frac_avg":sum(r["mean_energy_frac"] for r in rows)/len(rows),
                   "rank90_over_N_avg":sum(r["rank90_over_N"] for r in rows)/len(rows)}
        print(f"proj={proj}: mean_energy_frac={res[proj]['mean_energy_frac_avg']:.3f} rank90/N={res[proj]['rank90_over_N_avg']:.3f}",flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(res,indent=2))
    print("[interp] mean_energy_frac high OR rank90/N low -> experts share structure -> transfer-once byte reduction possible")

if __name__=="__main__":
    main()
