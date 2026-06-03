"""VS-CPU kill-test (Plan Phase 1, decision gate): is SPICE's draft approximate expert-input z_draft
close enough to the true z_true that expert_e(z_draft) ~= expert_e(z_true)? If yes for horizon d>=2,
speculative CPU pre-compute on the draft hidden can be USED (verify passes) -> VS-CPU lives. If not,
VS-CPU degrades to plain cpu_serve (Fiddler) -> NO-GO.

Mechanism: SPICE's training-free draft (qwen_spice_draft) rolls shared-only layers from true hs[anchor];
the expert INPUT at a future layer f (post-attention hidden = the MoE block input) is z_draft(anchor,f).
We compare the REAL Qwen expert output on z_draft vs on the true z_true(f), for the truly-selected
experts, stratified by horizon d = f-anchor+1.

Metrics per horizon d:
  value_rel_err = ||E_e(z_draft) - E_e(z_true)|| / ||E_e(z_true)||  over true top-k experts
  value_accept@tau = fraction with value_rel_err <= tau (tau in {1e-2, 1e-3})
  also report ||z_draft - z_true||/||z_true|| (input drift) for diagnosis.
GO if value_accept@1e-2 >= ~0.5 at some d>=2. All printed English. Core params: no defaults.
"""
import argparse, json, types
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from qwen_spice_draft import shared_only_mlp_forward, true_forward, topk_sets_from_logits


def parse_args():
    ap = argparse.ArgumentParser(description="VS-CPU kill-test: draft-hidden expert-output accept rate vs horizon")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text_file", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--max_horizon", type=int, required=True)
    ap.add_argument("--max_samples", type=int, required=True)
    ap.add_argument("--max_length", type=int, required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    device = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, local_files_only=True).to(device).eval()
    base = model.model; layers = base.layers; n_layers = len(layers)

    # capture the MoE-block INPUT (= post-attention hidden = expert input z) per layer via pre-hook
    captured = {}
    def mk_hook(idx):
        def hook(mod, inp):
            captured[idx] = inp[0].detach()  # (B,S,d) expert input at this layer
        return hook
    hooks = [layers[l].mlp.register_forward_pre_hook(mk_hook(l)) for l in range(n_layers)]

    texts = [l.strip() for l in Path(a.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:a.max_samples]
    by_d_relerr = defaultdict(list); by_d_drift = defaultdict(list)

    for ti, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=a.max_length).to(device)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            # TRUE forward: captures true expert inputs z_true[l] (real MLP runs)
            captured.clear()
            true_topk, hs = true_forward(model, ids, attn, a.top_k)
            z_true = {l: captured[l].clone() for l in range(n_layers)}  # (B,S,d)

            # DRAFT rollout (shared-only) anchored at true hs[anchor]; capture z_draft(anchor,f)
            s = ids.shape[1]
            position_ids = torch.arange(s, device=device).unsqueeze(0)
            cos, sin = base.rotary_emb(hs[0], position_ids); pos_emb = (cos, sin)
            causal = torch.triu(torch.full((s, s), float("-inf"), device=device, dtype=hs[0].dtype), 1)[None, None]
            originals = []
            for layer in layers:
                originals.append(layer.mlp.forward)
                layer.mlp.forward = types.MethodType(shared_only_mlp_forward, layer.mlp)
            try:
                for anchor in range(n_layers):
                    h = hs[anchor]
                    for d_off in range(a.max_horizon):
                        f = anchor + d_off
                        if f >= n_layers: break
                        captured.clear()
                        layer_out = layers[f](hidden_states=h, attention_mask=causal, position_ids=position_ids,
                                              position_embeddings=pos_emb, output_router_logits=True, use_cache=False)
                        z_draft = captured[f]                      # draft expert input at layer f (post-attn, shared-only rolled)
                        d = d_off + 1                              # horizon (d=1 exact: draft=true attn+router on true z)
                        # NOTE: moe.experts[e] are the REAL expert submodules; patching moe.forward does NOT
                        # affect calling moe.experts[e](z) directly, so no restore/re-patch needed.
                        moe = layers[f].mlp
                        zt = z_true[f][:, -1, :]                   # (B,d) true expert input (last token, decode-relevant)
                        zd = z_draft[:, -1, :]                     # (B,d) draft expert input
                        sel_set = true_topk[f][-1]                 # last token's true top-k expert set at layer f
                        for e in sel_set:
                            ot = moe.experts[int(e)](zt)            # real expert on TRUE input
                            od = moe.experts[int(e)](zd)            # real expert on DRAFT input
                            rel = (od.float() - ot.float()).norm() / ot.float().norm().clamp_min(1e-9)
                            by_d_relerr[d].append(float(rel))
                        by_d_drift[d].append(float((zd.float() - zt.float()).norm() / zt.float().norm().clamp_min(1e-9)))
                        h = layer_out[0]
            finally:
                for layer, orig in zip(layers, originals):
                    layer.mlp.forward = orig
        if ti % 5 == 0:
            print(f"sample {ti}/{len(texts)} done", flush=True)
    for h in hooks: h.remove()

    rows = []
    for d in sorted(by_d_relerr):
        errs = np.array(by_d_relerr[d]); drift = np.array(by_d_drift[d])
        rows.append({"horizon_d": d, "n": len(errs),
                     "value_rel_err_mean": float(errs.mean()), "value_rel_err_p50": float(np.percentile(errs, 50)),
                     "value_accept@1e-2": float((errs <= 1e-2).mean()), "value_accept@1e-3": float((errs <= 1e-3).mean()),
                     "input_drift_mean": float(drift.mean())})
        print(f"d={d:>2} n={len(errs):>6} rel_err mean={errs.mean():.4f} p50={np.percentile(errs,50):.4f} "
              f"accept@1e-2={ (errs<=1e-2).mean():.3f} accept@1e-3={(errs<=1e-3).mean():.3f} "
              f"input_drift={drift.mean():.4f}", flush=True)
    go = any(r["horizon_d"] >= 2 and r["value_accept@1e-2"] >= 0.5 for r in rows)
    print(f"\n[VS-CPU KILL-TEST] GO={go} (need value_accept@1e-2>=0.5 at some horizon d>=2). "
          f"NO-GO -> VS-CPU degrades to cpu_serve; ship verified controller (cpu_serve/fetch/drop).", flush=True)
    Path(a.out).write_text(json.dumps({"rows": rows, "go": go}, indent=2))


if __name__ == "__main__":
    main()
