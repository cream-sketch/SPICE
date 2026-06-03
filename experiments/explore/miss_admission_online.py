"""Core experiment: online teacher-forced verified miss-admission (Qwen MoE).

核心实验 (codex 方法论修正): token-by-token teacher-forced decode, live KV;
每个 MoE 层在线施加 cache + controller 决策, drop 改变下游轨迹(真 on-policy);
latency sim 与模型前向锁步. 扫 importance 阈值 -> (TPOT, PPL) Pareto.

Controller (verified post-router gate weight):
  per routed expert on a DEMAND access:
    hit  -> use (no stall)
    miss & gate_weight >= threshold -> FETCH (stall += fetch_ms, admit, use) [exact]
    miss & gate_weight <  threshold -> DROP (zero its contribution, record gate mass)
  eviction = SpecMD Least-Stale (cyclic layer distance), held fixed.
  correctness: fetched experts run exactly; only low-verified-gate misses dropped.
threshold=0 -> fetch-all = SPICE (max latency, exact). threshold=inf -> drop-all-miss.
"""
from __future__ import annotations
import argparse, json, math, types
from collections import OrderedDict
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class Controller:
    def __init__(self, capacity, fetch_ms, threshold, num_layers, policy='gate', rank_keep=4):
        self.capacity = capacity; self.fetch_ms = fetch_ms
        self.threshold = threshold; self.num_layers = num_layers
        self.policy = policy; self.rank_keep = rank_keep
        self.reset()
    def reset(self):
        self.reset_cache()
        self.stall_ms = 0.0
        self.total = 0; self.hits = 0; self.fetched = 0; self.dropped = 0
        self.dropped_mass = 0.0; self.tokens = 0
    def reset_cache(self):
        # per-text cold cache; counters (stall/quality) accumulate across texts
        self.cache = OrderedDict()
        self.step = 0
    def _evict(self, cur_layer, protect):
        cand = [k for k in self.cache if k not in protect] or list(self.cache.keys())
        # Least-Stale: farthest cyclic reuse first, LRU tie
        victim = max(cand, key=lambda k: (((self.cache[k]["layer"] - cur_layer) % self.num_layers) or self.num_layers,
                                          -self.cache[k]["last"]))
        self.cache.pop(victim)
    def access(self, layer, experts, weights):
        """experts: list[int], weights: list[float] (verified gate weights this token).
        returns set of expert ids to KEEP (others dropped)."""
        self.step += 1
        protect = {(layer, e) for e in experts}
        keep = set()
        for rank, (e, w) in enumerate(zip(experts, weights)):
            self.total += 1
            key = (layer, e)
            admit_ok = (w >= self.threshold) if self.policy == 'gate' else (rank < self.rank_keep)
            if key in self.cache:
                self.hits += 1; self.cache[key]["last"] = self.step
                self.cache.move_to_end(key); keep.add(e)
            elif admit_ok:
                self.fetched += 1; self.stall_ms += self.fetch_ms
                while len(self.cache) >= self.capacity and self.cache:
                    self._evict(layer, protect)
                self.cache[key] = {"last": self.step, "layer": layer}
                self.cache.move_to_end(key); keep.add(e)
            else:
                self.dropped += 1; self.dropped_mass += float(w)
        return keep


CTRL = None  # global controller

def make_qwen_moe_forward(layer_idx):
    def fwd(mlp, hidden_states):
        b, s, d = hidden_states.shape
        h = hidden_states.view(-1, d)
        router_logits = mlp.gate(h)
        rw = F.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, mlp.top_k, dim=-1)
        if mlp.norm_topk_prob:
            rw = rw / rw.sum(dim=-1, keepdim=True)
        rw = rw.to(h.dtype)
        # apply controller per row (token); token-by-token decode -> N==1 typically
        N = h.shape[0]
        rw_eff = rw.clone()
        for r in range(N):
            experts = sel[r].tolist(); weights = rw[r].float().tolist()
            keep = CTRL.access(layer_idx, experts, weights)
            for j, e in enumerate(experts):
                if e not in keep:
                    rw_eff[r, j] = 0.0
        final = torch.zeros((N, d), dtype=h.dtype, device=h.device)
        mask = F.one_hot(sel, num_classes=mlp.num_experts).permute(2, 1, 0)
        for ei in range(mlp.num_experts):
            idx, topx = torch.where(mask[ei])
            if topx.numel() == 0: continue
            cur = h[None, topx].reshape(-1, d)
            out = mlp.experts[ei](cur) * rw_eff[topx, idx, None]
            final.index_add_(0, topx, out.to(h.dtype))
        shared = mlp.shared_expert(h)
        shared = F.sigmoid(mlp.shared_expert_gate(h)) * shared
        final = final + shared
        return final.view(b, s, d), router_logits
    return fwd


@torch.no_grad()
def run_text(model, tok, text, device, max_tokens):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens + 1).to(device)
    ids = enc["input_ids"]
    if ids.shape[1] < 2: return 0.0, 0
    past = None; nll = 0.0; n = 0
    for t in range(ids.shape[1] - 1):
        out = model(input_ids=ids[:, t:t+1], past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logp = F.log_softmax(out.logits[:, -1].float(), dim=-1)
        nll += -logp[0, ids[0, t+1]].item(); n += 1
    return nll, n


def main():
    global CTRL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--capacity", type=int, default=144)
    ap.add_argument("--expert_mb", type=float, default=17.0)
    ap.add_argument("--bandwidth_gbps", type=float, default=12.0)
    ap.add_argument("--t_layer_ms", type=float, default=0.4)
    ap.add_argument("--thresholds", type=str, default="0,0.02,0.05,0.1,0.2,1.0")
    ap.add_argument("--policy", choices=["gate","rank"], default="gate")
    ap.add_argument("--rank_keeps", type=str, default="4,3,2,1")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,
                                                 local_files_only=True, low_cpu_mem_usage=True).to(device).eval()
    import transformers.models.qwen2_moe.modeling_qwen2_moe as Mq
    layers = model.model.layers
    num_layers = len(layers)
    for li, lyr in enumerate(layers):
        if isinstance(lyr.mlp, Mq.Qwen2MoeSparseMoeBlock):
            lyr.mlp.forward = types.MethodType(make_qwen_moe_forward(li), lyr.mlp)
    fetch_ms = args.expert_mb / (args.bandwidth_gbps * 1024) * 1000.0  # MB / (GB/s*1024 MB/s) *1000 ms
    texts = [l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][:args.max_samples]
    rows = []
    knobs = [float(x) for x in args.thresholds.split(",")] if args.policy=='gate' else [int(x) for x in args.rank_keeps.split(",")]
    for kn in knobs:
        if args.policy=='gate':
            CTRL = Controller(args.capacity, fetch_ms, kn, num_layers, policy='gate')
        else:
            CTRL = Controller(args.capacity, fetch_ms, -1.0, num_layers, policy='rank', rank_keep=kn)
        th = kn
        nll_tot = 0.0; ntok = 0
        for text in texts:
            CTRL.reset_cache()  # per-text cold cache
            nll, n = run_text(model, tok, text, device, args.max_tokens)
            nll_tot += nll; ntok += n
        ppl = math.exp(nll_tot / max(1, ntok))
        # latency: stall over all decode tokens
        rows.append({"threshold": th, "ppl": ppl,
                     "stall_ms_per_token": CTRL.stall_ms / max(1, ntok),
                     "hit_rate": CTRL.hits / max(1, CTRL.total),
                     "fetch_rate": CTRL.fetched / max(1, CTRL.total),
                     "drop_rate": CTRL.dropped / max(1, CTRL.total),
                     "dropped_gate_mass_per_token": CTRL.dropped_mass / max(1, ntok),
                     "decode_tokens": ntok})
        print(f"th={th:<5} ppl={ppl:8.3f} stall/tok={rows[-1]['stall_ms_per_token']:7.3f} "
              f"hit={rows[-1]['hit_rate']:.3f} fetch={rows[-1]['fetch_rate']:.3f} drop={rows[-1]['drop_rate']:.3f} "
              f"dropmass/tok={rows[-1]['dropped_gate_mass_per_token']:.4f}", flush=True)
    out = {"experiment": "miss_admission_online_qwen", "model": args.model_dir,
           "config": {"capacity": args.capacity, "expert_mb": args.expert_mb,
                      "bandwidth_gbps": args.bandwidth_gbps, "fetch_ms": fetch_ms,
                      "max_tokens": args.max_tokens, "num_texts": len(texts)}, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()
