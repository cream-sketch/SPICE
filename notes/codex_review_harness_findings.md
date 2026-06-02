Findings:

- [experiments/miss_admission_online.py:73](/home/abc/Placement/HLS/spice/experiments/miss_admission_online.py:73): current `transformers` Qwen2MoE has `mlp.gate(h)` return `(router_logits, routing_weights, selected_experts)`, not logits. `F.softmax(router_logits, ...)` on line 74 will fail or be wrong. This patch assumes an older `gate = Linear` API.

- [experiments/miss_admission_online.py:75](/home/abc/Placement/HLS/spice/experiments/miss_admission_online.py:75): same API mismatch: current block stores `top_k/norm_topk_prob/num_experts` on `mlp.gate`, not directly on `mlp`. Lines 75-89 can fail on current Qwen2MoE.

- [experiments/miss_admission_online.py:94](/home/abc/Placement/HLS/spice/experiments/miss_admission_online.py:94): current `mlp.experts` is a fused `Qwen2MoeExperts`, not an indexable `ModuleList`; `mlp.experts[ei](cur)` is invalid for this installed model.

- [experiments/miss_admission_online.py:99](/home/abc/Placement/HLS/spice/experiments/miss_admission_online.py:99): current `Qwen2MoeSparseMoeBlock.forward` returns only hidden states; decoder layer does `hidden_states = residual + self.mlp(...)`. Returning `(hidden, router_logits)` will make residual addition fail.

- [experiments/miss_admission_online.py:56](/home/abc/Placement/HLS/spice/experiments/miss_admission_online.py:56): `threshold=0` fetches only experts with `w >= 0`. That is fine for softmax weights, so threshold 0 is fetch-all for routed experts. But shared expert is always local/unmetered at lines 96-98; if shared expert offload latency is in-scope, Pareto latency is biased low.

- [experiments/miss_admission_online.py:58](/home/abc/Placement/HLS/spice/experiments/miss_admission_online.py:58): no guard for `capacity == 0`; a fetched miss still gets inserted at line 60, creating hits in a zero-slot cache. If capacity is always positive, no issue.

Checked points with no correctness bug found: drop does perturb downstream hidden/KV via `rw_eff` zeroing before `final` returns; fetch-ms formula matches `expert_mb/(bw_gbps*1024)*1000`; LS protect/tie-break matches repo’s simulator; `reset_cache()` per text keeps counters; no obvious stall/hit double count; zeroing after `norm_topk_prob` does not renormalize survivors, which matches “zero contribution”; `N==1` is correct for the token-by-token loop.
