# First trained-LoRE on a REAL model (Qwen1.5-MoE) -- rebuild grounded in earliest scripts

Previously trained LoRE existed only on SYNTHETIC random data (core/draft_model.py); real Qwen/DeepSeek
evidence was the TRAINING-FREE draft (shared-only rollout). This is the first trained-LoRE-on-real.

Pipeline (all earliest/original scripts, only the missing aligned-hidden capture was added):
- core/data/collect_hf_moe_traces.py -- FIX: capture the router's INPUT hidden (post-attention residual,
  aligned per MoE layer) as `moe_hidden` (the original only saved misaligned output.hidden_states / None).
- core/draft/train_real_lore.py (restored from git a0af2b7) -- RealLoREDraft = frozen router + trainable
  low-rank transition (z = h + transition(h), NO attention) + route-history GRU; route-KL + hidden-align.

Result (Qwen, 16 traces -> 12 train / 4 val, 400 steps, rank 64):
  step 100: route_kl 0.115  slot_hit 0.688  exact_set 0.24
  step 400: route_kl 0.0088 slot_hit 0.910  exact_set 0.68
  val route_kl 0.504 (initial) -> 0.0088 (final)

The cheap trained LoRE (NO attention) reaches slot_hit ~0.91 / exact-set ~0.68 on real Qwen, comparable to
the linear cross-layer probe (~0.93) and the training-free draft -- supporting the thesis that a cheap
low-rank transition matches recall WITHOUT running attention (so it does not eat the long-context attention
shadow that would hide the prefetch). Small data (16 traces); scale up + DeepSeek next.

Next: run the SAME anchored eval (eval_hf_trace_prefetch) with LoRE vs training-free shared-only vs
anchor_repeat vs popularity -> recall-by-lead + forecast COST, to quantify LoRE's value (match recall at
far lower cost than the attention-running shared-only).
