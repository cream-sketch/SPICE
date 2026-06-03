"""P1: REAL greedy speculative-decoding capture for SPICE-W (real-trajectory, pure argmax).

P1: 为 SPICE-W 采集真实 greedy speculative decoding 数据 (真实轨迹 + 纯 argmax).

Design (addresses codex review of the teacher-forced draft):
  We simulate the REAL speculative trajectory directly, advancing the committed
  sequence by (accept_count + 1) each verify step -- exactly real greedy spec decoding,
  with NO separate teacher-forced ref and NO KV rewind. Each verify step does ONE target
  forward over (committed_prefix + draft_block); from that single pass we read BOTH:
    - verification: target's greedy argmax at each candidate position (pure argmax, no
      generation_config side-effects); accept_count = leading prefix where draft == target argmax.
    - expert demand: the K candidate positions' MoE router top-K (the verify-window demand).
  EOS is handled explicitly (trajectory stops when a committed token is EOS). The draft
  proposes K tokens by a hand-written greedy argmax loop (no generate() surprises).
  设计 (回应 codex 对 teacher-forced 版的审查):
    直接模拟真实 speculative 轨迹, 每个 verify step 把已提交序列推进 (accept_count+1),
    与真实 greedy spec decoding 完全一致, 无需单独 ref, 无需 KV rewind. 每步对
    (已提交前缀 + draft 块) 做一次 target 前向, 同时读出验证 argmax 与候选位置路由.
    显式处理 EOS; draft 用手写贪心 argmax 循环提议 K 个 token.

Outputs:
  1. Per-position accept rate P(accept candidate i | reached i) -- REAL conditional curve.
  2. Verify-window per-layer routes (K candidate positions' target top-K) dumped to windows_*.pt.
  3. Reuse C/U per verify window per layer (the SPICE-W lever).
  4. Real mean tokens advanced per verify step.

All printed strings English. Core params: no defaults. Bilingual comments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser(description="Real greedy speculative-decoding capture (real-trajectory)")
    ap.add_argument("--target_dir", required=True, help="target MoE model dir (Qwen1.5-MoE-A2.7B)")
    ap.add_argument("--draft_dir", required=True, help="draft model dir (Qwen1.5-0.5B-Chat)")
    ap.add_argument("--text_file", required=True, help="one prompt per line")
    ap.add_argument("--out_json", required=True, help="summary JSON (accept curve + reuse)")
    ap.add_argument("--out_windows", required=True, help="path to dump windows .pt for P2 runtime")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--K", type=int, required=True, help="speculative draft block length")
    ap.add_argument("--max_steps", type=int, required=True, help="max verify steps per prompt")
    ap.add_argument("--max_samples", type=int, required=True, help="number of prompts")
    ap.add_argument("--max_prompt_len", type=int, required=True)
    return ap.parse_args()


@torch.no_grad()
def draft_propose(draft, seq_ids: torch.Tensor, K: int, eos_id: int) -> torch.Tensor:
    """Hand-written greedy argmax: propose K tokens conditioned on seq_ids [1, S]. Returns [<=K].

    手写贪心 argmax: 在 seq_ids 条件下提议 K 个 token. 命中 EOS 提前停止 (draft 也可能产出 EOS).
    Recompute (no KV cache) -- simple and bug-resistant at measurement scale.
    """
    cur = seq_ids
    toks = []
    for _ in range(K):
        logits = draft(input_ids=cur, use_cache=False, return_dict=True).logits[0, -1]
        nxt = int(torch.argmax(logits))
        toks.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device, dtype=cur.dtype)], dim=1)
        if nxt == eos_id:
            break
    return torch.tensor(toks, device=seq_ids.device, dtype=seq_ids.dtype)


@torch.no_grad()
def verify_forward(target, seq_ids: torch.Tensor, draft_block: torch.Tensor, top_k: int):
    """ONE target forward over (seq_ids + draft_block). batch=1.

    一次 target 前向 (已提交前缀 + draft 块). batch=1.
    Returns:
      verify_argmax [K]   : target greedy token at each candidate verification position.
      cand_routes  list[L] of [K, top_k] LongTensor (CPU): router top-K for the K candidate tokens.
      bonus_logits_row    : logits row used for the bonus token when all K accepted.
    """
    S = seq_ids.shape[1]
    Kb = draft_block.shape[0]
    full = torch.cat([seq_ids[0], draft_block], dim=0).unsqueeze(0)  # [1, S+K]
    out = target(input_ids=full, output_router_logits=True, use_cache=False, return_dict=True)
    logits = out.logits[0]  # [S+K, vocab]
    # verification logits: index S-1+j predicts candidate j (j in 0..K-1); index S+K-1 -> bonus if all accepted
    verify_rows = logits[S - 1: S - 1 + Kb]          # [K, vocab]
    verify_argmax = torch.argmax(verify_rows, dim=-1)  # [K]
    bonus_row_argmax = int(torch.argmax(logits[S + Kb - 1]))
    # candidate-token routes: router_logits at indices [S .. S+K-1] (the K candidate positions)
    # 候选 token 路由: router_logits 在索引 [S .. S+K-1] (K 个候选位置)
    cand_routes = []
    for rl in out.router_logits:  # each [S+K, n_experts] for batch=1
        assert rl.shape[0] == S + Kb, f"router_logits shape {tuple(rl.shape)} != [{S + Kb}, E]; expected batch=1 flattened"
        probs = F.softmax(rl[S: S + Kb].float(), dim=-1)
        cand_routes.append(torch.topk(probs, k=top_k, dim=-1).indices.cpu())  # [K, top_k]
    return verify_argmax, cand_routes, bonus_row_argmax


def main():
    a = parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)

    tok = AutoTokenizer.from_pretrained(a.target_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok_d = AutoTokenizer.from_pretrained(a.draft_dir, local_files_only=True)
    # strong tokenizer-identity check: full vocab mapping must match for shared-vocab spec decoding
    # 强 tokenizer 一致性检查: 共享词表 spec decoding 要求完整 vocab 映射一致
    assert tok.get_vocab() == tok_d.get_vocab(), "draft/target vocab maps differ; spec decoding requires identical vocab"
    eos_id = tok.eos_token_id

    print(f"[load] target={a.target_dir}", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        a.target_dir, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).to(dev).eval()
    print(f"[load] draft={a.draft_dir}", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(
        a.draft_dir, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).to(dev).eval()

    top_k = int(target.config.num_experts_per_tok)
    n_layers = int(target.config.num_hidden_layers)
    n_experts = int(target.config.num_experts)
    print(f"[cfg] layers={n_layers} experts={n_experts} top_k={top_k} K={a.K} eos={eos_id}", flush=True)

    texts = [l.strip() for l in Path(a.text_file).read_text(encoding="utf-8").splitlines() if l.strip()][: a.max_samples]

    reached = [0] * a.K
    accepted = [0] * a.K
    accept_counts = []          # accept_count per verify step (real trajectory)
    reuse_per_window = []       # mean over layers of C/U per window
    total_tokens = 0            # committed tokens advanced (sum of ac+1, truncated at EOS)
    windows = []                # {routes:[L,K,top_k] long, accept_count:int}

    for ti, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=a.max_prompt_len).to(dev)
        seq = enc["input_ids"]                                # committed sequence [1, S]
        steps_done = 0
        stop = False
        while steps_done < a.max_steps and not stop:
            draft_block = draft_propose(draft, seq, a.K, eos_id)
            if draft_block.shape[0] < a.K:
                # draft hit EOS before K; pad-skip: still verify what it proposed, then stop after commit
                pass
            Kb = draft_block.shape[0]
            if Kb == 0:
                break
            verify_argmax, cand_routes, bonus_argmax = verify_forward(target, seq, draft_block, top_k)
            # accept_count: leading prefix where draft == target greedy argmax
            ac = 0
            for j in range(Kb):
                if int(draft_block[j]) == int(verify_argmax[j]):
                    ac += 1
                else:
                    break
            accept_counts.append(ac)
            # per-position conditional accept tallies (only over the full-K windows for a clean curve)
            # 仅对完整 K 窗口统计 per-position 条件接受率
            if Kb == a.K:
                for i in range(a.K):
                    if ac >= i:
                        reached[i] += 1
                    if ac > i:
                        accepted[i] += 1
                # reuse C/U per layer for this window
                reuse_layers = []
                for idx in cand_routes:  # [K, top_k]
                    flat = idx.reshape(-1).tolist()
                    C = len(flat); U = len(set(flat))
                    reuse_layers.append(C / U if U > 0 else 0.0)
                reuse_per_window.append(sum(reuse_layers) / len(reuse_layers))
                windows.append({"routes": torch.stack(cand_routes, dim=0), "accept_count": ac})

            # commit accepted draft tokens + 1 bonus (target's correction at first reject, or extra token)
            # 提交已接受 draft token + 1 个 bonus (首个拒绝位置的 target 修正, 或全接受后的额外 token)
            committed = [int(draft_block[j]) for j in range(ac)]
            bonus = int(verify_argmax[ac]) if ac < Kb else bonus_argmax
            committed.append(bonus)
            # truncate at EOS
            cut = None
            for k, t in enumerate(committed):
                if t == eos_id:
                    cut = k + 1
                    break
            if cut is not None:
                committed = committed[:cut]
                stop = True
            total_tokens += len(committed)
            seq = torch.cat([seq, torch.tensor([committed], device=dev, dtype=seq.dtype)], dim=1)
            steps_done += 1
        print(f"[{ti+1}/{len(texts)}] steps={steps_done} "
              f"mean_accept={sum(accept_counts)/max(1,len(accept_counts)):.2f}", flush=True)

    accept_rate_by_pos = [accepted[i] / max(1, reached[i]) for i in range(a.K)]
    mean_accept = sum(accept_counts) / max(1, len(accept_counts))
    mean_reuse = sum(reuse_per_window) / max(1, len(reuse_per_window))
    n_steps = len(accept_counts)
    tokens_per_step = total_tokens / max(1, n_steps)

    summary = {
        "experiment": "spec_decode_capture_real_trajectory",
        "target_dir": a.target_dir, "draft_dir": a.draft_dir,
        "K": a.K, "n_layers": n_layers, "n_experts": n_experts, "top_k": top_k,
        "num_prompts": len(texts), "num_verify_steps": n_steps,
        "accept_rate_by_position": accept_rate_by_pos,
        "mean_accept_count": mean_accept,
        "mean_tokens_per_step": tokens_per_step,
        "mean_reuse_C_over_U": mean_reuse,
    }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({"windows": windows, "K": a.K, "n_layers": n_layers, "n_experts": n_experts,
                "top_k": top_k, "accept_rate_by_position": accept_rate_by_pos,
                "mean_tokens_per_step": tokens_per_step}, a.out_windows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] summary -> {a.out_json}  windows -> {a.out_windows}", flush=True)


if __name__ == "__main__":
    main()
