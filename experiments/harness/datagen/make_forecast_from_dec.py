"""Build a forecast dump (true_top + fcast) from decode routing traces.

从 decode 路由 trace 构造 forecast dump (true_top 真实 + fcast 占位).
For depth=0 negative-admission runs the fcast is unused; only true_top (gate-descending
top-k routing) drives residual misses. Lets the shallow runtime run on models that have
decode traces but no draft-forecast dump (e.g. DeepSeek-V2-Lite).

By default fcast is a placeholder (-1).  With --oracle_future_fcast, fcast[l, lead, t]
is filled with the true top-k experts at layer l+lead for the same token t.  This is
NOT an implementable predictor; it is a resource-scheduler upper bound used to separate
"can the hardware scheduler help if predictions are perfect?" from draft accuracy.
"""
import argparse, glob, json
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_horizon", type=int, default=6)
    ap.add_argument("--max_files", type=int, required=True)
    ap.add_argument("--model_dir", default="deepseek-v2-lite")
    ap.add_argument("--oracle_future_fcast", action="store_true",
                    help="fill fcast with same-token future true routes; upper-bound only, not a real predictor")
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.trace_dir) / "dec_*.pt")))[: args.max_files]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    names = []
    L = K = None
    total_tokens = 0
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        steps = [s for s in d["steps"] if all(x is not None for x in s[1])]
        if not steps:
            continue
        L = len(steps[0][1]); K = len(steps[0][1][0]); T = len(steps)
        total_tokens += T
        # true_top[L, T, K] gate-descending (gen_decode_traces used torch.topk)
        true_top = torch.zeros(L, T, K, dtype=torch.long)
        for t, (_tid, per_layer) in enumerate(steps):
            for l in range(L):
                true_top[l, t] = torch.tensor([int(e) for e in per_layer[l]][:K], dtype=torch.long)
        fcast = torch.full((L, args.max_horizon, T, K), -1, dtype=torch.long)
        if args.oracle_future_fcast:
            for l in range(L):
                for lead in range(args.max_horizon):
                    target_l = l + lead
                    if target_l < L:
                        fcast[l, lead] = true_top[target_l]
        name = f"fc_{i:05d}.pt"
        torch.save({"true_top": true_top, "fcast": fcast, "num_layers": L,
                    "top_k": K, "max_horizon": args.max_horizon}, out / name)
        names.append(name)
    note = ("true_top from decode traces; fcast is oracle same-token future true routes; "
            "upper-bound only, not an implementable draft predictor") if args.oracle_future_fcast else (
            "true_top from decode traces; fcast placeholder (-1), valid only for depth=0 runs")
    (out / "manifest.json").write_text(json.dumps(
        {"files": names, "num_files": len(names), "total_tokens": total_tokens,
         "num_layers": L, "top_k": K, "max_horizon": args.max_horizon, "model_dir": args.model_dir,
         "oracle_future_fcast": bool(args.oracle_future_fcast), "note": note}, indent=2))
    print(f"[done] wrote {len(names)} forecast files (L={L}, K={K}) -> {out}")


if __name__ == "__main__":
    main()
