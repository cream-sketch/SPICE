from __future__ import annotations

import sys, pathlib  # bootstrap: resolve sibling core modules regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import build_arg_parser, device_from_arg, ensure_dir, measure_gpu_power_watts, set_seed, write_json
from prefetch_system_sim import SimConfig, make_trace, measure_copy_bandwidth, simulate_policy


def main() -> None:
    parser = build_arg_parser("Controlled proxy comparison for recent MoE prefetch systems")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--expert_mb", type=int, default=8)
    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = device_from_arg(args.gpu)
    base = SimConfig(steps=args.steps, expert_mb=args.expert_mb)
    trace = make_trace(base, args.seed)
    variants = [
        (
            "ExpertFlow_proxy",
            SimConfig(**{**base.__dict__, "predictor_acc": 0.70, "draft_ms": 0.08, "online_ms": 0.0}),
            "routing-path predictor proxy with no LoRE correction",
        ),
        (
            "AdapMoE_proxy",
            SimConfig(**{**base.__dict__, "predictor_acc": 0.60, "draft_ms": 0.02, "online_ms": 0.0, "lookahead_max": 2}),
            "sensitivity/cache-management proxy with shallow prefetch horizon",
        ),
        (
            "SP-MoE_proxy",
            SimConfig(**{**base.__dict__, "predictor_acc": 0.66, "draft_ms": 0.10, "online_ms": 0.0, "lookahead_max": 4}),
            "token-coupled speculative MoE proxy; lower routing reliability under token mismatch",
        ),
        (
            "MoE-SpeQ_proxy",
            SimConfig(**{**base.__dict__, "predictor_acc": 0.68, "draft_ms": 0.04, "online_ms": 0.0, "lookahead_max": 5}),
            "quantized speculative decoding proxy; cheaper draft but still token-coupled",
        ),
        (
            "SPICE_verified",
            base,
            "routing-only verified expert-residency speculation with LoRE correction",
        ),
    ]
    rows = []
    for name, cfg, note in variants:
        row = simulate_policy("spice", trace, cfg, args.seed)
        row["variant"] = name
        row["note"] = note
        rows.append(row)
    result = {
        "experiment": "sota_proxy",
        "device": str(device),
        "copy_microbench": measure_copy_bandwidth(device, args.expert_mb, copies=128),
        "gpu_power_watts_sample": measure_gpu_power_watts(args.gpu),
        "important_caveat": "These are controlled same-harness proxy variants, not official reproductions of the cited systems.",
        "rows": rows,
    }
    write_json(out_dir / "sota_proxy.json", result)
    print(result)


if __name__ == "__main__":
    main()
