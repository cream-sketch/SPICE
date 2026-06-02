from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

from common import build_arg_parser, ensure_dir, write_json


def query_power(gpu: int) -> tuple[float | None, float | None]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        p, u = [x.strip() for x in out.split(",")[:2]]
        return float(p), float(u)
    except Exception:
        return None, None


def main() -> None:
    parser = build_arg_parser("GPU power sampler")
    parser.add_argument("--duration_s", type=float, default=20.0)
    parser.add_argument("--interval_s", type=float, default=0.2)
    args = parser.parse_args()
    out_dir = ensure_dir(args.out_dir)
    rows = []
    start = time.perf_counter()
    while time.perf_counter() - start < args.duration_s:
        t = time.perf_counter() - start
        power, util = query_power(args.gpu)
        rows.append({"t_s": t, "power_w": power, "util_gpu_pct": util})
        time.sleep(args.interval_s)
    csv_path = out_dir / "power_trace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t_s", "power_w", "util_gpu_pct"])
        writer.writeheader()
        writer.writerows(rows)
    vals = [r["power_w"] for r in rows if r["power_w"] is not None]
    duration = rows[-1]["t_s"] - rows[0]["t_s"] if len(rows) > 1 else 0.0
    summary = {
        "experiment": "power_trace",
        "gpu": args.gpu,
        "samples": len(rows),
        "duration_s": duration,
        "avg_power_w": sum(vals) / len(vals) if vals else None,
        "min_power_w": min(vals) if vals else None,
        "max_power_w": max(vals) if vals else None,
        "energy_j_est": (sum(vals) / len(vals) * duration) if vals else None,
        "csv": str(csv_path),
    }
    write_json(out_dir / "power_trace.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
