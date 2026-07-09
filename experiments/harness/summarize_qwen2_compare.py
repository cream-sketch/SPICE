"""Summarize Qwen2 memory-limited SPICE/baseline JSON outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Qwen2-MoE comparison results")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out", default="")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: Any, digits: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def add_spice_rows(rows: list[dict[str, Any]], path: Path, label: str) -> None:
    data = load_json(path)
    if not data:
        return
    for row in data.get("rows", []):
        rows.append(
            {
                "method": label,
                "variant": row.get("policy", ""),
                "tpot_ms": row.get("tpot_ms"),
                "tokens": row.get("tokens"),
                "fetch_tok": row.get("residual_fetches_per_tok"),
                "cpu_tok": row.get("cpu_served_per_tok"),
                "sub_tok": row.get("substituted_per_tok", 0.0),
                "source": path.name,
            }
        )


def add_adapmoe_rows(rows: list[dict[str, Any]], path: Path) -> None:
    data = load_json(path)
    if not data:
        return
    for row in data.get("rows", []):
        rows.append(
            {
                "method": "AdapMoE-style",
                "variant": f"active_m={row.get('active_m')}",
                "tpot_ms": row.get("tpot_ms"),
                "tokens": row.get("tokens"),
                "fetch_tok": row.get("fetches_per_token"),
                "cpu_tok": 0.0,
                "sub_tok": 0.0,
                "source": path.name,
            }
        )


def add_hybrimoe_rows(rows: list[dict[str, Any]], path: Path) -> None:
    data = load_json(path)
    if not data:
        return
    for row in data.get("rows", []):
        rows.append(
            {
                "method": "HybriMoE-style",
                "variant": f"prefetch={row.get('prefetch_size', '-')}",
                "tpot_ms": row.get("tpot_ms"),
                "tokens": row.get("tokens"),
                "fetch_tok": row.get("demand_fetches_per_tok"),
                "cpu_tok": row.get("cpu_served_per_tok"),
                "sub_tok": 0.0,
                "source": path.name,
            }
        )


def decode_timing(run: Path) -> str:
    manifest = load_json(run / "decode" / "manifest.json")
    if not manifest:
        return ""
    timing = [r for r in manifest.get("timing", []) if not r.get("resumed")]
    ttft = [float(r["ttft_s"]) * 1000.0 for r in timing if r.get("ttft_s") is not None]
    tpot = [float(r["tpot_s"]) * 1000.0 for r in timing if r.get("tpot_s") is not None]
    if not ttft and not tpot:
        return ""
    return (
        "Trace collection timing from HF CPU-offload generation "
        "(not policy runtime): "
        f"TTFT={fmt(mean(ttft))} ms, TPOT={fmt(mean(tpot))} ms over "
        f"{len(timing)} newly generated prompt(s)."
    )


def main() -> None:
    args = parse_args()
    run = Path(args.run_dir)
    rows: list[dict[str, Any]] = []

    add_adapmoe_rows(rows, run / "adapmoe.json")
    add_hybrimoe_rows(rows, run / "hybrimoe.json")
    add_spice_rows(rows, run / "spice_exact.json", "SPICE exact residual orchestration")
    for path in sorted(run.glob("spice_sub_rank*.json")):
        if "cpu_only" in path.name:
            continue
        add_spice_rows(rows, path, "SPICE + low-confidence substitution")

    spice_exact = next((r for r in rows if r["method"] == "SPICE exact residual orchestration"), None)
    spice_tpot = spice_exact.get("tpot_ms") if spice_exact else None

    lines = [f"# Qwen2 Memory-Limited Comparison: `{run}`", ""]
    cfg = load_json(run / "spice_exact.json") or load_json(run / "adapmoe.json") or {}
    if cfg.get("forecast_manifest"):
        man = cfg["forecast_manifest"]
        lines.append(
            f"Forecast files: {len(man.get('files', []))}; layers={man.get('num_layers')}; "
            f"experts={man.get('experts')}; top_k={man.get('top_k')}."
        )
        lines.append("")
    timing_note = decode_timing(run)
    if timing_note:
        lines.append(timing_note)
        lines.append("")

    lines.extend(
        [
            "| method | variant | TPOT ms/token | vs SPICE exact | fetch/tok | CPU/tok | substituted/tok | source |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda r: (r.get("tpot_ms") is None, r.get("tpot_ms") or 1e9)):
        if spice_tpot and row.get("tpot_ms"):
            ratio = row["tpot_ms"] / spice_tpot
            if row["method"] == "SPICE exact residual orchestration":
                vs = "1.00x"
            elif ratio < 1.0:
                vs = f"{1.0 / ratio:.2f}x faster than SPICE exact"
            else:
                vs = f"{ratio:.2f}x slower than SPICE exact"
        else:
            vs = "-"
        lines.append(
            "| {method} | {variant} | {tpot} | {vs} | {fetch} | {cpu} | {sub} | `{source}` |".format(
                method=row["method"],
                variant=row.get("variant") or "-",
                tpot=fmt(row.get("tpot_ms")),
                vs=vs,
                fetch=fmt(row.get("fetch_tok")),
                cpu=fmt(row.get("cpu_tok")),
                sub=fmt(row.get("sub_tok")),
                source=row.get("source", ""),
            )
        )
    lines.append("")
    lines.append(
        "Note: AdapMoE and HybriMoE are method-level replays on the same Qwen2 trace and "
        "hardware cost measurements. This avoids mixing policy comparisons with unsupported "
        "model runtimes, quantization formats, or custom kernels."
    )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
