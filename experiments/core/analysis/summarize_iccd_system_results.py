from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_jsons(root: Path):
    for path in sorted(root.rglob("*.json")):
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue


def pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.2f}"


def f2(x: float | None) -> str:
    return "--" if x is None else f"{x:.2f}"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    energy_rows = []
    cache_rows = []
    timeline_rows = []
    for path, obj in load_jsons(root):
        exp = obj.get("experiment", "")
        if exp == "energy_per_token_replay":
            for row in obj.get("rows", []):
                sim = row.get("sim_row", {})
                energy_rows.append(
                    {
                        "policy": row["policy"],
                        "sim_tpot_ms": sim.get("sim_tpot_ms"),
                        "fallback_rate": sim.get("fallback_rate"),
                        "h2d_gb": sim.get("h2d_gb"),
                        "measured_tpot_ms": row.get("measured_tpot_ms"),
                        "avg_power_w": row.get("avg_power_w"),
                        "peak_power_w": row.get("peak_power_w"),
                        "energy_per_token_j": row.get("energy_per_token_j"),
                        "samples": row.get("num_power_samples"),
                        "file": str(path.relative_to(root)),
                    }
                )
        elif exp == "prefetch_system_cache_sweep":
            for row in obj.get("rows", []):
                cache_rows.append({**row, "file": str(path.relative_to(root))})
        elif exp == "timeline_replay":
            timeline_rows.append({**obj, "file": str(path.relative_to(root))})

    lines = ["# ICCD System Experiment Results\n\n"]
    lines.append(f"Root: `{root}`\n\n")

    if energy_rows:
        if any("energy_paired" in r["file"] for r in energy_rows):
            energy_rows = [r for r in energy_rows if "energy_paired" in r["file"]]
        df = pd.DataFrame(energy_rows).sort_values(["policy"])
        df.to_csv(root / "energy_per_token_summary.csv", index=False)
        lines.append("## Energy Replay\n\n")
        lines.append("| Policy | Sim TPOT (ms) | Fallback (%) | H2D (GB) | Replay TPOT (ms) | Avg W | Peak W | J/token |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in df.iterrows():
            lines.append(
                f"| {r.policy} | {f2(r.sim_tpot_ms)} | {pct(r.fallback_rate)} | "
                f"{f2(r.h2d_gb)} | {f2(r.measured_tpot_ms)} | {f2(r.avg_power_w)} | "
                f"{f2(r.peak_power_w)} | {f2(r.energy_per_token_j)} |\n"
            )
        lines.append("\n")

    if cache_rows:
        df = pd.DataFrame(cache_rows)
        df.to_csv(root / "cache_sweep_summary.csv", index=False)
        focus = df[df["policy"].isin(["lru", "pregated", "spice"])].copy()
        focus = focus.sort_values(["cache_capacity", "policy"])
        lines.append("## Cache Budget Sweep\n\n")
        lines.append("| Cache slots | Budget (GB) | Policy | TPOT (ms) | Fallback (%) | H2D (GB) | PCIe active (%) |\n")
        lines.append("|---:|---:|---|---:|---:|---:|---:|\n")
        for _, r in focus.iterrows():
            lines.append(
                f"| {int(r.cache_capacity)} | {f2(r.cache_budget_gb)} | {r.policy} | "
                f"{f2(r.sim_tpot_ms)} | {pct(r.fallback_rate)} | {f2(r.h2d_gb)} | "
                f"{pct(r.pcie_active_fraction)} |\n"
            )
        lines.append("\n")

    if timeline_rows:
        df = pd.DataFrame(timeline_rows).sort_values(["policy"])
        df.to_csv(root / "timeline_summary.csv", index=False)
        lines.append("## Timeline Replay\n\n")
        lines.append("| Policy | Overlap | H2D (GB) | Elapsed (s) | Effective H2D (GB/s) | Report |\n")
        lines.append("|---|---|---:|---:|---:|---|\n")
        for _, r in df.iterrows():
            report = str(r.file).replace(".json", ".nsys-rep")
            lines.append(
                f"| {r.policy} | {r.overlap} | {f2(r.h2d_gb)} | {f2(r.elapsed_s)} | "
                f"{f2(r.effective_h2d_gbps)} | `{report}` |\n"
            )
        lines.append("\n")

    (root / "ICCD_SYSTEM_RESULTS.md").write_text("".join(lines), encoding="utf-8")
    print(root / "ICCD_SYSTEM_RESULTS.md")


if __name__ == "__main__":
    main()
