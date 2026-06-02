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

    latex = []
    latex.append("% Paste-ready ICCD system supplement. Requires \\usepackage{booktabs}.\n\n")
    if energy_rows:
        df = pd.DataFrame(energy_rows).sort_values(["policy"])
        latex.append("\\begin{table}[t]\n\\centering\n\\caption{Hardware replay energy telemetry for offloaded MoE serving. Lower J/token is better.}\n\\label{tab:energy_replay}\n\\scriptsize\n\\begin{tabular}{lrrrr}\n\\toprule\nPolicy & TPOT (ms) & Avg W & Peak W & J/token \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            name = str(r.policy).replace("_", "-")
            latex.append(f"{name} & {f2(r.measured_tpot_ms)} & {f2(r.avg_power_w)} & {f2(r.peak_power_w)} & {f2(r.energy_per_token_j)} \\\\\n")
        latex.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    if cache_rows:
        df = pd.DataFrame(cache_rows)
        focus = df[df["policy"].isin(["lru", "pregated", "spice"])].copy()
        pivot = focus.pivot_table(index="cache_capacity", columns="policy", values="sim_tpot_ms", aggfunc="first")
        latex.append("\\begin{table}[t]\n\\centering\n\\caption{Cache-budget sensitivity under fixed Top-$K{=}6$. All methods use the same expert-transfer model. Lower TPOT is better.}\n\\label{tab:cache_budget}\n\\scriptsize\n\\begin{tabular}{rrrr}\n\\toprule\nCache slots & LRU & Pre-gated & SPICE \\\\\n\\midrule\n")
        for cap, row in pivot.sort_index().iterrows():
            latex.append(f"{int(cap)} & {f2(row.get('lru'))} & {f2(row.get('pregated'))} & {f2(row.get('spice'))} \\\\\n")
        latex.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    latex.append("\\textbf{System telemetry.} We add two ICCD-oriented measurements: a hardware replay for energy-per-token telemetry and a cache-budget sweep for memory-constrained serving. These experiments use synthetic routing traces and do not transfer datasets. Table~\\ref{tab:energy_replay} shows that SPICE reduces replay energy relative to Naive offloading, but it does not dominate every cache-based baseline; we therefore avoid claiming universal energy reduction. Table~\\ref{tab:cache_budget} shows that SPICE is most useful in the memory-constrained region (256--512 cache slots), where verified prefetching reduces fallback traffic without requiring the large resident expert set needed by cache-only policies. When the cache budget is large enough to hold most hot experts, all methods converge and the advantage naturally disappears.\n")
    (root / "ICCD_SYSTEM_RESULTS.md").write_text("".join(lines), encoding="utf-8")
    (root / "overleaf_iccd_system_results.tex").write_text("".join(latex), encoding="utf-8")
    print(root / "ICCD_SYSTEM_RESULTS.md")


if __name__ == "__main__":
    main()
