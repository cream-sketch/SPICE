from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def load_jsons(root: Path):
    for p in sorted(root.rglob("*.json")):
        try:
            yield p, json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    lines = ["# SPICE ICCD Supplemental Experiment Summary\n"]
    for path, obj in load_jsons(root):
        rel = path.relative_to(root)
        exp = obj.get("experiment", rel.stem)
        lines.append(f"\n## {exp}: `{rel}`\n")
        if exp.startswith("prefetch_system") or exp == "sota_proxy":
            for row in obj.get("rows", []):
                row = {**row, "experiment": exp, "file": str(rel)}
                rows.append(row)
            lines.append(f"- Copy microbench: {obj.get('copy_microbench')}\n")
            lines.append(f"- GPU power sample: {obj.get('gpu_power_watts_sample')} W\n")
            if "important_caveat" in obj:
                lines.append(f"- Caveat: {obj['important_caveat']}\n")
        else:
            for k, v in obj.items():
                if isinstance(v, (int, float, str)):
                    lines.append(f"- {k}: {v}\n")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(root / "prefetch_system_summary.csv", index=False)
        keep = [
            c
            for c in [
                "experiment",
                "policy",
                "variant",
                "top_k",
                "cache_hit_rate",
                "fallback_rate",
                "h2d_gb",
                "sim_tpot_ms",
                "pcie_active_fraction",
                "draft_overhead_ms",
                "online_overhead_ms",
            ]
            if c in df.columns
        ]
        lines.append("\n## Prefetch System Table\n\n")
        lines.append(markdown_table(df[keep]))
        lines.append("\n")
    (root / "SUMMARY.md").write_text("".join(lines), encoding="utf-8")
    print(root / "SUMMARY.md")


if __name__ == "__main__":
    main()
