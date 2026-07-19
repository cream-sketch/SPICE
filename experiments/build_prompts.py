"""Build a diverse prompt file for trace collection from a wikitext parquet."""
import sys

import pandas as pd

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_parquet(src)
col = "text" if "text" in df.columns else df.columns[0]
paras = []
for t in df[col]:
    t = " ".join(str(t).split())
    if len(t) >= 600 and not t.startswith("="):
        paras.append(t[:4000])
    if len(paras) >= n:
        break
assert len(paras) >= n, f"only {len(paras)} long paragraphs found"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(paras[:n]) + "\n")
print(f"wrote {n} prompts to {out}; median len={sorted(len(p) for p in paras[:n])[n//2]} chars")
