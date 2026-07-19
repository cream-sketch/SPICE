import glob
import json
import sys

rows = []
for f in sorted(glob.glob(sys.argv[1] + "/bench_*.json")):
    d = json.load(open(f))
    c = d["cache"]
    rows.append((d["policy"], d["cache_experts"], d["tpot_ms_mean"], d["ttft_s_mean"],
                 c["hit_rate"], c["fallback_rate"], c["h2d_gb"], d["cpu_computed_experts"],
                 d.get("draft_prefetch_ms_total", 0), d["per_prompt"][0]["generated"][:40]))
print("%-8s %5s %9s %7s %6s %6s %8s %7s %8s  %s" % ("policy", "cache", "tpot_ms", "ttft_s", "hit", "fallbk", "h2d_gb", "cpu_exp", "draft_ms", "gen"))
for r in rows:
    print("%-8s %5d %9.1f %7.2f %6.3f %6.3f %8.1f %7d %8.0f  %s" % r)
