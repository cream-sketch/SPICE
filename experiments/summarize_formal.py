import glob
import json
import sys

rows = []
for f in sorted(glob.glob(sys.argv[1] + "/bench_*.json")):
    d = json.load(open(f))
    c = d["cache"]
    t = c.get("miss_by_tier", {})
    rows.append((d["policy"], d["cache_experts"], d["tpot_ms_mean"],
                 d["ttft_s_mean"], c["hit_rate"], c["fallback_rate"], c["h2d_gb"],
                 d.get("avg_power_w") or 0, d.get("energy_per_token_j") or 0,
                 t.get("gpu_store", 0), t.get("ram", 0), t.get("nvme", 0),
                 d["per_prompt"][0]["generated"][:24]))
rows.sort(key=lambda r: (r[1], r[0]))
hdr = ("policy", "cache", "tpot_ms", "ttft_s", "hit", "fallbk", "h2d_gb",
       "avg_W", "J_tok", "m_gpu", "m_ram", "m_nvme", "gen")
print("%-8s %5s %8s %7s %6s %6s %7s %6s %6s %6s %6s %6s  %s" % hdr)
for r in rows:
    print("%-8s %5d %8.1f %7.2f %6.3f %6.3f %7.1f %6.1f %6.2f %6d %6d %6d  %s" % r)
