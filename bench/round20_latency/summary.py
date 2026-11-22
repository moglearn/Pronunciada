"""Round 20 (latency): collect the run_latency.sh results into one table.

    python bench/round20_latency/summary.py <latency dir> <out json>
"""
from __future__ import annotations

import glob
import json
import os
import sys

D = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(D, "*.json"))):
    d = json.load(open(f))
    lang, rest = os.path.basename(f)[:-5].split("_", 1)
    s, prec, old, new = rest.rsplit("_", 3)
    rows.append({"lang": lang, "set": s, "precision": prec, "old": old, "new": new, "n": d["n"],
                 "clip_s_median": round(d["clip_seconds_median"], 2),
                 "old_median_ms": round(d["old_median_ms"], 2), "new_median_ms": round(d["new_median_ms"], 2),
                 "paired_median_diff_ms": round(d["paired_median_diff_ms"], 2),
                 "ci95": [round(x, 2) for x in d["paired_median_diff_ci"]],
                 "old_p90_ms": round(d["old_p90_ms"], 2), "new_p90_ms": round(d["new_p90_ms"], 2),
                 "old_max_rss_mb": round(d["old_max_rss_mb"]), "new_max_rss_mb": round(d["new_max_rss_mb"]),
                 "load_avg_1min_start_end": [round(x, 1) for x in d["load_avg_1min_start_end"]],
                 "wall_s": round(d["wall_s"], 1)})
trees = {"pre19": "c9773cc (before round 19)", "main": "main 8c74789 (round 19)", "new": "this branch"}
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump({"tool": "bench/round19_merge/latency.py (two warm workers, one scorer running at a time, "
                       "clip order alternating; 4 threads; posterior cache cleared per call; 95% clip-bootstrap "
                       "interval of the median paired difference)", "trees": trees, "rows": rows}, fh, indent=1)
for r in rows:
    print(f"{r['lang']} {r['set']:<15} {r['precision']} {r['old']:>5} -> {r['new']:<4} n={r['n']:<3} "
          f"{r['clip_s_median']:5.2f} s  {r['old_median_ms']:7.1f} -> {r['new_median_ms']:7.1f} ms  "
          f"diff {r['paired_median_diff_ms']:+6.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  "
          f"load {r['load_avg_1min_start_end']}")
