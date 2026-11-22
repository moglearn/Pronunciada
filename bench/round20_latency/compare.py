"""Round 20 (latency): compare the dumps of run_equiv.sh (main 8c74789 vs this branch); exact equality.

    python bench/round20_latency/compare.py <dump dir> <out json>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "round19_merge"))
from equiv import compare  # noqa: E402

D = sys.argv[1]
rows = []
for p in ("fp32", "int8"):
    for lang in ("ko", "ja", "en", "de", "zh", "es", "fr", "it", "pt", "ru"):
        a, b = (os.path.join(D, f"{t}_{lang}_fleurs-val_{p}.jsonl") for t in ("old", "new"))
        if not (os.path.exists(a + ".done") and os.path.exists(b + ".done")):
            rows.append({"lang": lang, "precision": p, "missing": True})
            continue
        r = compare([a, b], 0.0, quiet=True)          # tolerance 0: any difference counts
        with open(a, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        rows.append({"lang": lang, "precision": p, "scores": len(recs),
                     "clips": len({x["id"].split("#")[0] for x in recs}),
                     "errors": sum("error" in x for x in recs), "values": r["values"],
                     "max_abs_diff": r["max_abs_diff"], "n_numeric_diff": r["n_numeric_over_tol"],
                     "n_other_diff": r["n_other_diff"], "identical": r["identical_within_tol"],
                     "examples": r["examples"][:4]})
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump({"old": "main 8c74789", "new": "this branch", "set": "fleurs-val (own text + one word swapped)",
               "outputs": "to_dict (without timing), rescore at strictness 0 and 1, details, coaching, to_azure",
               "rows": rows}, fh, indent=1, ensure_ascii=False)
for r in rows:
    if r.get("missing"):
        print(f"{r['lang']} {r['precision']} MISSING")
    else:
        print(f"{r['lang']} {r['precision']}: {r['clips']} clips, {r['scores']} scores, {r['values']} values, "
              f"max|diff| {r['max_abs_diff']:.1e}, differing {r['n_numeric_diff']} + {r['n_other_diff']} "
              f"-> {'IDENTICAL' if r['identical'] else 'DIFFERENT'}")
