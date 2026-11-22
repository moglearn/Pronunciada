"""Round 19 merge: compare the dumps of bench/round19_merge/run_equiv.sh (checks B and C).

    python bench/round19_merge/compare_all.py <dump dir> <out json>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equiv import compare  # noqa: E402

D = sys.argv[1]
SETS = {"en": ["fleurs-val", "so762-train"], "ko": ["fleurs-val"], "de": ["fleurs-val"], "ja": ["fleurs-val"]}
SIX = ("zh", "es", "fr", "it", "pt", "ru")


def f(name, lang, s, p):
    return os.path.join(D, f"{name}_{lang}_{s}_{p}.jsonl")


def n_items(path):
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    return len(rows), sum(1 for r in rows if "error" in r)


rows = []
for p in ("fp32", "int8"):
    for lang, sets in SETS.items():
        for s in sets:
            for check, a, b in (("B merged = branch", "merged", "branch"),
                                ("C merged, switches off = main", "mergedoff", "main"),
                                ("(info) merged vs main", "merged", "main")):
                pa, pb = f(a, lang, s, p), f(b, lang, s, p)
                if not (os.path.exists(pa + ".done") and os.path.exists(pb + ".done")):
                    rows.append({"check": check, "lang": lang, "set": s, "precision": p, "missing": True})
                    continue
                r = compare([pa, pb], 1e-6, quiet=True)
                n, ne = n_items(pa)
                rows.append({"check": check, "lang": lang, "set": s, "precision": p, "scores": n, "errors": ne,
                             "values": r["values"], "max_abs_diff": r["max_abs_diff"],
                             "n_numeric_over_1e-6": r["n_numeric_over_tol"], "n_other_diff": r["n_other_diff"],
                             "identical_within_1e-6": r["identical_within_tol"], "examples": r["examples"][:4]})
    for lang in SIX:
        pa, pb = f("merged", lang, "fleurs-val", p), f("main", lang, "fleurs-val", p)
        if not (os.path.exists(pa + ".done") and os.path.exists(pb + ".done")):
            rows.append({"check": "B unchanged language: merged = main", "lang": lang, "precision": p, "missing": True})
            continue
        r = compare([pa, pb], 1e-6, quiet=True)
        n, ne = n_items(pa)
        rows.append({"check": "B unchanged language: merged = main", "lang": lang, "set": "fleurs-val", "precision": p,
                     "scores": n, "errors": ne, "values": r["values"], "max_abs_diff": r["max_abs_diff"],
                     "n_numeric_over_1e-6": r["n_numeric_over_tol"], "n_other_diff": r["n_other_diff"],
                     "identical_within_1e-6": r["identical_within_tol"], "examples": r["examples"][:4]})
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=1, ensure_ascii=False)
for r in rows:
    if r.get("missing"):
        print(r["check"], r["lang"], r.get("set", ""), r["precision"], "MISSING")
    else:
        print(f"{r['check']:<34} {r['lang']} {r['set']:<12} {r['precision']} scores {r['scores']} values {r['values']} "
              f"max|d| {r['max_abs_diff']:.2e} over {r['n_numeric_over_1e-6']} other {r['n_other_diff']} "
              f"-> {'OK' if r['identical_within_1e-6'] else 'DIFFERENT'}")
