"""Round 19 merge: compare the dumps of run_equiv2.sh (fully merged code; checks B and C, second pass).

    python bench/round19_merge/compare_all2.py <dump dir> <out json>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equiv import compare  # noqa: E402

D = sys.argv[1]
LANGS = ("en", "ko", "de", "ja", "zh", "es", "fr", "it", "pt", "ru")
BRANCH = {"en": ["branch"], "ko": ["branch"], "de": ["branch", "branchid"], "ja": ["branchjaf"],
          "es": ["branchid"], "ru": ["branchid"], "pt": ["branchid"], "zh": ["branchid"],
          "fr": ["main"], "it": ["main"]}
NAMES = {"branch": "own branch (first pass dump)", "branchid": "identity rollout 51ba321",
         "branchjaf": "Japanese final c5eb79c", "main": "main c9773cc (nothing shipped)"}


def path(name, lang, s, p):
    return os.path.join(D, f"{name}_{lang}_{s}_{p}.jsonl")


rows = []


def add(check, a, b, lang, s, p):
    pa, pb = path(a, lang, s, p), path(b, lang, s, p)
    if not (os.path.exists(pa + ".done") and os.path.exists(pb + ".done")):
        rows.append({"check": check, "lang": lang, "set": s, "precision": p, "missing": True})
        return
    r = compare([pa, pb], 1e-6, quiet=True)
    with open(pa, encoding="utf-8") as fh:
        n = sum(1 for _ in fh)
    rows.append({"check": check, "lang": lang, "set": s, "precision": p, "scores": n, "values": r["values"],
                 "max_abs_diff": r["max_abs_diff"], "n_numeric_over_1e-6": r["n_numeric_over_tol"],
                 "n_other_diff": r["n_other_diff"], "identical_within_1e-6": r["identical_within_tol"],
                 "examples": r["examples"][:4]})


for p in ("fp32", "int8"):
    for lang in LANGS:
        sets = ["fleurs-val", "so762-train"] if lang == "en" else ["fleurs-val"]
        for s in sets:
            for b in BRANCH[lang]:
                add(f"B merged = {NAMES[b]}", "merged2", b, lang, s, p)
            add("C merged, all switches off = main", "merged2off", "main", lang, s, p)
            add("(info) merged vs first-pass merged", "merged2", "merged", lang, s, p)
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=1, ensure_ascii=False)
for r in rows:
    if r.get("missing"):
        print(f"{r['check']:<44} {r['lang']} {r['set']:<12} {r['precision']} MISSING")
    else:
        print(f"{r['check']:<44} {r['lang']} {r['set']:<12} {r['precision']} scores {r['scores']} "
              f"max|d| {r['max_abs_diff']:.2e} over {r['n_numeric_over_1e-6']} other {r['n_other_diff']} "
              f"-> {'OK' if r['identical_within_1e-6'] else 'DIFFERENT'}")
