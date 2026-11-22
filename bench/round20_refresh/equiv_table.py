"""Round 20 release check, step 1 table: collect the comparisons of run_equiv.sh.

    python bench/round20_refresh/equiv_table.py <out dir of run_equiv.sh> results/tables/round20_release_equiv.json

For Mandarin (expected to differ: the round-20 word step) it also lists which output keys differ.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

D = Path(sys.argv[1])
rows = []
for f in sorted(D.glob("cmp_*.json")):
    lang, s, p = f.stem[4:].split("_")
    r = json.loads(f.read_text(encoding="utf-8"))
    row = {"lang": lang, "set": s, "precision": p, "items": r["items_a"], "values": r["values"],
           "max_abs_diff": r["max_abs_diff"], "n_numeric_diff": r["n_numeric_over_tol"],
           "n_other_diff": r["n_other_diff"], "identical": r["identical_within_tol"]}
    if lang == "zh":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round19_merge"))
        from equiv import _flat

        A = [json.loads(l) for l in open(D / f"main_{lang}_{s}_{p}.jsonl", encoding="utf-8")]
        B = [json.loads(l) for l in open(Path(r["b"]), encoding="utf-8")]
        keys = Counter()
        for ra, rb in zip(A, B):
            fa, fb = _flat(ra), _flat(rb)
            for k in set(fa) | set(fb):
                if fa.get(k) != fb.get(k):
                    keys[re.sub(r"\[\d+\]", "[]", k)] += 1
        row["differing_keys"] = dict(keys.most_common())
    rows.append(row)
out = {"release": "main d7434aa", "reference": "round-19 frozen run 2 code 836dba6 (dumps of the round-19 merge check)",
       "tolerance": 0, "rows": rows}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
for r in rows:
    print(r["lang"], r["set"], r["precision"], r["items"], r["values"], r["max_abs_diff"], r["identical"],
          r.get("differing_keys", ""))
