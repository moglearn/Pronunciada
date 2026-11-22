"""Dev check of combining the shipped word probability with the round-19 recalibrated one.

    python -m bench.round19_ja.combine SHIPPED.jsonl RECAL.jsonl OUT.jsonl [--mode min|geo]

Both dumps must be of the same clips and front-end (word lists equal). Writes a dump-like file whose
word prob is min(shipped, recal) (or their geometric mean) for unit_eval / native rates.
"""
from __future__ import annotations

import argparse
import json
import math


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--mode", default="min")
    a = ap.parse_args(argv)
    A = {json.loads(x)["id"]: json.loads(x) for x in open(a.a, encoding="utf-8")}
    n_bad = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for x in open(a.b, encoding="utf-8"):
            r = json.loads(x)
            s = A.get(r["id"])
            if r.get("error") or not s or s.get("error") or len(s["words"]) != len(r["words"]):
                n_bad += 1
                continue
            for w, v in zip(r["words"], s["words"]):
                p, q = w["prob"], v["prob"]
                w["prob"] = min(p, q) if a.mode == "min" else math.sqrt(max(p, 1e-12) * max(q, 1e-12))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("skipped", n_bad)


if __name__ == "__main__":
    main()
