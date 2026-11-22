"""Round 19 (German): which factor of the word probability sinks the flagged correct words?

    python -m bench.round19.chain dev-base [--source cv]

word prob = sounds (word model) x stress factor x present factor (x length / contrast: off in de).
For flagged correct words and for changed words, the share whose largest log-loss comes from each
factor, and the mean log of each factor.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

from bench.round19.audit import threshold, word_rows


def factors(r):
    out = {"sounds": 0.0, "stress": 0.0, "present": 0.0}
    for w in r["iw"]:
        for k in out:
            v = w.get(k)
            if v is not None:
                out[k] += math.log(max(v, 1e-4))
    return out


def run(name, fold="all", source=None):
    rows = word_rows(name, fold, source)
    thr = threshold(rows)
    groups = {"flagged correct": [r for r in rows if not r["label"] and r["err"] >= thr],
              "other correct": [r for r in rows if not r["label"] and r["err"] < thr],
              "changed caught": [r for r in rows if r["label"] and r["err"] >= thr],
              "changed missed": [r for r in rows if r["label"] and r["err"] < thr]}
    for g, rs in groups.items():
        c = Counter()
        mean = Counter()
        for r in rs:
            f = factors(r)
            c[min(f, key=f.get)] += 1
            for k, v in f.items():
                mean[k] += v / max(1, len(rs))
        print(f"{g:16s} n={len(rs):5d}  main factor: " + ", ".join(f"{k} {v}" for k, v in c.most_common())
              + "  | mean log: " + ", ".join(f"{k} {v:.3f}" for k, v in mean.items()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--fold", default="all")
    ap.add_argument("--source")
    a = ap.parse_args(argv)
    run(a.name, a.fold, a.source)


if __name__ == "__main__":
    main()
