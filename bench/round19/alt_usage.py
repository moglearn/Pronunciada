"""Round 19 (German): which rule-based alternatives does the alignment take, on correct words vs
on changed words?

    python -m bench.round19.alt_usage dev-base [--fold fit]

Per (canonical sound -> chosen alternative) pair: how often natives' correct words take it (share
of that canonical sound's occurrences), and how often it is taken inside a changed word, split
into changed words that the system catches at the FA@80 threshold and those it misses. A pair
that is frequent in missed changed words but rare in natives lets real changes through; a pair
frequent in natives is needed.
"""
from __future__ import annotations

import argparse
from collections import Counter

from bench.round19.audit import threshold, word_rows


def chosen(p):
    a = p["alt"]
    if a == 0:
        return None
    if a == -1:
        return ""
    return p["alts"][a - 1][0]


def run(name, fold="all", source=None, top=40):
    rows = word_rows(name, fold, source)
    thr = threshold(rows)
    occ, nat, miss, hit = Counter(), Counter(), Counter(), Counter()
    n_miss = sum(1 for r in rows if r["label"] and r["err"] < thr)
    n_hit = sum(1 for r in rows if r["label"] and r["err"] >= thr)
    for r in rows:
        for w in r["iw"]:
            for p in w["phones"]:
                c = chosen(p)
                if not r["label"]:
                    occ[p["ipa"]] += 1
                    if c is not None:
                        nat[(p["ipa"], c)] += 1
                elif c is not None:
                    (miss if r["err"] < thr else hit)[(p["ipa"], c)] += 1
    keys = set(nat) | set(miss)
    print(f"{name}: changed words missed at FA@80 {n_miss}, caught {n_hit}")
    print(f"{'canon':8s} {'alt':8s} {'native n':>8s} {'share':>6s} {'missed':>6s} {'caught':>6s}")
    for k in sorted(keys, key=lambda k: -(miss[k] * 50 + nat[k]))[:top]:
        print(f"{k[0]:8s} {k[1] or '∅':8s} {nat[k]:8d} {100 * nat[k] / max(1, occ[k[0]]):5.1f}% {miss[k]:6d} {hit[k]:6d}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--fold", default="all")
    ap.add_argument("--source")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv)
    run(a.name, a.fold, a.source, a.top)


if __name__ == "__main__":
    main()
