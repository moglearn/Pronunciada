"""Round 19 (German): frequent native realisations that the reference does not allow.

    python -m bench.round19.native_subs dev-text1ea [--fold fit] [--min-n 30] [--min-rate 0.04]

Over the CONTROL items of a dump (native speakers reading their own prompt), every reference sound
in context (stressed / unstressed nucleus or consonant; word-initial / medial / final) is counted,
and so are the cases where the sound scored low (shown log-odds < 1) and the decoder heard another
symbol over its span. Prints (sound, context, heard): count / occurrences, and the mean shown
log-odds; candidates for a native-variant table are frequent ones.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from bench.round19.collect import manifest
from bench.round19.evaluate import load_dump


def ctx(ph, i):
    p = ph[i]
    pos = "I" if i == 0 else ("F" if i == len(ph) - 1 else "M")
    if p["nuc"]:
        return ("S" if p["stress"] == 1 else "U") + pos
    return "C" + pos


def count(name, fold="fit", sources=("cv", "fleurs")):
    M = {m["id"]: m for m in manifest("dev")}
    D = load_dump(name)
    occ, sub = Counter(), Counter()
    zsum = defaultdict(float)
    for k, r in D.items():
        m = M[k]
        if r.get("error") or m["label"] or (fold != "all" and m["fold"] != fold) or m["source"] not in sources:
            continue
        for w in r["rwords"]:
            ph = w["phones"]
            for i, p in enumerate(ph):
                c = ctx(ph, i)
                occ[(p["ipa"], c)] += 1
                if p["zl"] is not None and p["zl"] < 1.0 and p["alt"] == 0:
                    h = p["heard"] or "∅"
                    if h != p["ipa"]:
                        sub[(p["ipa"], c, h)] += 1
                        zsum[(p["ipa"], c, h)] += p["zl"]
    return occ, sub, zsum


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--fold", default="fit")
    ap.add_argument("--min-n", type=int, default=4)
    ap.add_argument("--top", type=int, default=60)
    a = ap.parse_args(argv)
    occ, sub, zsum = count(a.name, a.fold)
    rows = [(k, n, occ[k[:2]]) for k, n in sub.items() if n >= a.min_n]
    rows.sort(key=lambda x: -x[1])
    for (ipa, c, h), n, o in rows[: a.top]:
        print(f"{ipa:6s} {c:3s} > {h:8s} {n:4d} / {o:5d} = {100 * n / o:5.1f}%  mean zl {zsum[(ipa, c, h)] / n:.2f}")


if __name__ == "__main__":
    main()
