"""Switches off = main: every numeric output of Scorer.score on the same clips, this branch vs a main checkout.

    python -m bench.round20_native_fa.equiv dump --out FILE [--n 24] [--precision fp32]   # run in each checkout
    python -m bench.round20_native_fa.equiv compare A B

The clips: the first n practice items per language (en, de, ko, es, fr). The round-20 switches are unset.
"""
from __future__ import annotations

import argparse
import json
import math
import os


def flat(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("timing_ms",):
                continue
            yield from flat(v, f"{pre}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from flat(v, f"{pre}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield pre, float(x)
    elif x is None or isinstance(x, (str, bool)):
        yield pre, x


def dump(out, n, precision):
    for k in list(os.environ):
        if k.startswith("PRONUNCIADA_R20"):
            del os.environ[k]

    from pronunciada import Scorer

    with open(os.environ["NFA_ITEMS"], encoding="utf-8") as f:     # {lang: [[id, audio, text], ...]}
        items = json.load(f)
    res = {}
    for lang, its in items.items():
        sc = Scorer(lang, threads=2, precision=precision)
        for iid, audio, text in its[:n]:
            r = sc.score(audio, text)
            res[f"{lang}:{iid}"] = dict(flat(r.to_dict()))
    with open(out, "w") as f:
        json.dump(res, f)


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    worst, nkeys, mism = 0.0, 0, []
    for k in A:
        for f, v in A[k].items():
            w = B[k].get(f, "MISSING")
            nkeys += 1
            if isinstance(v, float) and isinstance(w, float):
                if math.isnan(v) and math.isnan(w):
                    continue
                worst = max(worst, abs(v - w))
            elif v != w:
                mism.append((k, f, v, w))
    print({"items": len(A), "values": nkeys, "max_abs_diff": worst, "other_mismatches": len(mism)}, mism[:5])
    return worst, mism


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "compare"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--precision", default="fp32")
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.out, a.n, a.precision)
    else:
        compare(*a.files)


if __name__ == "__main__":
    main()
