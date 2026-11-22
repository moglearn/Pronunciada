"""Parity of the offline word recalibration (bench/round19_ja/wordcal.py on collected evidence) with
the runtime Scorer (a dump.py file of the same clips).

    python -m bench.round19_ja.parity DUMP.jsonl [--tag trees]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from bench.round19_ja.collect import load
from bench.round19_ja.cv_pool import WORK
from bench.round19_ja.wordcal import mapped, model_logit, word_rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--tag", default="trees")
    ap.add_argument("-n", type=int, default=60)
    a = ap.parse_args(argv)
    M = json.loads((WORK / f"wordcal_{a.tag}.json").read_text())
    dump = {json.loads(x)["id"]: json.loads(x) for x in open(a.dump, encoding="utf-8")}
    diffs = []
    for it in load("dev")[:a.n]:
        rows = word_rows(it)
        X = np.array([[r[f] for f in M["features"]] for r in rows])
        p = 1 / (1 + np.exp(-mapped(M, model_logit(M, X))))
        q = np.array([w["prob"] for w in dump[it["id"]]["words"]])
        if len(p) != len(q):
            print("len mismatch", it["id"])
            continue
        d = np.abs(p - q)
        diffs.append(d.max())
        if d.max() > 1e-3:
            k = int(d.argmax())
            dw = dump[it["id"]]["words"][k]
            print(it["id"], it["texts"][k], f"offline {p[k]:.4f} runtime {q[k]:.4f}",
                  "sounds", round(float(it["sounds"][k]), 4), round(dw["sounds"], 4),
                  "present", it["present"][k], dw["present"], "con", it["con"][k], dw["con"])
    print("max abs diff", max(diffs), "items", len(diffs), "with diff>1e-3", sum(d > 1e-3 for d in diffs))


if __name__ == "__main__":
    main()
