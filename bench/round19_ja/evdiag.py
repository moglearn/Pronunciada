"""Single-evidence view on the dev set: AUC of each word evidence, planted words of each bucket
against words of control clips.

    python -m bench.round19_ja.evdiag [--length r19]
"""
from __future__ import annotations

import argparse
import math
import pickle

import numpy as np

from bench.round19_ja.cv_pool import WORK
from bench.round19_ja.step import word_labels


def main(argv=None):
    from bench.round19_ja.unit_eval import roc_auc

    ap = argparse.ArgumentParser()
    ap.add_argument("--length", default="r19")
    a = ap.parse_args(argv)
    D = pickle.loads((WORK / f"sel2_{a.length}.pkl").read_bytes())
    ev = {"base": lambda w: math.log(max(w["base"], 1e-9) / max(1 - w["base"], 1e-9)),
          "id_min": lambda w: min(w["sids"]) if w["sids"] else 5.0,
          "len_min": lambda w: min(w["lens"]) if w["lens"] else 8.0,
          "mm_tot": lambda w: -(w["mm"]["ins"] + w["mm"]["del"] + w["mm"]["sub"]),
          "mm_ins": lambda w: -w["mm"]["ins"]}
    neg = [w for r in D["dev"] if not r["label"] for w in r["words"]]
    pos = {}
    for r in D["dev"]:
        if r["label"]:
            for w, y in zip(r["words"], word_labels(r)):
                if y:
                    pos.setdefault(r["label"]["bucket"], []).append(w)
    for name, f in ev.items():
        nv = [f(w) for w in neg]
        line = f"{name:8s}"
        for b, ws in sorted(pos.items()):
            pv = [f(w) for w in ws]
            line += f" {b[:8]}={roc_auc([1] * len(pv) + [0] * len(nv), [-x for x in pv] + [-x for x in nv]):.3f}"
        print(line)


if __name__ == "__main__":
    main()
