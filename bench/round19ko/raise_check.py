"""Round 19 (Korean, part 3): how often does the two-sided step RAISE a word that carries a planted
error, and do a lower cap or an aligner gate on raising remove those cases without giving back the
native gain? Dev only, speaker-grouped 4-fold out of fold (the part-2 fit per fold).

    python -m bench.round19ko.raise_check [--write]

Variants: cap on raising in {3, 2, 1.5, 1, 0.5} x raise gate in {none, tau 1, 2, 4} (a word with any
sound whose aligner llr_sub < -tau is never raised).
Rule (fixed before looking): the part-2 guards (each native-pool source keeps or lowers its words
below 0.6 and keeps or raises its accepted words; FLEURS-validation planted AUC not lower); among
the guarded variants with out-of-fold dev AUC >= the best guarded AUC - 0.002, the lowest share of
changed words that are raised; ties by AUC.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round19ko import devmetrics as DM
from bench.round19ko import fit2 as F2
from bench.round19ko import gate as G

QC, L2, WPOS = 0.1, 10.0, 2.0
CAPS = (3.0, 2.0, 1.5, 1.0, 0.5)
GATES = (None, 1.0, 2.0, 4.0, "id0", "id1", "id2")


def main(write=False):
    dev, zt, vc = F2.data()
    k = 4
    fits = []
    for f in range(k):
        dtr = {a: r for a, r in dev.items() if F2.fold_of(r["meta"]["speaker"], k) != f}
        ztr = {a: r for a, r in zt.items() if F2.fold_of(r["meta"]["speaker"], k) != f}
        pool = {**{a: r for a, r in dtr.items() if not r["meta"].get("label")}, **ztr}
        thr = G.class_pos_thresholds(pool, QC)
        X, off, y = F2.stack({**dtr, **ztr}, thr, lambda r: r["changed"])
        fits.append((thr, F2.fit_delta(X, off, y, L2, np.where(y == 0, WPOS, 1.0))))
    feats = {}
    for src in (dev, zt, vc):
        for a, r in src.items():
            thr, b = fits[F2.fold_of(r["meta"]["speaker"], k)]
            feats[a] = (F2.feats(r, thr), b)
    res = []
    for cap, gate in itertools.product(CAPS, GATES):
        P, Pz, Pc, D = {}, {}, {}, {}
        for src, dst in ((dev, P), (zt, Pz), (vc, Pc)):
            for a, r in src.items():
                X, b = feats[a]
                d = F2.deltas(r, X, b, cap, gate)
                D[a] = d
                dst[a] = 1 / (1 + np.exp(-(F2.lg(r["prob"]) + d)))
        m = DM.metrics(F2.rows_for(dev), P)
        ps = F2.per_source(dev, zt, P, Pz)
        pl = F2.planted(vc, Pc)
        ok = all(v["below06"][1] <= v["below06"][0] and v["accept05"][1] >= v["accept05"][0] for v in ps.values()) \
            and pl[1] >= pl[0] - 1e-4
        raised = defaultdict(lambda: [0, 0])
        crossed = defaultdict(lambda: [0, 0])
        for a, r in dev.items():
            for j in r["changed"]:
                bk = r["meta"]["label"]["bucket"]
                for key in (bk, "all"):
                    raised[key][0] += int(D[a][j] > 0)
                    crossed[key][0] += int(r["prob"][j] < 0.5 <= P[a][j])
                    raised[key][1] += 1
        res.append({"cap": cap, "gate": gate, "ok": bool(ok), "auc": m["all"]["auc"], "fa80": m["all"]["fa80"],
                    "bucket": m["all"]["bucket"], "native": ps, "planted": pl,
                    "raised_changed": {k2: [v[0], v[1], v[0] / v[1]] for k2, v in sorted(raised.items())},
                    "raised_across_0.5": {k2: v[0] for k2, v in sorted(crossed.items())}})
        print(ok, cap, gate, round(m["all"]["auc"], 4), round(m["all"]["fa80"], 4),
              "raised", {k2: f"{v[0]}/{v[1]}" for k2, v in sorted(raised.items())}, "across0.5", dict(crossed) and {k2: v[0] for k2, v in crossed.items()},
              {s: (v["below06"], v["accept05"]) for s, v in ps.items()}, flush=True)
    oks = [r for r in res if r["ok"]]
    top = max(r["auc"] for r in oks)
    near = [r for r in oks if r["auc"] >= top - 0.002]
    best = min(near, key=lambda r: (r["raised_changed"]["all"][2], -r["auc"]))
    print("chosen", {k2: best[k2] for k2 in ("cap", "gate", "auc", "fa80")}, best["raised_changed"]["all"])
    Path("results/tables/round19_ko3_raise.json").write_text(json.dumps({"variants": res, "chosen": best}, indent=1, default=float))
    if write:
        p = Path("pronunciada/data/calib/ko_identity_word.json")
        par = json.loads(p.read_text())
        par["cap_up"] = best["cap"]
        par["raise_gate"] = best["gate"]
        p.write_text(json.dumps(par, indent=1, ensure_ascii=False))
        print("wrote", p)


if __name__ == "__main__":
    main("--write" in sys.argv[1:])
