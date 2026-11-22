"""Round 19 (Korean, final): dev table of both variants, out of fold (speaker-grouped 4-fold, the
part-2 fit per fold): cap +3 without a gate, and the adopted no-raise rule (cap +3, no raise when
the word's lowest identity margin < -2).

    python -m bench.round19ko.dev_final      # -> results/tables/round19_ko_final_dev.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round19ko import devmetrics as DM
from bench.round19ko import fit2 as F2
from bench.round19ko import gate as G


def main():
    dev, zt, vc = F2.data()
    k = 4
    fits = []
    for f in range(k):
        dtr = {a: r for a, r in dev.items() if F2.fold_of(r["meta"]["speaker"], k) != f}
        ztr = {a: r for a, r in zt.items() if F2.fold_of(r["meta"]["speaker"], k) != f}
        pool = {**{a: r for a, r in dtr.items() if not r["meta"].get("label")}, **ztr}
        thr = G.class_pos_thresholds(pool, 0.1)
        X, off, y = F2.stack({**dtr, **ztr}, thr, lambda r: r["changed"])
        fits.append((thr, F2.fit_delta(X, off, y, 10.0, np.where(y == 0, 2.0, 1.0))))
    rows = F2.rows_for(dev)
    out = {"before": DM.metrics(rows, boot=1000)}
    for name, gate in (("cap3_no_gate", None), ("cap3_no_raise_below_-2", "id2")):
        P, Pz, Pc = {}, {}, {}
        raised = defaultdict(lambda: [0, 0, 0])
        for src, dst in ((dev, P), (zt, Pz), (vc, Pc)):
            for a, r in src.items():
                thr, b = fits[F2.fold_of(r["meta"]["speaker"], k)]
                X = F2.feats(r, thr)
                d = F2.deltas(r, X, b, 3.0, gate)
                dst[a] = 1 / (1 + np.exp(-(F2.lg(r["prob"]) + d)))
                if src is dev:
                    for j in r["changed"]:
                        for key in (r["meta"]["label"]["bucket"], "all"):
                            raised[key][0] += int(d[j] > 0)
                            raised[key][1] += int(r["prob"][j] < 0.5 <= dst[a][j])
                            raised[key][2] += 1
        m = DM.metrics(rows, P, boot=1000)
        out[name] = {"dev": m, "paired_vs_before": DM.paired(rows, None, P), "native": F2.per_source(dev, zt, P, Pz),
                     "planted_val": F2.planted(vc, Pc),
                     "raised_changed": {kk: {"raised": v[0], "raised_past_0.5": v[1], "n": v[2]} for kk, v in sorted(raised.items())}}
        print(name, DM.line(m), out[name]["native"], out[name]["planted_val"], dict(out[name]["raised_changed"]["all"]), flush=True)
    print("before", DM.line(out["before"]))
    Path("results/tables/round19_ko_final_dev.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
