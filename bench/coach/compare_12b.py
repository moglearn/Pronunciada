"""Round 12b: coaching score on speechocean762 TRAIN with sound_B vs sound_E inputs (out-of-fold
logits of each calibrator; coaching logistic refitted out-of-fold, 5-fold speaker-grouped, exactly
as bench/coach/fit_v2.py). Precision at recall 0.20 / 0.33 / 0.47 and at the fitted thresholds.

    python -m bench.coach.compare_12b
"""
from __future__ import annotations

import json
import os

import numpy as np


def prec_at_recall(y, s, recalls=(0.20, 0.33, 0.47)):
    o = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[o].astype(float))
    return [float(tp[int(np.searchsorted(tp, r * y.sum()))] / (int(np.searchsorted(tp, r * y.sum())) + 1))
            for r in recalls]


def main():
    import pandas as pd

    from bench.coach import fit_v2 as V
    from pronunciada import coach as Co

    R = V.so762_rows("so762-train", "train")
    m = np.isfinite(R["acc"])
    R = {k: (v[m] if isinstance(v, np.ndarray) else [x for x, keep in zip(v, m) if keep]) for k, v in R.items()}
    y = R["acc"] < 2
    pq = os.path.join(os.environ.get("PRONUNCIADA_CACHE", os.path.expanduser("~/.cache/pronunciada")),
                      "calib", "sound_oof.parquet")
    df = pd.read_parquet(pq, columns=["utt", "phone_index", "B_z_len", "B_z_str", "E_z_len", "E_z_str"])
    key = {(u, int(i)): j for j, (u, i) in enumerate(zip(df.utt.values, df.phone_index.values))}
    j = np.array([key[(u, int(i))] for u, i in zip(R["utt"], R["i"])])
    out = {}
    for pre in ("B", "E"):
        X = dict(R)
        X["zl"] = df[f"{pre}_z_len"].values[j]
        X["zs"] = np.minimum(df[f"{pre}_z_str"].values[j], X["zl"])
        F = Co.feature_matrix(X)
        p = V.oof(F, y, R["spk"])
        th = {s: V.thr_for_precision(p, y, t) for s, t in V.TARGETS.items()}
        from sklearn.metrics import roc_auc_score

        out[pre] = {"auc_oof": float(roc_auc_score(y, p)), "prec_at_recall": prec_at_recall(y, p),
                    "operating_points": {s: V.pr_at(p, y, th[s]) for s in V.TARGETS}}
    print(json.dumps(out, indent=1, default=float))
    from bench.calib.common import save_json

    save_json("coaching_12b_compare.json", out)


if __name__ == "__main__":
    main()
