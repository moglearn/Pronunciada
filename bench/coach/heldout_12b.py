"""Round 12b HELD-OUT (run once): English coaching score on speechocean762 TEST, round 11
(dump-so762-test-r11 = the dump made before round 12b, global coaching model) vs round 12b
(fresh dump, English sound_E inputs and logistic) vs Azure (cached phoneme accuracy): precision at
recall 0.20 / 0.33 / 0.47 and PR-AUC on the phones all three cover, speaker-bootstrap 95 % CIs.

    python -m bench.coach.heldout_12b
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

RECALLS = (0.20, 0.33, 0.47)


def prec_at(score, y, r):
    o = np.argsort(-score, kind="stable")
    tp = np.cumsum(y[o])
    k = int(np.searchsorted(tp, r * y.sum()))
    return float(tp[k] / (k + 1))


def main():
    from sklearn.metrics import average_precision_score

    from bench.calib.common import save_json
    from bench.coach.azure_yardstick import azure_phones
    from bench.coach.fit_v2 import so762_rows
    from pronunciada import coach as Co

    root = os.path.expanduser(os.path.join(os.environ.get("PRONUNCIADA_CACHE", "~/.cache/pronunciada"), "azure",
                                           "speechocean762"))
    AZP = azure_phones(root)
    mdl = Co.model()
    m11 = copy.deepcopy(mdl)
    m11["by_lang"].pop("en", None)
    S = {}
    for name, dump, m in (("round11", "so762-test-r11", m11), ("round12b", "so762-test", mdl)):
        R = so762_rows(dump, "test")
        ep = Co.error_prob(R, m, "en")
        S[name] = {(u, j, k): (e, a, s) for u, j, k, e, a, s in zip(R["utt"], R["tj"], R["tk"], ep, R["acc"], R["spk"])
                   if np.isfinite(a)}
    keys = sorted(set(S["round11"]) & set(S["round12b"]) & set(AZP))
    y = np.array([S["round12b"][k][1] < 2 for k in keys], float)
    spk = np.array([S["round12b"][k][2] for k in keys])
    sc = {"round11": np.array([S["round11"][k][0] for k in keys]), "round12b": np.array([S["round12b"][k][0] for k in keys]),
          "azure": np.array([100.0 - (AZP[k][0] or 0.0) for k in keys])}
    pt = {n: {"ap": float(average_precision_score(y, s)), **{str(r): prec_at(s, y, r) for r in RECALLS}} for n, s in sc.items()}
    rng = np.random.default_rng(0)
    us = np.unique(spk)
    idx = {u: np.flatnonzero(spk == u) for u in us}
    B = {n: [] for n in sc}
    D = []
    for _ in range(1000):
        b = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        row = {}
        for n, s in sc.items():
            row[n] = [average_precision_score(y[b], s[b])] + [prec_at(s[b], y[b], r) for r in RECALLS]
            B[n].append(row[n])
        D.append(np.array(row["round12b"]) - np.array(row["round11"]))
    out = {"n_phones": len(keys), "error_rate": float(y.mean()), "point": pt}
    for n in sc:
        a = np.array(B[n])
        out[f"ci_{n}"] = {k: [float(np.quantile(a[:, j], 0.025)), float(np.quantile(a[:, j], 0.975))]
                          for j, k in enumerate(["ap"] + [str(r) for r in RECALLS])}
    D = np.array(D)
    out["delta_12b_minus_11"] = {k: [float(pt["round12b"][k] - pt["round11"][k]), float(np.quantile(D[:, j], 0.025)),
                                     float(np.quantile(D[:, j], 0.975))]
                                 for j, k in enumerate(["ap"] + [str(r) for r in RECALLS])}
    save_json("coaching_12b_heldout.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
