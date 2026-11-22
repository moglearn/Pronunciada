"""Round 20 (insdel): why the greedy-decode feature adds so little (dev aggregates only).

    python -m bench.round20_insdel.diag --lang ja [--cfg weighted_top2] [--feat rate]

For the dev set: the shipped unit signal's 80 %-recall threshold; per bucket the planted units
caught / missed there; how often the feature fires (> 0) on missed and caught planted units and
on correct units (all, and those already above the threshold); the in-sample best lower-only term
WITHOUT the native guard (an upper bound) and how many native control words it pushes below 0.6;
a two-sided variant (also raising words whose feature is 0). Cost tables: dev controls (in sample;
diagnosis only). Writes results/tables/round20_insdel_<lang>_diag.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from bench.round20_insdel import core as K
from bench.round20_insdel.ablate import A_GRID, K_GRID, ROOT, Lang, control_recs, core_metrics, feats_for


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--cfg", default="weighted_top2")
    ap.add_argument("--feat", default="rate")
    a = ap.parse_args(argv)
    L = Lang(a.lang)
    U = L.units()
    y = np.array([u[2] for u in U])
    b = np.array([u[3] for u in U])
    base = {it: np.asarray(L.dev[it]["words"]["prob"], np.float64) for it in L.items}
    cfg = K.CFGS[a.cfg]
    T = K.native_table(control_recs(L, L.items), cfg) if cfg["weighted"] else None
    F = feats_for(L, L.dev, a.cfg, T)
    s0 = L.unit_signal(base)
    fu = np.array([max((F[it][a.feat][j] for j in ws), default=0.0) for it, _, _, _, ws in U])
    pos = np.sort(s0[y == 1])[::-1]
    thr = pos[int(np.ceil(0.8 * len(pos))) - 1]
    out = {"lang": a.lang, "cfg": a.cfg, "feat": a.feat, "thr80": float(thr), "buckets": {}}
    for bk in sorted(set(b[y == 1])):
        m = (y == 1) & (b == bk)
        miss = m & (s0 < thr)
        caught = m & (s0 >= thr)
        out["buckets"][bk] = {"n": int(m.sum()), "missed": int(miss.sum()),
                              "fires_on_missed": float(np.mean(fu[miss] > 0)) if miss.any() else None,
                              "fires_on_caught": float(np.mean(fu[caught] > 0)) if caught.any() else None}
    neg = y == 0
    out["fires_on_correct"] = float(np.mean(fu[neg] > 0))
    out["fires_on_correct_above_thr"] = float(np.mean(fu[neg & (s0 >= thr)] > 0))
    out["fires_on_correct_below_thr"] = float(np.mean(fu[neg & (s0 < thr)] > 0))
    out["n_correct_units"] = int(neg.sum())
    # native control words: how many sit near 0.6 and have the feature
    ctl = [i for i in L.items if L.is_control(i)]
    pc = np.concatenate([base[i] for i in ctl])
    fc = np.concatenate([F[i][a.feat] for i in ctl])
    out["control_words"] = {"n": int(len(pc)), "lt06": int(np.sum(pc < 0.6)),
                            "fires": float(np.mean(fc > 0)),
                            "fires_and_06_09": int(np.sum((fc > 0) & (pc >= 0.6) & (pc < 0.9)))}
    # unconstrained upper bound and the two-sided variant
    best, best2 = (core_metrics(y, s0, b)["auc"], None, 0), (core_metrics(y, s0, b)["auc"], None, 0)
    n0 = int(np.sum(pc < 0.6))
    for aa in A_GRID:
        for kk in K_GRID:
            P = {i: K.apply_term(base[i], F[i][a.feat], aa, kk) for i in L.items}
            auc = core_metrics(y, L.unit_signal(P), b)["auc"]
            nn = int(np.sum(np.concatenate([P[i] for i in ctl]) < 0.6)) - n0
            if auc > best[0]:
                best = (auc, (aa, kk), nn)
            for up in (0.25, 0.5, 1.0):
                P2 = {i: K.apply_term(base[i], F[i][a.feat], aa, kk, two_sided=up) for i in L.items}
                auc2 = core_metrics(y, L.unit_signal(P2), b)["auc"]
                nn2 = int(np.sum(np.concatenate([P2[i] for i in ctl]) < 0.6)) - n0
                if auc2 > best2[0] and nn2 <= 0:
                    best2 = (auc2, (aa, kk, up), nn2)
    out["shipped_auc"] = core_metrics(y, s0, b)
    out["unguarded_lower_only_best"] = {"auc": best[0], "a_k": best[1], "native_ctl_lt06_rise": best[2]}
    out["two_sided_guarded_best_in_sample"] = {"auc": best2[0], "a_k_up": best2[1], "native_ctl_lt06_rise": best2[2]}
    # agreement: Spearman between feature and shipped signal on planted indel units
    from scipy.stats import spearmanr

    m = (y == 1) & (b == "indel")
    out["indel_spearman_feature_vs_shipped"] = float(spearmanr(fu[m], s0[m]).correlation)
    print(json.dumps(out, indent=1, default=float))
    p = ROOT / "results" / "tables" / f"round20_insdel_{a.lang}_diag.json"
    p.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
