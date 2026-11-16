"""Generic CV-fit-export of one calibrator level (regression + probability heads)."""
from __future__ import annotations

import time

import numpy as np

from bench.calib.common import (choose_calib, ece, export_trees, fit, raw, speaker_folds,
                                walker_parity)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def head_metrics(kind, y, pred, groups=None):
    from bench.metrics import brier, pearson, roc_auc, spearman

    m = np.isfinite(y)
    y, pred = y[m], pred[m]
    if kind == "reg":
        return {"pearson": pearson(pred, y), "spearman": spearman(pred, y), "n": int(m.sum())}
    yb = y.astype(int)
    return {"auc": roc_auc(yb, pred), "ece": ece(yb, pred), "brier": brier(yb, pred),
            "pos_rate": float(yb.mean()), "mean_p": float(pred.mean()), "n": int(m.sum())}


def head_params(params, h):
    """params: one hyper-parameter dict for every head, or {head: dict}."""
    return params[h] if h in params and isinstance(params[h], dict) else params


def cv_heads(X, labels: dict, kinds: dict, groups, params, cat_mask, weights=None, folds=None,
             predict_mask=None, label_masks=None):
    """5-fold grouped CV. labels: {head: y (NaN = no label)}; kinds: {head: "reg"|"prob"}.
    Rows with a NaN label are not used to fit that head but still receive out-of-fold
    predictions. Returns {head: oof raw prediction (reg value / log-odds)}."""
    folds = folds or speaker_folds(groups)
    out = {h: np.full(len(X), np.nan) for h in labels}
    for k, (tr, te) in enumerate(folds):
        for h, y in labels.items():
            ok = tr[np.isfinite(y[tr])]
            w = None if weights is None else weights[h][ok]
            m = fit(kinds[h], head_params(params, h), X[ok], y[ok], cat_mask, w)
            out[h][te] = raw(m, X[te])
    return out, folds


def calibrate_oof(oof, labels, kinds, groups, weights=None):
    """Pick Platt vs isotonic for every probability head on its OOF log-odds (grouped CV ECE).
    Returns ({head: calib dict}, {head: {platt: ece, isotonic: ece}}, {head: oof calibrated p
    from the nested CV of the chosen kind})."""
    from bench.calib.common import apply_calib, fit_calib
    from sklearn.model_selection import GroupKFold

    cal, cmp, pcv = {}, {}, {}
    for h, y in labels.items():
        if kinds[h] != "prob":
            continue
        m = np.isfinite(y) & np.isfinite(oof[h])
        c, res = choose_calib(oof[h][m], y[m], groups[m])
        cal[h], cmp[h] = c, res
        p = np.full(len(y), np.nan)
        z, yy = oof[h][m], y[m].astype(int)
        pp = np.zeros(m.sum())
        for tr, te in GroupKFold(5).split(z, yy, groups[m]):
            pp[te] = apply_calib(fit_calib(z[tr], yy[tr], c["kind"]), z[te])
        p[m] = pp
        pcv[h] = p
    return cal, cmp, pcv


def fit_final(X, labels, kinds, params, cat_mask, weights=None, clips=None, parity_rows=20000):
    """Fit every head on all labelled rows; export trees; numpy-walker parity."""
    heads, parity = {}, {}
    rng = np.random.default_rng(0)
    for h, y in labels.items():
        ok = np.flatnonzero(np.isfinite(y))
        w = None if weights is None else weights[h][ok]
        m = fit(kinds[h], head_params(params, h), X[ok], y[ok], cat_mask, w)
        sel = rng.choice(len(X), min(parity_rows, len(X)), replace=False)
        parity[h] = walker_parity(m, X[sel])
        heads[h] = {"kind": kinds[h], "trees": export_trees(m, X.shape[1]),
                    "clip": (clips or {}).get(h), "_model": m}
    return heads, parity
