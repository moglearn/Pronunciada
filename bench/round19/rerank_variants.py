"""Round 19 (German): feature variants of the word recalibration, fit fold -> check fold.

    python -m bench.round19.rerank_variants dev-text1ea devcor-ea

Each variant is fitted on the fit fold (planted real-word edits + controlled sound errors, equal
weight) and reported on the check fold: the Lingolix-style metric (AUC, FA@80) and the controlled
sound errors (word AUC / hit @ 5 % native flags vs check control words).
"""
from __future__ import annotations

import math
import sys

import numpy as np

from bench.round19 import fit_rerank as R

BASE = R.word_feats


def v_base(w, o):
    return BASE(w, o)


def v_inter(w, o):
    f = BASE(w, o)
    return f + [f[0] * f[5]]


def v_cap6(w, o):
    f = BASE(w, o)
    f[5] = min(f[5], math.log(6))
    return f


def v_cap8_inter(w, o):
    f = BASE(w, o)
    f[5] = min(f[5], math.log(8))
    return f + [f[1] * f[5]]


def v_min2(w, o):
    f = BASE(w, o)
    W = w["W"]
    g = lambda k: float(np.clip(W[k] if W.get(k) is not None else 0.0, -R.CLIP, R.CLIP))  # noqa: E731
    return f + [g("wl_min2"), g("ws_min2")]


VARIANTS = {"base": v_base}


def run(name, cor):
    from bench.metrics import roc_auc
    from bench.round19.evaluate import _metrics, load_dump, units
    from bench.round19.collect import manifest
    from bench.round19.whatif import rebuild

    R.COR = cor
    check = [it for it in manifest("dev") if it["fold"] == "check"]
    D = load_dump(name)
    for vn, fn in VARIANTS.items():
        R.word_feats = fn
        nf = len(fn(next(iter(D.values()))["rwords"][0], False))
        R.FEATS = ("z_word", "wl_min", "ws_min", "wl_mean", "ws_mean", "log_n") + tuple(f"x{i}" for i in range(nf - 6))
        par = R.fit(name, 0.1)
        coef = np.array(par["coef"])

        def wfn(w, c, coef=coef, par=par, fn=fn):
            o = R.is_oov(w, R.oov_set(c["rec"]))
            z = float(np.array(fn(w, o)) @ coef + par["b0"]) + (par["oov_shift"] if o else 0.0)
            return w["prob"] / max(w["sounds"], 1e-6) / (1 + math.exp(-z))

        d = {k: (rebuild(v, wfn) if not v.get("error") else v) for k, v in D.items()}
        m = _metrics(units(check, d))
        Xc = R.cor_table(cor, "check")
        X, y, ids = R.table(name, "check")
        Xn = X[np.array([i[2] for i in ids]) & (y == 0)]
        zc, zn = Xc @ coef + par["b0"], Xn @ coef + par["b0"]
        yy = np.r_[np.ones(len(zc)), np.zeros(len(zn))]
        thr = np.quantile(zn, 0.05)
        print(f"{vn:11s} check AUC {m['auc']:.4f} FA@80 {100 * m['fa80']:.2f} | controlled AUC "
              f"{roc_auc(yy, -np.r_[zc, zn]):.4f} hit@5 {np.mean(zc < thr):.4f} | coef {np.round(coef, 2).tolist()}", flush=True)
    R.word_feats = BASE


if __name__ == "__main__":
    for cw in [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1"])]:
        R.COR_W = cw
        print("controlled-error weight", cw)
        run(sys.argv[1], sys.argv[2])
