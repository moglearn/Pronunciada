"""Round 19 (Korean, part 2): a two-sided word adjustment from identity + insertion / deletion /
duration evidence, fitted on the dev set with speaker-grouped cross-validation.

    python -m bench.round19ko.fit2 cv          # 4-fold speaker-grouped CV over the dev + native pool
    python -m bench.round19ko.fit2 write       # fit on all of it, write data/calib/ko_identity_word.json

Word log-odds after = word log-odds before + delta(features), delta linear in hinge features:
  min(s_min, 0), min(s_min, -4) (a second slope for strong evidence), mean margin, number of sounds
  with s < 0, min(ins + c, 0), min(dele, 0), |dur|, and 1.
s = identity margin minus the class x position native threshold (q_c quantile, gate.py).
Fitted by L2-penalised logistic regression with the old log-odds as a fixed offset (target:
"the word was read as written"): changed dev words = 0; every other dev word and every word of
the native pool (dev controls are dev rows already; Zeroth test natives are added) = 1.
Delta may be positive: a word whose every sound is clearly the expected one is raised.
Guards for the operating point (strictness 0.5): on the native pool (FLEURS validation +
CV dev controls + Zeroth test), words below 0.6 must not rise and acceptance (>= 0.5) must
not fall; FLEURS-validation planted-error AUC must not fall.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from bench.round19ko import devmetrics as DM
from bench.round19ko import gate as G
from bench.round19ko.cv_pool import work

FEATS = ("s_min0", "s_min4", "s_mean", "n_neg", "ins", "dele", "adur", "one")


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def feats(r, thr, ins_c=2.0):
    st = r["st"]
    s = np.array([x - thr.get(f"{c}|{p}", thr.get(f"{c}|*", 0.0)) if c is not None else np.nan
                  for x, c, p in zip(st["s"], st["cls"], st["pos"])])
    nw = len(r["prob"])
    X = np.zeros((nw, len(FEATS)))
    for j in range(nw):
        v = s[st["word"] == j]
        v = v[np.isfinite(v)]
        smin = float(v.min()) if len(v) else 0.0
        X[j] = [min(smin, 0.0), min(smin + 4.0, 0.0), float(np.clip(v, -10, 10).mean()) if len(v) else 0.0,
                float(np.clip(-v, 0.0, 1.0).sum()), min(r["wf"]["ins"][j] + ins_c, 0.0), min(r["wf"]["dele"][j], 0.0),
                abs(r["wf"]["dur"][j]), 1.0]
    return X


def fit_delta(X, off, y, l2=1.0, w=None):
    from scipy.optimize import minimize

    w = np.ones(len(y)) if w is None else w

    def f(b):
        z = off + X @ b
        ll = np.sum(w * (np.logaddexp(0, z) - y * z))
        g = X.T @ (w * (1 / (1 + np.exp(-z)) - y))
        return ll + 0.5 * l2 * (b[:-1] @ b[:-1]), g + l2 * np.append(b[:-1], 0.0)

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def word_llr_min(r):
    st = r["st"]
    out = np.zeros(len(r["prob"]))
    for j in range(len(out)):
        v = st["llr"][st["word"] == j]
        out[j] = float(v.min()) if len(v) else 0.0
    return out


def deltas(r, X, b, cap_up=1.0, raise_gate=None):
    """The word step: X b, capped at cap_up; with raise_gate = tau, a word with any sound whose
    aligner llr_sub < -tau is never raised (its step is min(step, 0))."""
    d = np.minimum(X @ b, cap_up)           # raising is bounded
    if isinstance(raise_gate, str) and raise_gate.startswith("id"):
        # identity gate: no raise when any sound's margin is below its class threshold by > g
        g = float(raise_gate[2:] or 0.0)
        d = np.where(X[:, 0] < -g, np.minimum(d, 0.0), d)
    elif raise_gate is not None:
        d = np.where(word_llr_min(r) < -raise_gate, np.minimum(d, 0.0), d)
    return d


def gate_of(cal: dict):
    """The shipped no-raise rule as a raise_gate argument ("id<g>": no raise when the lowest margin < -g)."""
    nr = cal.get("no_raise_below")
    return None if nr is None else f"id{-float(nr)}"


def apply(r, X, b, cap_up=1.0, raise_gate=None):
    return 1 / (1 + np.exp(-(lg(r["prob"]) + deltas(r, X, b, cap_up, raise_gate))))


def fold_of(spk: str, k: int) -> int:
    return int(hashlib.sha1(f"r19ko2|{spk}".encode()).hexdigest(), 16) % k


def data():
    D = pickle.loads((work() / "words_r2.pkl").read_bytes())
    return D["dev"], D["zeroth"], D["val_cor"]


def rows_for(dev):
    return {k: {"meta": r["meta"], "words": {"prob": r["prob"], "span": r["span"]}} for k, r in dev.items()}


def native_counts(pool, P):
    b0 = b1 = a0 = a1 = n = 0
    for k, r in pool.items():
        p0, p1 = r["prob"], P[k]
        b0 += int(np.sum(p0 < 0.6)); b1 += int(np.sum(p1 < 0.6))
        a0 += int(np.sum(p0 >= 0.5)); a1 += int(np.sum(p1 >= 0.5))
        n += len(p0)
    return {"n": n, "below06": [b0, b1], "accept05": [a0, a1]}


def stack(rows, thr, labels):
    Xs, offs, ys, keys = [], [], [], []
    for k, r in rows.items():
        X = feats(r, thr)
        y = np.ones(len(r["prob"]))
        for j in labels(r):
            y[j] = 0.0
        Xs.append(X); offs.append(lg(r["prob"])); ys.append(y); keys += [(k, j) for j in range(len(y))]
    return np.vstack(Xs), np.concatenate(offs), np.concatenate(ys)


def train(dev_tr, zt_tr, qc, l2, wpos):
    pool = {**{k: r for k, r in dev_tr.items() if not r["meta"].get("label")}, **zt_tr}
    thr = G.class_pos_thresholds(pool, qc)
    X, off, y = stack({**dev_tr, **zt_tr}, thr, lambda r: r["changed"])
    w = np.where(y == 0, wpos, 1.0)
    return thr, fit_delta(X, off, y, l2, w)


def cv(k=4):
    dev, zt, _ = data()
    out = {}
    for qc in (0.02, 0.05, 0.1):
        for l2 in (1.0, 10.0):
            for wpos in (1.0, 5.0, 20.0):
                for cap in (0.0, 0.5, 1.0, 2.0):
                    P = {}
                    Pz = {}
                    for f in range(k):
                        dtr = {a: r for a, r in dev.items() if fold_of(r["meta"]["speaker"], k) != f}
                        ztr = {a: r for a, r in zt.items() if fold_of(r["meta"]["speaker"], k) != f}
                        thr, b = train(dtr, ztr, qc, l2, wpos)
                        for a, r in dev.items():
                            if fold_of(r["meta"]["speaker"], k) == f:
                                P[a] = apply(r, feats(r, thr), b, cap)
                        for a, r in zt.items():
                            if fold_of(r["meta"]["speaker"], k) == f:
                                Pz[a] = apply(r, feats(r, thr), b, cap)
                    m = DM.metrics(rows_for(dev), P)
                    pool = {**{a: r for a, r in dev.items() if not r["meta"].get("label")}, **zt}
                    nc = native_counts(pool, {**P, **Pz})
                    key = (qc, l2, wpos, cap)
                    out[key] = (m, nc)
                    print(key, DM.line(m)[:150], nc, flush=True)
    return out


def oof(qc, l2, wpos, cap, k=4, drop=()):
    """Out-of-fold probabilities for dev, Zeroth test and fleurs-val-cor (fold = speaker / FLEURS
    sentence), optionally with features zeroed (ablation)."""
    dev, zt, vc = data()
    keep = np.array([f not in drop for f in FEATS], float)
    P, Pz, Pc = {}, {}, {}
    for f in range(k):
        dtr = {a: r for a, r in dev.items() if fold_of(r["meta"]["speaker"], k) != f}
        ztr = {a: r for a, r in zt.items() if fold_of(r["meta"]["speaker"], k) != f}
        pool = {**{a: r for a, r in dtr.items() if not r["meta"].get("label")}, **ztr}
        thr = G.class_pos_thresholds(pool, qc)
        X, off, y = stack({**dtr, **ztr}, thr, lambda r: r["changed"])
        b = fit_delta(X * keep, off, y, l2, np.where(y == 0, wpos, 1.0))
        for src, dst in ((dev, P), (zt, Pz), (vc, Pc)):
            for a, r in src.items():
                if fold_of(r["meta"]["speaker"], k) == f:
                    dst[a] = apply(r, feats(r, thr) * keep, b, cap)
    return dev, zt, vc, P, Pz, Pc


def per_source(dev, zt, P, Pz):
    out = {}
    for src in ("fleurs-val", "cv", "zeroth-test"):
        pool = {a: r for a, r in {**dev, **zt}.items() if r["src"] == src and not r["meta"].get("label")}
        out[src] = native_counts(pool, {**P, **Pz})
    return out


def planted(vc, Pc):
    from bench.third_party.lingolix import roc_auc

    def auc(P):
        pos, neg = [], []
        for a, r in vc.items():
            if r["meta"]["cor_kind"] == "clean":
                neg += list(P[a])
            else:
                pos += [p for p, w in zip(P[a], r["word_wrong"] or []) if w]
        return roc_auc([1] * len(pos) + [0] * len(neg), [1 - x for x in pos + neg])
    return auc({a: r["prob"] for a, r in vc.items()}), auc(Pc)


def report(qc, l2, wpos, cap):
    rep = {"setting": {"qc": qc, "l2": l2, "wpos": wpos, "cap_up": cap}}
    for name, drop in (("full", ()), ("identity_only", ("ins", "dele", "adur")), ("no_identity",
                       ("s_min0", "s_min4", "s_mean", "n_neg"))):
        dev, zt, vc, P, Pz, Pc = oof(qc, l2, wpos, cap, drop=drop)
        m0, m1 = DM.metrics(rows_for(dev), boot=1000), DM.metrics(rows_for(dev), P, boot=1000)
        rep[name] = {"dev_before": m0, "dev_after": m1, "paired": DM.paired(rows_for(dev), None, P),
                     "native_by_source": per_source(dev, zt, P, Pz), "planted_auc_val": planted(vc, Pc)}
        print(name, "\n  before", DM.line(m0), "\n  after ", DM.line(m1), "\n ", rep[name]["native_by_source"],
              "planted", rep[name]["planted_auc_val"], flush=True)
    Path("results/tables/round19_ko2_cv.json").write_text(json.dumps(rep, indent=1, default=float))


def grid():
    """Selection rule (fixed before looking): every source of the native pool (FLEURS validation,
    CV dev controls, Zeroth test) has no more words below 0.6 and no fewer accepted words than
    before (out of fold); the FLEURS-validation planted AUC does not fall; then the highest
    out-of-fold dev AUC, ties by lower FA@80."""
    res = []
    for qc in (0.02, 0.05, 0.1):
        for l2 in (10.0,):
            for wpos in (2.0, 5.0, 10.0):
                for cap in (1.0, 2.0, 3.0):
                    for drop in ((), ("ins", "dele", "adur")):
                        dev, zt, vc, P, Pz, Pc = oof(qc, l2, wpos, cap, drop=drop)
                        m = DM.metrics(rows_for(dev), P)
                        ps = per_source(dev, zt, P, Pz)
                        pl = planted(vc, Pc)
                        ok = all(v["below06"][1] <= v["below06"][0] and v["accept05"][1] >= v["accept05"][0]
                                 for v in ps.values()) and pl[1] >= pl[0] - 1e-4
                        res.append({"qc": qc, "l2": l2, "wpos": wpos, "cap": cap, "indel_feats": not drop, "ok": ok,
                                    "auc": m["all"]["auc"], "fa80": m["all"]["fa80"], "bucket": m["all"]["bucket"],
                                    "native": ps, "planted": pl})
                        print(res[-1]["ok"], qc, wpos, cap, not drop, round(m["all"]["auc"], 4), round(m["all"]["fa80"], 4),
                              {s: (v["below06"], v["accept05"]) for s, v in ps.items()}, flush=True)
    oks = [r for r in res if r["ok"]]
    best = max(oks, key=lambda r: (round(r["auc"], 4), -r["fa80"])) if oks else None
    Path("results/tables/round19_ko2_grid.json").write_text(json.dumps({"grid": res, "best": best}, indent=1, default=float))
    print("best", best)


def write(qc, l2, wpos, cap):
    dev, zt, _ = data()
    pool = {**{a: r for a, r in dev.items() if not r["meta"].get("label")}, **zt}
    thr = G.class_pos_thresholds(pool, qc)
    X, off, y = stack({**dev, **zt}, thr, lambda r: r["changed"])
    b = fit_delta(X, off, y, l2, np.where(y == 0, wpos, 1.0))
    par = {"form": "delta", "version": 2, "features": list(FEATS), "coef": [float(x) for x in b], "cap_up": cap,
           "flag_delta": -1.0, "no_raise_below": -2.0,                  # bench/round19ko/flagcheck.py (dev flag precision rule)
           "ins_c": 2.0, "thr": thr, "qc": qc, "l2": l2, "wpos": wpos,
           "fitted_on": "round 19 Korean dev (FLEURS ko validation + Common Voice dev pool, Whisper-filtered, "
                        "planted text edits) + Zeroth-Korean test natives (bench/round19ko/fit2.py)"}
    Path("pronunciada/data/calib/ko_identity_word.json").write_text(json.dumps(par, indent=1, ensure_ascii=False))
    print(dict(zip(FEATS, np.round(b, 3))))


if __name__ == "__main__":
    SET = (0.1, 10.0, 2.0, 3.0)        # the grid() choice (results/tables/round19_ko2_grid.json)
    {"cv": lambda: cv(), "grid": grid, "report": lambda: report(*SET), "write": lambda: write(*SET)}[sys.argv[1]]()
