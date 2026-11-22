"""Round 19, second structure: a capped two-sided step on the round-18 word log-odds (the Korean
round-19 form), fitted on the dev set with speaker-grouped 2-fold cross-fitting.

    python -m bench.round19_ja.step run --length r19|shipped [--cap 3] [--c 1.0]

z_word = logit(round-18 word prob with fixes 1-2, without the round-15 length factor)
       + min(b . f(word), cap)
f(word) = identity margins of the word's sounds (pronunciada.ja_identity: min(s, 0), min(s + 4, 0),
mean s, count of s < 0, has), the word's length evidence (lowest logit of the length model:
min(z, 0), min(z + 4, 0), has), the free-decode counts (pronunciada.word_match: extra, missing,
different sounds, their rate) and 1. b: L2 logistic regression with z_word as a fixed offset (word
matches its text = 1), planted dev words against all other dev words. Reported dev numbers are
out-of-fold (fold = hash of the speaker / clip group); the guard sets are scored by the fold model
that did not see their group (FLEURS clip / CV speaker), cv-real dev by the full model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle

import numpy as np

from bench.round19_ja.cv_pool import WORK

ID_KNOTS = (-2.0, 2.0, 6.0)
LEN_KNOTS = (-4.0, -1.0, 2.0)
NAMES = tuple(f"id_k{k:g}" for k in ID_KNOTS) + ("id_neg", "id_has") + tuple(f"len_k{k:g}" for k in LEN_KNOTS) \
    + ("len_has", "mm_ins", "mm_del", "mm_sub", "mm_rate", "one")
# signs: lower evidence lowers the word; has-flags and the constant are kept inside the cap's range
# (an unbounded constant saturates every word at the cap, where the fit has no gradient)
BOUNDS = [(0, None)] * len(ID_KNOTS) + [(None, 0), (-3, 3)] + [(0, None)] * len(LEN_KNOTS) + [(-3, 3)] \
    + [(None, 0)] * 4 + [(-3, 3)]


def feats(w) -> list[float]:
    """Hinge basis min(x - knot, 0) of the word's lowest identity margin and lowest length logit
    (monotone piecewise-linear), the count of sounds whose margin is negative, the free-decode counts."""
    s = w.get("sids") or []
    L = w.get("lens") or []
    m = w["mm"]
    n = max(m["n"], 1)
    ms = float(np.clip(min(s), -12, 12)) if s else 12.0
    ml = float(np.clip(min(L), -12, 12)) if L else 12.0
    return [min(ms - k, 0.0) for k in ID_KNOTS] + [float(sum(x < 0 for x in s)), float(bool(s))] \
        + [min(ml - k, 0.0) for k in LEN_KNOTS] + [float(bool(L)),
           m["ins"], m["del"], m["sub"], (m["ins"] + m["del"] + m["sub"]) / n, 1.0]


def _z(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fold_of(g) -> int:
    return int(hashlib.sha1(str(g).encode()).hexdigest()[:8], 16) % 2


def word_labels(r):
    lab = r.get("label")
    y = [0] * len(r["words"])
    if lab:
        u = r["units"][lab["word_index"]]
        a, b = u["start"], u["start"] + len(lab["replacement"])
        for k, w in enumerate(r["words"]):
            if w["span"][0] < b and w["span"][1] > a:
                y[k] = 1
    return y


def fit_b(Z, F, y, C=1.0, cap=3.0):
    from scipy.optimize import minimize

    t = 1.0 - y                                  # 1 = matches its text
    wpos = np.ones(len(t))                       # natural prior: the offset is a calibrated log-odds

    def loss(b):
        u = F @ b
        # the cap inside the fit, smoothed: step = cap - softplus(cap - u)
        sp = np.logaddexp(0.0, cap - u)
        step = cap - sp
        dstep = 1.0 / (1.0 + np.exp(-(cap - u)))          # d step / d u = sigmoid(cap - u)
        z = Z + step
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        l = -(wpos * (t * np.log(p + 1e-12) + (1 - t) * np.log(1 - p + 1e-12))).sum() / len(t)
        g = (F * (wpos * (p - t) * dstep)[:, None]).sum(0) / len(t)
        return l + 0.5 / C * (b[:-1] @ b[:-1]) / len(t), g + np.r_[b[:-1], 0] / C / len(t)

    # monotone: evidence of a mismatch can only lower the word (signs fixed), the constant is free
    r = minimize(loss, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B", bounds=BOUNDS)
    return r.x


def apply(recs, B, cap, fold_models=None):
    out = []
    for r in recs:
        b = B if fold_models is None else fold_models[1 - fold_of(r.get("group") or r["id"])]
        ws = []
        for w in r["words"]:
            step = min(float(np.dot(b, feats(w))), cap)
            ws.append({"text": w["text"], "span": w["span"], "prob": 1 / (1 + math.exp(-(_z(w["base"]) + step))),
                       "step": step})
        out.append({**{k: v for k, v in r.items() if k != "words"}, "words": ws})
    return out


def run(length: str, cap: float, C: float) -> dict:
    from bench.round19_ja.select2 import guard_ok, metrics, round18_baseline

    D = pickle.loads((WORK / f"sel2_{length}.pkl").read_bytes())
    folds = {0: ([], [], []), 1: ([], [], [])}
    for r in D["dev"]:
        f = fold_of(r.get("group") or r["id"])
        for w, y in zip(r["words"], word_labels(r)):
            folds[f][0].append(_z(w["base"]))
            folds[f][1].append(feats(w))
            folds[f][2].append(y)
    models = {}
    for f in (0, 1):
        Z, F, y = (np.array(x, float) for x in folds[f])
        models[f] = fit_b(Z, F, y, C, cap)
    Zall = np.r_[folds[0][0], folds[1][0]]
    Fall = np.r_[np.array(folds[0][1]), np.array(folds[1][1])]
    yall = np.r_[folds[0][2], folds[1][2]]
    full = fit_b(Zall, Fall, yall, C, cap)
    # dev / guard sets with cross-fitted models; cv-real dev with the full model
    E = {"dev": apply(D["dev"], None, cap, models), "devnat": apply(D["devnat"], None, cap, models),
         "devswap": apply(D["devswap"], None, cap, models), "cvdev": apply(D["cvdev"], full, cap)}
    for k in E:
        for r in E[k]:
            for w in r["words"]:
                w["base"] = w["prob"]
                w["zlen"] = w["zid"] = None
                w["mm"] = None
    m = metrics(E, {}, None)
    b18 = round18_baseline()
    res = {"length": length, "cap": cap, "C": C, "names": NAMES, "b_full": full.tolist(),
           "b_folds": [models[0].tolist(), models[1].tolist()], "m": m, "round18": b18, "ok": guard_ok(m, b18)}
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}))
    print("round18", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b18.items()}))
    print("guards ok:", res["ok"], "b_full:", dict(zip(NAMES, np.round(full, 3))))
    (WORK / f"step_{length}_cap{cap}_C{C}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--length", default="r19")
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--c", type=float, default=1.0)
    a = ap.parse_args(argv)
    run(a.length, a.cap, a.c)


if __name__ == "__main__":
    main()
