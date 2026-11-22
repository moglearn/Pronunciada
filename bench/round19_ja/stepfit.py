"""Round 19, part 3: the word step fitted as a logistic regression with the base log-odds as offset, on
planted edits against a large native pool (bench/round19_ja/stepfeat.py), then checked on dev.

    python -m bench.round19_ja.stepfit --pool cv|cv+fleurs [--cap 3] [--C 1]

step = min(cap, b . f(word)); f built so that every non-constant input reads "evidence of an error"
with a sign-constrained weight (an input can only lower the word), the constant is free within
[-3, 3]. Identity margins enter as native-relative z-scores per class (mean / sd of the class on the
fit pool's native words). Fit pool natives: the control clips and the untouched words of edited clips.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict

import numpy as np

from bench.round19_ja.cv_pool import WORK
from bench.round19_ja.step import word_labels

NAMES = ("id_min", "id_min_h2", "id_neg2", "id_mean",
         "len_long", "len_short", "len_cons", "cf_exp", "ratio_hi", "ratio_lo",
         "ins", "extra", "del", "present", "one")
# sign of each weight: +1 = weight >= 0 on an input that is <= 0 for errors (hinge min(x, 0) form) or a
# z-score; -1 = weight <= 0 on a count / magnitude; 0 = free
SIGN = (1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 0)


def _lg(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def class_stats(recs):
    v = defaultdict(list)
    for r in recs:
        y = word_labels(r)
        for w, l in zip(r["words"], y):
            if l:
                continue
            for s in w["f"]["sounds"]:
                if s["s"] is not None:
                    v[s["cls"]].append(s["s"])
    allv = np.concatenate([np.array(x) for x in v.values()])
    g = (float(allv.mean()), float(allv.std() + 1e-6))
    return {c: ((float(np.mean(x)), float(np.std(x) + 1e-6)) if len(x) >= 200 else g) for c, x in v.items()}, g


def feats(w, stats) -> list[float]:
    st, g = stats
    f = w["f"]
    zs = [(s["s"] - st.get(s["cls"], g)[0]) / st.get(s["cls"], g)[1] for s in f["sounds"] if s["s"] is not None]
    zmin = float(np.clip(min(zs), -8, 4)) if zs else 4.0
    L = {"long": [], "short": [], "cons": []}
    cfe = []
    for s in f["sounds"]:
        if "lp" not in s:
            continue
        k = "cons" if s.get("lcls") == "c" else s["lexp"]
        L[k].append(_lg(s["lp"]))
        if s.get("cf") is not None:
            cfe.append(s["cf"] if s["lexp"] == "long" else -s["cf"])     # evidence for the expected form
    m = lambda xs: min(min(xs), 0.0) if xs else 0.0  # noqa: E731
    r = f["word_ratio"]
    pres = f["present"]
    return [zmin, min(zmin + 2, 0.0), float(sum(z < -2 for z in zs)), float(np.clip(np.mean(zs), -6, 4)) if zs else 0.0,
            m(L["long"]), m(L["short"]), m(L["cons"]), float(np.clip(m(cfe), -20, 0)),
            max(r - 0.3, 0.0), max(-r - 0.3, 0.0),
            float(np.clip(min([x for x in (f["ins_gap"], f["ins_in"]) if x is not None] or [0.0]), -30, 0)),
            float(f["extra"]), min(f["del_min"], 0.0), min(_lg(pres), 0.0) if pres is not None else 0.0, 1.0]


def fit_b(Z, F, y, C, cap, pos_weight=1.0):
    from scipy.optimize import minimize

    t = 1.0 - y
    wt = np.where(y == 1, pos_weight, 1.0)
    wt = wt / wt.mean()
    sd = F.std(0) + 1e-6
    sd[-1] = 1.0
    Fs = F / sd

    def loss(b):
        u = Fs @ b
        step = cap - np.logaddexp(0.0, cap - u)
        dstep = 1.0 / (1.0 + np.exp(-(cap - u)))
        p = 1 / (1 + np.exp(-np.clip(Z + step, -30, 30)))
        l = -(wt * (t * np.log(p + 1e-12) + (1 - t) * np.log(1 - p + 1e-12))).mean()
        g = (Fs * (wt * (p - t) * dstep)[:, None]).mean(0)
        reg = np.r_[b[:-1], 0.0] / (C * len(t))
        return l + 0.5 * (b[:-1] @ b[:-1]) / (C * len(t)), g + reg

    bounds = [((0, None) if s > 0 else (None, 0) if s < 0 else (-3, 3)) for s in SIGN]
    r = minimize(loss, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds)
    return r.x / sd


def load(which):
    return pickle.loads((WORK / "stepfeat" / f"{which}.pkl").read_bytes())


def apply(recs, b, stats, cap):
    out = []
    for r in recs:
        ws = []
        for w in r["words"]:
            st = min(float(np.dot(b, feats(w, stats))), cap)
            ws.append({"text": w["text"], "span": w["span"], "base": 1 / (1 + math.exp(-(_lg(w["base"]) + st))),
                       "zlen": None, "zid": None, "mm": None})
        out.append({**{k: v for k, v in r.items() if k != "words"}, "words": ws})
    return out


def cand2(recs):
    return [{**{k: v for k, v in r.items() if k != "words"},
             "words": [{"text": w["text"], "span": w["span"], "base": w["prob"], "zlen": None, "zid": None, "mm": None}
                       for w in r["words"]]} for r in recs]


def main(argv=None):
    from bench.round19_ja.select2 import guard_ok, metrics, round18_baseline

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="cv")
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--posw", type=float, default=1.0, help="weight of planted words in the fit")
    a = ap.parse_args(argv)
    fit = [r for r in load("fit2") if a.pool == "cv+fleurs" or r.get("source") == "cv"]
    stats = class_stats(fit)
    Z, F, y = [], [], []
    for r in fit:
        for w, l in zip(r["words"], word_labels(r)):
            Z.append(_lg(w["base"]))
            F.append(feats(w, stats))
            y.append(l)
    Z, F, y = np.array(Z), np.array(F), np.array(y)
    print(f"fit words {len(y)} planted {int(y.sum())}", flush=True)
    b = fit_b(Z, F, y, a.C, a.cap, a.posw)
    print("b:", {k: round(float(v), 3) for k, v in zip(NAMES, b)}, flush=True)
    D = {k: load(k) for k in ("dev", "devnat", "devswap", "cvdev")}
    b18 = round18_baseline()
    m2 = metrics({k: cand2(v) for k, v in D.items()}, {}, None)
    m3 = metrics({k: apply(v, b, stats, a.cap) for k, v in D.items()}, {}, None)
    keys = ("dev_auc_all", "dev_fa80_all", "dev_auc_cv", "dev_length", "dev_sub_subtle", "dev_sub_moderate",
            "dev_sub_gross", "dev_indel", "nat_fleurs_lt05", "nat_fleurs_lt06", "nat_cv_lt05", "nat_cv_lt06",
            "swap_far:auc", "swap_near:auc", "cvdev_word_mean", "cvdev_word_min")
    for name, m in (("round18", b18), ("cand2", m2), ("fitted", m3)):
        print(f"{name:8s}", " ".join(f"{k.replace('dev_', '')}={m[k]:.4f}" for k in keys),
              "" if name == "round18" else f"guards_ok={guard_ok(m, b18)}", flush=True)
    res = {"pool": a.pool, "cap": a.cap, "C": a.C, "names": NAMES, "b": b.tolist(),
           "class_stats": {k: list(v) for k, v in stats[0].items()}, "global": list(stats[1]),
           "round18": b18, "cand2": m2, "fitted": m3, "guards_ok": guard_ok(m3, b18)}
    res["posw"] = a.posw
    (WORK / f"stepfit_{a.pool}_cap{a.cap}_C{a.C}_w{a.posw}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
