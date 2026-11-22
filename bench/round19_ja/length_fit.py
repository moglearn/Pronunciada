"""Round 19: refit the Japanese length model on more varied native audio and re-select the word
factor on the round-19 dev set.

    python -m bench.round19_ja.length_fit fit        # trees on fit natives -> <work>/len_r19_model.json
    python -m bench.round19_ja.length_fit select     # word-factor grid on dev -> <work>/len_r19_select.json

Model: boosted depth-3 trees per class (vowel / consonant), as round 15 (bench/length/fit_length.py)
but (a) with the round-19 labels (pronunciada.length.labels with PRONUNCIADA_JA_ROUND19 parts
"merge" and "labels" on), (b) fitted on FLEURS ja train AND Common Voice ja train natives (CC0;
speakers disjoint from the dev pool and from every held-out set, bench/round19_ja/cv_pool.py),
(c) with the round-17 attribute-head evidence s as a feature of the trees (round 17 combined it
afterwards with a logistic fitted on FLEURS validation; here it is learnt with the timing features,
from natives only), plus a "has head" indicator.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from bench.round19_ja.cv_pool import WORK

V_EXTRA = ("head_s", "has_head")


def table(items):
    from pronunciada import length as L

    out = []
    for k, it in enumerate(items):
        lab = L.labels(it["ref"])
        idx, cls, rows = L.features(it["ref"], lab, it["inp"])
        hs = it.get("hs")
        for i, c, r in zip(idx, cls, rows):
            s = float(hs[i]) if hs is not None and np.isfinite(hs[i]) else 0.0
            s = s if lab[i] == "long" else -s          # head evidence for "long", whatever the text expects
            r = dict(r, head_s=s, has_head=float(hs is not None and np.isfinite(hs[i])))
            out.append((k, i, c, lab[i], r))
    return out


def fit_model(items, n_trees=150, depth=3, lr=0.1, use_head=True) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier

    from bench.length.fit_length import _complete
    from pronunciada import length as L

    tab = table(items)
    model = {}
    for c, names in (("v", L.V_FEATURES), ("c", L.C_FEATURES)):
        names = list(names) + (list(V_EXTRA) if use_head else [])
        rows = [t for t in tab if t[2] == c]
        y = np.array([t[3] == "long" for t in rows], dtype=int)
        X = np.array([[t[4][f] for f in names] for t in rows], dtype=np.float64)
        w = np.where(y == 1, (len(y) - y.sum()) / y.sum(), 1.0)
        m = GradientBoostingClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=lr, subsample=0.8,
                                       random_state=0).fit(X, y, sample_weight=w)
        F, T, V = zip(*(_complete(e[0], depth) for e in m.estimators_))
        init = float(m.decision_function(X[:1])[0] - lr * sum(e[0].predict(X[:1])[0] for e in m.estimators_))
        M = {"features": names, "init": round(init, 6), "learning_rate": lr,
             "feat": np.array(F).tolist(), "thr": [[float(f"{v:.7g}") for v in r] for r in np.array(T)],
             "leaf": np.round(np.array(V), 6).tolist(), "n_long": int(y.sum()), "n_short": int(len(y) - y.sum())}
        chk = L.p_long({c: dict(M)}, c, [dict(zip(names, x)) for x in X[:300]])
        ref = 1 / (1 + np.exp(-m.decision_function(X[:300])))
        assert np.max(np.abs(chk - ref)) < 1e-3, np.max(np.abs(chk - ref))
        model[c] = M
        print(f"class {c}: {len(y)} sounds ({y.sum()} long)", file=sys.stderr)
    return model


def sound_probs(items, model, use_head=True):
    """Per item: list of length_prob (None where not judged), with the model reading head_s."""
    from pronunciada import length as L

    out = []
    for it in items:
        lab = L.labels(it["ref"])
        idx, cls, rows = L.features(it["ref"], lab, it["inp"])
        hs = it.get("hs")
        pr = [None] * len(it["ref"].phones)
        for c in ("v", "c"):
            sel = [k for k, x in enumerate(cls) if x == c]
            if not sel:
                continue
            rr = []
            for k in sel:
                i = idx[k]
                s = float(hs[i]) if hs is not None and np.isfinite(hs[i]) else 0.0
                s = s if lab[i] == "long" else -s
                rr.append(dict(rows[k], head_s=s, has_head=float(hs is not None and np.isfinite(hs[i]))))
            pl = L.p_long(model, c, rr)
            for k, p in zip(sel, pl):
                i = idx[k]
                pr[i] = float(p if lab[i] == "long" else 1 - p)
        out.append(pr)
    return out


def shipped_probs(items):
    """Length probs of the shipped round-15/17 channel (timing trees + head combine)."""
    from pronunciada import length as L

    par = L._shipped_params("ja") if hasattr(L, "_shipped_params") else L.params("ja")
    return [L.score(it["ref"], it["inp"], par, head_s=it.get("hs"))["prob"] for it in items]


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def word_factor(z_min: float, knee: float, width: float, floor: float) -> float:
    t = min(max((knee - z_min) / width, 0.0), 1.0)
    return 1.0 - (1.0 - floor) * t


def recs_with(items, probs, knee, width, floor):
    """Dump-like records (unit_eval) with the word prob recomputed from a new length factor."""
    out = []
    for it, pr in zip(items, probs):
        words = []
        for wi, (w, a, b) in enumerate(zip(it["ref"].words, it["wprob"], it["wlen"])):
            base = a / b if b > 0 else a
            v = [pr[i] for i in range(*w.phones) if pr[i] is not None]
            f = word_factor(min(logit(x) for x in v), knee, width, floor) if v else 1.0
            words.append({"text": w.text, "prob": base * f})
        out.append({"id": it["id"], "text": it["text"], "units": it.get("units"), "label": it.get("label"),
                    "group": it.get("group"), "source": it.get("source"), "words": words})
    return out


def cmd_fit(use_head=True) -> None:
    from bench.round19_ja.collect import load

    items = load("fit")
    print(len(items), "fit items", file=sys.stderr)
    model = fit_model(items, use_head=use_head)
    (WORK / ("len_r19_model.json" if use_head else "len_r19_model_nohead.json")).write_text(json.dumps(model))


def native_word_rates(recs) -> dict:
    """FLEURS / CV natives (the text as said): share of words with prob < 0.5 and < 0.6."""
    out = {}
    for src in ("fleurs", "cv"):
        p = np.array([w["prob"] for r in recs if r["source"] == src for w in r["words"]])
        out[src] = {"n": int(len(p)), "below_0.5": float(np.mean(p < 0.5)), "below_0.6": float(np.mean(p < 0.6))}
    return out


def cmd_select(model_name="len_r19_model.json") -> None:
    from bench.round19_ja.collect import load
    from bench.round19_ja.unit_eval import fa_at, roc_auc, units_of

    model = json.loads((WORK / model_name).read_text())
    dev = load("dev")
    nat = load("devnat")
    P_dev = sound_probs(dev, model)
    P_nat = sound_probs(nat, model)
    S_dev = shipped_probs(dev)
    S_nat = shipped_probs(nat)
    # native length logits (the word's lowest), dev natives: knee quantiles
    zn = np.array([min(logit(x) for x in (pr[i] for i in range(*w.phones)) if x is not None)
                   for it, pr in zip(nat, P_nat) for w in it["ref"].words
                   if any(pr[i] is not None for i in range(*w.phones))])
    res = {"grid": []}

    def evaluate(recs, nrecs):
        rows = units_of(recs)
        y = [x[2] for x in rows]
        s = [x[3] for x in rows]
        L = [x for x in rows if x[2] and x[4] == "length"]
        neg = [x for x in rows if not x[2]]
        auc_len = roc_auc([1] * len(L) + [0] * len(neg), [x[3] for x in L] + [x[3] for x in neg])
        cv = [x for x, r in zip(rows, rows) if True]
        return {"auc": roc_auc(y, s), "fa80": fa_at(y, s), "auc_length": auc_len, "native": native_word_rates(nrecs)}

    # shipped channel with its own knee / floor (reference point) and with the base (no length)
    from pronunciada import length as Lm

    par = Lm.params("ja")
    W = par["word"]
    base = evaluate(recs_with(dev, S_dev, W["knee"][1], W["width"], W["floor"]),
                    recs_with(nat, S_nat, W["knee"][1], W["width"], W["floor"]))
    res["shipped"] = base
    print("shipped", json.dumps(base), flush=True)
    off = evaluate(recs_with(dev, S_dev, -1e9, 1, 1.0), recs_with(nat, S_nat, -1e9, 1, 1.0))
    res["no_length"] = off
    print("no_length", json.dumps(off), flush=True)
    for q in (0.002, 0.005, 0.01, 0.02):
        knee = float(np.quantile(zn, q))
        for width in (1.0, 2.0, 4.0):
            for floor in (0.4, 0.2, 0.1, 0.05, 0.02):
                e = evaluate(recs_with(dev, P_dev, knee, width, floor), recs_with(nat, P_nat, knee, width, floor))
                e.update(q=q, knee=knee, width=width, floor=floor)
                res["grid"].append(e)
                print(f"q={q} w={width} floor={floor} AUC {e['auc']:.4f} FA80 {e['fa80']:.3f} "
                      f"lenAUC {e['auc_length']:.3f} fleurs<0.5 {e['native']['fleurs']['below_0.5']:.4f} "
                      f"<0.6 {e['native']['fleurs']['below_0.6']:.4f} cv<0.6 {e['native']['cv']['below_0.6']:.4f}",
                      flush=True)
    (WORK / f"len_r19_select_{model_name.replace('.json', '')}.json").write_text(json.dumps(res, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "select"))
    ap.add_argument("--no-head", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        cmd_fit(not a.no_head)
    else:
        cmd_select("len_r19_model_nohead.json" if a.no_head else "len_r19_model.json")


if __name__ == "__main__":
    main()
