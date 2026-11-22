"""Round 19: the length model fitted discriminatively, natives against planted long <-> short swaps
scored on their own alignment (bench/round19_ja/swapcollect.py), like the round-17 head combination
but on Common Voice + FLEURS TRAIN audio and with the round-19 duration inputs.

    python -m bench.round19_ja.length_fit3 fit [--set NAME]   # -> <work>/len3_<NAME>.json
    python -m bench.round19_ja.length_fit3 dev [NAMES]        # sound-level AUC on the dev pool swaps

Model: boosted depth-3 trees per class (vowel / consonant) predicting "the sound matches its text"
(length_prob) from the round-15 features, the expected length, the round-17 head evidence relative
to the expected value, and pronunciada.length_r19 extras. Natives are weighted so each class's
natives and swaps weigh the same.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from bench.round19_ja.cv_pool import WORK

EXTRA = {"base": ("exp_long", "head_rel", "has_head"),
         "dur": ("exp_long", "head_rel", "has_head", "v_stable", "c_dip", "rate_loc", "word_ratio"),
         "all": ("exp_long", "head_rel", "has_head", "v_stable", "c_dip", "rate_loc", "word_ratio", "cf_llr")}


def flat(items):
    out = []
    for it in items:
        for x in it["rows"]:
            r = dict(x["row"], exp_long=float(x["lab"] == "long"))
            out.append((x["cls"], r, int(x["swap"]), x["shipped"], it["source"], it["group"], it["kind"]))
    return out


def names_for(c, name):
    from pronunciada import length as L

    base = list(L.V_FEATURES if c == "v" else L.C_FEATURES)
    ex = [f for f in EXTRA[name] if not (c == "c" and f == "v_stable") and not (c == "v" and f == "c_dip")]
    return base + ex


def fit(name, n_trees=300, depth=3, lr=0.08):
    from sklearn.ensemble import GradientBoostingClassifier

    from bench.length.fit_length import _complete
    from bench.round19_ja.swapcollect import load

    rows = flat(load("fit"))
    model = {}
    for c in ("v", "c"):
        R = [x for x in rows if x[0] == c]
        names = names_for(c, name)
        X = np.array([[x[1][f] for f in names] for x in R])
        y = np.array([1 - x[2] for x in R])        # 1 = matches its text
        w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
        m = GradientBoostingClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=lr, subsample=0.8,
                                       random_state=0).fit(X, y, sample_weight=w)
        F, T, V = zip(*(_complete(e[0], depth) for e in m.estimators_))
        init = float(m.decision_function(X[:1])[0] - lr * sum(e[0].predict(X[:1])[0] for e in m.estimators_))
        model[c] = {"features": names, "init": round(init, 6), "learning_rate": lr, "feat": np.array(F).tolist(),
                    "thr": [[float(f"{v:.7g}") for v in r] for r in np.array(T)],
                    "leaf": np.round(np.array(V), 6).tolist(), "target": "match", "n_native": int(y.sum()),
                    "n_swap": int((y == 0).sum())}
        print(c, len(y), int((y == 0).sum()), "swaps", file=sys.stderr)
    (WORK / f"len3_{name}.json").write_text(json.dumps(model))


def dev(names):
    from bench.round19_ja.swapcollect import load
    from bench.round19_ja.unit_eval import roc_auc
    from pronunciada import length as L

    rows = flat(load("dev"))
    out = {}
    for name in ["shipped"] + list(names):
        M = None if name == "shipped" else json.loads((WORK / f"len3_{name}.json").read_text())
        p = np.empty(len(rows))
        for c in ("v", "c"):
            ii = [k for k, x in enumerate(rows) if x[0] == c]
            if M is None:
                p[ii] = [rows[k][3] for k in ii]
            else:
                p[ii] = L.p_long(M, c, [rows[k][1] for k in ii])     # trees' P(class 1) = P(match)
        for src in ("cv", "fleurs"):
            for c in ("all", "v", "c"):
                sel = [k for k, x in enumerate(rows) if x[4] == src and (c == "all" or x[0] == c)]
                y = [rows[k][2] for k in sel]
                auc = roc_auc(y, list(-p[sel]))
                nat = p[[k for k in sel if not rows[k][2]]]
                pos = p[[k for k in sel if rows[k][2]]]
                hit = float(np.mean(pos <= np.quantile(nat, 0.02)))
                out[f"{name}:{src}:{c}"] = {"auc": auc, "hit_at_2pct": hit, "n_swap": len(pos), "n_native": len(nat)}
                print(f"{name:8s} {src:6s} {c:3s} swaps {len(pos):5d} natives {len(nat):6d} AUC {auc:.4f} hit@2% {hit:.3f}",
                      flush=True)
    (WORK / "len3_dev.json").write_text(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "dev"))
    ap.add_argument("names", nargs="*")
    ap.add_argument("--set", default="all")
    a = ap.parse_args(argv)
    fit(a.set) if a.cmd == "fit" else dev(a.names)


if __name__ == "__main__":
    main()
