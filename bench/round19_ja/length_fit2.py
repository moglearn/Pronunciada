"""Round 19, second structure: the length model with filterbank durations, local mora rate, word
duration ratio and the counterfactual long / short likelihood ratio (pronunciada.length_r19).

    python -m bench.round19_ja.length_fit2 fit  [--set base|head|extra|all]   # fit natives (v2 collection)
    python -m bench.round19_ja.length_fit2 diag [names...]                      # CV / FLEURS dev view

Fit: boosted depth-3 trees per class on natives of Common Voice ja train + FLEURS ja train with the
text's own long / short labels (round-19 labels). Dev view: planted length words of the round-19 dev
set (lowest length probability over the planted unit's judged sounds) against every judged sound
of the control clips (sound level) and against control words (word level).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from bench.round19_ja.cv_pool import WORK

SETS = {"base": (), "head": ("head_s", "has_head"), "extra": ("v_stable", "c_dip", "rate_loc", "word_ratio", "cf_llr"),
        "all": ("head_s", "has_head", "v_stable", "c_dip", "rate_loc", "word_ratio", "cf_llr"),
        "nocf": ("head_s", "has_head", "v_stable", "c_dip", "rate_loc", "word_ratio")}


def rows_of(it):
    """(idx, cls, rows, labels) with every candidate feature."""
    from pronunciada import length as L
    from pronunciada import length_r19 as R

    lab = L.labels(it["ref"])
    idx, cls, rows = L.features(it["ref"], lab, it["inp"])
    hs = it.get("hs")
    for i, r in zip(idx, rows):
        s = float(hs[i]) if hs is not None and np.isfinite(hs[i]) else 0.0
        r["head_s"] = s if lab[i] == "long" else -s
        r["has_head"] = float(hs is not None and np.isfinite(hs[i]))
    R.extra(it["ref"], it["inp"], idx, rows, lpa=it.get("lpa"), slot_se=it.get("slot_se"), fb=it.get("fb"))
    return idx, cls, rows, lab


def cache_rows(which):
    from bench.round19_ja.collect import load

    p = WORK / f"rows2_{which}.pkl"
    import pickle

    if p.exists():
        return pickle.loads(p.read_bytes())
    out = []
    for it in load(which, "v2"):
        idx, cls, rows, lab = rows_of(it)
        out.append({"id": it["id"], "source": it.get("source"), "label": it.get("label"), "units": it.get("units"),
                    "spans": it["spans"], "ref": it["ref"], "idx": idx, "cls": cls, "rows": rows, "lab": lab,
                    "hs": it.get("hs"), "inp": it["inp"]})
    p.write_bytes(pickle.dumps(out))
    return out


def fit(name: str, n_trees=200, depth=3, lr=0.1) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier

    from bench.length.fit_length import _complete
    from pronunciada import length as L

    items = cache_rows("fit")
    model = {}
    for c, base in (("v", L.V_FEATURES), ("c", L.C_FEATURES)):
        names = list(base) + [f for f in SETS[name] if not (c == "c" and f == "v_stable") and not (c == "v" and f == "c_dip")]
        X, y = [], []
        for it in items:
            for i, cc, r in zip(it["idx"], it["cls"], it["rows"]):
                if cc == c:
                    X.append([r[f] for f in names])
                    y.append(it["lab"][i] == "long")
        X, y = np.array(X), np.array(y, int)
        w = np.where(y == 1, (len(y) - y.sum()) / y.sum(), 1.0)
        m = GradientBoostingClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=lr, subsample=0.8,
                                       random_state=0).fit(X, y, sample_weight=w)
        F, T, V = zip(*(_complete(e[0], depth) for e in m.estimators_))
        init = float(m.decision_function(X[:1])[0] - lr * sum(e[0].predict(X[:1])[0] for e in m.estimators_))
        model[c] = {"features": names, "init": round(init, 6), "learning_rate": lr, "feat": np.array(F).tolist(),
                    "thr": [[float(f"{v:.7g}") for v in r] for r in np.array(T)],
                    "leaf": np.round(np.array(V), 6).tolist(), "n_long": int(y.sum()), "n_short": int(len(y) - y.sum())}
        print(c, len(y), y.sum(), file=sys.stderr)
    (WORK / f"len2_{name}.json").write_text(json.dumps(model))
    return model


def probs(it, model):
    from pronunciada import length as L

    pr = [None] * len(it["ref"].phones)
    for c in ("v", "c"):
        sel = [k for k, x in enumerate(it["cls"]) if x == c]
        if not sel:
            continue
        pl = L.p_long(model, c, [it["rows"][k] for k in sel])
        for k, p in zip(sel, pl):
            i = it["idx"][k]
            pr[i] = float(p if it["lab"][i] == "long" else 1 - p)
    return pr


def shipped(it):
    from pronunciada import length as L

    return L.score(it["ref"], it["inp"], L.params("ja"), head_s=it.get("hs"))["prob"]


def _unit_words(it):
    u = it["units"][it["label"]["word_index"]]
    return [k for k, (a, b) in enumerate(it["spans"]) if a < u["end"] and b > u["start"]]


def diag(names) -> dict:
    from bench.round19_ja.unit_eval import roc_auc

    items = cache_rows("dev")
    out = {}
    models = {"shipped": None, **{n: json.loads((WORK / f"len2_{n}.json").read_text()) for n in names}}
    for name, M in models.items():
        P = [shipped(it) if M is None else probs(it, M) for it in items]
        for src in ("cv", "fleurs"):
            pos, nat_s, nat_w, nj = [], [], [], 0
            for it, pr in zip(items, P):
                if it["source"] != src:
                    continue
                ref = it["ref"]
                if it["label"] and it["label"]["bucket"] == "length":
                    v = [pr[i] for k in _unit_words(it) for i in range(*ref.words[k].phones) if pr[i] is not None]
                    if v:
                        pos.append(min(v))
                    else:
                        nj += 1
                if not it["label"]:
                    nat_s += [p for p in pr if p is not None]
                    for w in ref.words:
                        v = [pr[i] for i in range(*w.phones) if pr[i] is not None]
                        if v:
                            nat_w.append(min(v))
            pos, nat_s, nat_w = map(np.array, (pos, nat_s, nat_w))
            r = {"n_pos": len(pos), "not_judged": nj,
                 "auc_sound": roc_auc([1] * len(pos) + [0] * len(nat_s), list(-pos) + list(-nat_s)),
                 "auc_word": roc_auc([1] * len(pos) + [0] * len(nat_w), list(-pos) + list(-nat_w)),
                 "hit_at_2pct_native_words": float(np.mean(pos <= np.quantile(nat_w, 0.02)))}
            out[f"{name}:{src}"] = r
            print(f"{name:10s} {src:6s} pos {len(pos):3d} (unjudged {nj}) sound AUC {r['auc_sound']:.3f} "
                  f"word AUC {r['auc_word']:.3f} hit@2% {r['hit_at_2pct_native_words']:.2f}", flush=True)
    (WORK / "len2_diag.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "diag"))
    ap.add_argument("names", nargs="*")
    ap.add_argument("--set", default="all")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit(a.set)
    else:
        diag(a.names)


if __name__ == "__main__":
    main()
