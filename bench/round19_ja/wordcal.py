"""Round 19: a Japanese word recalibration fitted on planted edits in Common Voice / FLEURS train
audio, selected on the round-19 dev set.

    python -m bench.round19_ja.wordcal fit [--kind trees|logistic]   # -> <work>/wordcal_<kind>.json
    python -m bench.round19_ja.wordcal dev [--kind ...]              # dev table vs shipped

Inputs per word (all already computed by the Scorer; nothing new is run on the audio except the
free-decode match of pronunciada.word_match): the shipped word pieces (sounds logit at s = 0.5, the
word's lowest length logit, contrast factor, present probability), word-table statistics of the
sound evidence, and the free-decode mismatch counts. Label: the word overlaps the replaced core of a
planted edit (fit set: bench/round19_ja/plant.py on the fit pool). The model's logit is then mapped,
by quantiles, onto the shipped word logit's distribution on the fit set's untouched words, so native
word rates at any threshold stay where they were and only the ranking changes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from bench.round19_ja.cv_pool import WORK

FEATURES = ("z_sounds", "z_len", "has_len", "log_con", "z_present", "mm_ins", "mm_del", "mm_sub", "mm_rate",
            "ws_min", "wl_min", "ws_mean", "word_min_llr_sub", "word_mean_llr_sub", "word_min_llr_del",
            "n_phones", "frames_per_phone", "utt_ws_mean", "word_pos_first", "word_pos_last")


def _logit(p, lo=1e-6):
    p = min(max(float(p), lo), 1 - lo)
    return math.log(p / (1 - p))


def length_logits(it):
    """Lowest length logit per word with the shipped length model (current label switches)."""
    from pronunciada import length as L

    par = L.params("ja")
    pr = L.score(it["ref"], it["inp"], par, head_s=it.get("hs"))["prob"]
    out = []
    for w in it["ref"].words:
        v = [pr[i] for i in range(*w.phones) if pr[i] is not None]
        out.append(min(_logit(x) for x in v) if v else None)
    return out


def word_rows(it) -> list[dict]:
    W = it["W"]
    lz = length_logits(it)
    rows = []
    for k in range(len(it["ref"].words)):
        m = it["mm"][k]
        n = max(m["n"], 1)
        pres = it["present"][k]
        rows.append({
            "z_sounds": _logit(it["sounds"][k]), "z_len": 8.0 if lz[k] is None else min(lz[k], 8.0),
            "has_len": float(lz[k] is not None), "log_con": math.log(max(it["con"][k], 1e-6)),
            "z_present": 8.0 if not np.isfinite(pres) else _logit(pres), "mm_ins": m["ins"], "mm_del": m["del"],
            "mm_sub": m["sub"], "mm_rate": (m["ins"] + m["del"] + m["sub"]) / n,
            "ws_min": W["ws_min"][k], "wl_min": W["wl_min"][k], "ws_mean": W["ws_mean"][k],
            "word_min_llr_sub": max(W["word_min_llr_sub"][k], -50), "word_mean_llr_sub": max(W["word_mean_llr_sub"][k], -50),
            "word_min_llr_del": float(np.clip(W["word_min_llr_del"][k], -50, 50)), "n_phones": W["n_phones"][k],
            "frames_per_phone": W["word_frames"][k] / max(W["n_phones"][k], 1), "utt_ws_mean": W["utt_ws_mean"][k],
            "word_pos_first": W["word_pos_first"][k], "word_pos_last": W["word_pos_last"][k]})
    return rows


def word_labels(it) -> list[int]:
    lab = it.get("label")
    y = [0] * len(it["ref"].words)
    if lab:
        u = it["units"][lab["word_index"]]
        a, b = u["start"], u["start"] + len(lab["replacement"])
        for k, (s, e) in enumerate(it["spans"]):
            if s < b and e > a:
                y[k] = 1
    return y


def matrix(items, feats=FEATURES):
    X, y, base, grp, gid = [], [], [], [], []
    for it in items:
        rows = word_rows(it)
        X += [[r[f] for f in feats] for r in rows]
        y += word_labels(it)
        base += [_logit(p) for p in it["wprob"]]
        grp += [it.get("source") or ""] * len(rows)
        gid += [it.get("group") or it["id"]] * len(rows)
    matrix.groups = np.array(gid)
    return np.array(X, dtype=np.float64), np.array(y), np.array(base), np.array(grp)


MAP_SOURCE = "fleurs"


def fit(kind: str = "trees", n_trees: int = 200, depth: int = 3, lr: float = 0.05, drop=(), tag=None) -> dict:
    from bench.round19_ja.collect import load

    feats = [f for f in FEATURES if not any(f.startswith(d) for d in drop)]
    items = load("fitset")
    X, y, base, src = matrix(items, feats)
    print(f"fit words {len(y)}, planted {y.sum()}", file=sys.stderr)
    if kind == "trees":
        from sklearn.ensemble import GradientBoostingClassifier

        from bench.length.fit_length import _complete

        m = GradientBoostingClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=lr, subsample=0.8,
                                       random_state=0).fit(X, y)
        F, T, V = zip(*(_complete(e[0], depth) for e in m.estimators_))
        init = float(m.decision_function(X[:1])[0] - lr * sum(e[0].predict(X[:1])[0] for e in m.estimators_))
        M = {"kind": "trees", "features": feats, "init": init, "learning_rate": lr,
             "feat": np.array(F).tolist(), "thr": [[float(f"{v:.7g}") for v in r] for r in np.array(T)],
             "leaf": np.round(np.array(V), 6).tolist(), "depth": depth}
        # out-of-fold logits (speaker / clip groups) for the native map: in-sample logits of the
        # trees are optimistic on their own natives
        from sklearn.model_selection import GroupKFold

        z = np.empty(len(y))
        for tr, te in GroupKFold(5).split(X, y, matrix.groups):
            mk = GradientBoostingClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=lr, subsample=0.8,
                                            random_state=0).fit(X[tr], y[tr])
            z[te] = -mk.decision_function(X[te])
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
        M = {"kind": "logistic", "features": feats, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
             "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}
        z = -(sc.transform(X) @ m.coef_[0] + m.intercept_[0])
    # quantile map: the model's "fine" logit on untouched words -> the shipped word logit there
    # read-speech natives (FLEURS train) define the map: the guards are on FLEURS natives, and the
    # Common Voice natives are not all clean (their own word rates are allowed to move)
    nat = (y == 0) & (src == MAP_SOURCE) if MAP_SOURCE else (y == 0)
    qs = np.linspace(0, 1, 201)
    M["q_new"] = np.round(np.quantile(z[nat], qs), 5).tolist()
    M["q_orig"] = np.round(np.quantile(base[nat], qs), 5).tolist()
    (WORK / f"wordcal_{tag or kind}.json").write_text(json.dumps(M))
    return M


def model_logit(M, X):
    if M["kind"] == "trees":
        from pronunciada.length import _z_trees

        Md = dict(M)
        Md.pop("_np", None)
        return -_z_trees(Md, X)
    return -(((X - np.array(M["mean"])) / np.array(M["scale"])) @ np.array(M["coef"]) + M["intercept"])


def mapped(M, z):
    qn, qo = np.array(M["q_new"]), np.array(M["q_orig"])
    out = np.interp(z, qn, qo)
    out = np.where(z < qn[0], qo[0] + (z - qn[0]), out)
    return np.where(z > qn[-1], qo[-1] + (z - qn[-1]), out)


def recs(items, probs_by_item):
    out = []
    for it, pw in zip(items, probs_by_item):
        out.append({"id": it["id"], "text": it["text"], "units": it.get("units"), "label": it.get("label"),
                    "group": it.get("group"), "source": it.get("source"),
                    "words": [{"text": t, "prob": float(p)} for t, p in zip(it["texts"], pw)]})
    return out


def evaluate(kind: str) -> None:
    from bench.round19_ja.collect import load
    from bench.round19_ja.length_fit import native_word_rates
    from bench.round19_ja.unit_eval import paired, summary, units_of

    M = json.loads((WORK / f"wordcal_{kind}.json").read_text())
    for which in ("dev",):
        items = load(which)
        nat = load("devnat")
        new_p, nat_p = [], []
        for its, dst in ((items, new_p), (nat, nat_p)):
            for it in its:
                X = np.array([[r[f] for f in M["features"]] for r in word_rows(it)], dtype=np.float64)
                dst.append(1 / (1 + np.exp(-mapped(M, model_logit(M, X)))))
        old = recs(items, [it["wprob"] for it in items])
        new = recs(items, new_p)
        for src in ("all", "cv", "fleurs"):
            o = units_of([r for r in old if src == "all" or r["source"] == src])
            n = units_of([r for r in new if src == "all" or r["source"] == src])
            so, sn = summary(o, 300), summary(n, 300)
            d = paired(o, n, 500)
            bb = " ".join(f"{b[:6]}={v['auc']:.3f}" for b, v in sn["by_bucket"].items())
            print(f"{kind} {src:6s} AUC {so['auc']:.4f} -> {sn['auc']:.4f} (d {d['auc']['delta']:+.4f} p={d['auc']['p']:.3f}) "
                  f"FA80 {so['fa80']:.3f} -> {sn['fa80']:.3f} | {bb}")
        print("natives before", json.dumps(native_word_rates(recs(nat, [it["wprob"] for it in nat]))))
        print("natives after ", json.dumps(native_word_rates(recs(nat, nat_p))))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "dev"))
    ap.add_argument("--kind", default="trees")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--drop", nargs="*", default=())
    ap.add_argument("--tag")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit(a.kind, n_trees=a.trees, drop=a.drop, tag=a.tag)
    elif a.tag:
        evaluate(a.tag)
    else:
        evaluate(a.kind)


if __name__ == "__main__":
    main()
