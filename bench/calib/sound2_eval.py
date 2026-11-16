"""Round 12: diagnosis + candidates a / b / c / d(ii) on FLEURS validation (dev), offline.

    python -m bench.calib.sound2_eval diagnose
    python -m bench.calib.sound2_eval candidates
    python -m bench.calib.sound2_eval so762            # d(i) + guards on speechocean762 TRAIN (CV)

Reads the per-language collections of bench/calib/sound2.py one language at a time (memory).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from bench.calib import sound2 as S2

Q_GRID = (0.002, 0.005, 0.01)
C_GRID = (-1.0, -2.0)
GUARD = 0.005          # native word / sound acceptance drop per language (0.5 points)
PROTO = ("proto_cos", "proto_other", "proto_margin")


def out_path(name):
    from bench.calib.common import tables_dir

    return tables_dir() / f"round12_{name}.json"


# ----------------------------------------------------------------------------- diagnosis

def diagnose():
    res = {}
    for lang in S2.LANGS:
        data = S2.load(lang)
        rows, nat = S2.diagnose(lang, data)
        del data
        r = {"native": nat}
        for src in ("swap", "phone_sub_near", "phone_sub_far"):
            x = [q for q in rows if q["src"] == src]
            mi = [q for q in x if q["miss"]]
            if not x:
                continue
            pos = Counter(("initial", "medial", "final", "only")[q["pos"]] for q in mi)
            pos_all = Counter(("initial", "medial", "final", "only")[q["pos"]] for q in x)
            r[src] = {"n": len(x), "miss": len(mi), "miss_share": len(mi) / len(x),
                      "cats": dict(Counter(q["cat"] for q in mi)),
                      "miss_pos": dict(pos), "all_pos": dict(pos_all),
                      "miss_vowel": float(np.mean([q["vowel"] for q in mi])) if mi else None,
                      "all_vowel": float(np.mean([q["vowel"] for q in x])),
                      "miss_short": float(np.mean([q["wlen"] <= 3 for q in mi])) if mi else None,
                      "all_short": float(np.mean([q["wlen"] <= 3 for q in x])),
                      "miss_median_llr": float(np.median([q["llr_sub"] for q in mi])) if mi else None,
                      "miss_median_ps": float(np.median([q["ps"] for q in mi])) if mi else None}
        res[lang] = r
        print(lang, json.dumps({k: (v if k == "native" else {kk: v[kk] for kk in ("n", "miss", "cats")})
                                for k, v in r.items()}, ensure_ascii=False), flush=True)
    out_path("diagnosis").write_text(json.dumps(res, indent=1))
    return res


# ----------------------------------------------------------------------------- candidates

def _feats(P, with_proto):
    X = S2.feature_matrix(P)
    if not with_proto:
        return X
    E = np.column_stack([np.asarray(P.get(k, np.full(len(X), np.nan)), np.float64) for k in PROTO])
    have = np.isfinite(E[:, 0]).astype(float)
    E = np.nan_to_num(E, nan=0.0)
    return np.column_stack([X, E, have])


def rows_for(lang, with_proto):
    data = S2.load(lang)
    X, y, nat = [], [], []
    for d in data.values():
        it = d["items"]["clean"]
        if not it["bad"]:
            X.append(_feats(it["P"], with_proto))
            y.append(np.ones(len(it["ipa"])))
        for key in ("cor:phone_sub_near", "cor:phone_sub_far"):
            it = d["items"].get(key)
            if it is None or it["bad"]:
                continue
            m = it["phone_wrong"]
            X.append(_feats(it["P"], with_proto)[m])
            y.append(np.zeros(int(m.sum())))
    return np.concatenate(X), np.concatenate(y)


def fit_lr(parts):
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    w = np.concatenate([np.where(p[1] == 1, 0.5 / max((p[1] == 1).sum(), 1), 0.5 / max((p[1] == 0).sum(), 1))
                        for p in parts]) * len(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=1.0, max_iter=5000).fit((X - mu) / sd, y, sample_weight=w)
    return {"coef": (m.coef_[0] / sd).tolist(), "b": float(m.intercept_[0] - (m.coef_[0] * mu / sd).sum())}


def score_model(model, X):
    return X @ np.asarray(model["coef"]) + model["b"]


def lolo_models(with_proto):
    """Leave-one-language-out models + the training natives' z (for the offset)."""
    parts = {lang: rows_for(lang, with_proto) for lang in S2.LANGS}
    out = {}
    for lang in S2.LANGS:
        tr = [parts[l] for l in S2.LANGS if l != lang]
        m = fit_lr(tr)
        zn = np.concatenate([score_model(m, X[y == 1]) for X, y in tr])
        m["nat_q"] = {str(q): float(np.quantile(zn, q)) for q in Q_GRID}
        out[lang] = m
    full = fit_lr(list(parts.values()))
    zn = np.concatenate([score_model(full, X[y == 1]) for X, y in parts.values()])
    full["nat_q"] = {str(q): float(np.quantile(zn, q)) for q in Q_GRID}
    return out, full


def transform_a(model, q, with_proto, level="sound"):
    off = -model["nat_q"][str(q)]

    def tr(it):
        P = it["P"]
        z = score_model(model, _feats(P, with_proto)) + off
        if level == "sound":
            return (np.minimum(P["uzl"], z), np.minimum(P["uzs"], z),
                    np.minimum(P["szl"], z), np.minimum(P["szs"], z))
        return None
    return tr


def metrics_word_level(lang, data, model, q, with_proto):
    """Candidate b: word logits capped by the lowest second-opinion z of the word's sounds."""
    off = -model["nat_q"][str(q)]
    orig = S2.outputs

    def outputs(it, transform, cal):
        ps, wp, snd = orig(it, None, cal)
        z = score_model(model, _feats(it["P"], with_proto)) + off
        wi = np.asarray(it["P"]["word_index"], int)
        zmin = np.full(len(wp), np.inf)
        np.minimum.at(zmin, wi, z)
        cap = S2._sig(zmin)
        ratio = np.minimum(1.0, cap / np.maximum(snd, 1e-9))
        return ps, wp * ratio, snd * ratio
    S2.outputs = outputs
    try:
        return S2.metrics(lang, data)
    finally:
        S2.outputs = orig


def transform_c(deltas):
    """Candidate c: position-specific additive bias (final / only sounds)."""
    def tr(it):
        P = it["P"]
        d = np.where(np.asarray(P["pos"]) >= 2, deltas, 0.0)
        return P["uzl"] + d, P["uzs"] + d, P["szl"] + d, P["szs"] + d
    return tr


KEYS = ("swap_near:auc", "near_gt09", "near_gt08", "swap_near:hit@5%FA", "swap_near:right_word", "swap_far:auc",
        "native_words", "native_sounds", "sound_auc", "word_auc", "pros_auc")


def _v(m, k):
    x = m.get(k)
    return x[0] if isinstance(x, list) else x


def _cand_lang(args):
    import os

    os.environ["OMP_NUM_THREADS"] = "2"
    lang, models = args
    data = S2.load(lang)
    r = {"base": S2.metrics(lang, data)}
    for name, wp in (("a", False), ("d2", True)):
        m = models[name]["lolo"][lang]
        r[name] = {str(q): S2.metrics(lang, data, transform_a(m, q, wp)) for q in Q_GRID}
    m = models["a"]["lolo"][lang]
    r["b"] = {str(q): metrics_word_level(lang, data, m, q, False) for q in Q_GRID}
    r["c"] = {str(dl): S2.metrics(lang, data, transform_c(dl)) for dl in C_GRID}
    return lang, r


def candidates():
    from concurrent.futures import ProcessPoolExecutor

    res = {"base": {}, "a": {}, "b": {}, "c": {}, "d2": {}}
    models = {}
    for name, wp in (("a", False), ("d2", True)):
        lolo, full = lolo_models(wp)
        models[name] = {"lolo": lolo, "full": full}
        print(name, "fitted", flush=True)
    out_path("models").write_text(json.dumps(models, indent=1))
    with ProcessPoolExecutor(2) as ex:
        for lang, r in ex.map(_cand_lang, [(l, models) for l in S2.LANGS]):
            for k, v in r.items():
                res[k][lang] = v
            print(lang, "base", {k: round(_v(r["base"], k), 4) for k in KEYS if _v(r["base"], k) is not None}, flush=True)
            for name in ("a", "b", "d2", "c"):
                for q, mm in r[name].items():
                    print(f"  {name} {q}", {k: round(_v(mm, k), 4) for k in KEYS if _v(mm, k) is not None}, flush=True)
    res["models"] = {k: v["full"] for k, v in models.items()}
    out_path("candidates").write_text(json.dumps(res, indent=1, default=float))
    return res


def select(res=None):
    """Per candidate: the setting (one for all languages; c: per language) with the best mean
    near AUC among those meeting the native guards in every language."""
    res = res or json.loads(out_path("candidates").read_text())
    base = res["base"]

    def ok(lang, m):
        return (base[lang]["native_words"] - m["native_words"] <= GUARD
                and base[lang]["native_sounds"] - m["native_sounds"] <= GUARD)

    summ = {}
    for name in ("a", "b", "d2"):
        best = None
        for q in map(str, Q_GRID):
            if all(ok(l, res[name][l][q]) for l in S2.LANGS):
                mean = np.mean([_v(res[name][l][q], "swap_near:auc") for l in S2.LANGS])
                if best is None or mean > best[1]:
                    best = (q, mean)
        summ[name] = best
    cpick = {}
    for l in S2.LANGS:
        c = [(dl, m) for dl, m in res["c"][l].items() if ok(l, m)]
        cpick[l] = max(c, key=lambda t: _v(t[1], "swap_near:auc"))[0] if c else None
    summ["c"] = cpick
    print(json.dumps(summ, indent=1))
    return summ


# ----------------------------------------------------------------------------- d(i): speechocean762

def prec_at_recall(y_bad, score_bad, recalls=(0.20, 0.33, 0.47)):
    """Precision when flagging the top-scored sounds until the given share of bad ones is caught."""
    o = np.argsort(-score_bad, kind="stable")
    yb = y_bad[o].astype(float)
    tp = np.cumsum(yb)
    out = {}
    for r in recalls:
        k = int(np.searchsorted(tp, r * yb.sum()))
        out[str(r)] = float(tp[k] / (k + 1))
    return out


def so762():
    import os
    import pickle

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    from bench.calib.common import fit, load_table, raw, speaker_folds, table_columns, vocab_for
    from bench.calib.train_sound import KINDS, labels_so762
    from bench.metrics import pearson, roc_auc
    from pronunciada import features as F

    df = load_table("speechocean762", "train")
    t = table_columns(df)
    lab = labels_so762(t)
    groups = df.speaker.astype(str).values
    spec = F.SOUND_SETS["B"]
    vocab = vocab_for(t, F.SOUND_SETS["A"]["cat"] + F.IDENTITY_CAT)
    XB = F.matrix(t, spec["num"], spec["cat"], vocab)
    cmB = F.categorical_mask(spec["num"], spec["cat"], vocab)
    vec = pickle.loads((S2.work_dir() / "so762_train_vectors.pkl").read_bytes())
    n = len(df)
    V = np.full((n, 1024), np.nan, np.float32)
    E = np.full((n, 4), np.nan)
    match = 0
    for i, (u, k, ph) in enumerate(zip(df.utt.values, df.phone_index.values.astype(int), df.phone.values)):
        it = vec.get(u)
        if it is None or k >= len(it["ipa"]):
            continue
        if it["ipa"][k] != ph:
            continue
        match += 1
        V[i] = it["V"][k]
        P = it["P"]
        e = [P[c][k] for c in PROTO]
        E[i] = e + [float(np.isfinite(e[0]))]
    del vec
    E[:, :3] = np.nan_to_num(E[:, :3], nan=0.0)
    E[:, 3] = np.nan_to_num(E[:, 3], nan=0.0)
    okV = np.isfinite(V[:, 0])
    print(f"rows {n}, matched vectors {match} ({match / n:.3f})", flush=True)
    folds = speaker_folds(groups)
    y = lab["reg"]
    params = {"reg": {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 40},
              "len": {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 40},
              "str": {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 40}}

    def proj_fold(tr, te):
        """in-fold ridge projection of V to the phone score; training rows get inner OOF values."""
        out = np.full(n, np.nan)
        lab_tr = tr[np.isfinite(y[tr]) & okV[tr]]
        mu, sd = V[lab_tr].mean(0), V[lab_tr].std(0) + 1e-6

        def fitp(rows):
            return Ridge(alpha=3e3).fit((V[rows] - mu) / sd, y[rows])
        m = fitp(lab_tr)
        te_ok = te[okV[te]]
        out[te_ok] = m.predict((V[te_ok] - mu) / sd)
        for itr, ite in GroupKFold(5).split(lab_tr, None, groups[lab_tr]):
            mi = fitp(lab_tr[itr])
            out[lab_tr[ite]] = mi.predict((V[lab_tr[ite]] - mu) / sd)
        rest = tr[~np.isin(tr, lab_tr) & okV[tr]]
        if len(rest):
            out[rest] = m.predict((V[rest] - mu) / sd)
        return out

    variants = {"B": (False, False), "B+proto": (False, True), "B+emb": (True, False), "B+emb+proto": (True, True)}
    res = {}
    for name, (use_v, use_p) in variants.items():
        oof = {h: np.full(n, np.nan) for h in ("reg", "len", "str")}
        for tr, te in folds:
            cols = [XB]
            if use_v:
                pj = proj_fold(tr, te)
                cols.append(np.nan_to_num(pj, nan=np.nanmean(pj))[:, None])
            if use_p:
                cols.append(E)
            X = np.column_stack(cols)
            cm = np.r_[cmB, np.zeros(X.shape[1] - XB.shape[1], bool)]
            for h in ("reg", "len", "str"):
                yy = lab[h]
                ok = tr[np.isfinite(yy[tr])]
                m = fit(KINDS[h], params[h], X[ok], yy[ok], cm)
                oof[h][te] = raw(m, X[te])
        m = np.isfinite(y)
        bad = (y[m] < 2).astype(int)
        r = {"phone_pearson": pearson(oof["reg"][m], y[m]),
             "auc_bad_lt2_reg": roc_auc(bad, -oof["reg"][m]),
             "auc_str": roc_auc((y[m] >= 1.8).astype(int), oof["str"][m]),
             "auc_len": roc_auc((y[m] >= 1.0).astype(int), oof["len"][m]),
             "prec_at_recall_reg": prec_at_recall(bad, -oof["reg"][m]),
             "prec_at_recall_str": prec_at_recall(bad, -oof["str"][m]),
             "n": int(m.sum()), "bad_rate": float(bad.mean())}
        res[name] = r
        print(name, json.dumps(r), flush=True)
    res["matched_share"] = match / n
    out_path("so762").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["diagnose", "candidates", "select", "so762"])
    a = ap.parse_args()
    {"diagnose": diagnose, "candidates": candidates, "select": select, "so762": so762}[a.step]()


if __name__ == "__main__":
    main()
