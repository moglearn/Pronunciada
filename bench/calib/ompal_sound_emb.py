"""Round 12b, step 4: the d(i) recipe for the Mandarin consonant / vowel channel, OMPAL DEV only.

    python -m bench.calib.ompal_sound_emb extract     # per-sound vectors + prototype features (zh)
    python -m bench.calib.ompal_sound_emb cv          # speaker-grouped CV, today vs + activations

Units: OMPAL dev characters (learners; units with a span), labels consonant wrong (initial
sounds of the character's syllable) / vowel wrong (final sounds), exactly as
bench/coach/fit_v2.ompal_units builds them. Today's score: the coaching score (max over the unit's
initial / final sounds). Candidate: logistic on [today's score, min prototype margin, max
in-fold ridge projection of the pooled activations] (5-fold speaker-grouped, projection fitted
inside each training fold with inner out-of-fold values). Adopt only with CV AUC gain >= 0.01.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")


def _path():
    from bench.calib.common import calib_cache

    return calib_cache() / "sound2" / "ompal_dev_vectors.pkl"


_SC = None
_PR = None


def _init():
    from bench.calib.sound_emb import Protos
    from pronunciada.scorer import Scorer

    global _SC, _PR
    _SC = Scorer("zh", threads=2, explain_away=False)
    _PR = Protos("zh")


def _one(rec):
    from bench.calib.sound_emb import last_emb, phone_vectors
    from pronunciada import coach as Co

    try:
        res = _SC.score(rec["audio"], rec["text"])
    except ValueError:
        return rec["id"], None
    c = res._pronunciada  # noqa: SLF001
    V = phone_vectors(last_emb(_SC), c["ev"], len(c["ref"].phones))
    X = Co.sound_inputs(res)
    return rec["id"], {"V": V.astype(np.float16), "F": _PR.features(V, c["ref"]),
                       "ep": Co.error_prob(X), "word": X["word"], "syl": X["syl"], "initial": X["initial"],
                       "words": [w.text for w in res.words], "n_syl": [len(w.syllables) for w in res.words]}


def extract(workers=2):
    from concurrent.futures import ProcessPoolExecutor

    from bench.data import load

    recs = load("ompal", split="dev")
    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for uid, v in ex.map(_one, recs, chunksize=4):
            if v is not None:
                out[uid] = v
    _path().write_bytes(pickle.dumps(out))
    print(f"ompal dev: {len(out)} utts ({time.time() - t0:.0f}s)", flush=True)


def units():
    from bench.data import load
    from bench.eval_ompal import overlap, syllable_char_spans, word_char_spans

    D = pickle.loads(_path().read_bytes())
    rows = []
    for rec in load("ompal", split="dev"):
        lab = rec["labels"]
        if lab["native"] or lab["word_level_dropped"]:
            continue
        o = D.get(rec["id"])
        if o is None:
            continue
        spans = word_char_spans(rec["text"], o["words"])
        syl_span = {}
        for wi in range(len(o["words"])):
            if spans[wi] is not None:
                for k, s in enumerate(syllable_char_spans(spans[wi], o["n_syl"][wi])):
                    syl_span[(wi, k)] = s
        for u in lab["units"]:
            if u["span"] is None:
                continue
            a, b = u["span"]
            keys = {kk for kk, s in syl_span.items() if overlap(a, b, s[0], s[1])}
            sel = np.array([(o["word"][i], o["syl"][i]) in keys for i in range(len(o["ep"]))], bool)
            for part, msk, lab_ in (("initial", sel & o["initial"], u["consonant"] == 0),
                                    ("final", sel & ~o["initial"], u["vowel"] == 0)):
                if not msk.any():
                    continue
                F = o["F"][msk]
                rows.append({"spk": rec["speaker"], "part": part, "y": bool(lab_),
                             "ep": float(np.max(o["ep"][msk])),
                             "margin": float(np.nanmin(F[:, 2])) if np.isfinite(F[:, 2]).any() else np.nan,
                             "V": o["V"][msk].astype(np.float32).mean(0)})
    return rows


def cv():
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    from bench.calib.common import save_json

    rows = units()
    res = {}
    for part in ("initial", "final"):
        R = [r for r in rows if r["part"] == part]
        y = np.array([r["y"] for r in R], float)
        g = np.array([r["spk"] for r in R])
        ep = np.array([r["ep"] for r in R])
        mg = np.array([r["margin"] for r in R])
        mg = np.where(np.isfinite(mg), mg, np.nanmedian(mg))
        V = np.stack([r["V"] for r in R])
        base = np.zeros(len(y))
        new = {"+margin": np.zeros(len(y)), "+activations": np.zeros(len(y)), "+both": np.zeros(len(y))}
        for tr, te in GroupKFold(5).split(V, y, g):
            mu, sd = V[tr].mean(0), V[tr].std(0) + 1e-6
            proj = np.zeros(len(y))
            r_ = Ridge(alpha=3e3).fit((V[tr] - mu) / sd, y[tr])
            proj[te] = r_.predict((V[te] - mu) / sd)
            for itr, ite in GroupKFold(5).split(tr, None, g[tr]):
                ri = Ridge(alpha=3e3).fit((V[tr[itr]] - mu) / sd, y[tr[itr]])
                proj[tr[ite]] = ri.predict((V[tr[ite]] - mu) / sd)
            lg = np.log(np.clip(ep, 1e-6, 1 - 1e-6) / (1 - np.clip(ep, 1e-6, 1 - 1e-6)))
            feats = {"base": [lg], "+margin": [lg, mg], "+activations": [lg, proj], "+both": [lg, mg, proj]}
            for k, cols in feats.items():
                X = np.column_stack(cols)
                m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
                if k == "base":
                    base[te] = p
                else:
                    new[k][te] = p
        r = {"n": int(len(y)), "n_wrong": int(y.sum()), "auc_today_raw": float(roc_auc_score(y, ep)),
             "auc_cv_base": float(roc_auc_score(y, base))}
        for k, p in new.items():
            r[f"auc_cv{k}"] = float(roc_auc_score(y, p))
        # paired speaker bootstrap of the best candidate vs base
        best = max(new, key=lambda k: r[f"auc_cv{k}"])
        uniq = np.unique(g)
        rng = np.random.default_rng(0)
        d = []
        for _ in range(1000):
            idx = np.concatenate([np.flatnonzero(g == s) for s in rng.choice(uniq, len(uniq))])
            if y[idx].min() == y[idx].max():
                continue
            d.append(roc_auc_score(y[idx], new[best][idx]) - roc_auc_score(y[idx], base[idx]))
        r["best"] = best
        r["delta_best"] = [r[f"auc_cv{best}"] - r["auc_cv_base"], float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
        res[part] = r
        print(part, json.dumps(r), flush=True)
    save_json("ompal_sound_emb_cv.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "cv", "ship"])
    a = ap.parse_args()
    {"extract": extract, "cv": cv, "ship": ship}[a.step]()



# ----------------------------------------------------------------------------- ship (step 4 adopted)

def _unit_rows_sounds():
    """Per unit: the per-sound arrays of its part (for the runtime form: score per sound, max)."""
    from bench.data import load
    from bench.eval_ompal import overlap, syllable_char_spans, word_char_spans

    D = pickle.loads(_path().read_bytes())
    out = []
    for rec in load("ompal", split="dev"):
        lab = rec["labels"]
        if lab["native"] or lab["word_level_dropped"]:
            continue
        o = D.get(rec["id"])
        if o is None:
            continue
        spans = word_char_spans(rec["text"], o["words"])
        syl_span = {}
        for wi in range(len(o["words"])):
            if spans[wi] is not None:
                for k, s in enumerate(syllable_char_spans(spans[wi], o["n_syl"][wi])):
                    syl_span[(wi, k)] = s
        for u in lab["units"]:
            if u["span"] is None:
                continue
            a, b = u["span"]
            keys = {kk for kk, s in syl_span.items() if overlap(a, b, s[0], s[1])}
            sel = np.array([(o["word"][i], o["syl"][i]) in keys for i in range(len(o["ep"]))], bool)
            for part, msk, lab_ in (("initial", sel & o["initial"], u["consonant"] == 0),
                                    ("final", sel & ~o["initial"], u["vowel"] == 0)):
                if msk.any():
                    out.append({"spk": rec["speaker"], "part": part, "y": bool(lab_), "ep": o["ep"][msk],
                                "margin": o["F"][msk, 2], "V": o["V"][msk].astype(np.float32)})
    return out


def _feats(ep, margin, proj, med):
    lg = np.log(np.clip(ep, 1e-6, 1 - 1e-6) / (1 - np.clip(ep, 1e-6, 1 - 1e-6)))
    return np.column_stack([lg, np.where(np.isfinite(margin), margin, med), proj])


def _fit_part(U, rows):
    """ridge on per-sound vectors (labels of their unit) + logistic on unit features
    [max logit(ep), min margin, max projection]; returns params."""
    from sklearn.linear_model import LogisticRegression, Ridge

    V = np.concatenate([U[i]["V"] for i in rows])
    yv = np.concatenate([np.full(len(U[i]["V"]), U[i]["y"], float) for i in rows])
    mu, sd = V.mean(0), V.std(0) + 1e-6
    r = Ridge(alpha=3e3).fit((V - mu) / sd, yv)
    allm = np.concatenate([U[i]["margin"] for i in rows])
    med = float(np.nanmedian(allm))
    X, y = [], []
    for i in rows:
        u = U[i]
        pj = ((u["V"] - mu) / sd) @ r.coef_ + r.intercept_
        f = _feats(u["ep"], u["margin"], pj, med)
        X.append([f[:, 0].max(), f[:, 1].min(), f[:, 2].max()])
        y.append(u["y"])
    m = LogisticRegression(C=1.0, max_iter=2000).fit(np.array(X), np.array(y))
    return {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "W": r.coef_.astype(np.float32),
            "b": float(r.intercept_), "margin_fill": med, "coef": m.coef_[0].astype(float), "intercept": float(m.intercept_[0])}


def unit_score(P, u):
    """Runtime form: the logistic applied PER SOUND (its own logit(ep), margin, projection), unit
    score = max over the part's sounds (pronunciada.coach_zh does the same per sound)."""
    pj = ((u["V"] - P["mu"]) / P["sd"]) @ P["W"] + P["b"]
    f = _feats(u["ep"], u["margin"], pj, P["margin_fill"])
    z = f @ P["coef"] + P["intercept"]
    return float(1 / (1 + np.exp(-z.max())))


def ship():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from bench.calib.common import save_json, ship_dir
    from bench.calib.sound_emb import proto_path
    from bench.coach.fit_v2 import TARGETS, pr_at, thr_for_precision

    U = _unit_rows_sounds()
    res, arrays, thr = {}, {}, {}
    for part in ("initial", "final"):
        idx = np.array([i for i, u in enumerate(U) if u["part"] == part])
        y = np.array([U[i]["y"] for i in idx], bool)
        g = np.array([U[i]["spk"] for i in idx])
        oof = np.zeros(len(idx))
        for tr, te in GroupKFold(5).split(idx, y, g):
            P = _fit_part(U, idx[tr])
            oof[te] = [unit_score(P, U[i]) for i in idx[te]]
        today = np.array([U[i]["ep"].max() for i in idx])
        th = {s: thr_for_precision(oof, y, t) for s, t in TARGETS.items()}
        res[part] = {"n": int(len(y)), "n_wrong": int(y.sum()), "auc_today": float(roc_auc_score(y, today)),
                     "auc_oof_runtime_form": float(roc_auc_score(y, oof)), "thresholds": th,
                     "at": {s: (pr_at(oof, y, x) if x is not None else None) for s, x in th.items()}}
        thr[part] = {s: (x if x is not None else 1.01) for s, x in th.items()}
        P = _fit_part(U, idx)
        for k, v in P.items():
            arrays[f"{part}_{k}"] = np.asarray(v, np.float32) if isinstance(v, np.ndarray) else np.array([v], np.float64) \
                if np.isscalar(v) else np.asarray(v)
        print(part, json.dumps(res[part], default=float), flush=True)
    z = np.load(proto_path("zh"))
    d = ship_dir().parent / "coach"
    np.savez_compressed(d / "zh_sound.npz", proto_mu=z["mu"], proto_sd=z["sd"], proto_P=z["P"].astype(np.float16),
                        proto_phones=z["phones"], **arrays)
    res["kb"] = round((d / "zh_sound.npz").stat().st_size / 1024, 1)
    res["thresholds"] = thr
    save_json("ompal_sound_emb_ship.json", res)
    return res


if __name__ == "__main__":
    main()
