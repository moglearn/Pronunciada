"""Round 7: a learned word-level combination of the sounds and tone channels for Mandarin, fitted
on real learner labels (OMPAL DEV) plus FLEURS-zh validation natives and planted errors.

    python -m bench.calib.fit_word_combo [--write]

Today:  WordResult.prob = sounds_prob(s) x P_tone(s) x present_prob.
Learned: WordResult.prob = sigmoid(a0 + a1 logit sounds_prob(s) + a2 logit P_tone(s)
                                   [+ a3 logit sounds x logit P_tone]) x present_prob
with a1, a2 >= 0 (monotone). Strictness keeps acting through both channels' own operating points,
so the lenient / strict ends move as today. Per-channel outputs (sounds_prob, tone_prob,
present_prob) are unchanged; only WordResult.prob changes, and only for languages listed in
calib/word_combo.json.

Data (never OMPAL TEST, never FLEURS test):
  OMPAL DEV: the cached `bench.eval_ompal --steps ours --split dev` outputs of the default tone
    model (per word: char span, prob, sounds_prob, tone_prob = P_tone; per syllable tone_prob);
    word label = any of its OMPAL units has an error; present_prob = prob / (sounds x tone).
  FLEURS validation: the bench.calib.fit_prosody rows of the same tone model (natives, planted
    tone errors, planted sound errors); P_tone recomputed with the shipped zh.json; present = 1.
Selection: 5-fold CV, OMPAL folds grouped by speaker, FLEURS folds grouped by utterance.
  Primary: OMPAL DEV unit-level AUC any-error vs correct (unit score = min over overlapping words,
  as bench.eval_ompal); also tone-only, consonant/vowel, hit at 10 % learner false alarms.
  Guards on FLEURS (held-out folds): native word acceptance at s = 0.5 drops <= 1 point vs the
  product, planted sound-error AUC drops <= 0.01, planted tone-error AUC does not drop.
  Adopt only if CV any-error AUC >= max(product word score, tone channel alone) + 0.01.
"""
from __future__ import annotations

import argparse
import json
import pickle
import zlib

import numpy as np

CLIP = 8.0


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1 - p)), -CLIP, CLIP)


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


# ----------------------------------------------------------------------------- data

def ompal_dev():
    import bench.eval_ompal as E

    recs = E.records("dev")
    out, meta = E.ours_outputs(recs)
    W = {k: [] for k in ("spk", "utt", "snd", "tone", "present", "prob", "wrong", "tmin")}
    U = {k: [] for k in ("spk", "word_idx", "any", "tone_only", "seg", "correct", "native")}
    for r in recs:
        lab = r["labels"]
        o = out.get(r["id"])
        if lab["word_level_dropped"] or not o or o.get("status") != "Success":
            continue
        base = len(W["snd"])
        units = [u for u in lab["units"] if u["span"] is not None]
        for a, b, prob, snd, tone in o["words"]:
            tn = 1.0 if tone is None else tone
            sy = [t for x0, x1, t in o["syllables"] if t is not None and x0 < b and a < x1]
            W["spk"].append(r["speaker"])
            W["utt"].append(r["id"])
            W["snd"].append(snd)
            W["tone"].append(tn)
            W["present"].append(float(np.clip(prob / max(snd * tn, 1e-12), 0, 1)) if snd * tn > 1e-9 else 1.0)
            W["prob"].append(prob)
            W["tmin"].append(min(sy) if sy else 1.0)
            W["wrong"].append(any(u["any_wrong"] for u in units if u["span"][0] < b and a < u["span"][1]))
        for u in units:
            a, b = u["span"]
            idx = [base + k for k, w in enumerate(o["words"]) if w[0] < b and a < w[1]]
            U["spk"].append(r["speaker"])
            U["word_idx"].append(idx)
            U["any"].append(u["any_wrong"])
            U["tone_only"].append(u["tone_only"])
            U["seg"].append(u["seg_wrong"])
            U["correct"].append(u["correct"])
            U["native"].append(lab["native"])
    W = {k: np.asarray(v) for k, v in W.items()}
    U = {k: (v if k == "word_idx" else np.asarray(v)) for k, v in U.items()}
    return W, U, meta


def fleurs_rows():
    from bench.calib import fit_prosody as FP
    from pronunciada.calibrate import data_dir

    rows, meta = FP.cached_collect("zh", 200)
    par = json.loads((data_dir() / "prosody" / "zh.json").read_text())
    assert par["prosody_model_meta"]["model"] == meta["model"], "zh.json is not fitted for this tone model"
    F = FP._feats(rows)
    pp = {s: FP._pp(par, F, s) for s in (0.0, 0.5, 1.0)}
    snd = np.array([r["snd"] for r in rows])
    return {"utt": np.array([r["utt"] for r in rows]), "ref": np.array([r["ref"] for r in rows]),
            "wrong": np.array([r["wrong"] for r in rows]), "snd": snd,
            "tone": np.stack([pp[s] for s in (0.0, 0.5, 1.0)], 1), "model": meta["model"]}


# ----------------------------------------------------------------------------- model

def design(snd, tone, kind):
    """lin / inter: logits of the two channels (+ their product); log / loginter: log-probabilities
    (the product's own space: log P = log sounds + log tone) (+ their product)."""
    if kind.startswith("log"):
        a = np.log(np.clip(np.asarray(snd, np.float64), 1e-4, 1.0))
        b = np.log(np.clip(np.asarray(tone, np.float64), 1e-4, 1.0))
        cols = [a, b] + ([a * b] if kind == "loginter" else [])
    else:
        a, b = lg(snd), lg(tone)
        cols = [a, b] + ([a * b / CLIP] if kind == "inter" else [])
    return np.stack(cols, 1)


def fit_lr(X, y, w, C=1.0):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=C, max_iter=5000).fit(X, y, sample_weight=w)
    coef = np.r_[m.intercept_, m.coef_[0]]
    if coef[1] < 0 or coef[2] < 0:     # (loginter: a3 sign free, monotone over the data range)
        raise RuntimeError(f"non-monotone combination {coef}")
    return coef


def predict(coef, snd, tone, kind):
    return sig(coef[0] + design(snd, tone, kind) @ coef[1:])


def unit_scores(Wp, U):
    return np.array([min(Wp[i] for i in ix) if ix else 0.0 for ix in U["word_idx"]])


def auc(y, s):
    from bench.metrics import roc_auc

    return roc_auc(np.asarray(y, int), -np.asarray(s))


def hit_at(pos, neg, alpha=0.10):
    thr = np.quantile(neg, alpha)
    return float(np.mean(pos < thr))


def ompal_metrics(us, U):
    L = U["native"] == 0
    c = (U["correct"] == 1)
    out = {}
    for k, m in (("any", U["any"] == 1), ("tone_only", U["tone_only"] == 1), ("seg", U["seg"] == 1)):
        sel = m | c
        out[f"auc_{k}"] = auc(m[sel], us[sel])
        out[f"hit10_{k}"] = hit_at(us[m & L], us[c & L])
    out["learner_correct_accepted"] = float(np.mean(us[c & L] >= 0.5))
    return out


def fleurs_metrics(pw, F):
    clean = F["ref"] == "clean"
    out = {"native_accept": float(np.mean(pw[clean] >= 0.5))}
    for tag in ("sound", "prosody"):
        m = (F["ref"] == tag) & F["wrong"]
        sel = m | clean
        out[f"auc_{tag}"] = auc(m[sel], pw[sel])
    return out


def folds(keys, k=5):
    return np.array([zlib.crc32(str(x).encode()) % k for x in keys])


def cv(W, U, F, kind, w_fleurs, w_ompal=1.0, C=1.0, k=5):
    fw, ff = folds(W["spk"], k), folds(F["utt"], k)
    pw_o = np.zeros(len(W["snd"]))
    pw_f = np.zeros(len(F["snd"]))
    coefs = []
    for i in range(k):
        tr_o, tr_f = fw != i, ff != i
        X = np.r_[design(W["snd"][tr_o], W["tone"][tr_o], kind), design(F["snd"][tr_f, 1], F["tone"][tr_f, 1], kind)]
        y = np.r_[~W["wrong"][tr_o], ~F["wrong"][tr_f]].astype(int)
        nf = tr_f.sum()
        fe = F["wrong"][tr_f]
        # FLEURS weights: planted errors at 10 % of the FLEURS weight (as fit_prosody), total w_fleurs x OMPAL
        wf = np.where(fe, 0.1 / max(fe.mean(), 1e-9), 0.9 / max(1 - fe.mean(), 1e-9)) / nf
        wo = np.full(tr_o.sum(), w_ompal / tr_o.sum())
        sw = np.r_[wo, wf * w_fleurs] * (tr_o.sum() + nf)
        c = fit_lr(X, y, sw, C)
        coefs.append(c)
        pw_o[~tr_o] = predict(c, W["snd"][~tr_o], W["tone"][~tr_o], kind) * W["present"][~tr_o]
        pw_f[~tr_f] = predict(c, F["snd"][~tr_f, 1], F["tone"][~tr_f, 1], kind)
    return ompal_metrics(unit_scores(pw_o, U), U), fleurs_metrics(pw_f, F), np.mean(coefs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    W, U, meta = ompal_dev()
    F = fleurs_rows()
    print(f"OMPAL DEV: {len(W['snd'])} words, {len(U['any'])} units ({int(U['any'].sum())} any-error, "
          f"{int(U['tone_only'].sum())} tone-only, {int(U['seg'].sum())} seg); runs {meta['dir']}; "
          f"FLEURS rows {len(F['snd'])} (tone model {F['model']})")
    base_o = {"word (product)": unit_scores(W["prob"], U),
              "tone channel alone": unit_scores(W["tmin"], U),
              "sounds alone": unit_scores(W["snd"], U)}
    res = {"baselines": {}, "cv": {}}
    for k, us in base_o.items():
        res["baselines"][k] = ompal_metrics(us, U)
        print(f"{k:22s}", {m: round(v, 4) for m, v in res["baselines"][k].items()})
    prod_f = fleurs_metrics(F["snd"][:, 1] * F["tone"][:, 1], F)
    res["baselines"]["fleurs product"] = prod_f
    print("FLEURS product      ", {m: round(v, 4) for m, v in prod_f.items()})
    # sounds-channel shift on the product (OMPAL DEV only, same CV not needed: one parameter per fold)
    for d in (0.0, 0.5, 1.0, 1.5):
        us = unit_scores(sig(lg(W["snd"]) + d) * W["tone"] * W["present"], U)
        pf = sig(lg(F["snd"][:, 1]) + d) * F["tone"][:, 1]
        res["baselines"][f"product, sounds shift {d}"] = {**ompal_metrics(us, U), **fleurs_metrics(pf, F)}
        print(f"product sounds+{d}", {m: round(v, 4) for m, v in res["baselines"][f"product, sounds shift {d}"].items()})
    best = None
    for kind in ("lin", "inter", "log", "loginter"):
        for wf in (0.25, 1.0, 4.0):
            mo, mf, coef = cv(W, U, F, kind, wf)
            ok = (prod_f["native_accept"] - mf["native_accept"] <= 0.010 and prod_f["auc_sound"] - mf["auc_sound"] <= 0.010
                  and mf["auc_prosody"] >= prod_f["auc_prosody"] - 1e-9)
            key = f"{kind} w_fleurs={wf}"
            res["cv"][key] = {"ompal": mo, "fleurs": mf, "coef_mean": coef.tolist(), "guards_ok": ok}
            print(f"CV {key:18s} guards {'OK ' if ok else 'FAIL'}", {m: round(v, 4) for m, v in mo.items()},
                  {m: round(v, 4) for m, v in mf.items()}, np.round(coef, 3))
            if ok and (best is None or mo["auc_any"] > res["cv"][best]["ompal"]["auc_any"]):
                best = key
    bar = max(res["baselines"]["word (product)"]["auc_any"], res["baselines"]["tone channel alone"]["auc_any"]) + 0.01
    res["bar"] = bar
    res["best"] = best
    adopt = best is not None and res["cv"][best]["ompal"]["auc_any"] >= bar
    res["adopt"] = adopt
    print("best", best, "bar", round(bar, 4), "ADOPT" if adopt else "keep the product")
    from bench.calib.common import save_json

    save_json("word_combo_zh_cv.json", res)
    if a.write and adopt:
        kind, wf = best.split(" w_fleurs=")
        wf = float(wf)
        X = np.r_[design(W["snd"], W["tone"], kind), design(F["snd"][:, 1], F["tone"][:, 1], kind)]
        y = np.r_[~W["wrong"], ~F["wrong"]].astype(int)
        nf, no = len(F["snd"]), len(W["snd"])
        fe = F["wrong"]
        wfv = np.where(fe, 0.1 / fe.mean(), 0.9 / (1 - fe.mean())) / nf
        sw = np.r_[np.full(no, 1.0 / no), wfv * wf] * (no + nf)
        coef = fit_lr(X, y, sw)
        from pronunciada.calibrate import data_dir

        p = data_dir() / "word_combo.json"
        cfg = json.loads(p.read_text()) if p.exists() else {}
        cfg["zh"] = {"kind": kind, "coef": coef.tolist(), "clip": CLIP, "tone_model": F["model"],
                     "features": "[1, logit sounds_prob(s), logit P_tone(s)" + (", logit x logit / clip]" if kind == "inter" else "]"),
                     "fit_on": "OMPAL DEV (28 learners) + FLEURS-zh validation (200 clips, fit_prosody rows)",
                     "w_fleurs": wf, "cv": res["cv"][best]}
        p.write_text(json.dumps(cfg, indent=1))
        print("wrote", p, np.round(coef, 4))
    _ = pickle


if __name__ == "__main__":
    main()
