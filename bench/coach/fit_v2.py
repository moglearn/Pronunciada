"""Coaching round 2: the coaching operating point fitted on REAL learner data.

    python -m bench.coach.fit_v2            # needs bench/coach/dump.py outputs

Fitting data (licence-clean, train / dev only):
  speechocean762 TRAIN  expert phone scores 0-2 (error = score < 2) and the `mispronunciations`
                        lists (canonical vs pronounced phone, <DEL>, <unk>)
  OMPAL DEV             per-character tone / consonant / vowel marks (Mandarin learners)
  FLEURS validation     native clips (false diagnoses per clip) + planted errors (sanity)
The English sound logits on speechocean762 train are replaced by the sound calibrator's
OUT-OF-FOLD logits (the shipped calibrator was fitted on that split). The first version of this
script used the in-sample logits; its held-out evaluation (once) showed the optimism (balanced
precision 0.51 on test against 0.70 on train), which is how the leak was found; see
results/coaching.md.
Rules (fixed before looking at results):
  1. sound model: logistic regression (monotone in each input by construction) on
     coach.SOUND_FEATURES; out-of-fold predictions by 5-fold speaker-grouped CV; thresholds =
     lowest score with OOF precision >= 0.8 / 0.7 / 0.6 (careful / balanced / eager).
  2. heard gate on flagged (balanced, OOF) sounds with a pronounced-phone substitution label:
     grid (log-ratio, heard posterior); per sound class, a class is ON when its precision
     (exact or within one feature) >= 0.75 with >= 15 items; the setting with the most correct
     named sounds among those reaching >= 0.75 pooled over the ON classes.
  3. missing: logistic on the flagged sounds, label <DEL>; enabled only when some threshold
     reaches OOF precision >= 0.7 with >= 20 items.
  4. Mandarin: OMPAL dev, per character; initial / final thresholds on the coaching score and
     a tone threshold on the syllable tone probability for type precision 0.8 / 0.7 / 0.6.
  5. other languages: the English thresholds; a language whose "balanced" setting gives more
     than 0.3 wrong diagnoses per FLEURS native clip gets the lowest threshold that meets it.
Writes pronunciada/data/coach/model.json and results/tables/coaching_v2_fit.json.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.coach.dump import load_dump
from pronunciada import articulatory as A
from pronunciada import coach as Co

FOLD = {"ʌ": "ə", "ə˞": "ɜ˞", "ɒ": "ɑ"}
TARGETS = {"careful": 0.8, "balanced": 0.7, "eager": 0.6}
LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")


def within1(t, h):
    if t == h:
        return True
    try:
        d = A.diff(t, h)
    except KeyError:
        return False
    return len(d) <= 1 and "type" not in d


def fold(p):
    return FOLD.get(p, p) if p else p


def add_initial(X):
    X["initial"] = (np.asarray(X["first_in_syl"], bool) & (np.asarray(X["nucleus"]) == 0)
                    & np.array([e not in Co.GLIDES for e in X["expected"]]))
    return X


# ----------------------------------------------------------------------------- so762 labels

def so762_rows(name, split):
    """Concatenated per-sound arrays of a so762 dump + labels aligned to the experts' phones."""
    from bench.data import load
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference, arpa_to_ours

    D = load_dump(name)
    cols = defaultdict(list)
    oofz = None
    if split == "train":
        # The English sound calibrator (sound_B) was itself fitted on speechocean762 TRAIN, so its
        # logits are in-sample there; use its out-of-fold logits (bench/calib/train_sound.py,
        # 5-fold speaker-grouped) instead, or the coaching model learns to trust them too much.
        import os

        import pandas as pd

        from pronunciada import calibrate as C

        name_, bias = C.sound_model_for("en")
        # round 12b: the coaching inputs come from the "detail" sound model when one is configured
        # (sound_E: its out-of-fold logits are the E_* columns, bench/calib/train_sound_emb.py)
        det = C.config().get("sound_detail_model", {}).get("en")
        pre = {"sound_B": "B", "sound_E": "E"}[det or name_]
        bias = 0.0 if det else bias
        pq = os.path.join(os.environ.get("PRONUNCIADA_CACHE", os.path.expanduser("~/.cache/pronunciada")),
                          "calib", "sound_oof.parquet")
        df = pd.read_parquet(pq, columns=["utt", "phone_index", f"{pre}_z_len", f"{pre}_z_str"])
        oofz = {(u, int(i)): (a + bias, b + bias) for u, i, a, b in
                zip(df.utt.values, df.phone_index.values, df[f"{pre}_z_len"].values, df[f"{pre}_z_str"].values)}
    for rec in load("speechocean762", split=split):
        o = D.get(rec["id"])
        if o is None or "error" in o:
            continue
        X = add_initial(dict(o["sounds"]))
        ref = build_ref("en", rec["text"])
        if ref is None or len(ref.phones) != len(X["expected"]):
            continue
        if oofz is not None:
            zz = [oofz.get((rec["id"], i)) for i in range(len(X["expected"]))]
            if any(z is None for z in zz):
                continue
            X["zl"] = np.array([z[0] for z in zz])
            X["zs"] = np.minimum(np.array([z[1] for z in zz]), X["zl"])
        W = rec["labels"]["words"]
        tp = [[arpa_to_ours(p) for p in w["phones"]] for w in W]
        o2t, _ = align_reference(ref, [w["text"] for w in W], tp)
        mis = {}
        for j, w in enumerate(W):
            for m in w["mispronunciations"]:
                pp = m["pronounced_phone"]
                mis[(j, m["index"])] = pp if pp in ("<DEL>", "<unk>") else fold(arpa_to_ours(pp))
        n = len(X["expected"])
        acc = np.full(n, np.nan)
        pron = [None] * n
        tj, tk = np.full(n, -1), np.full(n, -1)
        for i, (j, k) in o2t.items():
            acc[i] = W[j]["phones_accuracy"][k]
            pron[i] = mis.get((j, k))
            tj[i], tk[i] = j, k
        cols["tj"].extend(tj)
        cols["tk"].extend(tk)
        for k, v in X.items():
            if v is None or k == "phone_vec":      # round 12b: per-sound vectors stay per utterance
                continue
            cols[k].extend(list(v))
        cols["acc"].extend(acc)
        cols["pron"].extend(pron)
        cols["spk"].extend([rec["speaker"]] * n)
        cols["utt"].extend([rec["id"]] * n)
    out = {k: (np.asarray(v) if k not in ("expected", "alts", "cand", "slot_heard", "pron") else v)
           for k, v in cols.items()}
    return out


def fit_logistic(F, y, C=1.0):
    from sklearn.linear_model import LogisticRegression

    mu, sd = F.mean(0), F.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=2000).fit((F - mu) / sd, y)
    return {"mean": mu.tolist(), "scale": sd.tolist(), "coef": m.coef_[0].tolist(),
            "intercept": float(m.intercept_[0])}


def oof(F, y, groups):
    from sklearn.model_selection import GroupKFold

    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(F, y, groups):
        M = fit_logistic(F[tr], y[tr])
        p[te] = Co._logistic(M, F[te])  # noqa: SLF001
    return p


def thr_for_precision(score, y, target, min_n=20):
    """Lowest threshold whose precision (score >= thr) >= target; None if unreachable."""
    o = np.argsort(-score)
    s, yy = score[o], y[o].astype(float)
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    ok = np.flatnonzero((prec >= target) & (np.arange(1, len(yy) + 1) >= min_n))
    if not len(ok):
        return None
    k = ok.max()
    return float(s[k])


def pr_at(score, y, thr):
    f = score >= thr
    return {"threshold": thr, "flagged": int(f.sum()), "precision": float(y[f].mean()) if f.any() else None,
            "recall": float(y[f].sum() / max(y.sum(), 1))}


# ----------------------------------------------------------------------------- heard gate

def heard_rows(R, flag):
    """Flagged sounds labelled with a real substituted phone."""
    idx = [i for i in np.flatnonzero(flag) if R["pron"][i] not in (None, "<DEL>", "<unk>")
           and R["cand"][i] is not None and A.covered(R["pron"][i])]
    return idx


def fit_heard(R, flag):
    idx = heard_rows(R, flag)
    lr = np.array([R["lh"][i] - R["le"][i] for i in idx])
    hp = np.exp(np.array([R["lh"][i] for i in idx]))
    ok = np.array([within1(R["pron"][i], fold(R["cand"][i])) for i in idx])
    ex = np.array([R["pron"][i] == fold(R["cand"][i]) for i in idx])
    cls = np.array([Co.sound_class(R["expected"][i]) for i in idx])
    grid, best = [], None
    for a in (-1e9, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        for b in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
            sel = (lr >= a) & (hp >= b)
            on, per = [], {}
            for c in sorted(set(cls)):
                m = sel & (cls == c)
                per[c] = {"n": int(m.sum()), "precision": float(ok[m].mean()) if m.any() else None}
                if m.sum() >= 15 and ok[m].mean() >= 0.75:
                    on.append(c)
            m = sel & np.isin(cls, on)
            pooled = float(ok[m].mean()) if m.any() else None
            right = int(ok[m].sum())
            row = {"min_log_ratio": a, "min_heard_post": b, "on": on, "per_class": per, "pooled": pooled,
                   "exact": float(ex[m].mean()) if m.any() else None, "named": int(m.sum()), "right": right,
                   "coverage": float(m.sum() / max(len(idx), 1))}
            grid.append(row)
            if pooled is not None and pooled >= 0.75 and (best is None or right > best["right"]):
                best = row
    all_cls = sorted(set(cls) | {"vowel", "stop", "fricative", "affricate", "nasal", "approximant", "flap",
                                 "trill", "implosive", "click"})
    if best is None:
        return {"off_classes": all_cls, "min_log_ratio": 1e9, "min_heard_post": 1.0, "confidence": {}}, grid, None
    conf = {c: round(best["per_class"][c]["precision"], 3) for c in best["on"]}
    conf["all"] = round(best["pooled"], 3)
    H = {"min_log_ratio": best["min_log_ratio"], "min_heard_post": best["min_heard_post"],
         "off_classes": [c for c in all_cls if c not in best["on"]], "confidence": conf}
    return H, grid, best


# ----------------------------------------------------------------------------- Mandarin

def ompal_units(name, split, mdl):
    from bench.data import load
    from bench.eval_ompal import overlap, syllable_char_spans, word_char_spans

    D = load_dump(name)
    rows = []
    for rec in load("ompal", split=split):
        lab = rec["labels"]
        if lab["native"] or lab["word_level_dropped"]:
            continue
        o = D.get(rec["id"])
        if o is None or "error" in o:
            continue
        X = add_initial(dict(o["sounds"]))
        ep = Co.error_prob(X, mdl, "zh")
        spans = word_char_spans(rec["text"], [w["text"] for w in o["words"]])
        syl_span = {}
        for wi, w in enumerate(o["words"]):
            if spans[wi] is not None:
                for k, s in enumerate(syllable_char_spans(spans[wi], w["n_syl"])):
                    syl_span[(wi, k)] = s
        tone = {(s["word"], s["k"]): s["tone_prob"] for s in o["syllables"]}
        for u in lab["units"]:
            if u["span"] is None:
                continue
            a, b = u["span"]
            keys = [kk for kk, s in syl_span.items() if overlap(a, b, s[0], s[1])]
            ini = [ep[i] for i in range(len(ep)) if (X["word"][i], X["syl"][i]) in keys and X["initial"][i]]
            fin = [ep[i] for i in range(len(ep)) if (X["word"][i], X["syl"][i]) in keys and not X["initial"][i]]
            tp = [tone[kk] for kk in keys if tone.get(kk) is not None]
            rows.append({"spk": rec["speaker"], "tone_wrong": u["tone"] == 0, "cons_wrong": u["consonant"] == 0,
                         "vowel_wrong": u["vowel"] == 0, "any": u["any_wrong"] == 1,
                         "ini": max(ini) if ini else np.nan, "fin": max(fin) if fin else np.nan,
                         "tone": min(tp) if tp else np.nan})
    return rows


def zh_fit(rows):
    out, curves = {}, {}
    for key, lab, sign in (("initial", "cons_wrong", 1), ("final", "vowel_wrong", 1), ("tone", "tone_wrong", -1)):
        col = {"initial": "ini", "final": "fin", "tone": "tone"}[key]
        v = np.array([r[col] for r in rows], float)
        y = np.array([r[lab] for r in rows], bool)
        m = np.isfinite(v)
        s = sign * v[m]
        yy = y[m]
        th = {}
        for sens, t in TARGETS.items():
            x = thr_for_precision(s, yy, t)
            th[sens] = None if x is None else sign * x
        curve = []
        for q in np.quantile(s, [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]):
            r = pr_at(s, yy, q)
            r["threshold"] = float(sign * q)
            curve.append(r)
        out[key] = th
        curves[key] = {"curve": curve, "n_units": int(m.sum()), "n_wrong": int(yy.sum()),
                       "at": {k: (pr_at(s, yy, sign * x) if x is not None else None) for k, x in th.items()}}
    return out, curves


def tone_logistic(rows):
    v = np.array([r["tone"] for r in rows], float)
    y = np.array([r["tone_wrong"] for r in rows], float)
    m = np.isfinite(v)
    q = np.clip(v[m], 1e-4, 1 - 1e-4)
    from sklearn.linear_model import LogisticRegression

    L = LogisticRegression(C=10).fit(np.log(q / (1 - q))[:, None], y[m])
    return float(L.intercept_[0]), float(L.coef_[0][0])


# ----------------------------------------------------------------------------- FLEURS native

def fleurs_native(mdl, langs=LANGS):
    """Per language and sensitivity: flagged sounds and wrong diagnoses (a named heard sound or
    "missing") per native clip (every flag on native speech is a false alarm)."""
    D = load_dump("fleurs-val")
    by = defaultdict(list)
    for uid, o in D.items():
        if "error" in o:
            continue
        lang = uid.split("-")[1]
        by[lang].append(add_initial(dict(o["sounds"])))
    out = {}
    for l in langs:
        out[l] = {}
        for sens in TARGETS:
            fl = dg = 0
            for X in by[l]:
                d = Co.decide(X, l, sens, mdl)
                fl += int(d["flag"].sum())
                dg += sum(1 for i in np.flatnonzero(d["flag"]) if d["heard"][i] is not None)
            n = max(len(by[l]), 1)
            out[l][sens] = {"clips": len(by[l]), "flags_per_clip": fl / n, "diagnoses_per_clip": dg / n}
    return out, by


def lang_threshold_for_native(mdl, l, Xs, cap=0.3):
    """Lowest balanced threshold for language l with <= cap wrong diagnoses per native clip."""
    base = Co.sound_threshold(mdl, l, "balanced", False)
    for t in np.r_[base, np.linspace(base, 0.99, 60)[1:]]:
        dg = 0
        for X in Xs:
            d = Co.decide(X, l, "balanced", mdl, threshold=float(t))
            dg += sum(1 for i in np.flatnonzero(d["flag"]) if d["heard"][i] is not None)
        if dg / max(len(Xs), 1) <= cap:
            return float(t)
    return 0.999


def planted_check(mdl):
    """FLEURS validation planted substitutions (round-1 gather): flagged share and heard precision."""
    from bench.coach.fit_gate import load

    rows, _ = load()
    out = {}
    for l in LANGS + ("pooled",):
        R = [r for r in rows if r["kind"] in ("phone_sub_near", "phone_sub_far") and r["truth"]
             and (l == "pooled" or r["lang"] == l)]
        if not R:
            continue
        X = {"zl": np.array([Co_logit(r["p_len"]) for r in R]), "zs": np.array([Co_logit(r["p_str"]) for r in R]),
             "llr_sub": np.array([r["llr_sub"] for r in R]), "z_llr_sub": np.array([r["z_llr_sub"] for r in R]),
             "log_ratio_comp": np.array([r["log_ratio_comp"] for r in R]),
             "log_frames": np.log1p([r["n_frames"] for r in R]), "nucleus": np.array([float(r["nucleus"]) for r in R]),
             "llr_del": np.array([r["llr_del"] for r in R]), "peak_post": np.array([r["peak_post"] for r in R]),
             "blank_mean": np.array([r["blank_mean"] for r in R])}
        ep = Co.error_prob(X, mdl)
        res = {}
        for sens in TARGETS:
            thr = np.array([Co.sound_threshold(mdl, r["lang"], sens, False) for r in R])
            f = ep >= thr
            named = [r for r, ff in zip(R, f) if ff and Co.heard_ok(mdl, r["lang"], r["expected"], r["cand"], r["lh"], r["lh"] - r["lr"])]
            res[sens] = {"flagged": float(f.mean()), "named": len(named) / len(R),
                         "within1": float(np.mean([within1(r["truth"], r["cand"]) for r in named])) if named else None}
        out[l] = res
    return out


def Co_logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# ----------------------------------------------------------------------------- main

def main():
    R = so762_rows("so762-train", "train")
    m = np.isfinite(R["acc"])
    R = {k: (v[m] if isinstance(v, np.ndarray) else [x for x, keep in zip(v, m) if keep]) for k, v in R.items()}
    y = R["acc"] < 2
    F = Co.feature_matrix(R)
    p_oof = oof(F, y, R["spk"])
    from sklearn.metrics import roc_auc_score

    rep = {"n_sounds": int(len(y)), "error_rate": float(y.mean()), "auc_oof": float(roc_auc_score(y, p_oof)),
           "auc_prob_only": float(roc_auc_score(y, -R["zs"]))}
    th = {s: thr_for_precision(p_oof, y, t) for s, t in TARGETS.items()}
    rep["operating_points"] = {s: pr_at(p_oof, y, th[s]) for s in TARGETS}
    # comparison: v1 rule (prob < 0.5) on the same sounds
    pv1 = 1 / (1 + np.exp(-0.5 * (R["zl"] + R["zs"])))
    rep["v1_prob_below_0.5"] = pr_at(-pv1, y, -0.5)
    rep["strict_error_target_acc_below_1.5"] = {s: pr_at(p_oof, R["acc"] < 1.5, th[s]) for s in TARGETS}
    M = fit_logistic(F, y)
    M["features"] = list(Co.SOUND_FEATURES)
    rep["coef"] = dict(zip(Co.SOUND_FEATURES, [round(c, 3) for c in M["coef"]]))
    mdl = {"version": 2, "fitted_on": "speechocean762 train (sounds), OMPAL dev (Mandarin), FLEURS validation (native cap)",
           "sound": M, "thresholds": th, "by_lang": {}}
    # heard gate (balanced OOF flags)
    flag = p_oof >= th["balanced"]
    H, hgrid, hbest = fit_heard(R, flag)
    H["by_lang"] = {"de": {"min_heard_post": max(0.25, H["min_heard_post"])}}
    mdl["heard"] = H
    rep["heard"] = {"best": hbest, "grid": hgrid, "n_labelled_flagged": len(heard_rows(R, flag))}
    # all flagged sounds by label (what kind of thing a flag points at)
    lab = defaultdict(int)
    for i in np.flatnonzero(flag):
        pr = R["pron"][i]
        lab["deleted" if pr == "<DEL>" else "unk" if pr == "<unk>" else "substituted" if pr else
            ("scored<2, no label" if y[i] else "correct")] += 1
    rep["flag_labels_balanced"] = dict(lab)
    # missing sounds
    fi = np.flatnonzero(flag)
    yd = np.array([R["pron"][i] == "<DEL>" for i in fi])
    Xm = {k: (v[fi] if isinstance(v, np.ndarray) else [v[i] for i in fi]) for k, v in R.items()}
    MF = ("zl", "zs", "llr_sub", "llr_del", "log_frames", "blank_mean", "peak_post", "nucleus")
    Fm = Co.feature_matrix(Xm, MF)
    pm = oof(Fm, yd, np.asarray(R["spk"])[fi]) if yd.sum() >= 10 else np.zeros(len(fi))
    tm = thr_for_precision(pm, yd, 0.7)
    rep["missing"] = {"n_flagged": int(len(fi)), "n_deleted": int(yd.sum()),
                      "auc_oof": float(roc_auc_score(yd, pm)) if 0 < yd.sum() < len(yd) else None,
                      "threshold_p70": tm, "at": pr_at(pm, yd, tm) if tm is not None else None,
                      "best_precision": float(max(np.cumsum(yd[np.argsort(-pm)])[19:] / np.arange(20, len(yd) + 1)))
                      if len(yd) > 20 else None}
    if tm is not None:
        Mm = fit_logistic(Fm, yd)
        Mm.update(features=list(MF), threshold=tm, confidence=round(rep["missing"]["at"]["precision"], 3))
        mdl["missing"] = Mm
    else:
        mdl["missing"] = {"threshold": None}
    # Mandarin (OMPAL dev)
    zr = ompal_units("ompal-dev", "dev", mdl)
    zth, zcurves = zh_fit(zr)
    big = 1.01
    mdl["by_lang"]["zh"] = {"initial": {s: (zth["initial"][s] if zth["initial"][s] is not None else big) for s in TARGETS},
                            "final": {s: (zth["final"][s] if zth["final"][s] is not None else big) for s in TARGETS}}
    a, b = tone_logistic(zr)
    mdl["tone"] = {"thresholds": {s: (zth["tone"][s] if zth["tone"][s] is not None else -1.0) for s in TARGETS},
                   "a": a, "b": b}
    rep["zh"] = {"thresholds": zth, "curves": zcurves, "n_units": len(zr)}
    # native cap (FLEURS validation clean): other languages
    nat, by = fleurs_native(mdl)
    rep["native_before_cap"] = nat
    for l in LANGS:
        if nat[l]["balanced"]["diagnoses_per_clip"] <= 0.3:
            continue
        # tighten the heard-sound bar of that language first (flags stay; fewer named sounds)
        hb = mdl["heard"]["by_lang"].setdefault(l, {})
        for hp in np.arange(max(mdl["heard"]["min_heard_post"], hb.get("min_heard_post", 0)), 1.0001, 0.025):
            hb["min_heard_post"] = round(float(hp), 3)
            dg = 0
            for X in by[l]:
                d = Co.decide(X, l, "balanced", mdl)
                dg += sum(1 for i in np.flatnonzero(d["flag"]) if d["heard"][i] is not None)
            if dg / max(len(by[l]), 1) <= 0.3:
                break
    nat2, _ = fleurs_native(mdl)
    rep["native"] = nat2
    rep["planted"] = planted_check(mdl)
    Path("pronunciada/data/coach/model.json").write_text(json.dumps(mdl, indent=1))
    Path("results/tables/coaching_v2_fit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: rep[k] for k in ("n_sounds", "error_rate", "auc_oof", "auc_prob_only", "operating_points",
                                           "v1_prob_below_0.5", "coef", "flag_labels_balanced", "missing")}, indent=1, default=float))
    print("heard", json.dumps({k: hbest[k] for k in ("min_log_ratio", "min_heard_post", "on", "pooled", "exact", "named", "coverage")} if hbest else None, default=float))
    print("heard per class", json.dumps(hbest["per_class"] if hbest else None))
    print("zh", json.dumps(zth), json.dumps({k: v["at"] for k, v in zcurves.items()}, default=float))
    print("native", json.dumps({l: {s: round(v[s]["diagnoses_per_clip"], 3) for s in v} for l, v in nat2.items()}))
    print("native flags", json.dumps({l: {s: round(v[s]["flags_per_clip"], 2) for s in v} for l, v in nat2.items()}))
    print("by_lang", json.dumps(mdl["by_lang"]))
    print("planted", json.dumps(rep["planted"], default=float))


def fit_stress():
    """Re-fit the stress heard-syllable bar for the current stress model (FLEURS validation planted
    stress moves, bench/coach/gather.py --only stress); updates gate.json stress_* only."""
    import pickle

    from bench.coach.fit_gate import add_word_channel, per_lang_pros, pros_curve
    from bench.coach.gather import out_dir
    from pronunciada.prosody import selected_model

    pros = []
    for l in ("en", "de", "es", "it", "pt", "ru"):
        p = out_dir() / f"gather-stress-{l}.pkl"
        if p.exists():
            pros += [dict(x, lang=l) for x in pickle.loads(p.read_bytes())["pros"]]
    add_word_channel(pros)
    pick, curve, met = pros_curve(pros, "stress", (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    gp = Path("pronunciada/data/coach/gate.json")
    g = json.loads(gp.read_text())
    g["thresholds"]["stress_min_post"] = pick["bar"]
    g["thresholds"]["stress_confidence"] = round(pick["planted"]["precision"] or 0, 3)
    g["stress_model"] = selected_model("en")
    gp.write_text(json.dumps(g, indent=1))
    rep = {"stress_model": selected_model("en"), "curve": curve, "target_met": met,
           "by_lang": per_lang_pros(pros, "stress", pick["bar"])}
    Path("results/tables/coaching_stress.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"pick": pick, "met": met}, default=float))


if __name__ == "__main__":
    import sys

    fit_stress() if "--stress" in sys.argv else main()
