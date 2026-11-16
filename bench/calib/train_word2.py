"""Round 2 word / sentence calibrators (designed AFTER the first held-out evaluation; all fitting
and selection below uses train splits and FLEURS validation only).

    python -m bench.calib.train_word2

Word acceptance = "would a rater accept every sound of this word?"
  speechocean762 train: lenient ok = every aligned phone score >= 1.0 AND word accuracy >= 5;
                        strict ok  = every aligned phone score >= 1.8 AND word accuracy >= 8
  l2arctic / epadb train: word with any annotated substitution / deletion -> not strict-ok,
                        all annotated phones correct -> strict-ok (no lenient label)
  FLEURS validation (10 languages, controlled errors, all 5 kinds): word containing a corrupted
                        sound -> not ok (both heads); untouched words of corrupted references and
                        words of the correct references -> ok; synthetic weight <= 40 % per head
Inputs: pronunciada.features.WORD2_NUM (min / second-min of both sound log-odds, mean, share below
0.5, count, llr sums / minima, spill-over context of the previous / next word and differences).
Sound inputs: out-of-fold universal-model predictions on speechocean762 train; the shipped
universal model (+ language bias) elsewhere (it never saw those corpora).

Candidates: old (round-1 word model, its speechocean762 OOF), so762, real (+ l2arctic / epadb),
full (+ FLEURS synthetic); hyper-parameters s1 / s2 / s3. Metrics: speechocean762 speaker CV
(other corpora always in training) AUC / ECE of the new labels; FLEURS-validation utterance CV
(candidates without FLEURS: trained on everything else) controlled word AUC, hit @ 5 % native
flags, lowest-word rate, native acceptance at s = 0.5. Rule: among configurations with FLEURS
native word acceptance >= 0.97, maximise mean(FLEURS controlled word AUC, so762 strict AUC).

Then: word-accuracy regression (inputs WORD_NUM, optionally + the new word log-odds; kept only if
CV Pearson improves), and the sentence heads (SENT_NUM vs SENT2_NUM = + min / second-min of the
new word log-odds; strict ok = sentence accuracy >= 8 AND every word lenient-ok).
Writes pronunciada/data/calib/word.npz, word_reg.npz, sentence.npz, results/tables/calib_round2.json.
"""
from __future__ import annotations

import json

import numpy as np

from bench.calib.common import (SMALL_GRID, apply_calib, calib_cache, choose_calib, fit, load_table,
                                raw, save_json, ship_dir, speaker_folds, table_columns)
from bench.calib.fitlib import calibrate_oof, cv_heads, fit_final, head_metrics, log
from pronunciada import features as F
from pronunciada.calibrate import LevelModel, blend_logit, config, logit, save_level

SYN_SHARE = 0.40
FLEURS_UTTS = 120
WL, WS = (1.0, 5.0), (1.8, 8.0)        # (phone threshold, word-accuracy threshold)


def u_outputs(t, lang_bias=None, chunk=20000):
    U = LevelModel(ship_dir() / "sound_U.npz")
    n = len(t["llr_sub"])
    zl, zs, rg = np.empty(n), np.empty(n), np.empty(n)
    X = U.matrix(t)
    for a in range(0, n, chunk):
        o = U.logits(X=X[a:a + chunk])
        zl[a:a + chunk], zs[a:a + chunk], rg[a:a + chunk] = o["z_len"], o["z_str"], o["reg"]
    if lang_bias is not None:
        zl, zs = zl + lang_bias, zs + lang_bias
    return zl, zs, rg


def _wfirst(W, arr):
    return np.asarray(arr)[W["row"]]


def _wgroup_reduce(t, W, x, how):
    """Reduce a per-phone array over each word (rows of a word are contiguous)."""
    n = len(t["llr_sub"])
    bounds = np.r_[W["row"], n]
    out = np.full(len(W["row"]), np.nan)
    for k in range(len(W["row"])):
        v = x[bounds[k]:bounds[k + 1]]
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = how(v)
    return out


def build():
    import pandas as pd

    from bench.pipeline.common import feat_dir

    parts = {}
    # ---- speechocean762 train (OOF universal sound outputs)
    df = load_table("speechocean762", "train")
    t = table_columns(df)
    od = pd.read_parquet(calib_cache() / "sound_oof.parquet")
    od = df[["utt", "phone_index"]].astype({"phone_index": int}).merge(od, on=["utt", "phone_index"], how="left")
    sl, ss, sr = od["U_z_len"].to_numpy(float), od["U_z_str"].to_numpy(float), np.clip(od["U_reg_raw"].to_numpy(float), 0, 2)
    W = F.word_table(t, sl, ss, sr, group=t["_group"])
    ps = np.asarray(t["phone_score"], float)
    pmin = _wgroup_reduce(t, W, ps, np.min)
    acc = _wfirst(W, t["word_accuracy"]).astype(float)
    ok_ph = lambda thr: np.where(np.isfinite(pmin), pmin >= thr, True)  # noqa: E731
    W["y_len"] = np.where(np.isfinite(acc), (ok_ph(WL[0]) & (acc >= WL[1])).astype(float), np.nan)
    W["y_str"] = np.where(np.isfinite(acc), (ok_ph(WS[0]) & (acc >= WS[1])).astype(float), np.nan)
    W["y_reg"] = acc
    W["grp"] = _wfirst(W, df.speaker.astype(str).values)
    W["utt"] = _wfirst(W, df.utt.values)
    W["corpus"] = np.array(["so762"] * len(acc))
    W["kind"] = np.array(["prompt"] * len(acc))
    W["lang"] = np.array(["en"] * len(acc))
    W["wrong"] = np.zeros(len(acc), bool)
    parts["so762"] = (W, t, (sl, ss, sr))
    # ---- l2arctic / epadb train
    for ds in ("l2arctic", "epadb"):
        d = load_table(ds, "train")
        tt = table_columns(d)
        a, b, c = u_outputs(tt)
        Wd = F.word_table(tt, a, b, c, group=tt["_group"])
        e = np.asarray(tt["label_err"], float)
        emax = _wgroup_reduce(tt, Wd, e, np.max)
        Wd["y_len"] = np.full(len(emax), np.nan)
        Wd["y_str"] = np.where(np.isfinite(emax), 1.0 - emax, np.nan)
        Wd["y_reg"] = np.full(len(emax), np.nan)
        Wd["grp"] = _wfirst(Wd, (ds + ":" + d.speaker.astype(str)).values)
        Wd["utt"] = _wfirst(Wd, d.utt.values)
        Wd["corpus"] = np.array([ds] * len(emax))
        Wd["kind"] = np.array(["prompt"] * len(emax))
        Wd["lang"] = np.array(["en"] * len(emax))
        Wd["wrong"] = emax == 1
        parts[ds] = (Wd, tt, None)
    # ---- FLEURS validation controlled errors
    fv = pd.read_parquet(feat_dir() / "fleurs-validation.parquet")
    rng = np.random.default_rng(0)
    keep = []
    for lang, g in fv.groupby("lang"):
        u = g.utt.unique()
        keep += list(rng.choice(u, min(FLEURS_UTTS, len(u)), replace=False))
    fv = fv[fv.utt.isin(set(keep))].sort_values(["utt", "ref_kind", "phone_index"], kind="stable").reset_index(drop=True)
    tt = table_columns(fv)
    bias = config()["bias"]
    lb = np.array([bias.get(x, 0.0) for x in fv.lang.values])
    a, b, c = u_outputs(tt, lb)
    Wf = F.word_table(tt, a, b, c, group=tt["_group"])
    wrong = _wfirst(Wf, fv.word_wrong.values).astype(bool)
    Wf["y_len"] = (~wrong).astype(float)
    Wf["y_str"] = (~wrong).astype(float)
    Wf["y_reg"] = np.full(len(wrong), np.nan)
    Wf["grp"] = _wfirst(Wf, ("fleurs:" + fv.utt.astype(str)).values)
    Wf["utt"] = _wfirst(Wf, (fv.utt.astype(str) + "|" + fv.ref_kind.astype(str)).values)
    Wf["corpus"] = np.array(["fleurs"] * len(wrong))
    Wf["kind"] = _wfirst(Wf, fv.ref_kind.astype(str).values)
    Wf["lang"] = _wfirst(Wf, fv.lang.astype(str).values)
    Wf["wrong"] = wrong
    parts["fleurs"] = (Wf, tt, None)
    return parts


KEYS = ["y_len", "y_str", "y_reg", "grp", "utt", "corpus", "kind", "lang", "wrong"]


def stack(parts, cols):
    T = {}
    for k in cols + KEYS:
        T[k] = np.concatenate([np.asarray(parts[c][0][k]) for c in ("so762", "l2arctic", "epadb", "fleurs")])
    return T


def weights(T, rows, head):
    y = T[f"y_{head}"][rows]
    lab = np.isfinite(y)
    syn = (T["corpus"][rows] == "fleurs") & lab
    real = lab & ~syn
    w = np.ones(len(rows))
    if syn.any() and real.any():
        w[syn] = SYN_SHARE / (1 - SYN_SHARE) * real.sum() / syn.sum()
    return w


CANDS = {"so762": ("so762",), "real": ("so762", "l2arctic", "epadb"),
         "full": ("so762", "l2arctic", "epadb", "fleurs")}


def fleurs_metrics(T, fl, zl, zs):
    """Controlled-error metrics on FLEURS-validation words (s = 0.5 blend)."""
    from bench.metrics import roc_auc

    zl, zs = np.asarray(zl)[fl], np.asarray(zs)[fl]
    p = 1 / (1 + np.exp(-blend_logit(zl, np.minimum(zs, zl), 0.5)))
    kind = T["kind"][fl]
    wrong = T["wrong"][fl]
    utt = T["utt"][fl]
    clean = kind == "clean"
    thr = np.quantile(p[clean], 0.05)
    cor = ~clean
    res = {"native_accept": float(np.mean(p[clean] >= 0.5)),
           "auc": roc_auc(wrong[cor].astype(int), 1 - p[cor]),
           "hit_at_5pct": float(np.mean(p[cor & wrong] < thr))}
    low = n = 0
    for u in np.unique(utt[cor]):
        m = utt == u
        if wrong[m].any() and (~wrong[m]).any():
            n += 1
            low += bool(wrong[m][int(np.argmin(p[m]))])
    res["lowest_word"] = low / max(n, 1)
    res["by_lang_auc"] = {l: roc_auc(wrong[cor & (T["lang"][fl] == l)].astype(int), 1 - p[cor & (T["lang"][fl] == l)])
                          for l in np.unique(T["lang"][fl])}
    return res


def evaluate(T, X, cands, params):
    corp = T["corpus"]
    so = np.flatnonzero(corp == "so762")
    fl = np.flatnonzero(corp == "fleurs")
    so_folds = speaker_folds(T["grp"][so])
    from sklearn.model_selection import GroupKFold

    fl_folds = list(GroupKFold(5).split(fl, None, T["grp"][fl]))
    inc = np.isin(corp, cands)
    z = {h: np.full(len(corp), np.nan) for h in ("len", "str")}
    for h in ("len", "str"):
        y = T[f"y_{h}"]
        for tr, te in so_folds:
            hold = np.zeros(len(corp), bool)
            hold[so[te]] = True
            m = np.flatnonzero(inc & ~hold & np.isfinite(y))
            mdl = fit("prob", params, X[m], y[m], [False] * X.shape[1], weights(T, m, h))
            z[h][so[te]] = raw(mdl, X[so[te]])
        if "fleurs" in cands:
            for tr, te in fl_folds:
                hold = np.zeros(len(corp), bool)
                hold[fl[te]] = True
                m = np.flatnonzero(inc & ~hold & np.isfinite(y))
                mdl = fit("prob", params, X[m], y[m], [False] * X.shape[1], weights(T, m, h))
                z[h][fl[te]] = raw(mdl, X[fl[te]])
        else:
            m = np.flatnonzero(inc & np.isfinite(y))
            mdl = fit("prob", params, X[m], y[m], [False] * X.shape[1], weights(T, m, h))
            z[h][fl] = raw(mdl, X[fl])
    return z


def calibrated(T, z):
    """Calibration maps on so762 OOF (human target) applied everywhere; returns logits + maps."""
    so = T["corpus"] == "so762"
    out, maps = {}, {}
    for h in ("len", "str"):
        y = T[f"y_{h}"]
        ok = so & np.isfinite(y) & np.isfinite(z[h])
        c, _ = choose_calib(z[h][ok], y[ok], T["grp"][ok])
        p = apply_calib(c, z[h])
        out[h], maps[h] = logit(p), c
    out["str"] = np.minimum(out["str"], out["len"])
    return out, maps


def so_metrics(T, zc):
    so = T["corpus"] == "so762"
    r = {}
    for h in ("len", "str"):
        y = T[f"y_{h}"][so]
        r[h] = head_metrics("prob", y, 1 / (1 + np.exp(-zc[h][so])))
    return r


def main():
    import pandas as pd

    parts = build()
    cols = sorted(set(F.WORD2_NUM) | set(F.WORD_NUM))
    T = stack(parts, cols)
    corp = T["corpus"]
    out = {"n_words": {c: int((corp == c).sum()) for c in CANDS["full"]}, "syn_share": SYN_SHARE,
           "fleurs_utts_per_lang": FLEURS_UTTS, "thresholds": {"len": WL, "str": WS}}
    so = corp == "so762"
    for h in ("len", "str"):
        y = T[f"y_{h}"]
        out[f"balance_{h}"] = {c: float(np.nanmean(y[corp == c])) for c in CANDS["full"]
                               if np.isfinite(y[corp == c]).any()}
    log("words", out["n_words"], "balance", out["balance_len"], out["balance_str"])
    X2 = np.column_stack([T[c].astype(float) for c in F.WORD2_NUM])
    # ---- baseline: round-1 word model (old labels/inputs), so762 OOF + FLEURS-val predictions
    old = LevelModel(calib_cache() / "round1_shipped" / "word.npz")
    Xo = np.column_stack([T[c].astype(float) for c in old.num])
    zo = old.logits(X=Xo)
    wo = pd.read_parquet(calib_cache() / "word_oof.parquet")
    key = dict(zip(zip(wo.utt, wo.word_index), zip(wo.z_len, wo.z_str)))
    so_idx = np.flatnonzero(so)
    W0 = parts["so762"][0]
    zl0, zs0 = zo["z_len"].copy(), zo["z_str"].copy()
    for k, i in enumerate(so_idx):
        a = key.get((W0["utt"][k], int(W0["word_index"][k])))
        if a:
            zl0[i], zs0[i] = a
    base = {"so762": so_metrics(T, {"len": zl0, "str": np.minimum(zs0, zl0)}),
            "fleurs_val": fleurs_metrics(T, np.flatnonzero(corp == "fleurs"), zl0, zs0)}
    out["configs"] = {"old (round 1)": base}
    log("old", base["so762"]["str"]["auc"], {k: v for k, v in base["fleurs_val"].items() if k != "by_lang_auc"})
    best, best_score, store = None, -1, {}
    for cname, cands in CANDS.items():
        for pname in ("s1", "s2", "s3"):
            z = evaluate(T, X2, cands, SMALL_GRID[pname])
            zc, maps = calibrated(T, z)
            r = {"so762": so_metrics(T, zc), "fleurs_val": fleurs_metrics(T, np.flatnonzero(corp == "fleurs"), zc["len"], zc["str"])}
            r["score"] = 0.5 * (r["fleurs_val"]["auc"] + r["so762"]["str"]["auc"])
            r["eligible"] = r["fleurs_val"]["native_accept"] >= 0.97
            name = f"{cname}/{pname}"
            out["configs"][name] = r
            store[name] = (zc, maps)
            log(name, "so762 str auc", round(r["so762"]["str"]["auc"], 4), "fleurs",
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["fleurs_val"].items() if k != "by_lang_auc"},
                "score", round(r["score"], 4))
            if r["eligible"] and r["score"] > best_score:
                best, best_score = name, r["score"]
            save_json("calib_round2.json", out)
    out["chosen_word"] = best
    log("chosen", best)
    cname, pname = best.split("/")
    cands, params = CANDS[cname], SMALL_GRID[pname]
    zc, maps = store[best]
    # ---- final word model (len / str)
    rows = np.flatnonzero(np.isin(corp, cands))
    labels = {h: T[f"y_{h}"][rows] for h in ("len", "str")}
    wts = {h: weights(T, rows, h) for h in ("len", "str")}
    heads, parity = fit_final(X2[rows], labels, {"len": "prob", "str": "prob"}, params, [False] * X2.shape[1], wts)
    for h in ("len", "str"):
        heads[h]["calib"] = maps[h]
    save_level(ship_dir() / "word.npz", F.WORD2_NUM, [], {},
               {h: {k: v for k, v in hh.items() if k != "_model"} for h, hh in heads.items()},
               {"level": "word", "round": 2, "trained_on": "+".join(cands),
                "definition": "every sound acceptable", "thresholds": {"len": WL, "str": WS}})
    out["word_final"] = {"walker_parity_max_abs": parity, "kb": (ship_dir() / "word.npz").stat().st_size / 1024,
                         "calib": {h: maps[h]["kind"] for h in maps}}
    # ---- word-accuracy regression (so762): old inputs vs + new word log-odds (OOF)
    sidx = np.flatnonzero(so)
    yreg = T["y_reg"][sidx]
    g = T["grp"][sidx]
    Xr_old = np.column_stack([T[c].astype(float)[sidx] for c in F.WORD_NUM])
    Xr_new = np.column_stack([Xr_old, zc["len"][sidx], zc["str"][sidx]])
    reg_res = {}
    oofs = {}
    for nm, Xr in (("old_inputs", Xr_old), ("plus_word_logodds", Xr_new)):
        o, _ = cv_heads(Xr, {"reg": yreg}, {"reg": "reg"}, g, SMALL_GRID["s2"], [False] * Xr.shape[1])
        reg_res[nm] = head_metrics("reg", yreg, o["reg"])
        oofs[nm] = np.clip(o["reg"], 0, 10)
        log("word reg", nm, round(reg_res[nm]["pearson"], 4))
    choice = "plus_word_logodds" if reg_res["plus_word_logodds"]["pearson"] > reg_res["old_inputs"]["pearson"] else "old_inputs"
    out["word_reg"] = {"cv": reg_res, "chosen": choice}
    Xr = Xr_new if choice == "plus_word_logodds" else Xr_old
    num_r = F.WORD_NUM + (["word_z_len", "word_z_str"] if choice == "plus_word_logodds" else [])
    rh, rpar = fit_final(Xr, {"reg": yreg}, {"reg": "reg"}, SMALL_GRID["s2"], [False] * Xr.shape[1], clips={"reg": (0.0, 10.0)})
    save_level(ship_dir() / "word_reg.npz", num_r, [], {},
               {h: {k: v for k, v in hh.items() if k != "_model"} for h, hh in rh.items()},
               {"level": "word_reg", "round": 2})
    out["word_reg"]["walker_parity_max_abs"] = rpar
    save_json("calib_round2.json", out)
    # ---- sentence (so762): SENT_NUM vs SENT2_NUM, new strict label
    Wso, tso, (sl, ss, sr) = parts["so762"]
    wzl, wzs, wreg = zc["len"][sidx], zc["str"][sidx], oofs[choice]
    gc, ng = F._codes(np.asarray(tso["_group"]).astype(object))
    firsts = np.r_[0, np.flatnonzero(np.diff(gc)) + 1]
    dur = np.asarray(tso["utt_dur"], float)[firsts]
    S = F.sentence_table(tso, sl, ss, sr, Wso, wzl, wzs, wreg, dur, group=tso["_group"])
    sspk = np.asarray(tso["speaker"])[firsts] if isinstance(tso["speaker"], np.ndarray) else np.array(tso["speaker"])[firsts]
    acc = np.asarray(tso["sent_accuracy"], float)[firsts]
    gw = np.asarray(Wso["group"])
    yl = T["y_len"][sidx]
    all_len_ok = np.array([np.all(np.nan_to_num(yl[gw == k], nan=1.0) == 1) for k in range(ng)])
    slab = {f"reg_{k}": np.asarray(tso[f"sent_{k}"], float)[firsts] for k in ("accuracy", "fluency", "prosodic", "total")}
    slab["len"] = (acc >= 6).astype(float)
    slab["str"] = ((acc >= 8) & all_len_ok).astype(float)
    out["sentence_balance"] = {"len": float(slab["len"].mean()), "str_round2": float(slab["str"].mean()),
                               "str_round1": float((acc >= 8).mean())}
    skinds = {**{f"reg_{k}": "reg" for k in ("accuracy", "fluency", "prosodic", "total")}, "len": "prob", "str": "prob"}
    sres = {}
    for nm, cols_s in (("SENT_NUM", F.SENT_NUM), ("SENT2_NUM", F.SENT2_NUM)):
        XS = np.column_stack([np.asarray(S[c], float) for c in cols_s])
        o, _ = cv_heads(XS, slab, skinds, sspk, SMALL_GRID["s2"], [False] * XS.shape[1])
        cal, cmp, pcv = calibrate_oof(o, slab, skinds, sspk)
        sres[nm] = {h: head_metrics("reg", slab[h], o[h]) if k == "reg" else head_metrics("prob", slab[h], pcv[h])
                    for h, k in skinds.items()}
        log("sentence", nm, {h: round(v.get("pearson", v.get("auc")), 4) for h, v in sres[nm].items()})
    score = lambda r: np.mean([r["reg_accuracy"]["pearson"], r["len"]["auc"], r["str"]["auc"]])  # noqa: E731
    sch = "SENT2_NUM" if score(sres["SENT2_NUM"]) >= score(sres["SENT_NUM"]) else "SENT_NUM"
    out["sentence"] = {"cv": sres, "chosen": sch}
    cols_s = F.SENT2_NUM if sch == "SENT2_NUM" else F.SENT_NUM
    XS = np.column_stack([np.asarray(S[c], float) for c in cols_s])
    o, _ = cv_heads(XS, slab, skinds, sspk, SMALL_GRID["s2"], [False] * XS.shape[1])
    cal, cmp, pcv = calibrate_oof(o, slab, skinds, sspk)
    heads, spar = fit_final(XS, slab, skinds, SMALL_GRID["s2"], [False] * XS.shape[1],
                            clips={f"reg_{k}": (0.0, 10.0) for k in ("accuracy", "fluency", "prosodic", "total")})
    for h in cal:
        heads[h]["calib"] = cal[h]
    save_level(ship_dir() / "sentence.npz", cols_s, [], {},
               {h: {k: v for k, v in hh.items() if k != "_model"} for h, hh in heads.items()},
               {"level": "sentence", "round": 2, "thresholds": {"len": "accuracy >= 6",
                                                                 "str": "accuracy >= 8 and every word lenient-ok"}})
    out["sentence"]["walker_parity_max_abs"] = spar
    save_json("calib_round2.json", out)
    log("done", json.dumps({"word": best, "word_reg": choice, "sentence": sch}))


if __name__ == "__main__":
    main()
