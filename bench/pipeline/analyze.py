"""Part 5: analysis tables and results/phase2_evidence.md.

    python -m bench.pipeline.analyze [--lang all] [--steps so762_adapter,calib,errdet,ctl,report]

Uses only: FLEURS validation (dev), speechocean762 TRAIN, l2arctic TRAIN, epadb TRAIN.
Held-out splits (FLEURS test, speechocean762 / l2arctic / epadb test) are never read here.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from bench.pipeline.common import (build_ref, evidence_arrays, feat_dir, load_lp, log, parse_langs,
                                   pmap, records, repo_root)

TAB = None


def tables_dir():
    d = repo_root() / "results" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(name, obj):
    (tables_dir() / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))


def load_json(name):
    p = tables_dir() / name
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- Part 2 English check

def _so_utt(args):
    rec, specs = args
    from bench.pipeline.evalctl import get_adapter
    from bench.pipeline.labels import align_reference, arpa_to_ours

    ref = build_ref("en", rec["text"])
    lp = load_lp("speechocean762", rec)
    if ref is None or lp is None:
        return []
    W = rec["labels"]["words"]
    o2t, _ = align_reference(ref, [w["text"] for w in W], [[arpa_to_ours(p) for p in w["phones"]] for w in W])
    if not o2t:
        return []
    keys = sorted(o2t)
    out = {k: [] for k in keys}
    for name, spec in specs:
        a = get_adapter(spec)
        s = evidence_arrays(lp if a is None else a(lp), ref)["llr_sub"]
        med = float(np.median(s))
        for k in keys:
            out[k] += [float(s[k]), float(s[k] - med)]
    rows = []
    for k in keys:
        j, q = o2t[k]
        rows.append((rec["id"], rec["speaker"], float(W[j]["phones_accuracy"][q]), *out[k]))
    return rows


def so762_adapter_check(workers=6):
    """speechocean762 TRAIN: correlation of llr_sub with the human phone score, raw vs each
    English adapter candidate (and the shipped one)."""
    import pandas as pd

    from bench.metrics import bootstrap_ci, pearson, spearman
    from bench.pipeline.common import g2p_version, pipe_dir

    ver = g2p_version("en")
    specs = [("raw", None)]
    for k in ("linear", "ctx", "conv"):
        p = pipe_dir("adapters", "en") / f"{k}-{ver}.npz"
        if p.exists():
            specs.append((k, str(p)))
    recs = records("speechocean762", split="train")
    res = pmap(_so_utt, [(r, specs) for r in recs], workers=workers, chunksize=8, desc="so762 adapters")
    cols = ["utt", "speaker", "score"] + [f"{n}_{v}" for n, _ in specs for v in ("llr", "med")]
    df = pd.DataFrame([x for r in res for x in r], columns=cols)
    ad = load_json("adapter_en.json")
    out = {"n_phones": len(df), "n_utts": int(df.utt.nunique()), "shipped": ad["choice"] if ad else None,
           "by": {}}
    for n, _ in specs:
        for v in ("llr", "med"):
            x = df[f"{n}_{v}"].values
            ci = bootstrap_ci(pearson, x, df.score.values, clusters=df.speaker.values, n_boot=300)
            out["by"][f"{n}_{v}"] = {"pearson": pearson(x, df.score), "spearman": spearman(x, df.score),
                                    "pearson_ci": [ci["lo"], ci["hi"]]}
    # paired: candidate - raw (speaker bootstrap)
    from bench.metrics import paired_bootstrap

    for n, _ in specs[1:]:
        b = paired_bootstrap(lambda y, s: pearson(s, y), df.score.values, df[f"{n}_llr"].values,
                             df["raw_llr"].values, clusters=df.speaker.values, n_boot=500)
        out["by"][f"{n}_llr"]["minus_raw"] = {"delta": b["delta"], "lo": b["lo"], "hi": b["hi"]}
    save("so762_adapter.json", out)
    log("so762 adapter check: " + ", ".join(f"{k} {v['pearson']:.3f}" for k, v in out["by"].items()))
    return out


# ----------------------------------------------------------------------------- calibration CV

NUM = ["llr_sub", "llr_del", "n_frames", "peak_post", "alt_nonzero", "z_llr_sub", "z_llr_del",
       "z_n_frames", "utt_mean_llr_sub", "utt_median_llr_sub", "utt_dur", "phones_per_s",
       "llr_sub_per_frame", "llr_sub_rel", "is_nucleus_f", "stress_f",
       # span posterior / competitor features (feature_version 2)
       "can_mean", "can_max", "comp_mean", "comp_max", "log_ratio_comp", "ent_nonblank",
       "blank_mean", "comp_dist", "span_frames_padded", "has_last", "last_can_mean",
       "last_can_max", "last_comp_mean", "last_comp_max", "last_log_ratio_comp",
       # neighbour / word context
       "prev_llr_sub", "next_llr_sub", "word_mean_llr_sub", "word_min_llr_sub"]
CAT = ["cls", "position", "comp_cls", "phone", "comp_tok"]
ID_CAT = ("phone", "comp_tok")          # dropped for the *_no_phone_id models


def _prep_X(df):
    df = df.copy()
    df["alt_nonzero"] = (df.alt_index != 0).astype(float)
    df["llr_sub_rel"] = df.llr_sub - df.utt_median_llr_sub
    df["is_nucleus_f"] = df.is_nucleus.astype(float)
    df["stress_f"] = df.stress.astype(float).fillna(-1)
    df["has_last"] = df.last_can_mean.notna().astype(float)
    for c in NUM:
        df[c] = df[c].astype(float)            # NaN kept (GBM handles it; linear fills 0)
    for c in CAT:
        df[c] = df[c].astype(object).where(df[c].notna(), "none").astype(str)
    return df


def _cats(kind):
    return CAT if kind == "gbm" else [c for c in CAT if c not in ID_CAT]


def _fit_predict(kind, task, dtr, ytr, dte, return_model=False):
    """Fit on (dtr, ytr), predict dte. kind: linear | gbm | gbm_nophone | phone_prior.
    task: 'score' (0-2, ordinal for linear) or 'err' (0/1)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    if kind == "phone_prior":
        m = pd.Series(ytr, index=dtr.index).groupby(dtr.phone.values).mean()
        return dte.phone.map(m).fillna(float(np.mean(ytr))).values
    if kind == "linear":
        Xtr = np.nan_to_num(dtr[NUM].values.astype(float))
        Xte = np.nan_to_num(dte[NUM].values.astype(float))
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))  # noqa: E731
        if task == "score":
            yr = np.rint(ytr)
            e = np.zeros(len(dte))
            for thr in (1, 2):
                m = mk().fit(Xtr, (yr >= thr).astype(int))
                e += m.predict_proba(Xte)[:, 1]
            return e
        m = mk().fit(Xtr, ytr.astype(int))
        return m.predict_proba(Xte)[:, 1]
    cats = _cats(kind)
    cols = NUM + cats
    both = pd.concat([dtr[cols], dte[cols]], ignore_index=True)
    for c in cats:
        both[c] = both[c].astype("category")
    Xtr, Xte = both.iloc[:len(dtr)], both.iloc[len(dtr):]
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
              categorical_features="from_dtype", random_state=0)
    if task == "score":
        m = HistGradientBoostingRegressor(**kw).fit(Xtr, ytr)
        p = m.predict(Xte)
    else:
        m = HistGradientBoostingClassifier(**kw).fit(Xtr, ytr.astype(int))
        p = m.predict_proba(Xte)[:, 1]
    return (p, m, Xte) if return_model else p


def _cv_predict(df, y, kind, groups, task, importances=None):
    """Out-of-fold predictions, 5-fold GroupKFold. If `importances` is a list, permutation
    importances on each held-out fold are appended (gbm kinds only)."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GroupKFold

    from bench.metrics import pearson

    pred = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(df, y, groups):
        dtr, dte = df.iloc[tr], df.iloc[te]
        if importances is not None and kind.startswith("gbm"):
            p, m, Xte = _fit_predict(kind, task, dtr, y[tr], dte, return_model=True)
            scorer = "roc_auc" if task == "err" else make_scorer(lambda a, b: pearson(b, a))
            sub = np.random.default_rng(0).permutation(len(te))[:20000]
            r = permutation_importance(m, Xte.iloc[sub], y[te][sub], scoring=scorer, n_repeats=3,
                                       random_state=0, n_jobs=1)
            importances.append(dict(zip(Xte.columns, r.importances_mean)))
        else:
            p = _fit_predict(kind, task, dtr, y[tr], dte)
        pred[te] = p
    return pred


def _top_importances(imps, k=15):
    import pandas as pd

    m = pd.DataFrame(imps).mean().sort_values(ascending=False)
    s = pd.DataFrame(imps).std()
    return [[f, float(m[f]), float(s[f])] for f in m.index[:k]]


def _prompts(df):
    """Prompt text per row (for a prompt-grouped CV check: same sentence never in train and test fold)."""
    txt = {}
    for ds in df.dataset.unique():
        for r in records(ds):
            txt[r["id"]] = r["text"].lower().strip()
    return df.utt.map(txt).values


def _so762_train(with_labels=True):
    import pandas as pd

    df = pd.read_parquet(feat_dir() / "speechocean762.parquet")
    df = df[df.split == "train"]
    if with_labels:
        df = df[df.phone_score.notna()]
    return _prep_X(df.reset_index(drop=True))


def calib_so762():
    import pandas as pd

    from bench.metrics import bootstrap_ci, pearson, spearman

    th = pd.read_parquet(feat_dir() / "speechocean762_theirs.parquet")
    th = th[th.split == "train"].reset_index(drop=True)
    lab = _so762_train()
    y = lab.phone_score.values
    g = lab.speaker.values
    imps = []
    preds = {"llr_sub": lab.llr_sub.values, "z_llr_sub": lab.z_llr_sub.fillna(0).values,
             "ordinal_logistic": _cv_predict(lab, y, "linear", g, "score"),
             "gbm": _cv_predict(lab, y, "gbm", g, "score", importances=imps),
             "gbm_no_phone_id": _cv_predict(lab, y, "gbm_nophone", g, "score"),
             "phone_prior_only": _cv_predict(lab, y, "phone_prior", g, "score")}
    out = {"n_phones_ours_matched": int(len(lab)), "n_utts": int(lab.utt.nunique()),
           "n_speakers": int(lab.speaker.nunique()), "features_numeric": NUM, "features_cat": CAT,
           "ours": {}, "theirs": {}, "gbm_permutation_importance_top15": _top_importances(imps)}
    for name, p in preds.items():
        ci = bootstrap_ci(pearson, p, y, clusters=g, n_boot=300)
        out["ours"][name] = {"pearson": pearson(p, y), "spearman": spearman(p, y),
                             "pearson_ci": [ci["lo"], ci["hi"]]}
        key = dict(zip(zip(lab.utt, lab.phone_index), p))
        vals, ys, gs, nfb, nmiss = [], [], [], 0, 0
        for r in th.itertuples():
            if r.our_phone_index >= 0 and (r.utt, r.our_phone_index) in key:
                v = key[(r.utt, r.our_phone_index)]
            else:
                c = [key[(r.utt, k)] for k in r.fallback_phone_indices if (r.utt, k) in key]
                if not c:
                    nmiss += 1
                    continue
                v = float(np.mean(c)); nfb += 1
            vals.append(v); ys.append(r.phone_score); gs.append(r.speaker)
        vals, ys = np.array(vals), np.array(ys)
        ci = bootstrap_ci(pearson, vals, ys, clusters=np.array(gs), n_boot=300)
        out["theirs"][name] = {"pearson": pearson(vals, ys), "spearman": spearman(vals, ys),
                               "pearson_ci": [ci["lo"], ci["hi"]], "n": int(len(vals)),
                               "n_fallback": nfb, "n_no_prediction": nmiss}
    out["n_their_phones"] = int(len(th))
    save("calib_so762.json", out)
    log("calib so762: " + ", ".join(f"{k} {v['pearson']:.3f}" for k, v in out["ours"].items()))
    return out


def _err_metrics(y, s, g, n_boot=300):
    from bench.metrics import best_f1, bootstrap_ci, roc_auc

    ci = bootstrap_ci(roc_auc, y, s, clusters=g, n_boot=n_boot)
    f = best_f1(y, s)
    return {"auc": roc_auc(y, s), "auc_ci": [ci["lo"], ci["hi"]], "best_f1": f.f1,
            "precision": f.precision, "recall": f.recall}


def errdet():
    import pandas as pd

    parts = []
    for ds in ("l2arctic", "epadb"):
        d = pd.read_parquet(feat_dir() / f"{ds}.parquet")
        parts.append(d[(d.split == "train") & d.label_err.notna()])
    df = _prep_X(pd.concat(parts, ignore_index=True))
    y = df.label_err.values.astype(int)
    g = (df.dataset + ":" + df.speaker.astype(str)).values
    gp = _prompts(df)
    imps = []
    base = {"llr_sub": -df.llr_sub.values, "z_llr_sub": -df.z_llr_sub.fillna(0).values}
    spk = dict(base)
    spk.update({"logistic": _cv_predict(df, y, "linear", g, "err"),
                "gbm": _cv_predict(df, y, "gbm", g, "err", importances=imps),
                "gbm_no_phone_id": _cv_predict(df, y, "gbm_nophone", g, "err"),
                "phone_prior_only": _cv_predict(df, y, "phone_prior", g, "err")})
    prm = {"logistic": _cv_predict(df, y, "linear", gp, "err"),
           "gbm": _cv_predict(df, y, "gbm", gp, "err"),
           "gbm_no_phone_id": _cv_predict(df, y, "gbm_nophone", gp, "err"),
           "phone_prior_only": _cv_predict(df, y, "phone_prior", gp, "err")}
    out = {"speaker_cv": {}, "prompt_cv": {},
           "cv": "5-fold GroupKFold over l2arctic train + epadb train pooled; speaker_cv groups = "
                 "corpus:speaker, prompt_cv groups = prompt text",
           "n_prompts": int(len(set(gp))),
           "gbm_permutation_importance_top15": _top_importances(imps)}
    for ds in ("l2arctic", "epadb", "both"):
        m = np.ones(len(df), bool) if ds == "both" else (df.dataset == ds).values
        out["speaker_cv"][ds] = {"n": int(m.sum()), "n_err": int(y[m].sum()),
                                 "n_ins_flag": int(df.ins_flag[m].sum()),
                                 **{k: _err_metrics(y[m], s[m], g[m]) for k, s in spk.items()}}
        out["prompt_cv"][ds] = {"n": int(m.sum()), "n_err": int(y[m].sum()),
                                **{k: _err_metrics(y[m], s[m], g[m]) for k, s in prm.items()}}
    # leave-one-corpus-out
    so = _so762_train()
    so_y = (so.phone_score.values < 2).astype(int)
    xc = {}
    sources = {"l2arctic": (df[df.dataset == "l2arctic"], y[(df.dataset == "l2arctic").values]),
               "epadb": (df[df.dataset == "epadb"], y[(df.dataset == "epadb").values]),
               "speechocean762 (score<2)": (so, so_y)}
    for src, (dtr, ytr) in sources.items():
        for tgt in ("l2arctic", "epadb"):
            if src == tgt:
                continue
            mt = (df.dataset == tgt).values
            dte, yte, gte = df[mt], y[mt], g[mt]
            res = {"n_train": int(len(dtr)), "train_err_rate": float(ytr.mean()), "n": int(mt.sum()),
                   "n_err": int(yte.sum())}
            for k in ("linear", "gbm", "gbm_nophone", "phone_prior"):
                p = _fit_predict(k, "err", dtr, ytr, dte)
                name = {"linear": "logistic", "gbm_nophone": "gbm_no_phone_id",
                        "phone_prior": "phone_prior_only"}.get(k, k)
                res[name] = _err_metrics(yte, p, gte)
            xc[f"{src} -> {tgt}"] = res
    out["cross_corpus"] = xc
    save("errdet.json", out)
    log("errdet: " + json.dumps({k: {n: round(v[n]["auc"], 3) for n in spk} for k, v in out["speaker_cv"].items()}))
    log("cross: " + json.dumps({k: {n: round(v[n]["auc"], 3) for n in ("logistic", "gbm", "gbm_no_phone_id", "phone_prior_only")} for k, v in xc.items()}))
    return out


# ----------------------------------------------------------------------------- controlled-error tables

def ctl_tables(langs):
    import pandas as pd

    from bench.metrics import roc_auc

    p = feat_dir() / "fleurs-validation.parquet"
    df = pd.read_parquet(p)
    out = {}
    for lang in langs:
        d = df[df.lang == lang]
        if d.empty:
            continue
        r = {"by_kind": {}, "by_class": {}, "config": {"adapter": d.adapter.iloc[0],
                                                       "variants": bool(d.variants.iloc[0])}}
        clean = d[d.ref_kind == "clean"]
        r["native_p05"] = float(clean.llr_sub.quantile(0.05))
        r["native_median"] = float(clean.llr_sub.median())
        for k in ("phone_sub_near", "phone_sub_far", "phone_ins", "word_swap"):
            x = d[(d.ref_kind == k) & (~d.dropped)]
            if x.label_ok.min() == 1:
                continue
            yy = 1 - x.label_ok.values
            r["by_kind"][k] = {"auc": roc_auc(yy, -x.llr_sub.values),
                               "auc_med": roc_auc(yy, -(x.llr_sub - x.utt_median_llr_sub).values),
                               "auc_z": roc_auc(yy, -x.z_llr_sub.fillna(0).values),
                               "n_wrong": int(yy.sum()), "n_ok": int(len(yy) - yy.sum())}
        x = d[d.ref_kind == "phone_del"]
        if x.near_gap.any():
            r["by_kind"]["phone_del (gap neighbours vs rest)"] = {
                "auc": roc_auc(x.near_gap.values.astype(int), -x.llr_sub.values),
                "auc_med": roc_auc(x.near_gap.values.astype(int), -(x.llr_sub - x.utt_median_llr_sub).values),
                "n_wrong": int(x.near_gap.sum()), "n_ok": int((~x.near_gap).sum())}
        x = d[d.ref_kind.isin(["phone_sub_near", "phone_sub_far", "word_swap", "phone_ins"]) & (~d.dropped)]
        for c, g in x.groupby("cls"):
            yy = 1 - g.label_ok.values
            if 0 < yy.sum() < len(yy):
                r["by_class"][c] = {"auc": roc_auc(yy, -g.llr_sub.values), "n_wrong": int(yy.sum()),
                                    "n_ok": int(len(yy) - yy.sum())}
        out[lang] = r
    save("ctl_fleurs_validation.json", out)
    return out


# ----------------------------------------------------------------------------- report

def f3(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def report(langs):
    L = []
    add = L.append
    add("# Phase 2 evidence\n")
    add("Generated by `python -m bench.pipeline.analyze --steps report` from `results/tables/*.json`. "
        "Model zipa-small-crctc-500k fp32. Dev set = FLEURS validation; fitting = FLEURS train / "
        "speechocean762 train / l2arctic train / epadb train. No held-out split (FLEURS test, "
        "speechocean762 / l2arctic / epadb test) was used for fitting, selection or any number below.\n")
    notes = tables_dir() / "notes_phase2.md"
    if notes.exists():
        add(notes.read_text())
    w = load_json("warm.json")
    if w:
        add("## 1. Posterior cache\n")
        add("| dataset | utts | hours | GB | missing |\n|---|---|---|---|---|")
        for k, v in w["datasets"].items():
            add(f"| {k} | {v['n']} | {v['hours']} | {v['gb']} | {v['missing']} |")
        t = w["total"]
        add(f"| **total** | {t['n']} | {t['hours']} | {t['gb']} | {t['missing']} |\n")
        if w.get("wall_s"):
            add(f"Wall time {w['wall_s'] / 60:.0f} min (2 processes x 3 threads, machine shared).\n")
    # adapters
    add("## 2. Output adapter (FLEURS validation, controlled errors, 2 errors/utt)\n")
    add("AUC = ROC AUC of −llr_sub, corrupted vs untouched slots of the same corrupted references; "
        "mean = mean over the three kinds; med = per-utterance-median-subtracted llr_sub; TER = token "
        "error rate of the greedy output vs the canonical reference; p05 = 5th percentile of native "
        "llr_sub on the correct references.\n")
    add("| lang | variant | near | far | swap | mean [95% CI] | mean med | TER | native p05 | epochs | val CTC/token |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    choices = {}
    for lang in langs:
        a = load_json(f"adapter_{lang}.json")
        if not a:
            continue
        choices[lang] = a["choice"]
        for k, t in a["table"].items():
            h = a["history"].get(k)
            ep = (len(h) - 1) if h else "–"
            vbest = min(x["val"] for x in h) if h else None
            v0 = h[0]["val"] if h else None
            mark = " **(chosen)**" if k == a["choice"] else ""
            add(f"| {lang} | {k}{mark} | {f3(t.get('auc_phone_sub_near'))} | {f3(t.get('auc_phone_sub_far'))} | "
                f"{f3(t.get('auc_word_swap'))} | {f3(t['mean_auc'])} [{f3(t['mean_auc_ci'][0])}, {f3(t['mean_auc_ci'][1])}] | "
                f"{f3(t['mean_auc_med'])} | {f3(t['ter'])} | {f3(t['native_llr_sub_p05'])} | {ep} | "
                f"{(f3(v0) + ' → ' + f3(vbest)) if h else '–'} |")
    add("")
    add("Selection: simplest variant whose paired-bootstrap (500 utterance resamples) 95 % CI of "
        "(best − variant) mean AUC includes 0; shipped only if its CI of (variant − raw) is > 0.\n")
    add("| lang | best | chosen | chosen − raw [95% CI] |\n|---|---|---|---|")
    for lang in langs:
        a = load_json(f"adapter_{lang}.json")
        if not a:
            continue
        s = a["selection"]
        vr = s.get("choice_vs_raw")
        add(f"| {lang} | {s['best']} | {s['choice']} | "
            + (f"{vr['delta']:+.4f} [{vr['lo']:+.4f}, {vr['hi']:+.4f}]" if vr else "–") + " |")
    add("")
    so = load_json("so762_adapter.json")
    if so:
        add("### English learners: speechocean762 TRAIN, llr_sub vs human phone score (0–2)\n")
        add(f"{so['n_phones']} aligned phones, {so['n_utts']} utterances. Shipped en adapter: {so['shipped']}.\n")
        add("| llr | Pearson [95% CI, speaker bootstrap] | Spearman | Pearson − raw [95% CI] |\n|---|---|---|---|")
        for k, v in so["by"].items():
            mr = v.get("minus_raw")
            add(f"| {k} | {f3(v['pearson'])} [{f3(v['pearson_ci'][0])}, {f3(v['pearson_ci'][1])}] | {f3(v['spearman'])} | "
                + (f"{mr['delta']:+.4f} [{mr['lo']:+.4f}, {mr['hi']:+.4f}]" if mr else "–") + " |")
        add("")
    # variants
    add("## 3. Learned variants\n")
    add("Decision: learned variants are shipped **off** for every language (they lowered the "
        "controlled-error AUC wherever they were evaluated). The off/on evaluation (FLEURS validation, "
        "shipped adapter, paired corruptions) was run for the languages below; for the others the "
        "variant tables are learned and stored but not evaluated (`native_stats --eval-variants`).\n")
    add("| lang | phones with variants | learned alts | AUC off | AUC on | on − off [95% CI] | med AUC off → on | native p05 off → on | dropped slots |")
    add("|---|---|---|---|---|---|---|---|---|")
    for lang in langs:
        n = load_json(f"native_{lang}.json")
        if not n:
            continue
        nalt = sum(len(v) for v in n["variants"].values())
        c = n.get("compare")
        if not c:
            add(f"| {lang} | {len(n['variants'])} | {nalt} | not evaluated | | | | | |")
            continue
        b = c["on_minus_off"]
        dropped = sum(c["on"]["dropped"].values())
        add(f"| {lang} | {len(n['variants'])} | {nalt} | {f3(c['off']['mean_auc'])} | {f3(c['on']['mean_auc'])} | "
            f"{b['delta']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] | {f3(c['off']['mean_auc_med'])} → {f3(c['on']['mean_auc_med'])} | "
            f"{f3(c['off']['native_llr_sub_p05'])} → {f3(c['on']['native_llr_sub_p05'])} | {dropped} |")
    add("")
    add("Per-kind AUC (off → on), evaluated languages:\n")
    add("| lang | near | far | swap |\n|---|---|---|---|")
    for lang in langs:
        n = load_json(f"native_{lang}.json")
        if not n or not n.get("compare"):
            continue
        c = n["compare"]
        add(f"| {lang} | " + " | ".join(f"{f3(c['off'].get('auc_' + k))} → {f3(c['on'].get('auc_' + k))}"
                                       for k in ("phone_sub_near", "phone_sub_far", "word_swap")) + " |")
    add("")
    add("Learned alternatives (phone → alt (rate)), ∅ = may be dropped:\n")
    for lang in langs:
        n = load_json(f"native_{lang}.json")
        if not n:
            continue
        items = [f"{p}→{a[0] or '∅'} ({a[3]:.1%})" for p, v in n["variants"].items() for a in v]
        add(f"- **{lang}**: " + (", ".join(items) if items else "none"))
    add("")
    # features / alignment
    cov = load_json("alignment_coverage.json")
    if cov:
        add("## 4. Feature tables and label alignment\n")
        add("| dataset | split | their phones | % of theirs matched | % of ours matched | % identical (of matched) | excluded utts |")
        add("|---|---|---|---|---|---|---|")
        for ds, v in cov.items():
            for sp, c in v.items():
                add(f"| {ds} | {sp} | {c.get('theirs', 0)} | {f3(c.get('pct_theirs_matched'))} | {f3(c.get('pct_ours_matched'))} | "
                    f"{f3(c.get('pct_identical_of_matched'))} | {c.get('excluded_utts', 0)} |")
        add("")
    fs = load_json("features_summary.json")
    if fs:
        add("| table | rows | utts | languages |\n|---|---|---|---|")
        for k, v in fs.items():
            add(f"| {k} | {v['rows']} | {v['utts']} | {v.get('langs', '')} |")
        add("")
    # calibration
    c = load_json("calib_so762.json")
    if c:
        add("## 5. Calibrator form\n")
        add("### speechocean762 TRAIN, 5-fold speaker-grouped CV, target = human phone score 0–2\n")
        add(f"{c['n_phones_ours_matched']} of our phones with a human score ({c['n_utts']} utts, "
            f"{c['n_speakers']} speakers). ordinal_logistic = two cumulative logistic regressions on the "
            "rounded score (prediction = expected score), numeric features, NaN→0; gbm = sklearn "
            "HistGradientBoostingRegressor on numeric + categorical; gbm_no_phone_id = same without the "
            "identity categoricals (phone, comp_tok); phone_prior_only = training-fold mean score of the "
            "canonical phone.\n")
        add(f"Numeric: {', '.join(c['features_numeric'])}. Categorical: {', '.join(c['features_cat'])}.\n")
        add("| predictor | Pearson (our phones) [95% CI] | Spearman | Pearson (all THEIR phones, word-mean fallback) | n fallback |")
        add("|---|---|---|---|---|")
        for k in c["ours"]:
            o, t = c["ours"][k], c["theirs"][k]
            add(f"| {k} | {f3(o['pearson'])} [{f3(o['pearson_ci'][0])}, {f3(o['pearson_ci'][1])}] | {f3(o['spearman'])} | "
                f"{f3(t['pearson'])} [{f3(t['pearson_ci'][0])}, {f3(t['pearson_ci'][1])}] | {t['n_fallback']} |")
        add("")
        add("GBM permutation importance (Pearson drop on held-out folds, mean ± sd over 5 folds, 3 repeats):\n")
        add(" | ".join(f"{f} {m:.3f}±{s:.3f}" for f, m, s in c["gbm_permutation_importance_top15"]) + "\n")
    e = load_json("errdet.json")
    if e:
        add("### Annotated error detection: l2arctic train + epadb train\n")
        add("Error = substituted or deleted (inserted sounds are only flagged on the neighbour, column "
            "`ins_flag`, not counted). AUC [95% CI, speaker-cluster bootstrap] / best F1.\n")
        ks = ("llr_sub", "z_llr_sub", "logistic", "gbm", "gbm_no_phone_id", "phone_prior_only")
        for title, key in (("5-fold speaker-grouped CV (pooled fit)", "speaker_cv"),
                           (f"5-fold prompt-grouped CV ({e.get('n_prompts')} prompts; a prompt never in both train and test fold)", "prompt_cv")):
            add(f"**{title}**\n")
            kk = [k for k in ks if k in next(iter(e[key].values()))]
            add("| data | n | errors | " + " | ".join(kk) + " |")
            add("|---|---|---|" + "---|" * len(kk))
            for ds, v in e[key].items():
                add(f"| {ds} | {v['n']} | {v['n_err']} | " + " | ".join(
                    f"{f3(v[k]['auc'])} [{f3(v[k]['auc_ci'][0])}, {f3(v[k]['auc_ci'][1])}] / {f3(v[k]['best_f1'])}" for k in kk) + " |")
            add("")
        add("**Leave-one-corpus-out** (fit on the whole source train split, evaluate on the target train split; "
            "raw llr_sub on the target for reference: see the speaker-CV table)\n")
        kk = ("logistic", "gbm", "gbm_no_phone_id", "phone_prior_only")
        add("| source → target | n train | train err rate | n | errors | " + " | ".join(kk) + " |")
        add("|---|---|---|---|---|" + "---|" * len(kk))
        for k2, v in e["cross_corpus"].items():
            add(f"| {k2} | {v['n_train']} | {v['train_err_rate']:.3f} | {v['n']} | {v['n_err']} | " + " | ".join(
                f"{f3(v[k]['auc'])} [{f3(v[k]['auc_ci'][0])}, {f3(v[k]['auc_ci'][1])}] / {f3(v[k]['best_f1'])}" for k in kk) + " |")
        add("")
        add("GBM permutation importance (AUC drop on held-out speaker folds, mean ± sd):\n")
        add(" | ".join(f"{f} {m:.3f}±{s:.3f}" for f, m, s in e["gbm_permutation_importance_top15"]) + "\n")
    ct = load_json("ctl_fleurs_validation.json")
    if ct:
        add("## 6. Controlled-error AUC by kind and class (FLEURS validation, shipped config)\n")
        add("AUC of −llr_sub (2 errors/utt). phone_del: no reference phone is wrong; reported is the AUC "
            "of the two phones adjacent to the gap vs the rest of the same references.\n")
        kinds = ["phone_sub_near", "phone_sub_far", "phone_ins", "word_swap", "phone_del (gap neighbours vs rest)"]
        add("| lang | config | " + " | ".join(k.split(" ")[0] for k in kinds) + " | near med | native p05 |")
        add("|---|---|" + "---|" * (len(kinds) + 2))
        for lang, r in ct.items():
            cfg = f"{r['config']['adapter']}{'+var' if r['config']['variants'] else ''}"
            add(f"| {lang} | {cfg} | " + " | ".join(f3(r["by_kind"].get(k, {}).get("auc")) for k in kinds)
                + f" | {f3(r['by_kind'].get('phone_sub_near', {}).get('auc_med'))} | {f3(r['native_p05'])} |")
        add("")
        from pronunciada.variants import CLASSES

        add("| lang | " + " | ".join(CLASSES) + " |\n|---|" + "---|" * len(CLASSES))
        for lang, r in ct.items():
            add(f"| {lang} | " + " | ".join(
                (f"{f3(r['by_class'][c]['auc'])} ({r['by_class'][c]['n_wrong']})" if c in r["by_class"] else "–")
                for c in CLASSES) + " |")
        add("\n(number of wrong slots in brackets; kinds near/far/swap/ins pooled)\n")
    add("## 7. Worst native phones (FLEURS train, variants off, shipped adapter)\n")
    add("Mean llr_sub per canonical phone (count ≥ 30), the 20 lowest; `heard` = top greedy "
        "realisations in the phone's span (∅ = llr_del < 0).\n")
    for lang in langs:
        n = load_json(f"native_{lang}.json")
        if not n:
            continue
        add(f"**{lang}**\n")
        add("| phone | n | mean | median | p05 | share llr_del<0 | top heard |\n|---|---|---|---|---|---|---|")
        for r in n["worst_phones"]:
            hh = ", ".join(f"{a or '·'}:{c}" for a, c in r["top_heard"])
            add(f"| {r['phone']} | {r['n']} | {r['mean']:.2f} | {r['median']:.2f} | {r['p05']:.2f} | {r['share_del']:.1%} | {hh} |")
        add("")
    (repo_root() / "results" / "phase2_evidence.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("wrote results/phase2_evidence.md")


def features_summary():
    import pyarrow.parquet as pq

    out = {}
    for p in sorted(feat_dir().glob("*.parquet")):
        cols = ["utt"] + (["lang"] if "lang" in pq.read_schema(p).names else [])
        t = pq.read_table(p, columns=cols)
        out[p.name] = {"rows": t.num_rows, "utts": len(set(t.column("utt").to_pylist())),
                       "langs": ",".join(sorted(set(t.column("lang").to_pylist()))) if "lang" in cols else ""}
    save("features_summary.json", out)


def main():
    from bench.pipeline.common import base_parser

    ap = base_parser(__doc__)
    ap.add_argument("--steps", default="so762_adapter,calib,errdet,ctl,report")
    a = ap.parse_args()
    langs = parse_langs(a.lang)
    steps = a.steps.split(",")
    t0 = time.time()
    if "so762_adapter" in steps:
        so762_adapter_check(a.workers)
    if "calib" in steps:
        calib_so762()
    if "errdet" in steps:
        errdet()
    if "ctl" in steps:
        ctl_tables(langs)
        features_summary()
    if "report" in steps:
        report(langs)
    log(f"analyze done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
