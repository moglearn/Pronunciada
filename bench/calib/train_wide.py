"""Universal sound model (feature set A) on a wider training mix, chosen by leave-one-corpus-out.

    python -m bench.calib.train_wide
    python -m bench.calib.train_wide --clean    # licence-clean re-selection (2026-11-17)

--clean: the same selection WITHOUT the non-commercial corpora (L2-ARCTIC and EpaDB are neither
loaded nor used for evaluation). Candidates so762 and so762+fleurs; the "mean real-corpus strict
AUC" is then the speechocean762 strict AUC alone; same 0.002 replacement margin. If "so762" wins,
the shipped sound_U.npz (trained on speechocean762 only) stays as it is and nothing is written
except results/tables/calib_wide_clean.json.

Candidates for the two probability heads of A (the regression head is always speechocean762):
  so762 : speechocean762 train only (human scores; len = score >= 1.0, str = score >= 1.8)
  real  : + l2arctic train + epadb train (annotated correct vs substituted / deleted -> strict
          label only; their lenient label is unknown)
  wide  : real + FLEURS-validation controlled errors in all 10 languages (untouched native phone
          -> ok, corrupted -> not ok for both heads; kinds phone_sub_near / phone_sub_far /
          phone_ins / word_swap; "dropped" rows excluded); synthetic rows down-weighted to 30 % of
          each head's total weight.
Evaluation (never touches a test split):
  * l2arctic, epadb (strict head, AUC of 1-P vs error): the target corpus is left out entirely
  * speechocean762 (len / str AUC + ECE): 5-fold speaker CV, the other corpora always in training
  * FLEURS-validation controlled errors (AUC): utterance-grouped 5-fold CV for "wide"; the
    other candidates never see FLEURS (reported for transfer)
Choice: highest mean of the three real-corpus strict AUCs (l2arctic, epadb, so762); a
candidate replaces the simpler one only if its mean is higher by > 0.002.
The chosen heads are calibrated on speechocean762 OOF predictions and written, with A's
regression head, to pronunciada/data/calib/sound_U.npz.
"""
from __future__ import annotations

import numpy as np

from bench.calib.common import (apply_calib, calib_cache, choose_calib, fit, load_json, load_table, raw,
                                save_json, ship_dir, speaker_folds, sub, table_columns, vocab_for)
from bench.calib.fitlib import fit_final, head_metrics, log
from bench.calib.train_sound import KINDS, LEN_THR, STR_THR
from pronunciada import features as F
from pronunciada.calibrate import LevelModel, save_level

SYN_SHARE = 0.30
FLEURS_UTTS_PER_LANG = 60          # x 5 references (clean + 4 kinds) ~ 2e4 rows per language
SYN_KINDS = ("phone_sub_near", "phone_sub_far", "phone_ins", "word_swap")


def load_corpora(clean=False):
    so = load_table("speechocean762", "train")
    so_t = table_columns(so)
    y = np.asarray(so_t["phone_score"], dtype=np.float64)
    so_t["y_len"] = np.where(np.isfinite(y), (y >= LEN_THR).astype(float), np.nan)
    so_t["y_str"] = np.where(np.isfinite(y), (y >= STR_THR).astype(float), np.nan)
    so_t["corpus"] = np.array(["so762"] * len(y))
    so_t["grp"] = so.speaker.astype(str).values
    parts = {"so762": so_t}
    for ds in (() if clean else ("l2arctic", "epadb")):
        d = load_table(ds, "train")
        t = table_columns(d)
        e = np.asarray(t["label_err"], dtype=np.float64)
        t["y_len"] = np.full(len(e), np.nan)
        t["y_str"] = np.where(np.isfinite(e), 1.0 - e, np.nan)
        t["corpus"] = np.array([ds] * len(e))
        t["grp"] = (ds + ":" + d.speaker.astype(str)).values
        parts[ds] = t
    import pandas as pd

    from bench.pipeline.common import feat_dir

    fv = pd.read_parquet(feat_dir() / "fleurs-validation.parquet")
    rng = np.random.default_rng(0)
    keep = []
    for lang, g in fv.groupby("lang"):
        utts = g.utt.unique()
        keep += list(rng.choice(utts, min(FLEURS_UTTS_PER_LANG, len(utts)), replace=False))
    fv = fv[fv.utt.isin(set(keep)) & fv.ref_kind.isin(("clean",) + SYN_KINDS)]
    fv = fv.sort_values(["utt", "ref_kind", "phone_index"], kind="stable").reset_index(drop=True)
    t = table_columns(fv)
    ok = np.asarray(t["label_ok"], dtype=np.float64)
    drop = np.asarray(fv.dropped.values, bool)
    ok = np.where(drop, np.nan, ok)
    t["y_len"], t["y_str"] = ok.copy(), ok.copy()
    t["corpus"] = np.array(["fleurs"] * len(ok))
    t["grp"] = ("fleurs:" + fv.utt.astype(str)).values
    parts["fleurs"] = t
    return parts


def weights(t, head):
    """1 for real rows, synthetic rows scaled to SYN_SHARE of the head's total weight."""
    y = t[f"y_{head}"]
    lab = np.isfinite(y)
    syn = (t["corpus"] == "fleurs") & lab
    real = lab & ~syn
    w = np.ones(len(y))
    if syn.any() and real.any():
        w[syn] = SYN_SHARE / (1 - SYN_SHARE) * real.sum() / syn.sum()
    return w


CANDIDATES = {"so762": ("so762",), "real": ("so762", "l2arctic", "epadb"),
              "wide": ("so762", "l2arctic", "epadb", "fleurs")}


CLEAN_CANDIDATES = {"so762": ("so762",), "so762+fleurs": ("so762", "fleurs")}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    clean = ap.parse_args().clean
    global CANDIDATES
    if clean:
        CANDIDATES = CLEAN_CANDIDATES
    corpora = ("so762", "fleurs") if clean else ("so762", "l2arctic", "epadb", "fleurs")
    spec = F.SOUND_SETS["A"]
    A = LevelModel(calib_cache() / "sound_A.npz")       # vocab of the so762 model
    vocab = A.vocab
    params = load_json("calib_sound.json")["params"]["str"]     # probability-head params
    parts = load_corpora(clean)
    keys = spec["num"] + spec["cat"] + ["y_len", "y_str", "corpus", "grp", "_group", "utt",
                                        "phone_index"]
    from bench.calib.common import concat

    T = concat([parts[k] for k in corpora], keys)
    T["corpus"] = np.asarray(T["corpus"])
    T["grp"] = np.asarray(T["grp"])
    X = F.matrix(T, spec["num"], spec["cat"], vocab, A.encoding)
    cm = F.categorical_mask(spec["num"], spec["cat"], vocab, A.encoding)
    corp = T["corpus"]
    log("rows", {c: int((corp == c).sum()) for c in corpora})
    out = {"n_rows": {c: int((corp == c).sum()) for c in corpora},
           "n_labelled_str": {c: int(np.isfinite(T["y_str"][corp == c]).sum()) for c in corpora},
           "err_rate_str": {c: float(1 - np.nanmean(T["y_str"][corp == c])) for c in corpora},
           "syn_share": SYN_SHARE, "fleurs_utts_per_lang": FLEURS_UTTS_PER_LANG, "eval": {}}
    so = np.flatnonzero(corp == "so762")
    so_folds = speaker_folds(T["grp"][so])
    fl = np.flatnonzero(corp == "fleurs")
    from sklearn.model_selection import GroupKFold

    fl_folds = list(GroupKFold(5).split(fl, None, T["grp"][fl]))

    def train_rows(cands, exclude=None, extra_mask=None):
        m = np.isin(corp, cands)
        if exclude is not None:
            m &= corp != exclude
        if extra_mask is not None:
            m &= extra_mask
        return m

    oof_so = {}
    for name, cands in CANDIDATES.items():
        res = {}
        # --- LOCO on l2arctic / epadb (strict head)
        for tgt in (() if clean else ("l2arctic", "epadb")):
            m = train_rows(cands, exclude=tgt) & np.isfinite(T["y_str"])
            w = weights(sub(T, np.flatnonzero(m)), "str")
            mdl = fit("prob", params, X[m], T["y_str"][m], cm, w)
            te = np.flatnonzero((corp == tgt) & np.isfinite(T["y_str"]))
            z = raw(mdl, X[te])
            res[tgt] = head_metrics("prob", 1 - T["y_str"][te], 1 / (1 + np.exp(z)))
            log(name, tgt, round(res[tgt]["auc"], 4))
        # --- so762 speaker CV (others always in)
        zs = {h: np.full(len(so), np.nan) for h in ("len", "str")}
        for tr, te in so_folds:
            hold = np.zeros(len(corp), bool)
            hold[so[te]] = True
            for h in ("len", "str"):
                m = train_rows(cands) & ~hold & np.isfinite(T[f"y_{h}"])
                w = weights(sub(T, np.flatnonzero(m)), h)
                mdl = fit("prob", params, X[m], T[f"y_{h}"][m], cm, w)
                zs[h][te] = raw(mdl, X[so[te]])
        g_so = T["grp"][so]
        for h in ("len", "str"):
            y = T[f"y_{h}"][so]
            ok = np.isfinite(y)
            c, cmp = choose_calib(zs[h][ok], y[ok], g_so[ok])
            p = apply_calib(c, zs[h])
            res[f"so762_{h}"] = head_metrics("prob", y, p)
            res[f"so762_{h}"]["auc_err"] = head_metrics("prob", 1 - y, 1 - p)["auc"]
            res[f"so762_{h}"]["calib"] = c["kind"]
            res[f"so762_{h}"]["calib_cv_ece"] = cmp
        oof_so[name] = zs
        log(name, "so762 str auc", round(res["so762_str"]["auc"], 4), "len", round(res["so762_len"]["auc"], 4))
        # --- FLEURS controlled errors
        zf = np.full(len(fl), np.nan)
        if "fleurs" in cands:
            for tr, te in fl_folds:
                hold = np.zeros(len(corp), bool)
                hold[fl[te]] = True
                m = train_rows(cands) & ~hold & np.isfinite(T["y_str"])
                mdl = fit("prob", params, X[m], T["y_str"][m], cm, weights(sub(T, np.flatnonzero(m)), "str"))
                zf[te] = raw(mdl, X[fl[te]])
        else:
            m = train_rows(cands) & np.isfinite(T["y_str"])
            mdl = fit("prob", params, X[m], T["y_str"][m], cm, weights(sub(T, np.flatnonzero(m)), "str"))
            zf = raw(mdl, X[fl])
        yf = T["y_str"][fl]
        okf = np.isfinite(yf)
        res["fleurs_val_controlled"] = head_metrics("prob", 1 - yf[okf], 1 / (1 + np.exp(zf[okf])))
        res["mean_real_strict_auc"] = float(np.mean(([] if clean else [res["l2arctic"]["auc"], res["epadb"]["auc"]])
                                                    + [res["so762_str"]["auc_err"]]))
        out["eval"][name] = res
        log(name, "fleurs auc", round(res["fleurs_val_controlled"]["auc"], 4), "mean real",
            round(res["mean_real_strict_auc"], 4))
        save_json("calib_wide_clean.json" if clean else "calib_wide.json", out)
    # --- choice
    order = list(CANDIDATES)
    chosen = order[0]
    for c in order[1:]:
        if out["eval"][c]["mean_real_strict_auc"] > out["eval"][chosen]["mean_real_strict_auc"] + 0.002:
            chosen = c
    out["chosen"] = chosen
    log("chosen", chosen)
    if clean:
        save_json("calib_wide_clean.json", out)
        if chosen == "so762":
            log("so762 wins: shipped sound_U.npz (trained_on so762) is kept unchanged")
            return
        raise SystemExit(f"clean selection chose {chosen}: export it with the round-2 recipe and cascade")
    # --- final models: reg head of A + the candidate's len / str heads, calibrated on its own
    # speechocean762 OOF log-odds. Chosen -> pronunciada/data/calib/sound_U.npz (shipped); the
    # other mixes -> $PRONUNCIADA_CACHE/calib/sound_R.npz (real) / sound_W.npz (wide), benchmark only.
    with np.load(calib_cache() / "sound_A.npz", allow_pickle=False) as z:
        reg_trees = {k[4:]: z[k] for k in z.files if k.startswith("reg/")}

    def export(cand, dest):
        cands = CANDIDATES[cand]
        idx = np.flatnonzero(np.isin(corp, cands))
        Tm = sub(T, idx)
        labels = {h: Tm[f"y_{h}"] for h in ("len", "str")}
        wts = {h: weights(Tm, h) for h in ("len", "str")}
        heads, parity = fit_final(X[idx], labels, {"len": "prob", "str": "prob"}, params, cm, wts)
        for h in ("len", "str"):
            y = T[f"y_{h}"][so]
            ok = np.isfinite(y)
            c, _ = choose_calib(oof_so[cand][h][ok], y[ok], T["grp"][so][ok])
            heads[h]["calib"] = c
        heads["reg"] = {"kind": "reg", "trees": reg_trees, "clip": (0.0, 2.0)}
        save_level(dest, spec["num"], spec["cat"], vocab,
                   {h: {k: v for k, v in hh.items() if k != "_model"} for h, hh in heads.items()},
                   {"level": "sound", "set": "A", "trained_on": "+".join(cands),
                    "thresholds": {"len": LEN_THR, "str": STR_THR}}, encoding=A.encoding)
        return {"file": str(dest), "kb": dest.stat().st_size / 1024, "walker_parity_max_abs": parity,
                "calib": {h: heads[h]["calib"]["kind"] for h in ("len", "str")}, "trained_on": cands}

    out["final"] = export(chosen, ship_dir() / "sound_U.npz")
    out["others"] = {}
    for cand, tag in (("real", "R"), ("wide", "W")):
        if cand != chosen:
            out["others"][cand] = export(cand, calib_cache() / f"sound_{tag}.npz")
    log("exported", out["final"], out["others"])
    # OOF z of the chosen universal model on so762 train, for the word / sentence calibrators
    import pandas as pd

    od = pd.read_parquet(calib_cache() / "sound_oof.parquet")
    key = pd.DataFrame({"utt": np.asarray(T["utt"])[so], "phone_index": np.asarray(T["phone_index"])[so].astype(int)})
    for h in ("len", "str"):
        y = T[f"y_{h}"][so]
        ok = np.isfinite(y)
        c, _ = choose_calib(oof_so[chosen][h][ok], y[ok], T["grp"][so][ok])
        p = apply_calib(c, oof_so[chosen][h])
        key[f"U_z_{h}"] = np.log(p / (1 - p))
    key["U_z_str"] = np.minimum(key["U_z_str"], key["U_z_len"])
    od = od.drop(columns=[c for c in od.columns if c.startswith("U_")]).merge(key, on=["utt", "phone_index"], how="left")
    od["U_reg_raw"] = od["A_reg_raw"]
    od.to_parquet(calib_cache() / "sound_oof.parquet", index=False)
    save_json("calib_wide.json", out)
    log("done", out["final"])


if __name__ == "__main__":
    main()
