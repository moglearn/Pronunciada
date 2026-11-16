"""Sound-level calibrators on speechocean762 TRAIN (5-fold speaker-grouped CV), feature sets A/B/C.

    python -m bench.calib.train_sound [--grid]

Heads: reg = human 0-2 phone score; len = P(score >= LEN_THR); str = P(score >= STR_THR).
Writes results/tables/calib_sound.json, the out-of-fold predictions
($PRONUNCIADA_CACHE/calib/sound_oof.parquet, every our-phone row of the train split, also unlabelled
ones) and the exported models: pronunciada/data/calib/sound_B.npz (English), $PRONUNCIADA_CACHE/calib/
sound_A.npz (input of bench.calib.train_wide) and sound_C.npz (benchmark transparency only,
never shipped).
"""
from __future__ import annotations

import argparse

import numpy as np

from bench.calib.common import (HGB_GRID, calib_cache, load_json, load_table, save_json, ship_dir,
                                table_columns, vocab_for)
from bench.calib.fitlib import calibrate_oof, cv_heads, fit_final, head_metrics, log
from pronunciada import features as F
from pronunciada.calibrate import save_level

LEN_THR, STR_THR = 1.0, 1.8
KINDS = {"reg": "reg", "len": "prob", "str": "prob"}


def labels_so762(t):
    y = np.asarray(t["phone_score"], dtype=np.float64)
    return {"reg": y, "len": np.where(np.isfinite(y), (y >= LEN_THR).astype(float), np.nan),
            "str": np.where(np.isfinite(y), (y >= STR_THR).astype(float), np.nan)}


def balance(y):
    y = y[np.isfinite(y)]
    return {f">={thr}": float((y >= thr).mean()) for thr in (0.5, 1.0, 1.2, 1.5, 1.6, 1.8, 2.0)}


def theirs_metric(df_ours, pred, split="train"):
    """Pearson / Spearman on THEIR phone list: matched phones take our prediction, unmatched
    ones the mean over our phones of the aligned word(s) (same convention as for Azure)."""
    import pandas as pd

    from bench.metrics import pearson, spearman
    from bench.pipeline.common import feat_dir

    th = pd.read_parquet(feat_dir() / "speechocean762_theirs.parquet")
    th = th[th.split == split]
    key = dict(zip(zip(df_ours.utt.values, df_ours.phone_index.values.astype(int)), pred))
    vals, ys, nfb = [], [], 0
    for r in th.itertuples():
        k = (r.utt, int(r.our_phone_index))
        if r.our_phone_index >= 0 and k in key and np.isfinite(key[k]):
            v = key[k]
        else:
            c = [key[(r.utt, int(j))] for j in r.fallback_phone_indices if (r.utt, int(j)) in key]
            if not c:
                continue
            v = float(np.mean(c))
            nfb += 1
        vals.append(v)
        ys.append(r.phone_score)
    return {"pearson": pearson(vals, ys), "spearman": spearman(vals, ys), "n": len(vals),
            "n_fallback": nfb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="re-run the hyper-parameter grid")
    a = ap.parse_args()
    df = load_table("speechocean762", "train")
    t = table_columns(df)
    lab = labels_so762(t)
    groups = df.speaker.astype(str).values
    out = load_json("calib_sound.json") or {}
    out["n_rows"] = int(len(df))
    out["n_labelled"] = int(np.isfinite(lab["reg"]).sum())
    out["n_speakers"] = int(len(set(groups)))
    out["balance"] = balance(lab["reg"])
    out["thresholds"] = {"len": LEN_THR, "str": STR_THR}
    log("balance", out["balance"])
    # ------------------------------------------------------------------ grid on A / reg
    specA = F.SOUND_SETS["A"]
    vocab = vocab_for(t, specA["cat"] + F.IDENTITY_CAT)
    XA = F.matrix(t, specA["num"], specA["cat"], vocab)          # "mixed" encoding
    cmA = F.categorical_mask(specA["num"], specA["cat"], vocab)
    if a.grid or "grid" not in out:
        g = {}
        for name, params in HGB_GRID.items():
            oof, _ = cv_heads(XA, {"reg": lab["reg"]}, KINDS, groups, params, cmA)
            g[name] = head_metrics("reg", lab["reg"], oof["reg"])["pearson"]
            log("grid", name, g[name])
        out["grid"] = {"pearson_A_reg": g, "chosen": max(g, key=g.get)}
        save_json("calib_sound.json", out)
    # regression head: grid choice; probability heads: 200 trees at lr 0.1 (g1). 400 trees gave
    # +0.002 AUC on CV at twice the runtime cost (see results/calibration.md).
    params = {"reg": HGB_GRID[out["grid"]["chosen"]], "len": HGB_GRID["g1"], "str": HGB_GRID["g1"]}
    out["params"] = params
    # ------------------------------------------------------------------ A / B / C
    import pandas as pd

    oof_df = pd.DataFrame({"utt": df.utt.values, "phone_index": df.phone_index.values.astype(int),
                           "speaker": groups, "phone_score": lab["reg"]})
    out["sets"] = {}
    for name, spec in F.SOUND_SETS.items():
        X = F.matrix(t, spec["num"], spec["cat"], vocab)
        cm = F.categorical_mask(spec["num"], spec["cat"], vocab)
        oof, folds = cv_heads(X, lab, KINDS, groups, params, cm)
        cal, cmp, pcv = calibrate_oof(oof, lab, KINDS, groups)
        res = {"reg": head_metrics("reg", lab["reg"], oof["reg"]),
               "reg_theirs": theirs_metric(df, oof["reg"])}
        for h in ("len", "str"):
            res[h] = head_metrics("prob", lab[h], pcv[h])
            res[h]["calib_cv_ece"] = cmp[h]
            res[h]["calib"] = cal[h]["kind"]
            res[h]["auc_raw"] = head_metrics("prob", lab[h], 1 / (1 + np.exp(-oof[h])))["auc"]
        # monotone consistency on OOF: share of rows where strict > lenient before capping
        from bench.calib.common import apply_calib

        pl, ps = apply_calib(cal["len"], oof["len"]), apply_calib(cal["str"], oof["str"])
        res["share_strict_gt_lenient_before_cap"] = float(np.mean(ps > pl + 1e-9))
        heads, parity = fit_final(X, lab, KINDS, params, cm, clips={"reg": (0.0, 2.0)})
        for h in ("len", "str"):
            heads[h]["calib"] = cal[h]
        res["walker_parity_max_abs"] = parity
        res["n_trees"] = {h: int(len(v["trees"]["roots"])) for h, v in heads.items()}
        res["depth"] = {h: int(v["trees"]["depth"]) for h, v in heads.items()}
        # B ships (English); A (speechocean762-only universal candidate) and C (benchmark
        # transparency) stay in the cache; the shipped universal model is sound_U (train_wide)
        dest = (ship_dir() if name == "B" else calib_cache()) / f"sound_{name}.npz"
        save_level(dest, spec["num"], spec["cat"], {c: vocab[c] for c in spec["cat"]},
                   {h: {k: v for k, v in hh.items() if k != "_model"} for h, hh in heads.items()},
                   {"level": "sound", "set": name, "trained_on": "speechocean762 train",
                    "thresholds": {"len": LEN_THR, "str": STR_THR}})
        res["file"] = str(dest)
        res["kb"] = dest.stat().st_size / 1024
        out["sets"][name] = res
        for h in ("reg", "len", "str"):
            oof_df[f"{name}_{h}_raw"] = oof[h]
        oof_df[f"{name}_z_len"] = np.log(pl / (1 - pl))
        oof_df[f"{name}_z_str"] = np.minimum(np.log(ps / (1 - ps)), oof_df[f"{name}_z_len"])
        log(name, "reg", round(res["reg"]["pearson"], 4), "theirs", round(res["reg_theirs"]["pearson"], 4),
            "len auc", round(res["len"]["auc"], 4), "ece", round(res["len"]["ece"], 4),
            "str auc", round(res["str"]["auc"], 4), "ece", round(res["str"]["ece"], 4),
            "parity", parity, "kb", round(res["kb"], 1))
        save_json("calib_sound.json", out)
    oof_df.to_parquet(calib_cache() / "sound_oof.parquet", index=False)
    # B vs A on CV (paired speaker bootstrap of the reg Pearson on our phones)
    from bench.metrics import paired_bootstrap, pearson

    m = np.isfinite(lab["reg"])
    pb = paired_bootstrap(lambda y, s: pearson(s, y), lab["reg"][m], oof_df["B_reg_raw"].values[m],
                          oof_df["A_reg_raw"].values[m], clusters=groups[m], n_boot=1000)
    out["B_minus_A_reg_pearson"] = pb
    out["english_choice"] = "B" if pb["lo"] > 0 else "A"
    log("B - A", pb, "->", out["english_choice"])
    save_json("calib_sound.json", out)


if __name__ == "__main__":
    main()
