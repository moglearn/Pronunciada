"""Word- and sentence-level calibrators on speechocean762 TRAIN (5-fold speaker-grouped CV).

    python -m bench.calib.train_levels [--sound U]

Inputs are built with pronunciada.features.word_table / sentence_table from OUT-OF-FOLD sound
predictions of the shipped universal sound model (column prefix U_ in
$PRONUNCIADA_CACHE/calib/sound_oof.parquet), and the sentence inputs from out-of-fold word
predictions, so no level sees predictions made on its own training labels.

Word heads: reg = human word accuracy 0-10; len = accuracy >= WORD_LEN; str = >= WORD_STR.
Sentence heads: reg_accuracy / reg_fluency / reg_prosodic / reg_total (0-10);
len = sentence accuracy >= SENT_LEN; str = >= SENT_STR.
Writes pronunciada/data/calib/word.npz, sentence.npz and results/tables/calib_levels.json.
"""
from __future__ import annotations

import argparse

import numpy as np

from bench.calib.common import (SMALL_GRID, calib_cache, load_table, save_json, ship_dir,
                                speaker_folds, table_columns)
from bench.calib.fitlib import calibrate_oof, cv_heads, fit_final, head_metrics, log
from pronunciada import features as F
from pronunciada.calibrate import save_level

WORD_LEN, WORD_STR = 5.0, 8.0
SENT_LEN, SENT_STR = 6.0, 8.0


def balance(y, thrs):
    y = y[np.isfinite(y)]
    return {f">={t}": float((y >= t).mean()) for t in thrs}


def run_level(name, X, labels, kinds, groups, clips, num, extra_meta, grid_head):
    g = {}
    for gname, params in SMALL_GRID.items():
        oof, _ = cv_heads(X, {grid_head: labels[grid_head]}, kinds, groups, params, [False] * X.shape[1])
        g[gname] = head_metrics("reg", labels[grid_head], oof[grid_head])["pearson"]
        log(name, "grid", gname, round(g[gname], 4))
    chosen = max(g, key=g.get)
    params = SMALL_GRID[chosen]
    oof, _ = cv_heads(X, labels, kinds, groups, params, [False] * X.shape[1])
    cal, cmp, pcv = calibrate_oof(oof, labels, kinds, groups)
    res = {"grid": g, "chosen": chosen, "params": params, "heads": {}}
    from bench.calib.common import apply_calib

    oof_out = {}
    for h, k in kinds.items():
        if k == "reg":
            res["heads"][h] = head_metrics("reg", labels[h], oof[h])
            oof_out[h] = np.clip(oof[h], *clips[h])
        else:
            res["heads"][h] = head_metrics("prob", labels[h], pcv[h])
            res["heads"][h]["calib"] = cal[h]["kind"]
            res["heads"][h]["calib_cv_ece"] = cmp[h]
            p = apply_calib(cal[h], oof[h])
            oof_out["z_" + h] = np.log(p / (1 - p))
    oof_out["z_str"] = np.minimum(oof_out["z_str"], oof_out["z_len"])
    heads, parity = fit_final(X, labels, kinds, params, [False] * X.shape[1], clips=clips)
    for h in cal:
        heads[h]["calib"] = cal[h]
    dest = ship_dir() / f"{name}.npz"
    save_level(dest, num, [], {}, {h: {k: v for k, v in hh.items() if k != "_model"} for h, hh in heads.items()},
               {"level": name, **extra_meta})
    res["walker_parity_max_abs"] = parity
    res["n_trees"] = {h: int(len(v["trees"]["roots"])) for h, v in heads.items()}
    res["kb"] = dest.stat().st_size / 1024
    for h, v in res["heads"].items():
        log(name, h, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "calib_cv_ece"})
    return res, oof_out


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--sound", default="U", help="prefix of the OOF sound predictions to use")
    a = ap.parse_args()
    df = load_table("speechocean762", "train")
    t = table_columns(df)
    od = pd.read_parquet(calib_cache() / "sound_oof.parquet")
    od = df[["utt", "phone_index"]].astype({"phone_index": int}).merge(od, on=["utt", "phone_index"], how="left")
    sl = od[f"{a.sound}_z_len"].to_numpy(float)
    ss = od[f"{a.sound}_z_str"].to_numpy(float)
    sr = np.clip(od[f"{a.sound}_reg_raw"].to_numpy(float), 0, 2)
    assert np.isfinite(sl).all() and np.isfinite(ss).all(), "missing OOF sound predictions"
    out = {"sound_inputs": a.sound}
    # ------------------------------------------------------------------ word
    W = F.word_table(t, sl, ss, sr, group=t["_group"])
    first = W["row"]
    wacc = np.asarray(t["word_accuracy"], dtype=np.float64)[first]
    wspk = df.speaker.astype(str).values[first]
    wutt = df.utt.values[first]
    assert len(wacc) == len(W["wl_mean"])
    lab = {"reg": wacc,
           "len": np.where(np.isfinite(wacc), (wacc >= WORD_LEN).astype(float), np.nan),
           "str": np.where(np.isfinite(wacc), (wacc >= WORD_STR).astype(float), np.nan)}
    out["word_balance"] = balance(wacc, (3, 4, 5, 6, 7, 8, 9, 10))
    out["word_n"] = int(len(wacc))
    out["word_n_labelled"] = int(np.isfinite(wacc).sum())
    out["word_omitted_rate"] = float(np.mean(W["omitted_f"]))
    log("word balance", out["word_balance"], "omitted", out["word_omitted_rate"])
    XW = np.column_stack([np.asarray(W[c], dtype=np.float64) for c in F.WORD_NUM])
    kinds = {"reg": "reg", "len": "prob", "str": "prob"}
    res, woof = run_level("word", XW, lab, kinds, wspk, {"reg": (0.0, 10.0)}, F.WORD_NUM,
                          {"thresholds": {"len": WORD_LEN, "str": WORD_STR}, "sound_inputs": a.sound},
                          "reg")
    out["word"] = res
    save_json("calib_levels.json", out)
    pd.DataFrame({"utt": wutt, "word_index": W["word_index"], "speaker": wspk, "accuracy": wacc,
                  **{k: v for k, v in woof.items()}}).to_parquet(calib_cache() / "word_oof.parquet", index=False)
    # ------------------------------------------------------------------ sentence
    gcodes, ng = F._codes(np.asarray(t["_group"]).astype(object))
    firsts = np.r_[0, np.flatnonzero(np.diff(gcodes)) + 1]
    dur = np.asarray(t["utt_dur"], dtype=np.float64)[firsts]
    S = F.sentence_table(t, sl, ss, sr, W, woof["z_len"], woof["z_str"], woof["reg"], dur, group=t["_group"])
    XS = np.column_stack([np.asarray(S[c], dtype=np.float64) for c in F.SENT_NUM])
    sspk = df.speaker.astype(str).values[firsts]
    slab = {}
    for k in ("accuracy", "fluency", "prosodic", "total"):
        slab[f"reg_{k}"] = np.asarray(t[f"sent_{k}"], dtype=np.float64)[firsts]
    acc = slab["reg_accuracy"]
    slab["len"] = (acc >= SENT_LEN).astype(float)
    slab["str"] = (acc >= SENT_STR).astype(float)
    out["sentence_balance"] = balance(acc, (4, 5, 6, 7, 8, 9))
    out["sentence_n"] = int(len(acc))
    skinds = {**{f"reg_{k}": "reg" for k in ("accuracy", "fluency", "prosodic", "total")},
              "len": "prob", "str": "prob"}
    sclips = {f"reg_{k}": (0.0, 10.0) for k in ("accuracy", "fluency", "prosodic", "total")}
    res, soof = run_level("sentence", XS, slab, skinds, sspk, sclips, F.SENT_NUM,
                          {"thresholds": {"len": SENT_LEN, "str": SENT_STR}, "sound_inputs": a.sound},
                          "reg_accuracy")
    out["sentence"] = res
    save_json("calib_levels.json", out)
    log("done")


if __name__ == "__main__":
    main()
