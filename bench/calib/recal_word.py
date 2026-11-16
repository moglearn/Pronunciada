"""Round 3, item 3: re-fit the word calibration maps on inputs produced like at runtime.

    python -m bench.calib.recal_word

Finding (train data only): the round-2 word model is trained on speechocean762 words whose sound
inputs are OUT-OF-FOLD universal-model outputs, but at runtime its inputs come from the FINAL
universal model, whose lenient log-odds are ~0.12 lower on average (train words: wl_mean 5.31 OOF
vs 5.19 final; test words 5.19). Applied to TRAIN words with final-model inputs the word model's
mean P(strict) is 0.695 vs 0.776 with OOF inputs — the same under-confidence seen on test. Fix:
keep the trees, re-fit the 1-D calibration maps on held-out-speaker predictions of the CV word
models evaluated on FINAL-model sound inputs (5-fold speaker CV on speechocean762 train, exactly
the round-2 training mix). Writes the new maps into pronunciada/data/calib/word.npz and
results/tables/calib_recal_word.json.
"""
from __future__ import annotations

import json

import numpy as np

from bench.calib.common import (SMALL_GRID, apply_calib, choose_calib, fit, raw, save_json, ship_dir,
                                speaker_folds)
from bench.calib.train_word2 import CANDS, build, stack, u_outputs, weights
from pronunciada import features as F
from pronunciada.calibrate import save_level


def main():
    from bench.calib.common import ece, load_json, load_table, table_columns

    r2 = load_json("calib_round2.json")
    cname, pname = r2["chosen_word"].split("/")
    cands, params = CANDS[cname], SMALL_GRID[pname]
    parts = build()
    cols = sorted(set(F.WORD2_NUM) | set(F.WORD_NUM))
    T = stack(parts, cols)
    corp = T["corpus"]
    X = np.column_stack([T[c].astype(float) for c in F.WORD2_NUM])
    # speechocean762 train words with FINAL universal-model sound inputs (as at runtime)
    df = load_table("speechocean762", "train")
    t = table_columns(df)
    a, b, c = u_outputs(t)
    Wf = F.word_table(t, a, b, c, group=t["_group"])
    Xf = np.column_stack([np.asarray(Wf[k], float) for k in F.WORD2_NUM])
    so = np.flatnonzero(corp == "so762")
    assert len(so) == len(Xf)
    folds = speaker_folds(T["grp"][so])
    inc = np.isin(corp, cands)
    out = {"mean_input_shift": {"wl_mean_oof": float(np.nanmean(X[so, F.WORD2_NUM.index("wl_mean")])),
                                "wl_mean_final": float(np.nanmean(Xf[:, F.WORD2_NUM.index("wl_mean")]))}}
    with np.load(ship_dir() / "word.npz", allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    meta = json.loads(bytes(arr.pop("meta")).decode())
    heads = {}
    for h in ("len", "str"):
        y = T[f"y_{h}"]
        z_oof, z_fin = np.full(len(so), np.nan), np.full(len(so), np.nan)
        for tr, te in folds:
            hold = np.zeros(len(corp), bool)
            hold[so[te]] = True
            m = np.flatnonzero(inc & ~hold & np.isfinite(y))
            mdl = fit("prob", params, X[m], y[m], [False] * X.shape[1], weights(T, m, h))
            z_oof[te], z_fin[te] = raw(mdl, X[so[te]]), raw(mdl, Xf[te])
        yy = y[so]
        ok = np.isfinite(yy)
        old_map = meta["heads"][h]["calib"]
        old = {"kind": old_map["kind"], "params": old_map.get("params", arr.get(f"{h}/calib_y", np.array([])).tolist()),
               "x": arr[f"{h}/calib_x"].tolist() if f"{h}/calib_x" in arr else None}
        new, cmp = choose_calib(z_fin[ok], yy[ok], T["grp"][so][ok])
        p_old_oof = apply_calib(old, z_oof[ok])
        p_old_fin = apply_calib(old, z_fin[ok])
        # nested-CV estimate of the new map's ECE on final-model inputs
        from sklearn.model_selection import GroupKFold

        from bench.calib.common import fit_calib

        p_new = np.zeros(ok.sum())
        for tr, te in GroupKFold(5).split(z_fin[ok], yy[ok], T["grp"][so][ok]):
            p_new[te] = apply_calib(fit_calib(z_fin[ok][tr], yy[ok][tr], new["kind"]), z_fin[ok][te])
        out[h] = {"pos_rate": float(yy[ok].mean()),
                  "old_map_oof_inputs": {"mean_p": float(p_old_oof.mean()), "ece": ece(yy[ok], p_old_oof)},
                  "old_map_final_inputs": {"mean_p": float(p_old_fin.mean()), "ece": ece(yy[ok], p_old_fin)},
                  "new_map_final_inputs_cv": {"mean_p": float(p_new.mean()), "ece": ece(yy[ok], p_new)},
                  "new_map": new["kind"], "map_cv_ece": cmp}
        print(h, json.dumps(out[h]), flush=True)
        heads[h] = new
    # rewrite word.npz with the new maps (trees unchanged)
    tree = {}
    for h in meta["heads"]:
        tree[h] = {"kind": meta["heads"][h]["kind"],
                   "trees": {k[len(h) + 1:]: v for k, v in arr.items() if k.startswith(h + "/") and "calib" not in k},
                   "calib": heads[h]}
    extra = {k: v for k, v in meta.items() if k not in ("num", "cat", "vocab", "encoding", "heads")}
    extra["calibration_inputs"] = "final universal-model sound outputs (round 3)"
    save_level(ship_dir() / "word.npz", meta["num"], meta["cat"], meta["vocab"], tree, extra, meta.get("encoding", "mixed"))
    save_json("calib_recal_word.json", out)


if __name__ == "__main__":
    main()
