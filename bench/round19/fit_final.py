"""Round 19 (English): fit the sentence heads on all of speechocean762 TRAIN, check the native
guard on FLEURS VALIDATION, export pronunciada/data/calib/<name>.npz.

    python -m bench.round19.fit_final guard --config CFG          # native guard only (dev)
    python -m bench.round19.fit_final export --config CFG --name sentence_r19

CFG uses the syntax of bench/round19/exp_utt.py (one proj: group at most). Tree inputs on train:
SENT_NUM (out-of-fold, as shipped), the listed delivery groups, the F0 summary, and the projection
of the utterance-pooled activations as 5-fold out-of-fold values (the shipped projection is fitted
on all train rows), exactly as round 5 (bench/calib/train_delivery.py fit_ship).

Native guard (results/licence_refit.md, round 5): on the 100 seeded FLEURS-validation English clips
of bench/calib/native_sentence_check.py, the median native fluency, prosodic and total (0-10) may
not drop by more than 0.3 against the shipped sentence_r4 heads on the same calls.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from bench.calib import train_delivery as TD
from bench.calib.common import SMALL_GRID, fit, raw
from bench.round19 import data as RD
from bench.round19 import exp_utt as EU
from pronunciada import delivery as DV
from pronunciada import features as F

HEADS = RD.HEADS
F0_COLS = ["f0_med_hz", "f0_p10_hz", "f0_p90_hz"]
VOICE_COLS = ["centroid_med_hz", "centroid_mean_hz", "rolloff85_med_hz"]
PROJ_NAMES = [f"emb_{h}" for h in HEADS]
MAX_DROP = 0.3


def columns(cfg):
    groups, projs, extra, opts = EU.parse(cfg)
    assert len(projs) <= 1, "one projection group at most"
    cols = list(F.SENT_NUM) + [c for g in groups for c in DV.GROUPS[g]]
    for e in extra:
        cols += {"f0": F0_COLS, "voice": VOICE_COLS}[e]
    return cols, (projs[0] if projs else None), opts


def runtime_rows(utts, cols):
    """Tree inputs (without projection) of utterances from the feature cache, as the runtime
    builds them (S table of the shipped models + delivery + F0 / voice summaries)."""
    X = np.full((len(utts), len(cols)), np.nan)
    for i, u in enumerate(utts):
        src = {**u["S"], **u["delivery"], **u["f0"], **(u.get("voice") or {})}
        X[i] = [src.get(c, np.nan) for c in cols]
    return X


def fit_all(cfg, seed=0):
    d = EU.load_train()
    cols, layers, opts = columns(cfg)
    U_cols = cols[len(F.SENT_NUM):]
    src = {**{c: d["deliv"][c] for c in DV.ALL}, **dict(zip(F0_COLS, d["f0"].T))}
    if "voice" in d:
        src.update(dict(zip(VOICE_COLS, d["voice"].T)))
    X = np.column_stack([d["XB"]] + [src[c] for c in U_cols]) if U_cols else d["XB"]
    Y, spk = d["Y"], d["spk"]
    proj = None
    if layers:
        E = np.column_stack([RD.pooled_part(d["emb"][RD.layer_of(k)], k) for k in layers])
        alpha, P, _ = TD.pick_alpha(E, Y, spk, seed=seed)
        path, mu, sdv = TD.ridge_path(E, Y, [alpha])
        proj = {"alpha": alpha, "mu": mu, "sd": sdv, "W": path[0][0], "b": path[0][1], "layers": list(layers)}
        X = np.column_stack([X, P])
        cols = cols + PROJ_NAMES
    models = []
    for j in range(len(HEADS)):
        models.append(fit("reg", SMALL_GRID["s2"], X, Y[:, j], [False] * X.shape[1],
                          EU.item_weights(d, np.arange(len(Y)), j, opts.get("weight"))))
    return {"cfg": cfg, "cols": cols, "proj": proj, "models": models, "X": X}


def predict(M, utts):
    base_cols = [c for c in M["cols"] if c not in PROJ_NAMES]
    X = runtime_rows(utts, base_cols)
    if M["proj"] is not None:
        p = M["proj"]
        E = np.stack([np.concatenate([RD.pooled_part(u["utt_layers"][RD.layer_of(k)], k) for k in p["layers"]]) for u in utts])
        X = np.column_stack([X, ((E - p["mu"]) / p["sd"]) @ p["W"] + p["b"]])
    return np.column_stack([np.clip(raw(m, X), 0, 10) for m in M["models"]])


def guard(M):
    from bench.data.common import cache_root

    import pickle

    d = cache_root() / "round19" / "english" / "feats" / "fleurs_validation_fp32"
    utts = [v for p in sorted(d.glob("shard*.pkl")) for v in pickle.loads(p.read_bytes()).values() if "error" not in v]
    new = predict(M, utts)
    old = np.array([[u["sent"][h] for h in HEADS] for u in utts])
    out = {"n": len(utts)}
    for j, h in enumerate(HEADS):
        out[h] = {"median_r4": float(np.median(old[:, j])), "median_new": float(np.median(new[:, j])),
                  "median_delta": float(np.median(new[:, j]) - np.median(old[:, j])),
                  "mean_r4": float(old[:, j].mean()), "mean_new": float(new[:, j].mean())}
    out["pass"] = bool(all(out[h]["median_delta"] >= -MAX_DROP for h in ("fluency", "prosodic", "total")))
    return out


def export(M, name):
    from bench.calib.common import export_trees, ship_dir, walker_parity

    r4 = np.load(ship_dir() / "sentence_r4.npz", allow_pickle=False)
    arrays = {k: r4[k] for k in r4.files if k != "meta" and not k.startswith("reg_")}
    meta = json.loads(bytes(r4["meta"]).decode("utf-8"))
    assert meta["num"] == list(F.SENT_NUM) and M["cols"][:len(F.SENT_NUM)] == list(F.SENT_NUM)
    meta["num"] = M["cols"]
    parity = {}
    X = M["X"]
    for h, m in zip(HEADS, M["models"]):
        tr = export_trees(m, X.shape[1])
        parity[h] = walker_parity(m, X)
        for k, v in tr.items():
            arrays[f"reg_{h}/{k}"] = np.asarray(v)
        meta["heads"][f"reg_{h}"] = {"kind": "reg", "n_trees": int(len(tr["roots"])), "depth": int(tr["depth"]),
                                     "clip": [0.0, 10.0]}
    p = M["proj"]
    if p is not None:
        for k in ("mu", "sd", "W", "b"):
            arrays[f"proj/{k}"] = np.asarray(p[k], dtype=np.float32 if k == "W" else np.float64)
        meta["projection"] = {"names": PROJ_NAMES, "layers": p["layers"], "alpha": p["alpha"]}
    meta["round"] = 19
    meta["config"] = M["cfg"]
    meta["reg_trained_on"] = "speechocean762 train (bench/round19/fit_final.py)"
    meta["len_str_from"] = "sentence_r4.npz (unchanged)"
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    dest = ship_dir() / f"{name}.npz"
    np.savez_compressed(dest, **arrays)
    from pronunciada.calibrate import LevelModel

    new, old = LevelModel(dest), LevelModel(ship_dir() / "sentence_r4.npz")
    for h in ("len", "str"):
        assert np.array_equal(new.heads[h].predict(X), old.heads[h].predict(X[:, :len(F.SENT_NUM)])), h
    return dest, parity


def main():
    from bench.calib.common import load_json, save_json

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("guard", "export"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="sentence_r19")
    a = ap.parse_args()
    M = fit_all(a.config)
    g = guard(M)
    print(json.dumps({h: g[h] for h in HEADS}, indent=0), "PASS" if g["pass"] else "FAIL", flush=True)
    R = load_json("round19_english_guard.json") or {}
    R[a.config] = g
    save_json("round19_english_guard.json", R)
    if a.cmd == "export":
        dest, parity = export(M, a.name)
        print("wrote", dest, round(dest.stat().st_size / 1024, 1), "kB; walker parity", parity)
        R2 = load_json("round19_english_fit.json") or {}
        R2[a.name] = {"config": a.config, "columns": M["cols"], "guard": g, "walker_parity_max_abs": parity,
                      "kb": dest.stat().st_size / 1024,
                      "projection_alpha": None if M["proj"] is None else M["proj"]["alpha"]}
        save_json("round19_english_fit.json", R2)


if __name__ == "__main__":
    main()
