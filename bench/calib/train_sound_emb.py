"""Round 12b: English sound heads with per-sound ZIPA-layer inputs (candidate d(i)), speechocean762 TRAIN.

    python -m bench.calib.train_sound_emb cv        # CV + paired speaker bootstrap vs sound_B, OOF to sound_oof.parquet
    python -m bench.calib.train_sound_emb fit       # writes pronunciada/data/calib/sound_E.npz + sound_E_emb.npz

Inputs: the shipped English feature set B (bench/calib/train_sound.py, same hyper-parameters, same
5-fold speaker-grouped folds) plus
  emb_proj      ridge projection of the per-sound pooled enc1/enc3/enc5 activations (1024-d,
                pronunciada.sound_emb.phone_vectors) to the expert phone score, fitted IN-FOLD
                (training rows get inner out-of-fold values; runtime uses the projection fitted on
                all labelled rows);
  proto_*       cosine to the native prototype of the expected sound (best over its accepted
                alternatives), best cosine to another phone, margin, has-prototype flag; prototypes
                from 500 FLEURS-en TRAIN clips (bench/calib/sound_emb.py protos).
Vectors: bench/calib/sound_emb.py so762 (the public Scorer on every train utterance).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")

EXTRA = ["emb_proj", "proto_cos", "proto_other", "proto_margin", "proto_has"]
ALPHA = 3e3
SPEC = "B"


def load():
    from bench.calib.common import calib_cache, load_table, table_columns, vocab_for
    from bench.calib.train_sound import labels_so762
    from pronunciada import features as F

    df = load_table("speechocean762", "train")
    t = table_columns(df)
    lab = labels_so762(t)
    groups = df.speaker.astype(str).values
    vocab = vocab_for(t, F.SOUND_SETS["A"]["cat"] + F.IDENTITY_CAT)
    vec = pickle.loads((calib_cache() / "sound2" / "so762_train_vectors.pkl").read_bytes())
    n = len(df)
    V = np.full((n, 1024), np.nan, np.float32)
    E = np.full((n, 3), np.nan)
    for i, (u, k, ph) in enumerate(zip(df.utt.values, df.phone_index.values.astype(int), df.phone.values)):
        it = vec.get(u)
        if it is None or k >= len(it["ipa"]) or it["ipa"][k] != ph:
            continue
        V[i] = it["V"][k]
        E[i] = [it["P"][c][k] for c in ("proto_cos", "proto_other", "proto_margin")]
    del vec
    t = dict(t)
    has = np.isfinite(E[:, 0])
    t["proto_cos"], t["proto_other"], t["proto_margin"] = (np.where(has, E[:, j], np.nan) for j in range(3))
    t["proto_has"] = has.astype(float)
    return df, t, lab, groups, vocab, V


def fit_projection(V, y, rows):
    from sklearn.linear_model import Ridge

    mu, sd = V[rows].mean(0), V[rows].std(0) + 1e-6
    m = Ridge(alpha=ALPHA).fit((V[rows] - mu) / sd, y[rows])
    return {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "W": m.coef_.astype(np.float32),
            "b": np.float32(m.intercept_)}


def apply_projection(P, V):
    return ((V - P["mu"]) / P["sd"]) @ P["W"] + P["b"]


def projection_for(V, y, groups, tr, te, okV):
    """emb_proj for rows te (projection fitted on labelled rows of tr) and inner-OOF for tr."""
    from sklearn.model_selection import GroupKFold

    out = np.full(len(y), np.nan)
    lab_tr = tr[np.isfinite(y[tr]) & okV[tr]]
    P = fit_projection(V, y, lab_tr)
    for rows in (te, tr):
        r = rows[okV[rows]]
        out[r] = apply_projection(P, V[r])
    for itr, ite in GroupKFold(5).split(lab_tr, None, groups[lab_tr]):
        Pi = fit_projection(V, y, lab_tr[itr])
        out[lab_tr[ite]] = apply_projection(Pi, V[lab_tr[ite]])
    return out


def prec_at_recall(y_bad, score_bad, recalls=(0.20, 0.33, 0.47)):
    o = np.argsort(-score_bad, kind="stable")
    yb = y_bad[o].astype(float)
    tp = np.cumsum(yb)
    return [float(tp[int(np.searchsorted(tp, r * yb.sum()))] / (int(np.searchsorted(tp, r * yb.sum())) + 1))
            for r in recalls]


def metrics(y, oof):
    from bench.metrics import pearson, roc_auc

    m = np.isfinite(y)
    bad = (y[m] < 2).astype(int)
    return {"pearson": pearson(oof["reg"][m], y[m]), "auc_lt2": roc_auc(bad, -oof["reg"][m]),
            "auc_str": roc_auc((y[m] >= 1.8).astype(int), oof["str"][m]),
            "auc_len": roc_auc((y[m] >= 1.0).astype(int), oof["len"][m]),
            "prec_str": prec_at_recall(bad, -oof["str"][m]), "prec_reg": prec_at_recall(bad, -oof["reg"][m])}


def cv():
    import pandas as pd

    from bench.calib.common import calib_cache, fit, raw, save_json, speaker_folds
    from bench.calib.fitlib import calibrate_oof
    from bench.calib.train_sound import KINDS
    from pronunciada import features as F

    df, t, lab, groups, vocab, V = load()
    okV = np.isfinite(V[:, 0])
    spec = F.SOUND_SETS[SPEC]
    folds = speaker_folds(groups)
    from bench.calib.common import load_json

    params = load_json("calib_sound.json")["params"]
    y = lab["reg"]
    res, oofs = {}, {}
    for name, num in (("B", list(spec["num"])), ("E", list(spec["num"]) + EXTRA)):
        oof = {h: np.full(len(df), np.nan) for h in ("reg", "len", "str")}
        for tr, te in folds:
            tt = dict(t)
            if name == "E":
                tt["emb_proj"] = projection_for(V, y, groups, tr, te, okV)
            X = F.matrix(tt, num, spec["cat"], vocab)
            cm = F.categorical_mask(num, spec["cat"], vocab)
            for h in ("reg", "len", "str"):
                yy = lab[h]
                ok = tr[np.isfinite(yy[tr])]
                mdl = fit(KINDS[h], params[h], X[ok], yy[ok], cm)
                oof[h][te] = raw(mdl, X[te])
        cal, _, pcv = calibrate_oof(oof, lab, KINDS, groups)
        oofs[name] = (oof, cal)
        res[name] = metrics(y, oof)
        print(name, json.dumps(res[name]), flush=True)
    # paired speaker bootstrap of E - B
    m = np.isfinite(y)
    spk = groups[m]
    uniq, inv = np.unique(spk, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(0)
    deltas = []
    oB = {h: v[m] for h, v in oofs["B"][0].items()}
    oE = {h: v[m] for h, v in oofs["E"][0].items()}
    yy = y[m]
    for _ in range(1000):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        a, b = metrics(yy[idx], {h: v[idx] for h, v in oE.items()}), metrics(yy[idx], {h: v[idx] for h, v in oB.items()})
        deltas.append([a["pearson"] - b["pearson"], a["auc_lt2"] - b["auc_lt2"], a["auc_str"] - b["auc_str"]]
                      + [x - z for x, z in zip(a["prec_str"], b["prec_str"])])
    D = np.array(deltas)
    names = ["pearson", "auc_lt2", "auc_str", "prec_str_0.20", "prec_str_0.33", "prec_str_0.47"]
    pt = [res["E"]["pearson"] - res["B"]["pearson"], res["E"]["auc_lt2"] - res["B"]["auc_lt2"],
          res["E"]["auc_str"] - res["B"]["auc_str"]] + [x - z for x, z in zip(res["E"]["prec_str"], res["B"]["prec_str"])]
    res["delta"] = {n: {"value": v, "lo": float(np.quantile(D[:, j], 0.025)), "hi": float(np.quantile(D[:, j], 0.975))}
                    for j, (n, v) in enumerate(zip(names, pt))}
    print(json.dumps(res["delta"], indent=1), flush=True)
    # OOF calibrated logits for the coaching fit (columns E_z_len / E_z_str next to B_*)
    from bench.calib.common import apply_calib

    p = calib_cache() / "sound_oof.parquet"
    od = pd.read_parquet(p)
    key = {(u, int(i)): j for j, (u, i) in enumerate(zip(df.utt.values, df.phone_index.values))}
    oof, cal = oofs["E"]
    pl = np.clip(apply_calib(cal["len"], oof["len"]), 1e-6, 1 - 1e-6)
    ps = np.clip(apply_calib(cal["str"], oof["str"]), 1e-6, 1 - 1e-6)
    zl = np.log(pl / (1 - pl))
    zs = np.minimum(np.log(ps / (1 - ps)), zl)
    j = np.array([key[(u, int(i))] for u, i in zip(od.utt.values, od.phone_index.values)])
    od["E_z_len"], od["E_z_str"], od["E_reg_raw"] = zl[j], zs[j], oof["reg"][j]
    od.to_parquet(p, index=False)
    save_json("calib_sound_emb.json", res)


def fit_final():
    from bench.calib.common import load_json, save_json, ship_dir
    from bench.calib.fitlib import calibrate_oof, cv_heads, fit_final as ff
    from bench.calib.sound_emb import proto_path
    from bench.calib.train_sound import KINDS
    from pronunciada import features as F
    from pronunciada.calibrate import save_level

    df, t, lab, groups, vocab, V = load()
    okV = np.isfinite(V[:, 0])
    y = lab["reg"]
    spec = F.SOUND_SETS[SPEC]
    num = list(spec["num"]) + EXTRA
    params = load_json("calib_sound.json")["params"]
    # training rows: inner out-of-fold projection; runtime: projection on all labelled rows
    allrows = np.arange(len(df))
    lab_rows = allrows[np.isfinite(y) & okV]
    from sklearn.model_selection import GroupKFold

    proj = np.full(len(df), np.nan)
    Pall = fit_projection(V, y, lab_rows)
    proj[okV] = apply_projection(Pall, V[okV])
    for itr, ite in GroupKFold(5).split(lab_rows, None, groups[lab_rows]):
        Pi = fit_projection(V, y, lab_rows[itr])
        proj[lab_rows[ite]] = apply_projection(Pi, V[lab_rows[ite]])
    tt = dict(t)
    tt["emb_proj"] = proj
    X = F.matrix(tt, num, spec["cat"], vocab)
    cm = F.categorical_mask(num, spec["cat"], vocab)
    oof, _ = cv_heads(X, lab, KINDS, groups, params, cm)
    cal, _, _ = calibrate_oof(oof, lab, KINDS, groups)
    heads, parity = ff(X, lab, KINDS, params, cm, clips={"reg": (0.0, 2.0)})
    for h in ("len", "str"):
        heads[h]["calib"] = cal[h]
    dest = ship_dir() / "sound_E.npz"
    save_level(dest, num, spec["cat"], {c: vocab[c] for c in spec["cat"]},
               {h: {k: v for k, v in hh.items() if k != "_model"} for h, hh in heads.items()},
               {"level": "sound", "set": "B+emb", "trained_on": "speechocean762 train",
                "thresholds": {"len": 1.0, "str": 1.8},
                "extra": "per-sound ZIPA enc1/enc3/enc5 pooled projection + FLEURS-en native prototypes "
                         "(sound_E_emb.npz, pronunciada/sound_emb.py)"})
    z = np.load(proto_path("en"))
    np.savez_compressed(ship_dir() / "sound_E_emb.npz", proj_mu=Pall["mu"], proj_sd=Pall["sd"],
                        proj_W=Pall["W"], proj_b=np.array([Pall["b"]], np.float32),
                        proto_mu=z["mu"], proto_sd=z["sd"], proto_P=z["P"].astype(np.float16),
                        proto_phones=z["phones"])
    out = {"walker_parity": parity, "kb": {p.name: round(p.stat().st_size / 1024, 1) for p in
                                           (dest, ship_dir() / "sound_E_emb.npz")}}
    save_json("calib_sound_emb_fit.json", out)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cv", "fit"])
    a = ap.parse_args()
    cv() if a.step == "cv" else fit_final()


if __name__ == "__main__":
    main()
