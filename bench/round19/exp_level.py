"""Round 19 (English): sound and word scores of the sequence head next to the shipped
sound_E / word_reg_r6 recipes, out-of-fold on the same folds (speechocean762 TRAIN only).

    python -m bench.round19.exp_level --seq NAME [--seeds 0,1,2]

Sound: the round-12b recipe (bench/calib/train_sound_emb.py: feature set B + in-fold emb_proj +
native-prototype inputs, the shipped hyper-parameters) refitted per fold of
bench.round19.data.folds; label = the expert phone score of the sound table. Reported: Pearson of
the 0-2 regression, AUC of "-score" for phone score < 2 (detection on train folds) and for < 1.
Word: the round-6 recipe (bench/calib/train_word_emb.py variant "word") on the same folds;
Pearson of the 0-10 score and AUC for word accuracy < 8 / < 6.
Both against the sequence head's OOF outputs (bench/round19/exp_seq.py, saved per seed) and
blends w * seq + (1 - w) * shipped recipe.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

from bench.round19 import data as RD

WS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def fold_of_speaker(seed):
    U = RD.load("train", keep=["speaker", "labels"])
    spk = np.array([u["speaker"] for u in U])
    age = np.array([u["labels"]["age"] for u in U])
    f = {}
    for j, (_, te) in enumerate(RD.folds(spk, age, seed)):
        for i in te:
            f[spk[i]] = j
    return f


def folds_from(groups, fmap):
    fid = np.array([fmap[g] for g in groups])
    return [(np.flatnonzero(fid != j), np.flatnonzero(fid == j)) for j in range(5)]


def auc(y_bad, s):
    from bench.metrics import roc_auc

    return float(roc_auc(np.asarray(y_bad).astype(int), np.asarray(s, float)))


def sound_oof(fmap):
    from bench.calib import train_sound_emb as TS
    from bench.calib.common import fit, load_json, raw
    from bench.calib.train_sound import KINDS
    from pronunciada import features as F

    df, t, lab, groups, vocab, V = TS.load()
    okV = np.isfinite(V[:, 0])
    spec = F.SOUND_SETS[TS.SPEC]
    params = load_json("calib_sound.json")["params"]
    y = lab["reg"]
    num = list(spec["num"]) + TS.EXTRA
    oof = np.full(len(df), np.nan)
    for tr, te in folds_from(groups, fmap):
        tt = dict(t)
        tt["emb_proj"] = TS.projection_for(V, y, groups, tr, te, okV)
        X = F.matrix(tt, num, spec["cat"], vocab)
        cm = F.categorical_mask(num, spec["cat"], vocab)
        ok = tr[np.isfinite(y[tr])]
        mdl = fit(KINDS["reg"], params["reg"], X[ok], y[ok], cm)
        oof[te] = raw(mdl, X[te])
    return df.utt.values.astype(str), df.phone_index.values.astype(int), y, oof


def word_oof(fmap):
    from bench.calib import train_word_emb as TW

    d = TW.load()
    from bench.calib.common import fit, raw

    X, y, spk = d["X"], d["y"], d["spk"]
    oof = np.zeros(len(y))
    for tr, te in folds_from(spk, fmap):
        _, ptr, p = TW._proj_fit(d["E"][tr], y[tr], spk[tr], 100)
        Xtr, Xte = np.column_stack([X[tr], ptr]), np.column_stack([X[te], TW._proj_apply(p, d["E"][te])])
        m = fit("reg", TW.PARAMS, Xtr, y[tr], [False] * Xtr.shape[1])
        oof[te] = np.clip(raw(m, Xte), 0, 10)
    return d["utt"], d["word_index"], y, oof


def main():
    from bench.calib.common import load_json, save_json
    from bench.data.common import cache_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    oofdir = cache_root() / "round19" / "english" / "oof"
    R = load_json("round19_english_cv_level.json") or {}
    res = {"sound": {}, "word": {}}
    for seed in [int(s) for s in a.seeds.split(",")]:
        fmap = fold_of_speaker(seed)
        seq = pickle.loads((oofdir / f"seq_{a.seq}_seed{seed}_pw.pkl").read_bytes())
        cache = oofdir / f"level_ship_seed{seed}.pkl"
        if cache.exists():
            S, Wd = pickle.loads(cache.read_bytes())
        else:
            S, Wd = sound_oof(fmap), word_oof(fmap)
            cache.write_bytes(pickle.dumps((S, Wd)))
        # sound
        utt, pi, y, ship = S
        sq = np.array([seq[u][0][k] * 2.0 if u in seq and k < len(seq[u][0]) else np.nan for u, k in zip(utt, pi)])
        m = np.isfinite(y) & np.isfinite(ship) & np.isfinite(sq)
        for w in WS:
            p = w * sq[m] + (1 - w) * ship[m]
            r = res["sound"].setdefault(f"w{w}", [])
            r.append({"pearson": RD.pearson(p, y[m]), "auc_lt2": auc(y[m] < 2, -p), "auc_lt1": auc(y[m] < 1, -p),
                      "n": int(m.sum())})
        # word: the sequence head's word value = its value on the word's first sound
        wutt, wi, wy, wship = Wd
        wq = np.full(len(wy), np.nan)
        U ={v["id"]: v for v in RD.load("train", keep=["phone_word"])}
        for j, (u, k) in enumerate(zip(wutt, wi)):
            if u in seq and u in U:
                pw = np.asarray(U[u]["phone_word"])
                hit = np.flatnonzero(pw == k)
                if len(hit):
                    wq[j] = seq[u][1][hit[0]] * 10.0
        m = np.isfinite(wy) & np.isfinite(wq)
        for w in WS:
            p = w * wq[m] + (1 - w) * wship[m]
            r = res["word"].setdefault(f"w{w}", [])
            r.append({"pearson": RD.pearson(p, wy[m]), "auc_lt8": auc(wy[m] < 8, -p), "auc_lt6": auc(wy[m] < 6, -p),
                      "n": int(m.sum())})
        print(f"seed {seed} done", flush=True)
    out = {}
    for lev, d in res.items():
        out[lev] = {}
        for k, rs in d.items():
            out[lev][k] = {m: float(np.mean([r[m] for r in rs])) for m in rs[0]}
            print(lev, k, " ".join(f"{m}={v:.4f}" for m, v in out[lev][k].items()), flush=True)
    R[a.seq] = {"mean": out, "seeds": res}
    save_json("round19_english_cv_level.json", R)


if __name__ == "__main__":
    main()
