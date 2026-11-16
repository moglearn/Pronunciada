"""Round 6: word 0-10 accuracy regression with word-pooled encoder activations, speechocean762 TRAIN.

    python -m bench.calib.train_word_emb ablate
    python -m bench.calib.train_word_emb fit --variant word     # writes pronunciada/data/calib/word_reg_r6.npz

What it fits on: speechocean762 TRAIN words (15,849 words, 125 speakers); nothing from test.
Inputs: the shipped word-regression inputs (WORD_NUM on out-of-fold universal sound predictions,
rebuilt as bench/calib/train_word2.py builds them) + a ridge projection of the mean ++ std of
the ZIPA stack-1 activations over the word's frames (pronunciada.delivery.pool_words, extracted
through the public Scorer by bench/calib/delivery_extract.py), fitted in-fold exactly like the
sentence projection of round 5 (inner out-of-fold values for the trees' training rows).
Variant "word+utt" also gives the trees the utterance-level ZIPA projection (emb_accuracy).
CV: 5-fold speaker-grouped x 3 seeds. Adopted if word Pearson improves by >= 0.010.
The word ACCEPTANCE heads (word.npz) are not touched; the sentence level keeps reading the
round-2 word regression (word_reg.npz), the new one is only the reported 0-10 word score.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from bench.calib.common import SMALL_GRID, calib_cache, fit, raw, save_json, ship_dir
from bench.calib.fitlib import log
from bench.calib.train_delivery import (SEEDS, pick_alpha, ridge_path, seeded_folds)
from pronunciada import features as F

PARAMS = SMALL_GRID["s2"]          # the shipped word_reg hyper-parameters (round 2)
ALPHAS = np.logspace(1, 5, 9)


def word_inputs():
    p = calib_cache() / "round2_word_inputs.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    from bench.calib import train_word2 as W2

    parts = W2.build()
    W = parts["so762"][0]
    out = {"X": np.column_stack([np.asarray(W[c], dtype=np.float64) for c in F.WORD_NUM]),
           "y": np.asarray(W["y_reg"], dtype=np.float64), "spk": np.asarray(W["grp"]).astype(str),
           "utt": np.asarray(W["utt"]).astype(str), "word_index": np.asarray(W["word_index"], dtype=np.int64)}
    np.savez(p, **out)
    return out


def load():
    d = word_inputs()
    al = pickle.loads((calib_cache() / "delivery_speechocean762_train_align.pkl").read_bytes())
    E = np.stack([al[u]["word_pooled"][k] for u, k in zip(d["utt"], d["word_index"])]).astype(np.float64)
    d["E"] = E
    z = np.load(calib_cache() / "delivery_speechocean762_train.npz")
    pos = {u: i for i, u in enumerate(z["utt"])}
    d["Eu"] = z["emb_enc1"][[pos[u] for u in d["utt"]]].astype(np.float64)
    d["ugroup"] = d["utt"]
    return d


def _proj_fit(E, y, spk, seed):
    a, oof, _ = pick_alpha(E, y[:, None], spk, alphas=ALPHAS, seed=seed)
    path, mu, sd = ridge_path(E, y[:, None], [a])
    W, b = path[0]
    return a, oof[:, 0], {"mu": mu, "sd": sd, "W": W, "b": b}


def _proj_apply(p, E):
    return (((E - p["mu"]) / p["sd"]) @ p["W"] + p["b"])[:, 0]


def cv(d, variant, seed, folds=None, params=PARAMS):
    X, y, spk = d["X"], d["y"], d["spk"]
    oof = np.zeros(len(y))
    folds = folds or seeded_folds(spk, seed)
    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        if variant in ("word", "word+utt"):
            _, ptr, p = _proj_fit(d["E"][tr], y[tr], spk[tr], seed + 100)
            Xtr, Xte = np.column_stack([Xtr, ptr]), np.column_stack([Xte, _proj_apply(p, d["E"][te])])
        if variant == "word+utt":
            _, ptr, p = _proj_fit(d["Eu"][tr], y[tr], spk[tr], seed + 200)
            Xtr, Xte = np.column_stack([Xtr, ptr]), np.column_stack([Xte, _proj_apply(p, d["Eu"][te])])
        m = fit("reg", params, Xtr, y[tr], [False] * Xtr.shape[1])
        oof[te] = np.clip(raw(m, Xte), 0, 10)
    return float(np.corrcoef(oof, y)[0, 1])


def ablate(a):
    from concurrent.futures import ProcessPoolExecutor

    d = load()
    log("words", d["X"].shape, d["E"].shape)
    jobs = [(v, s) for v in ("base", "word", "word+utt") for s in SEEDS]
    with ProcessPoolExecutor(4) as ex:
        res = dict(zip(jobs, ex.map(_job, [(d, v, s) for v, s in jobs])))
    out = {"seeds": list(SEEDS), "variants": {}}
    for v in ("base", "word", "word+utt"):
        r = np.array([res[(v, s)] for s in SEEDS])
        dl = r - np.array([res[("base", s)] for s in SEEDS])
        out["variants"][v] = {"mean": float(r.mean()), "per_seed": r.tolist(), "delta": float(dl.mean()),
                              "delta_se": float(dl.std(ddof=1) / np.sqrt(len(dl)))}
        log(v, out["variants"][v])
    save_json("round6_word_ablation.json", out)


def _job(args):
    d, v, s = args[:3]
    folds = args[3] if len(args) > 3 else None
    params = args[4] if len(args) > 4 else PARAMS
    return cv(d, v, s, folds=folds, params=params)


def harsh(a):
    """Robustness: train on adult speakers -> evaluate on children (< 15) and vice versa,
    pooled OOF Pearson; shipped (s2) and a more regularised (s3) tree setting."""
    from concurrent.futures import ProcessPoolExecutor

    from bench.pipeline.common import records

    d = load()
    age = {r["id"]: r["labels"]["age"] for r in records("speechocean762") if r["split"] == "train"}
    child = np.array([age[u] < 15 for u in d["utt"]])
    folds = [(np.flatnonzero(~child), np.flatnonzero(child)), (np.flatnonzero(child), np.flatnonzero(~child))]
    jobs = [(v, p) for v in ("base", "word", "word+utt") for p in ("s2", "s3")]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_job, [(d, v, 0, folds, SMALL_GRID[p]) for v, p in jobs]))
    out = {f"{v}/{p}": r for (v, p), r in zip(jobs, res)}
    for k, r in out.items():
        log("age split", k, round(r, 4))
    save_json("round6_word_harsh.json", out)


def fit_ship(a):
    from bench.calib.common import export_trees, walker_parity
    from pronunciada.calibrate import save_level

    d = load()
    X, y, spk = d["X"], d["y"], d["spk"]
    alpha, P, proj = _proj_fit(d["E"], y, spk, seed=0)        # trees see 5-fold OOF projections
    Xf = np.column_stack([X, P])
    m = fit("reg", PARAMS, Xf, y, [False] * Xf.shape[1])
    tr = export_trees(m, Xf.shape[1])
    dest = ship_dir() / "word_reg_r6.npz"
    num = list(F.WORD_NUM) + ["emb_word"]
    save_level(dest, num, [], {}, {"reg": {"kind": "reg", "trees": tr, "clip": (0.0, 10.0)}},
               {"level": "word_reg", "round": 6, "projection": {"names": ["emb_word"], "layers": ["enc1"],
                                                                 "alpha": alpha}})
    z = dict(np.load(dest, allow_pickle=False))
    for k in ("mu", "sd", "W", "b"):
        z[f"proj/{k}"] = np.asarray(proj[k], dtype=np.float32 if k == "W" else np.float64)
    np.savez_compressed(dest, **z)
    out = {"alpha": alpha, "walker_parity_max_abs": walker_parity(m, Xf), "kb": dest.stat().st_size / 1024}
    save_json("round6_word_fit.json", out)
    log("wrote", dest, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("ablate", "fit", "harsh"))
    ap.add_argument("--variant", default="word")
    a = ap.parse_args()
    {"ablate": ablate, "fit": fit_ship, "harsh": harsh}[a.cmd](a)


if __name__ == "__main__":
    main()
