"""Round 5: sentence regression heads with delivery features, fitted on speechocean762 TRAIN only.

    python -m bench.calib.train_delivery ablate        # CV ablation -> results/tables/round5_ablation.json
    python -m bench.calib.train_delivery fit --groups rhythm,pitch,...,emb [--layers final]

What it fits on: speechocean762 TRAIN (2,500 utterances, 125 speakers); nothing from test.
Inputs per utterance:
  * SENT_NUM: the shipped (round-2, unchanged since) sentence inputs, rebuilt exactly as
    bench/calib/train_word2.py builds them: pronunciada.features.sentence_table on OUT-OF-FOLD
    universal sound predictions ($PRONUNCIADA_CACHE/calib/sound_oof.parquet), OUT-OF-FOLD
    round-2 word log-odds (config full/s3, calibrated on speechocean762) and OUT-OF-FOLD word
    accuracy regressions (round2_sentence_inputs(); cached as calib/round2_sentence_inputs.npz);
  * delivery features (pronunciada.delivery) and pooled encoder activations, extracted through
    the public Scorer by bench/calib/delivery_extract.py ($PRONUNCIADA_CACHE/calib/
    delivery_speechocean762_train.npz).
Encoder activations enter through a supervised ridge projection onto the four targets
(accuracy, fluency, prosodic, total). It is fitted IN-FOLD: the trees of each outer fold see
projection values that are out-of-fold for their own training rows (inner 5-fold) and a
projection fitted on the outer training rows for the held-out rows; the shipped trees are trained
on 5-fold out-of-fold projections of all train rows, the shipped projection on all train rows.

Evaluation: 5-fold speaker-grouped CV, repeated with 3 seeds (speakers shuffled into folds), on
the four regression heads. A group is adopted when it raises the CV Pearson of prosodic or total
by >= 0.003 on average over the seeds and by more than 2 standard errors of the per-seed paired
difference, and lowers no head by 0.002 or more on average.

`fit` writes pronunciada/data/calib/sentence.npz: the four reg_* heads retrained on SENT_NUM +
the adopted columns, blended with a ridge model on the same inputs (NaN -> train median, each
input clipped to its train 1-99 % range; output = clip(0.7 * trees + 0.3 * ridge, 0, 10); the
blend is kept only because it raised the 3-seed CV of every head; arrays lin/*), the acceptance heads (len, str) copied unchanged from sentence_r4.npz
(their trees index only the first len(SENT_NUM) columns, which keep their positions), and the
projection (proj/mu, proj/sd, proj/W, proj/b + meta["projection"]).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from bench.calib.common import SMALL_GRID, calib_cache, fit, raw, save_json, ship_dir, speaker_folds
from bench.calib.fitlib import log
from pronunciada import delivery as DV
from pronunciada import features as F

HEADS = ("accuracy", "fluency", "prosodic", "total")
SEEDS = (0, 1, 2)
PARAMS = SMALL_GRID["s2"]          # chosen for the sentence level in round 4 (train_levels.py)
ALPHAS = np.logspace(0, 5, 11)
LAYERS = ("enc1", "enc3", "enc5", "final")
EMB_NAMES = [f"emb_{h}" for h in HEADS]
BLEND_W = 0.7                      # trees' weight in the tree / ridge blend (CV: 0.6-0.7 best, 1.0 = trees only)
LIN_ALPHAS = np.logspace(-2, 3, 11)


# ----------------------------------------------------------------------------- data

def round2_sentence_inputs():
    """SENT_NUM matrix of speechocean762 train as bench/calib/train_word2.py built it for the
    shipped sentence model (cached)."""
    p = calib_cache() / "round2_sentence_inputs.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    from bench.calib import train_word2 as W2
    from bench.calib.common import SMALL_GRID as G
    from bench.calib.fitlib import cv_heads

    parts = W2.build()
    cols = sorted(set(F.WORD2_NUM) | set(F.WORD_NUM))
    T = W2.stack(parts, cols)
    X2 = np.column_stack([T[c].astype(float) for c in F.WORD2_NUM])
    # round 2 chose "full/s3"; only the speechocean762 out-of-fold logits are needed here
    corp = T["corpus"]
    so = np.flatnonzero(corp == "so762")
    inc = np.isin(corp, W2.CANDS["full"])
    z = {h: np.full(len(corp), np.nan) for h in ("len", "str")}
    for h in ("len", "str"):
        y = T[f"y_{h}"]
        for tr, te in speaker_folds(T["grp"][so]):
            hold = np.zeros(len(corp), bool)
            hold[so[te]] = True
            m = np.flatnonzero(inc & ~hold & np.isfinite(y))
            mdl = fit("prob", G["s3"], X2[m], y[m], [False] * X2.shape[1], W2.weights(T, m, h))
            z[h][so[te]] = raw(mdl, X2[so[te]])
    zc, _ = W2.calibrated(T, z)
    yreg = T["y_reg"][so]
    Xr = np.column_stack([T[c].astype(float)[so] for c in F.WORD_NUM])
    o, _ = cv_heads(Xr, {"reg": yreg}, {"reg": "reg"}, T["grp"][so], G["s2"], [False] * Xr.shape[1])
    Wso, tso, (sl, ss, sr) = parts["so762"]
    gc, _ = F._codes(np.asarray(tso["_group"]).astype(object))
    firsts = np.r_[0, np.flatnonzero(np.diff(gc)) + 1]
    dur = np.asarray(tso["utt_dur"], float)[firsts]
    S = F.sentence_table(tso, sl, ss, sr, Wso, zc["len"][so], zc["str"][so], np.clip(o["reg"], 0, 10), dur,
                         group=tso["_group"])
    out = {"XB": np.column_stack([np.asarray(S[c], dtype=np.float64) for c in F.SENT_NUM]),
           "utt": np.asarray(tso["utt"])[firsts].astype(str),
           "spk": np.asarray(tso["speaker"])[firsts].astype(str),
           "Y": np.column_stack([np.asarray(tso[f"sent_{k}"], dtype=np.float64)[firsts] for k in HEADS])}
    np.savez(p, **out)
    return out


def load_data():
    r2 = round2_sentence_inputs()
    XB, utt, spk, Y = r2["XB"], r2["utt"], r2["spk"], r2["Y"]
    z = np.load(calib_cache() / "delivery_speechocean762_train.npz")
    pos = {u: i for i, u in enumerate(z["utt"])}
    ix = np.array([pos[u] for u in utt])
    names = [str(x) for x in z["names"]]
    Dm = z["D"][ix]
    deliv = {c: Dm[:, names.index(c)] for c in DV.ALL}
    emb = {k: z[f"emb_{k}"][ix].astype(np.float64) for k in LAYERS if f"emb_{k}" in z.files}
    return {"XB": XB, "Y": Y, "spk": spk, "utt": utt, "deliv": deliv, "emb": emb}


def seeded_folds(spk, seed, k=5):
    u = np.unique(spk)
    rng = np.random.default_rng(seed)
    f = {s: i % k for i, s in enumerate(rng.permutation(u))}
    fid = np.array([f[s] for s in spk])
    return [(np.flatnonzero(fid != j), np.flatnonzero(fid == j)) for j in range(k)]


# ----------------------------------------------------------------------------- ridge projection

def _std(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def ridge_path(X, Y, alphas):
    """Ridge on standardised X, centred Y for every alpha: list of (W, b) plus (mu, sd)."""
    mu, sd = _std(X)
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    UtY = U.T @ (Y - ym)
    out = []
    for a in alphas:
        Wt = Vt.T @ ((s / (s * s + a))[:, None] * UtY)
        out.append((Wt, ym))
    return out, mu, sd


def ridge_oof(X, Y, groups, alphas, k=5, seed=0):
    """OOF predictions for every alpha (grouped k-fold)."""
    P = np.zeros((len(alphas), len(X), Y.shape[1]))
    for tr, te in seeded_folds(groups, seed, k):
        path, mu, sd = ridge_path(X[tr], Y[tr], alphas)
        Xt = (X[te] - mu) / sd
        for i, (Wt, b) in enumerate(path):
            P[i, te] = Xt @ Wt + b
    return P


def pick_alpha(X, Y, groups, alphas=ALPHAS, seed=0):
    P = ridge_oof(X, Y, groups, alphas, seed=seed)
    score = [np.mean([np.corrcoef(P[i][:, j], Y[:, j])[0, 1] for j in range(Y.shape[1])])
             for i in range(len(alphas))]
    i = int(np.argmax(score))
    return float(alphas[i]), P[i], score


def fit_projection(X, Y, groups, seed=0):
    a, _, _ = pick_alpha(X, Y, groups, seed=seed)
    path, mu, sd = ridge_path(X, Y, [a])
    Wt, b = path[0]
    return {"alpha": a, "mu": mu, "sd": sd, "W": Wt, "b": b}


def apply_projection(p, X):
    return ((X - p["mu"]) / p["sd"]) @ p["W"] + p["b"]


def stacked_projection(E, Y, spk, tr, te, seed):
    """(projection features for rows tr: inner OOF; for rows te: fitted on all of tr)."""
    a, oof_tr, _ = pick_alpha(E[tr], Y[tr], spk[tr], seed=seed + 100)
    path, mu, sd = ridge_path(E[tr], Y[tr], [a])
    Wt, b = path[0]
    return oof_tr, ((E[te] - mu) / sd) @ Wt + b, a


# ----------------------------------------------------------------------------- CV

def build_X(data, groups):
    cols = list(F.SENT_NUM)
    mats = [data["XB"]]
    for g in groups:
        if g == "emb":
            continue
        cols += DV.GROUPS[g]
        mats.append(np.column_stack([data["deliv"][c] for c in DV.GROUPS[g]]))
    return np.column_stack(mats), cols


def fit_linear(Xtr, Ytr, spk, seed=0):
    """Ridge on all sentence inputs (NaN -> train median; every input clipped to its train
    1st-99th percentile so the linear part cannot extrapolate on unusual audio), alpha by
    grouped CV."""
    med = np.nanmedian(Xtr, 0)
    med = np.where(np.isfinite(med), med, 0.0)
    lo = np.nanpercentile(Xtr, 1, axis=0)
    hi = np.nanpercentile(Xtr, 99, axis=0)
    lo, hi = np.where(np.isfinite(lo), lo, -np.inf), np.where(np.isfinite(hi), hi, np.inf)
    A = np.clip(np.where(np.isnan(Xtr), med, Xtr), lo, hi)
    a, _, _ = pick_alpha(A, Ytr, spk, alphas=LIN_ALPHAS, seed=seed)
    path, mu, sd = ridge_path(A, Ytr, [a])
    W, b = path[0]
    return {"alpha": a, "med": med, "lo": lo, "hi": hi, "mu": mu, "sd": sd, "W": W, "b": b}


def apply_linear(lin, X):
    A = np.clip(np.where(np.isnan(X), lin["med"], X), lin["lo"], lin["hi"])
    return ((A - lin["mu"]) / lin["sd"]) @ lin["W"] + lin["b"]


def cv_score(data, groups, seed, layers=("final",), params=PARAMS, ridge_blend=None, return_oof=False):
    """ridge_blend: trees' weight w of the blend clip(w * clip(trees) + (1 - w) * ridge) (None = trees)."""
    X, cols = build_X(data, groups)
    Y, spk = data["Y"], data["spk"]
    use_emb = "emb" in groups
    E = np.column_stack([data["emb"][k] for k in layers]) if use_emb else None
    oof = np.zeros_like(Y)
    for tr, te in seeded_folds(spk, seed):
        Xtr, Xte = X[tr], X[te]
        if use_emb:
            ptr, pte, _ = stacked_projection(E, Y, spk, tr, te, seed)
            Xtr, Xte = np.column_stack([Xtr, ptr]), np.column_stack([Xte, pte])
        for j in range(len(HEADS)):
            m = fit("reg", params, Xtr, Y[tr, j], [False] * Xtr.shape[1])
            oof[te, j] = np.clip(raw(m, Xte), 0, 10)
        if ridge_blend is not None:
            lin = fit_linear(Xtr, Y[tr], spk[tr], seed=seed + 7)
            oof[te] = ridge_blend * oof[te] + (1 - ridge_blend) * apply_linear(lin, Xte)
    oof = np.clip(oof, 0, 10)
    r = {h: float(np.corrcoef(oof[:, j], Y[:, j])[0, 1]) for j, h in enumerate(HEADS)}
    return (r, oof) if return_oof else r


def multi_seed(data, groups, **kw):
    return {s: cv_score(data, groups, s, **kw) for s in SEEDS}


def compare(res, base):
    """Per head: mean over seeds, paired delta vs base (mean, se)."""
    out = {}
    for h in HEADS:
        v = np.array([res[s][h] for s in SEEDS])
        d = v - np.array([base[s][h] for s in SEEDS])
        out[h] = {"mean": float(v.mean()), "delta": float(d.mean()),
                  "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))), "delta_per_seed": d.tolist()}
    return out


def adopt(cmp) -> bool:
    gain = any(cmp[h]["delta"] >= 0.003 and cmp[h]["delta"] > 2 * cmp[h]["delta_se"]
               for h in ("prosodic", "total"))
    hurt = any(cmp[h]["delta"] <= -0.002 for h in HEADS)
    return bool(gain and not hurt)


def ablate(a):
    from concurrent.futures import ProcessPoolExecutor

    data = load_data()
    log("data", data["XB"].shape, {k: v.shape for k, v in data["emb"].items()})
    jobs = {"base": ([], {})}
    for g in DV.GROUPS:
        jobs[g] = ([g], {})
    for name, lay in (("emb_final", ("final",)), ("emb_enc3", ("enc3",)), ("emb_enc1", ("enc1",)),
                      ("emb_enc5", ("enc5",)), ("emb_all", LAYERS), ("emb_enc3_final", ("enc3", "final"))):
        jobs[name] = (["emb"], {"layers": lay})
    jobs["all_delivery"] = (list(DV.GROUPS), {})
    with ProcessPoolExecutor(min(8, len(jobs))) as ex:
        futs = {k: ex.submit(multi_seed, data, g, **kw) for k, (g, kw) in jobs.items()}
        res = {k: f.result() for k, f in futs.items()}
    out = {"seeds": list(SEEDS), "params": PARAMS, "single": {}}
    for k in jobs:
        c = compare(res[k], res["base"])
        out["single"][k] = {"groups": jobs[k][0], **jobs[k][1], "heads": c, "adopt": adopt(c) if k != "base" else None}
        log(k, {h: (round(c[h]["mean"], 4), round(c[h]["delta"], 4), round(c[h]["delta_se"], 4)) for h in HEADS},
            "ADOPT" if out["single"][k]["adopt"] else "")
    # embedding layer choice: best of the emb_* configs on mean(prosodic, total) delta
    emb_cfgs = [k for k in jobs if k.startswith("emb_")]
    best_emb = max(emb_cfgs, key=lambda k: np.mean([out["single"][k]["heads"][h]["delta"] for h in ("prosodic", "total")]))
    out["best_emb"] = best_emb
    save_json("round5_ablation.json", out)
    # forward selection over the groups that qualified alone (emb = its best layer config)
    cand = [g for g in DV.GROUPS if out["single"][g]["adopt"]]
    if out["single"][best_emb]["adopt"]:
        cand.append("emb")
    lay = tuple(jobs[best_emb][1]["layers"])
    order = sorted(cand, key=lambda g: -np.mean([out["single"][best_emb if g == "emb" else g]["heads"][h]["delta"]
                                                   for h in ("prosodic", "total")]))
    chosen, cur = [], res["base"]
    steps = []
    for g in order:
        trial = multi_seed(data, chosen + [g], layers=lay)
        c = compare(trial, cur)
        ok = adopt(c)
        steps.append({"add": g, "to": list(chosen), "heads": c, "adopt": ok})
        log("forward", chosen, "+", g, {h: (round(c[h]["mean"], 4), round(c[h]["delta"], 4)) for h in HEADS}, ok)
        if ok:
            chosen.append(g)
            cur = trial
    out["forward"] = steps
    out["chosen"] = chosen
    out["layers"] = list(lay)
    out["final_vs_base"] = compare(cur, res["base"])
    save_json("round5_ablation.json", out)
    log("chosen", chosen, lay, {h: round(out["final_vs_base"][h]["mean"], 4) for h in HEADS})


# ----------------------------------------------------------------------------- final fit

def fit_ship(a):
    from bench.calib.common import export_trees, walker_parity

    groups = [g for g in a.groups.split(",") if g]
    layers = tuple(a.layers.split(","))
    data = load_data()
    X, cols = build_X(data, groups)
    Y, spk = data["Y"], data["spk"]
    proj = None
    if "emb" in groups:
        E = np.column_stack([data["emb"][k] for k in layers])
        # tree inputs: 5-fold OOF projections over all train rows (alpha chosen on all train)
        alpha, P, _ = pick_alpha(E, Y, spk, seed=0)
        proj = fit_projection(E, Y, spk, seed=0)
        X = np.column_stack([X, P])
        cols = cols + EMB_NAMES
        log("projection alpha", alpha, "oof pearson",
            {h: round(float(np.corrcoef(P[:, j], Y[:, j])[0, 1]), 4) for j, h in enumerate(HEADS)})
    # CV of the exact shipped configuration (3 seeds) for the record, trees alone and blended
    cvt = multi_seed(data, groups, layers=layers)
    cvr = multi_seed(data, groups, layers=layers, ridge_blend=a.blend) if a.blend < 1 else cvt
    lin = fit_linear(X, Y, spk, seed=7) if a.blend < 1 else None
    r4 = np.load(ship_dir() / "sentence_r4.npz", allow_pickle=False)
    arrays = {k: r4[k] for k in r4.files if k != "meta" and not k.startswith("reg_")}
    meta = json.loads(bytes(r4["meta"]).decode("utf-8"))
    assert meta["num"] == list(F.SENT_NUM) and cols[:len(F.SENT_NUM)] == list(F.SENT_NUM)
    meta["num"] = cols
    parity = {}
    for j, h in enumerate(HEADS):
        m = fit("reg", PARAMS, X, Y[:, j], [False] * X.shape[1])
        tr = export_trees(m, X.shape[1])
        parity[h] = walker_parity(m, X)
        for k, v in tr.items():
            arrays[f"reg_{h}/{k}"] = np.asarray(v)
        meta["heads"][f"reg_{h}"] = {"kind": "reg", "n_trees": int(len(tr["roots"])),
                                     "depth": int(tr["depth"]), "clip": [0.0, 10.0]}
    if proj is not None:
        for k in ("mu", "sd", "W", "b"):
            arrays[f"proj/{k}"] = np.asarray(proj[k], dtype=np.float32 if k == "W" else np.float64)
        meta["projection"] = {"names": EMB_NAMES, "layers": list(layers), "alpha": proj["alpha"]}
    if lin is not None:
        for k in ("med", "lo", "hi", "mu", "sd", "W", "b"):
            arrays[f"lin/{k}"] = np.asarray(lin[k], dtype=np.float64)
        meta["linear_blend"] = {"heads": [f"reg_{h}" for h in HEADS], "tree_weight": a.blend,
                                "alpha": lin["alpha"]}
    meta["round"] = 5
    meta["delivery_groups"] = groups
    meta["reg_trained_on"] = "speechocean762 train (bench/calib/train_delivery.py)"
    meta["len_str_from"] = "sentence_r4.npz (unchanged)"
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    dest = ship_dir() / "sentence.npz"
    np.savez_compressed(dest, **arrays)
    # len / str heads must be bit-identical to round 4
    from pronunciada.calibrate import LevelModel

    new, old = LevelModel(dest), LevelModel(ship_dir() / "sentence_r4.npz")
    Xo = data["XB"]
    for h in ("len", "str"):
        assert np.array_equal(new.heads[h].predict(X), old.heads[h].predict(Xo)), h
    out = {"groups": groups, "layers": list(layers), "columns": cols, "params": PARAMS,
           "blend_tree_weight": a.blend,
           "cv": {h: {"mean": float(np.mean([cvr[s][h] for s in SEEDS])),
                      "per_seed": [cvr[s][h] for s in SEEDS]} for h in HEADS},
           "cv_trees_only": {h: float(np.mean([cvt[s][h] for s in SEEDS])) for h in HEADS},
           "walker_parity_max_abs": parity, "kb": dest.stat().st_size / 1024,
           "projection_alpha": None if proj is None else proj["alpha"]}
    save_json("round5_fit.json", out)
    log("wrote", dest, round(out["kb"], 1), "kB", {h: round(out["cv"][h]["mean"], 4) for h in HEADS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("ablate", "fit"))
    ap.add_argument("--groups", default="")
    ap.add_argument("--layers", default="final")
    ap.add_argument("--blend", type=float, default=BLEND_W, help="trees' weight (1 = no ridge blend)")
    a = ap.parse_args()
    ablate(a) if a.cmd == "ablate" else fit_ship(a)


if __name__ == "__main__":
    main()
