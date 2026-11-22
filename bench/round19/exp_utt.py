"""Round 19 (English): utterance-level sentence-head experiments, speechocean762 TRAIN CV only.

    python -m bench.round19.exp_utt [--configs base,deliv,...] [--out round19_english_cv_utt.json]

Every config = the shipped round-2/4 sentence inputs (SENT_NUM, out-of-fold, as
bench/calib/train_delivery.py rebuilds them) + optional groups:
  deliv:<group>  delivery features of pronunciada.delivery (rhythm, pauses, pitch, loudness,
                 disfluency; "all" = every group)
  f0             speaker cues: median / p10 / p90 F0 (Hz) of voiced frames
  proj:<layers>  in-fold ridge projection (nested, as round 5) of the utterance-pooled
                 activations (mean ++ std over speech frames) of the listed layers ("+"-joined)
s2 trees per head (the shipped recipe), optionally blended with ridge (blend=0.7).
Folds: bench.round19.data.folds (5-fold, speaker-grouped, children balanced), seeds 0-2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.calib import train_delivery as TD
from bench.calib.common import SMALL_GRID, calib_cache, fit, raw
from bench.round19 import data as RD
from pronunciada import delivery as DV

HEADS = RD.HEADS


def load_train():
    U = RD.load("train", keep=["speaker", "labels", "delivery", "f0", "utt_layers", "S", "sent", "dur",
                               "tab_num", "tab_names"])
    z = np.load(calib_cache() / "round2_sentence_inputs_clean.npz", allow_pickle=False)
    pos = {u: i for i, u in enumerate(z["utt"])}
    U = [u for u in U if u["id"] in pos]
    ix = np.array([pos[u["id"]] for u in U])
    d = {"XB": z["XB"][ix], "Y": z["Y"][ix], "spk": np.array([u["speaker"] for u in U]),
         "age": np.array([u["labels"]["age"] for u in U]), "ids": [u["id"] for u in U]}
    Ych = np.array([[u["labels"]["sentence"][h] for h in HEADS] for u in U])
    assert np.allclose(Ych, d["Y"]), "label mismatch"
    d["deliv"] = {c: np.array([u["delivery"].get(c, np.nan) for u in U]) for c in DV.ALL}
    d["f0"] = np.array([[u["f0"].get(k, np.nan) for k in ("f0_med_hz", "f0_p10_hz", "f0_p90_hz")] for u in U])
    d["emb"] = {k: np.stack([u["utt_layers"][k] for u in U]).astype(np.float64) for k in RD.LAYERS}
    d["r4_runtime"] = np.array([[u["sent"][h] for h in HEADS] for u in U])
    import json as _j

    vp = RD.feat_dir("train").parent / "voice_speechocean762_train.json"
    if vp.exists():
        V = _j.loads(vp.read_text())
        d["voice"] = np.array([[V[u["id"]][k] for k in ("centroid_med_hz", "centroid_mean_hz", "rolloff85_med_hz")]
                               for u in U])
    d["worst"] = np.stack([worst_k(u["tab_num"], u["tab_names"]) for u in U])
    sd = Path.home() / ".cache" / "pronunciada" / "round19" / "english" / "download" / "scores-detail.json"
    if sd.exists():
        R = _j.loads(sd.read_text())
        d["rater_sd"] = np.array([[np.std(R[u["id"].split("-")[1]][h]) for h in HEADS] for u in U])
    return d


WORST_COLS = ("llr_sub", "z_llr_sub", "llr_del", "peak_post")


def worst_k(X, names) -> np.ndarray:
    """Worst-k statistics of the aligner evidence over the utterance's sounds: 5th / 10th
    percentile and mean of the worst three, per column of WORST_COLS."""
    out = []
    for c in WORST_COLS:
        v = np.asarray(X[:, list(names).index(c)], float)
        v = np.sort(v[np.isfinite(v)])
        if not len(v):
            out += [np.nan] * 3
            continue
        out += [np.percentile(v, 5), np.percentile(v, 10), v[:3].mean()]
    return np.array(out)


def parse(cfg):
    groups, layers, extra, opts = [], [], [], {}
    for part in cfg.split(","):
        if part.startswith("deliv:"):
            g = part[6:]
            groups += list(DV.GROUPS) if g == "all" else g.split("+")
        elif part.startswith("proj:"):
            layers.append(part[5:].split("+"))
        elif part in ("f0", "voice", "worst"):
            extra.append(part)
        elif part.startswith("w:"):
            opts["weight"] = part[2:]
    return groups, layers, extra, opts


def item_weights(d, tr, j, kind):
    """Rater-agreement item weights (train rows only): 1 / (rater sd^2 + c), c = median sd^2."""
    if kind != "rater" or "rater_sd" not in d:
        return None
    v = d["rater_sd"][tr, j] ** 2
    c = max(float(np.median(v)), 0.05)
    w = 1.0 / (v + c)
    return w / w.mean()


def cv(d, cfg, seed, blend=None, params="s2", return_oof=False):
    groups, projs, extra, opts = parse(cfg)
    mats = [d["XB"]] + [np.column_stack([d["deliv"][c] for c in DV.GROUPS[g]]) for g in groups]
    mats += [d[e] for e in extra]
    X = np.column_stack(mats)
    Y, spk = d["Y"], d["spk"]
    Es = [np.column_stack([RD.pooled_part(d["emb"][RD.layer_of(k)], k) for k in L]) for L in projs]
    oof = np.zeros_like(Y)
    for tr, te in RD.folds(spk, d["age"], seed):
        Xtr, Xte = X[tr], X[te]
        for E in Es:
            ptr, pte, _ = TD.stacked_projection(E, Y, spk, tr, te, seed)
            Xtr, Xte = np.column_stack([Xtr, ptr]), np.column_stack([Xte, pte])
        for j in range(len(HEADS)):
            m = fit("reg", SMALL_GRID[params], Xtr, Y[tr, j], [False] * Xtr.shape[1],
                    item_weights(d, tr, j, opts.get("weight")))
            oof[te, j] = np.clip(raw(m, Xte), 0, 10)
        if blend is not None:
            lin = TD.fit_linear(Xtr, Y[tr], spk[tr], seed=seed + 7)
            oof[te] = blend * oof[te] + (1 - blend) * TD.apply_linear(lin, Xte)
    oof = np.clip(oof, 0, 10)
    r = {h: RD.pearson(oof[:, j], Y[:, j]) for j, h in enumerate(HEADS)}
    ch = d["age"] <= 12
    r.update({f"total_child": RD.pearson(oof[ch, 3], Y[ch, 3]), f"total_adult": RD.pearson(oof[~ch, 3], Y[~ch, 3])})
    return (r, oof) if return_oof else r


def child_prob(d, tr, te):
    """P(child voice) from F0 + spectral cues only (logistic, fitted on training rows; the age
    label is used for fitting, never as an input)."""
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([d["f0"], d["voice"]])
    med = np.nanmedian(X[tr], 0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    m = LogisticRegression(C=1.0, max_iter=1000).fit(Z[tr], (d["age"][tr] <= 12).astype(int))
    return m.predict_proba(Z)[:, 1]


def cv_gated(d, cfg, seed, sharp=1.0, return_oof=False):
    """Two copies of the config's model, one on all training rows and one on the training rows
    of children; prediction = p * child model + (1 - p) * all-rows model, p = child_prob ** sharp."""
    groups, projs, extra, opts = parse(cfg)
    mats = [d["XB"]] + [np.column_stack([d["deliv"][c] for c in DV.GROUPS[g]]) for g in groups]
    mats += [d[e] for e in extra]
    X = np.column_stack(mats)
    Y, spk = d["Y"], d["spk"]
    Es = [np.column_stack([RD.pooled_part(d["emb"][RD.layer_of(k)], k) for k in L]) for L in projs]
    oof = np.zeros_like(Y)
    pch = np.zeros(len(Y))
    for tr, te in RD.folds(spk, d["age"], seed):
        p = child_prob(d, tr, te)
        pch[te] = p[te]
        preds = []
        for rows in (tr, tr[d["age"][tr] <= 12]):
            Xtr, Xte = X[rows], X[te]
            for E in Es:
                ptr, pte, _ = TD.stacked_projection(E, Y, spk, rows, te, seed)
                Xtr, Xte = np.column_stack([Xtr, ptr]), np.column_stack([Xte, pte])
            P = np.zeros((len(te), 4))
            for j in range(len(HEADS)):
                m = fit("reg", SMALL_GRID["s2"], Xtr, Y[rows, j], [False] * Xtr.shape[1],
                        item_weights(d, rows, j, opts.get("weight")))
                P[:, j] = np.clip(raw(m, Xte), 0, 10)
            preds.append(P)
        g = (p[te] ** sharp)[:, None]
        oof[te] = g * preds[1] + (1 - g) * preds[0]
    ch = d["age"] <= 12
    r = {h: RD.pearson(oof[:, j], Y[:, j]) for j, h in enumerate(HEADS)}
    r.update({"total_child": RD.pearson(oof[ch, 3], Y[ch, 3]), "total_adult": RD.pearson(oof[~ch, 3], Y[~ch, 3]),
              "child_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(ch, pch))})
    return (r, oof) if return_oof else r


DEFAULT = [
    "base", "deliv:all", "deliv:disfluency", "f0", "deliv:all,f0",
    "deliv:disfluency,proj:enc1",                    # round 5 adopted set
    *[f"deliv:disfluency,proj:{k}" for k in RD.LAYERS if k != "enc1"],
    "deliv:disfluency,proj:enc0+enc1+enc2+enc3+enc4+enc5+final",
    "deliv:all,f0,proj:enc0+enc1+enc2+enc3+enc4+enc5+final",
    "deliv:all,f0,proj:enc0+enc1+enc2,proj:enc3+enc4+enc5+final",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None, help="';'-separated")
    ap.add_argument("--blend", type=float, default=None)
    ap.add_argument("--out", default="round19_english_cv_utt.json")
    a = ap.parse_args()
    d = load_train()
    print(f"{len(d['Y'])} utts; runtime r4 vs labels (in-sample, info only):",
          {h: round(RD.pearson(d['r4_runtime'][:, j], d['Y'][:, j]), 4) for j, h in enumerate(HEADS)}, flush=True)
    cfgs = a.configs.split(";") if a.configs else DEFAULT
    from bench.calib.common import load_json, save_json

    R = load_json(a.out) or {}
    base = None
    for cfg in cfgs:
        key = f"{cfg}|blend={a.blend}"
        if cfg.startswith("gated:"):
            per = [cv_gated(d, cfg[6:], s) for s in RD.SEEDS]
        else:
            per = [cv(d, cfg, s, blend=a.blend) for s in RD.SEEDS]
        mean = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        R[key] = {"seeds": per, "mean": mean}
        if cfg == "base":
            base = per
        b = R.get(f"base|blend={a.blend}", {}).get("seeds") or base
        dl = ""
        if b:
            dd = {k: [p[k] - q[k] for p, q in zip(per, b)] for k in per[0] if k in b[0]}
            R[key]["delta_vs_base"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))] for k, v in dd.items()}
            dl = " ".join(f"{k[:5]}{np.mean(v):+.4f}" for k, v in dd.items())
        print(f"{key:70s} " + " ".join(f"{k[:5]}={v:.4f}" for k, v in mean.items()) + "  | " + dl, flush=True)
        save_json(a.out, R)


if __name__ == "__main__":
    main()
