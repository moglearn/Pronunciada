"""Which encoder stack carries vowel duration? A linear probe per stack and window width.

    python -m bench.round20_ja_length.probe            # reads win_fit_probe/*.npz -> probe.json

Examples: natural native vowels (every long vowel, a third of the short ones) of the fit pool. Per
stack: PCA to 24 dimensions per frame (fitted inside each training fold), the frames within +-w of
the CTC peak flattened, logistic regression; 4-fold cross-validation grouped by speaker
(Common Voice) / sentence (FLEURS). Target: the text's long / short label (natives say what they
read). Also: the shipped round-19 length probability as P(long) on the same vowels (it was fitted
on these clips, so its figure is optimistic).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from bench.round20_ja_length.common import OUT, W


def load(name: str):
    files = sorted((OUT / name).glob("shard_*.npz"))
    wins, rows = [], {}
    layers = None
    for f in files:
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        layers = m["layers"]
        wins.append(z["win"])
        for k, v in m["rows"].items():
            rows.setdefault(k, []).extend(v)
    return np.concatenate(wins), {k: np.array(v) for k, v in rows.items()}, layers


def dims(layers):
    from pronunciada.acoustic import AcousticModel  # noqa: F401
    D = {"enc0": 192, "enc1": 256, "enc2": 384, "enc3": 512, "enc4": 384, "enc5": 256, "final": 512}
    out, a = {}, 0
    for k in layers:
        out[k] = (a, a + D[k])
        a += D[k]
    return out


def folds(groups, k=4, seed=0):
    g = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(g)
    f = {x: i % k for i, x in enumerate(g)}
    return np.array([f[x] for x in groups])


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X, R, layers = load("win_fit_probe")
    y = R["text_long"].astype(int)
    fo = folds(R["group"])
    print(len(y), "vowels", int(y.sum()), "long", file=sys.stderr)
    res = {"n": int(len(y)), "n_long": int(y.sum())}
    lp = R["lprob"]
    pl = np.where(y == 1, lp, 1 - lp)
    res["shipped_r19_auc_in_sample"] = float(roc_auc_score(y, pl))
    rate = np.log(R["md_loc"])
    sl = dims(layers)
    for name in list(layers) + ["enc0+enc1+enc5"]:
        parts = name.split("+")
        for w in (3, 5, 8, 10):
            pred = np.zeros(len(y))
            for k in range(4):
                tr, te = fo != k, fo == k
                feats_tr, feats_te = [], []
                for L in parts:
                    a, b = sl[L]
                    A = X[:, W - w: W + w + 1, a:b].astype(np.float32)
                    pca = PCA(24, random_state=0).fit(A[tr][:, ::2].reshape(-1, b - a)[::3])
                    Z = pca.transform(A.reshape(-1, b - a)).reshape(len(A), 2 * w + 1, 24)
                    feats_tr.append(Z[tr].reshape(tr.sum(), -1))
                    feats_te.append(Z[te].reshape(te.sum(), -1))
                Ftr = np.concatenate(feats_tr + [rate[tr, None]], 1)
                Fte = np.concatenate(feats_te + [rate[te, None]], 1)
                sc = StandardScaler().fit(Ftr)
                m = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(Ftr), y[tr])
                pred[te] = m.decision_function(sc.transform(Fte))
            auc = float(roc_auc_score(y, pred))
            res[f"{name}|w{w}"] = auc
            print(f"{name:16s} w=+-{w * 20} ms  AUC {auc:.4f}", flush=True)
    (OUT / "probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
