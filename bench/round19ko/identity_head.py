"""Round 19 (Korean): fit the sound-identity head (pronunciada/ko_identity.py).

    python -m bench.round19ko.identity_head [--C 0.05] [--save]

Training rows: FLEURS ko TRAIN natives (bench.round19ko.collect --set fleurs-train), every sound with a
class (pronunciada.ko_identity.classes on the front-end's reference of the prompt), input = pooled
enc1|enc3|enc5 vector + neighbours' mean, standardised; multinomial logistic regression.
Report: accuracy on the dev controls (FLEURS validation + Common Voice dev pool), by source.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bench.round19ko import collect as CO
from pronunciada import attr_heads as AH
from pronunciada import ko_identity as KI


def rows_xy(rows: dict, controls_only: bool = True, dtype=np.float32, every: int = 1):
    from pronunciada.text import get_g2p

    g = get_g2p("ko")
    X, y, src = [], [], []
    for n, (item, r) in enumerate(rows.items()):
        if n % every:
            continue
        if "error" in r or r["phones"]["V"] is None:
            continue
        m = r["meta"]
        if controls_only and m.get("label"):
            continue
        if m.get("cor_kind") not in (None, "clean"):
            continue
        ref = g(m["target"])
        if [p.ipa for p in ref.phones] != list(r["phones"]["ipa"]):
            continue
        cls = KI.classes(ref)
        V = r["phones"]["V"].astype(np.float32)
        C = AH.context_vectors(V)
        for i, c in enumerate(cls):
            if c is not None:
                X.append(np.concatenate([V[i], C[i]]).astype(dtype))
                y.append(c)
                src.append(m["src"])
    return np.asarray(X, dtype), np.asarray(y), np.asarray(src)


def fit(C: float):
    from sklearn.linear_model import LogisticRegression

    X, y, _ = rows_xy(CO.load("fleurs-train", "base"))
    classes = sorted(set(y))
    yi = np.array([classes.index(c) for c in y])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=C, max_iter=1500).fit((X - mu) / sd, yi)
    return {"mu": mu, "sd": sd, "W": m.coef_.T.astype(np.float32), "b": m.intercept_.astype(np.float32),
            "classes": np.array(classes), "index": {c: j for j, c in enumerate(classes)},
            "counts": dict(Counter(y.tolist()))}


def report(H):
    X, y, src = rows_xy(CO.load("dev", "base"))
    keep = np.array([c in H["index"] for c in y])
    X, y, src = X[keep], y[keep], src[keep]
    Z = ((X - H["mu"]) / H["sd"]) @ H["W"] + H["b"]
    pred = H["classes"][Z.argmax(1)]
    out = {"n_train": int(sum(H["counts"].values())), "train_counts": H["counts"], "dev_n": int(len(y)),
           "dev_acc": float(np.mean(pred == y))}
    for s in sorted(set(src)):
        out[f"dev_acc_{s}"] = float(np.mean(pred[src == s] == y[src == s]))
    out["dev_acc_by_class"] = {c: float(np.mean(pred[y == c] == c)) for c in H["classes"]}
    return out


def save(H, path="pronunciada/data/coach/ko_identity.npz"):
    np.savez_compressed(path, mu=H["mu"].astype(np.float16), sd=H["sd"].astype(np.float16),
                        W=H["W"].astype(np.float16), b=H["b"].astype(np.float32), classes=H["classes"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    H = fit(a.C)
    rep = report(H)
    print(json.dumps({k: v for k, v in rep.items() if k != "train_counts"}, ensure_ascii=False, indent=0))
    if a.save:
        save(H)
        Path("results/tables/round19_ko_identity_head.json").write_text(
            json.dumps({"C": a.C, **rep}, ensure_ascii=False, indent=1))
