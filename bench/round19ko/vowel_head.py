"""Round 19 (Korean): vowel-identity head.

    python -m bench.round19ko.vowel_head [--C 0.05]     # fit on FLEURS ko TRAIN natives, report dev

Mechanism: on Common Voice and FLEURS Korean, ZIPA writes ㅓ as ɔ / o / a and ㅗ as o / u, so the
frame posteriors (and the sound probability built on them) barely separate 자기 from 저기 or 것 from
곳. The pooled encoder activations of the vowel carry more: like the round-18 lax / tense head, a
linear softmax on the sound's enc1|enc3|enc5 vector plus its neighbours' mean predicts which of the
seven Korean vowel classes was said (a ʌ o u ɯ i e; ㅐ / ㅔ merged; ㅢ not judged). Labels: the vowel
of the syllable in the prompt (natives read their own prompt). Fitted on FLEURS ko TRAIN only.
Writes pronunciada/data/coach/ko_vowel.npz (float16 weights) and results/tables/round19_ko_vowel_head.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.round19ko import collect as CO
from pronunciada import attr_heads as AH

CLASSES = ("a", "ʌ", "o", "u", "ɯ", "i", "e")


def xy(rows: dict, labelled_only: bool = True):
    X, y, grp = [], [], []
    for item, r in rows.items():
        if "error" in r or r["phones"]["V"] is None:
            continue
        if labelled_only and r["meta"].get("label"):
            continue                                    # changed items: the prompt is not what was said
        ph = r["phones"]
        V = ph["V"].astype(np.float32)
        C = AH.context_vectors(V)
        for i, vc in enumerate(ph["vclass"]):
            if vc in CLASSES:
                X.append(np.concatenate([V[i], C[i]]))
                y.append(CLASSES.index(vc))
                grp.append(r["meta"]["speaker"])
    return np.asarray(X, np.float32), np.asarray(y), grp


def fit(C: float = 0.05):
    from sklearn.linear_model import LogisticRegression

    X, y, _ = xy(CO.load("fleurs-train", "base"))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=C, max_iter=1000).fit((X - mu) / sd, y)
    return {"mu": mu, "sd": sd, "W": m.coef_.T.astype(np.float32), "b": m.intercept_.astype(np.float32),
            "n": int(len(y)), "counts": np.bincount(y, minlength=len(CLASSES)).tolist()}


def predict(H: dict, X: np.ndarray) -> np.ndarray:
    Z = ((X - H["mu"]) / H["sd"]) @ H["W"] + H["b"]
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def report(H: dict) -> dict:
    X, y, grp = xy(CO.load("dev", "base"))
    P = predict(H, X)
    acc = float(np.mean(P.argmax(1) == y))
    src = np.array(["cv" if not g.startswith("fleurs") else "fleurs-val" for g in grp])
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(y, P.argmax(1)):
        conf[t, p] += 1
    out = {"n_train": H["n"], "train_counts": dict(zip(CLASSES, H["counts"])), "dev_n": int(len(y)),
           "dev_acc": acc, "dev_acc_by_src": {s: float(np.mean(P.argmax(1)[src == s] == y[src == s])) for s in set(src)},
           "dev_confusion": {CLASSES[i]: dict(zip(CLASSES, conf[i].tolist())) for i in range(len(CLASSES))}}
    return out


def save(H: dict, path="pronunciada/data/coach/ko_vowel.npz"):
    np.savez_compressed(path, mu=H["mu"].astype(np.float16), sd=H["sd"].astype(np.float16),
                        W=H["W"].astype(np.float16), b=H["b"].astype(np.float32), classes=np.array(CLASSES))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    H = fit(a.C)
    rep = report(H)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.save:
        save(H)
        Path("results/tables/round19_ko_vowel_head.json").write_text(json.dumps({"C": a.C, **rep}, ensure_ascii=False, indent=1))
