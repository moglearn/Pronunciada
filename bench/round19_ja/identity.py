"""Train the Japanese sound-identity head (pronunciada/ja_identity.py; training only, torch).

    python -m bench.round19_ja.identity [--hidden 256] [--epochs 15] [--save]

Rows: every aligned sound of the v2 fit natives (Common Voice ja train + FLEURS ja train), input =
phone vector + neighbours' mean (float16), label = pronunciada.ja_identity.phone_class. Classes with
fewer than MIN_N rows are dropped. Accuracy is reported on the dev-pool natives (CV dev / FLEURS
validation). The training loop is the Korean head's (bench/round19ko/identity_mlp.py in that branch).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

MIN_N = 300


def rows_xy(items, dtype=np.float16):
    from pronunciada import attr_heads as AH
    from pronunciada.ja_identity import classes

    X, y, src = [], [], []
    for it in items:
        V = it.get("V")
        if V is None:
            continue
        V = np.asarray(V, np.float32)
        Xc = np.concatenate([V, AH.context_vectors(V)], axis=1)
        ok = it["slot_se"][:, 2] >= 0
        for i, c in enumerate(classes(it["ref"])):
            if c is not None and ok[i]:
                X.append(Xc[i].astype(dtype))
                y.append(c)
                src.append(it.get("source") or "")
    return np.array(X), np.array(y), np.array(src)


def main():
    from bench.round19_ja.collect import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    import sys

    from bench.round19_ja import _ko_mlp as mod
    X, y, _ = rows_xy(load("fit", "v2"))
    cnt = Counter(y.tolist())
    keep = np.array([cnt[c] >= MIN_N for c in y])
    X, y = X[keep], y[keep]
    classes = sorted(set(y.tolist()))
    print(len(y), "rows", len(classes), "classes", X.shape, file=sys.stderr)
    H = mod.train(X, y, classes, a.hidden, a.epochs)
    del X
    from pronunciada.ja_identity import _mlp

    Xd, yd, sd = rows_xy(load("devnat", "v2"), dtype=np.float32)
    m = np.isin(yd, classes)
    pred = H["classes"][_mlp(H, Xd[m]).argmax(1)]
    rep = {"n_train": int(len(y)), "classes": classes, "hidden": a.hidden, "epochs": a.epochs,
           "dev_acc": float(np.mean(pred == yd[m]))}
    for s in sorted(set(sd)):
        mm = sd[m] == s
        rep[f"dev_acc_{s}"] = float(np.mean(pred[mm] == yd[m][mm]))
    print(json.dumps(rep))
    if a.save:
        root = Path(__file__).resolve().parents[2]
        np.savez_compressed(root / "pronunciada/data/coach/ja_identity.npz", kind=np.array(["mlp"]),
                            **{k: (v.astype(np.float16) if k in ("mu", "sd", "W1", "W2") else v) for k, v in H.items()})
        (root / "results/tables/round19_ja_identity_head.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
