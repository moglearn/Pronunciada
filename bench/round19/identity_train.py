"""Round 19b (German): train the sound-identity head (pronunciada/de_identity.py; training uses torch).

    python -m bench.round19.identity_train [--hidden 512] [--epochs 20] [--cap 8000] [--save]

Rows: FLEURS de TRAIN natives and Common Voice de TRAIN natives (bench.round19.cv_train), minus the
Common Voice speakers held out as the native pool (speaker hash, 20 %; used only to place the knee).
Every judged sound (de_identity.classes on the front-end's reference of the prompt); input = pooled
vector + neighbours' mean. At most --cap rows per class (seeded). Report: class accuracy on the dev
controls (Common Voice dev / FLEURS validation) and on the pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.round19.identity_collect import load
from pronunciada import attr_heads as AH
from pronunciada import de_identity as DI


def pool_speaker(row) -> bool:
    """Common Voice train speakers held out of the head's training: the native pool for the knee."""
    return int(hashlib.sha256(f"pool|{row['meta']['speaker']}".encode()).hexdigest()[:8], 16) % 5 == 0


def _g2p():
    from pronunciada.text import get_g2p

    return get_g2p("de")


def rows_xy(rows, controls_only=True, dtype=np.float16):
    g = _g2p()
    X, y, src = [], [], []
    for r in rows:
        m = r["meta"]
        if controls_only and (m.get("label") or r.get("wrong") is not None):
            continue
        ref = g(r["text"])
        if [p.ipa for p in ref.phones] != list(r["ipa"]):
            continue
        cls = DI.classes(ref)
        V = r["V"].astype(np.float32)
        C = AH.context_vectors(V)
        for i, c in enumerate(cls):
            if c is not None:
                X.append(np.concatenate([V[i], C[i]]).astype(dtype))
                y.append(c)
                src.append(m["src"])
    return np.asarray(X, dtype), np.asarray(y), np.asarray(src)


def training_rows(cap, seed=0):
    fl = load("fleurs-train")
    cv = [r for r in load("cv-train") if not pool_speaker(r)]
    X1, y1, _ = rows_xy(fl)
    X2, y2, _ = rows_xy(cv)
    X, y = np.concatenate([X1, X2]), np.concatenate([y1, y2])
    del X1, X2
    rng = np.random.default_rng(seed)
    keep = []
    for c in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == c)
        if len(idx) < 150:
            continue                               # too rare to learn: not judged
        keep.append(rng.permutation(idx)[:cap])
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def train(X, y, classes, hidden, epochs, wd=1e-4, drop=0.3, seed=0):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    mu = X.astype(np.float32).mean(0)
    sd = X.astype(np.float32).std(0) + 1e-6
    Xt = torch.from_numpy(X)
    mut, sdt = torch.from_numpy(mu), torch.from_numpy(sd)
    yt = torch.tensor([classes.index(c) for c in y])
    net = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Dropout(drop), torch.nn.Linear(hidden, len(classes)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(yt)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            xb = (Xt[b].float() - mut) / sdt
            loss = torch.nn.functional.cross_entropy(net(xb), yt[b], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        print("epoch", ep, float(loss), flush=True)
    net.eval()
    return {"mu": mu, "sd": sd, "W1": net[1].weight.detach().numpy().T.copy(), "b1": net[1].bias.detach().numpy().copy(),
            "W2": net[4].weight.detach().numpy().T.copy(), "b2": net[4].bias.detach().numpy().copy(),
            "classes": np.array(classes)}


def accuracy(H, rows):
    X, y, src = rows_xy(rows, dtype=np.float32)
    keep = np.array([c in set(H["classes"].tolist()) for c in y])
    X, y, src = X[keep], y[keep], src[keep]
    P = DI._mlp(H, X)
    pred = H["classes"][P.argmax(1)]
    out = {"n": int(len(y)), "acc": float(np.mean(pred == y))}
    for s in sorted(set(src.tolist())):
        out[f"acc_{s}"] = float(np.mean(pred[src == s] == y[src == s]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    X, y = training_rows(a.cap)
    classes = sorted(set(y.tolist()))
    print("rows", len(y), "classes", len(classes), flush=True)
    H = train(X, y, classes, a.hidden, a.epochs)
    del X
    rep = {"n_train": int(len(y)), "classes": classes, "hidden": a.hidden, "epochs": a.epochs, "cap": a.cap,
           "dev": accuracy(H, load("dev")),
           "pool": accuracy(H, [r for r in load("cv-train") if pool_speaker(r)])}
    print(json.dumps({k: v for k, v in rep.items() if k != "classes"}))
    if a.save:
        np.savez_compressed("pronunciada/data/coach/de_identity.npz",
                            **{k: (v.astype(np.float16) if k in ("mu", "sd", "W1", "W2") else v) for k, v in H.items()})
        Path("results/tables/round19_de_identity_head.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
