"""Round 19 (Korean): a one-hidden-layer version of the sound-identity head (training only; torch).

    python -m bench.round19ko.identity_mlp [--hidden 256] [--epochs 20] [--save] [--extra NAME ...]

Same rows and labels as identity_head.py (FLEURS ko TRAIN natives, plus optional extra native sets
collected with bench.round19ko.collect, e.g. zeroth-train). The trained weights are stored as numpy
arrays and run in pronunciada/ko_identity.py without torch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.round19ko import collect as CO
from bench.round19ko.identity_head import rows_xy


def train(X, y, classes, hidden, epochs, wd=1e-4, drop=0.3, seed=0):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    mu = np.zeros(X.shape[1], np.float64)
    sq = np.zeros(X.shape[1], np.float64)
    for i in range(0, len(X), 20000):                  # chunked: X stays float16
        c = X[i:i + 20000].astype(np.float64)
        mu += c.sum(0)
        sq += (c ** 2).sum(0)
    mu /= len(X)
    sd = np.sqrt(np.maximum(sq / len(X) - mu ** 2, 0)) + 1e-6
    mu, sd = mu.astype(np.float32), sd.astype(np.float32)
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
    net.eval()
    W1, b1 = net[1].weight.detach().numpy().T, net[1].bias.detach().numpy()
    W2, b2 = net[4].weight.detach().numpy().T, net[4].bias.detach().numpy()
    return {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "W1": W1, "b1": b1, "W2": W2, "b2": b2,
            "classes": np.array(classes)}


def forward(H, X):
    from pronunciada.ko_identity import _mlp

    return _mlp(H, X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="use every n-th clip of the extra sets")
    ap.add_argument("--out", default="pronunciada/data/coach/ko_identity.npz")
    a = ap.parse_args()
    X, y, _ = rows_xy(CO.load("fleurs-train", "base"), dtype=np.float16)
    for name in a.extra:
        X2, y2, _ = rows_xy(CO.load(name, "base"), dtype=np.float16, every=a.every)
        X, y = np.concatenate([X, X2]), np.concatenate([y, y2])
        del X2
    classes = sorted(set(y.tolist()))
    H = train(X, y, classes, a.hidden, a.epochs)
    Xd, yd, sd = rows_xy(CO.load("dev", "base"))
    P = forward(H, Xd)
    pred = H["classes"][P.argmax(1)]
    rep = {"n_train": int(len(y)), "hidden": a.hidden, "epochs": a.epochs, "extra": a.extra,
           "dev_acc": float(np.mean(pred == yd))}
    for s in sorted(set(sd)):
        rep[f"dev_acc_{s}"] = float(np.mean(pred[sd == s] == yd[sd == s]))
    print(json.dumps(rep))
    if a.save:
        np.savez_compressed(a.out, kind=np.array(["mlp"]),
                            **{k: (v.astype(np.float16) if k in ("mu", "sd", "W1", "W2") else v)
                               for k, v in H.items()})
        Path("results/tables/round19_ko_identity_head.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
