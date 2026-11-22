"""One-hidden-layer sound-identity head training loop (torch; training only), as used for the
Korean round-19 head (bench/round19ko/identity_mlp.py on that branch)."""
from __future__ import annotations

import numpy as np


def train(X, y, classes, hidden, epochs, wd=1e-4, drop=0.3, seed=0):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    mu = np.zeros(X.shape[1], np.float64)
    sq = np.zeros(X.shape[1], np.float64)
    for i in range(0, len(X), 20000):
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
    for _ in range(epochs):
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
    return {"mu": mu, "sd": sd, "W1": W1, "b1": b1, "W2": W2, "b2": b2, "classes": np.array(classes)}
