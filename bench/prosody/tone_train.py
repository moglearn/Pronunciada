"""Train the Mandarin tone classifier (ProsodyNet) or the contour-MLP baseline.

    .venv/bin/python -m bench.prosody.tone_train --kind ctx --name tone_ctx
    .venv/bin/python -m bench.prosody.tone_train --kind mlp --name tone_mlp

Train: aishell3 train (syllables where our tone and AISHELL's agree or differ only by sandhi) +
fleurs zh train (front-end labels). Dev (model selection): FLEURS validation zh.
Augmentation (fresh every epoch): tempo 0.85-1.15, pitch-range scaling 0.7-1.3 of the normalised
contour, noisy pitch track (12-30 dB SNR) for 30 % of utterances, boundary jitter +-20 ms.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from bench.prosody.common import (OUT, agree_map, build_arrays, load_prepped, n_extra,  # noqa: E402
                                  set_opts)

from bench.prosody.common import MODELS_DIR as MODELS  # noqa: E402
_W = {}


def _init(kind, task="tone", opts=None, frac=1.0):
    _W["kind"] = kind
    _W["task"] = task
    set_opts(**(opts or {}))
    _W["train"] = subsample(load_train(task), frac)
    _W["agree"] = agree_map(_W["train"])


def subsample(utts, frac):
    """Keep a fraction of the training speakers (aishell3) / utterances (FLEURS: no speaker ids)."""
    if frac >= 1:
        return utts
    rng = np.random.default_rng(123)
    spk = sorted({u["speaker"] for u in utts if u["speaker"]})
    keep = set(rng.permutation(spk)[:max(1, int(round(frac * len(spk))))])
    return [u for k, u in enumerate(utts)
            if ((u["speaker"] in keep) if u["speaker"] else (rng.random() < frac))]


def load_train(task="tone"):
    a = load_prepped("aishell3", "zh", "train")
    try:
        f = load_prepped("fleurs", "zh", "train")
    except FileNotFoundError:
        f = []
    return a + f


def _gen(args):
    seed, lo, hi = args
    rng = np.random.default_rng(seed)
    X, y, _ = build_arrays(_W["train"][lo:hi], _W["kind"], rng=rng, aug=True, agree=_W["agree"],
                           train_only=True, task=_W["task"])
    return X.astype(np.float16), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="ctx")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ch", type=int, default=96)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--task", default="tone", choices=["tone"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--arch", default="gru")
    ap.add_argument("--extra", default="")
    ap.add_argument("--target", default="syllable")
    ap.add_argument("--frac", type=float, default=1.0)
    a = ap.parse_args()
    name = a.name or f"tone_{a.kind}"
    import torch

    torch.set_num_threads(1)
    from bench.prosody.model import ContourMLP, ProsodyNet, n_params

    opts = {"extra": a.extra, "target": a.target}
    set_opts(**opts)
    dev_u = load_prepped("fleurs_val", "zh", "dev")
    Xd, yd, _ = build_arrays(dev_u, a.kind, task=a.task)
    n_train = len(subsample(load_train(a.task), a.frac))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    n_out = 5
    net = (ProsodyNet(n_in=10 + n_extra(), ch=a.ch, hid=a.hid, n_out=n_out, arch=a.arch, stride=a.stride) if a.kind == "ctx" else ContourMLP(n_out=n_out)).to(dev)
    print(f"{name}: {n_params(net)} params, dev {len(yd)} syllables, {n_train} train utts", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * 1000, pct_start=0.1)
    ctx = get_context("spawn")
    pool = ctx.Pool(a.workers, initializer=_init, initargs=(a.kind, a.task, opts, a.frac))
    chunks = np.linspace(0, n_train, a.workers * 4 + 1).astype(int)

    def submit(ep):
        return pool.map_async(_gen, [(ep * 1000 + k, chunks[k], chunks[k + 1]) for k in range(len(chunks) - 1)])

    fut = submit(0)
    Xd_t = torch.tensor(Xd, dtype=torch.float32)
    best, curve, best_state = -1, [], None
    steps = 0
    for ep in range(a.epochs):
        t0 = time.time()
        parts = fut.get()
        if ep + 1 < a.epochs:
            fut = submit(ep + 1)
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        perm = np.random.default_rng(ep).permutation(len(y))
        net.train()
        tl, nb = 0.0, 0
        nbatch = len(y) // a.bs
        for b in range(nbatch):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            xb = torch.tensor(X[idx], dtype=torch.float32, device=dev)
            yb = torch.tensor(y[idx], device=dev)
            loss = torch.nn.functional.cross_entropy(net(xb), yb, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if steps < a.epochs * 1000 - 1:
                sched.step()
                steps += 1
            tl += float(loss)
            nb += 1
        net.eval()
        with torch.no_grad():
            lo = torch.cat([net(Xd_t[i:i + 2048].to(dev)).cpu() for i in range(0, len(Xd_t), 2048)])
        acc = float((lo.argmax(1).numpy() == yd).mean())
        nll = float(torch.nn.functional.cross_entropy(lo, torch.tensor(yd)))
        curve.append({"epoch": ep, "train_loss": tl / max(nb, 1), "dev_acc": acc, "dev_nll": nll,
                      "n_train": int(len(y)), "sec": time.time() - t0})
        print(json.dumps(curve[-1]), flush=True)
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    pool.terminate()
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "kind": a.kind, "task": a.task, "ch": a.ch, "hid": a.hid, "arch": a.arch, "stride": a.stride, "opts": opts, "frac": a.frac, "n_train_utts": n_train, "curve": curve,
                "best_dev_acc": best}, MODELS / f"{name}.pt")
    print(f"saved {MODELS / f'{name}.pt'} best dev acc {best:.4f}")


if __name__ == "__main__":
    main()
