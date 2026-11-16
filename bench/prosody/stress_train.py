"""Word-stress model: per-syllable score (ProsodyNet family + word-relative cues), softmax over
the syllables of a word, trained to put the mass on the front-end's primary stress.

    .venv/bin/python -m bench.prosody.stress_train --langs en,de,es,it,pt,ru --name stress_multi
    .venv/bin/python -m bench.prosody.stress_train --langs es --name stress_es --no-lang-emb

Train: FLEURS train (labels from our front-ends; natives assumed correct). Dev: FLEURS validation.
Augmentation: tempo 0.85-1.15, pitch-range scaling, noisy track (30 %), boundary jitter +-20 ms.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from bench.prosody.common import MODELS_DIR, load_prepped, utt_inputs  # noqa: E402
from bench.prosody.model import ProsodyNet  # noqa: E402
from pronunciada.prosody.features import N_STRESS_EXTRA, context_input, stress_extra  # noqa: E402

MAX_SYL = 8


class StressNet(nn.Module):
    def __init__(self, ch=64, hid=48, n_lang=0, stride=2):
        super().__init__()
        self.net = ProsodyNet(n_out=1, ch=ch, hid=hid, n_lang=n_lang, n_extra=N_STRESS_EXTRA, stride=stride)

    def forward(self, x, extra, lang=None):
        return self.net(x, lang, extra)


def word_groups(u):
    """[(s0, s1, primary_index_local)] for words with 2..MAX_SYL syllables and one primary stress."""
    ref = u["ref"]
    out = []
    for w in ref.words:
        s0, s1 = w.syllables
        syl = ref.syllables[s0:s1]
        if not (2 <= s1 - s0 <= MAX_SYL) or any(s.stress is None for s in syl):
            continue
        prim = [i for i, s in enumerate(syl) if s.stress == 1]
        if len(prim) != 1:
            continue
        out.append((s0, s1, prim[0]))
    return out


def build_words(utts, lang_ids, rng=None, aug=False, jitter=0):
    """Padded arrays: X [W, S, L, C] f16, E [W, S, 7], M [W, S], y [W], lang [W], meta."""
    X, E, M, Y, LG, meta = [], [], [], [], [], []
    for ui, (u, lid) in enumerate(zip(utts, lang_ids)):
        g = word_groups(u)
        if not g:
            continue
        feats, wins = utt_inputs(u, rng, aug, jitter)
        for s0, s1, p in g:
            n = s1 - s0
            x = np.zeros((MAX_SYL,) + context_input(feats, wins, s0).shape, np.float16)
            e = np.zeros((MAX_SYL, N_STRESS_EXTRA), np.float32)
            for k in range(n):
                x[k] = context_input(feats, wins, s0 + k)
            e[:n] = stress_extra(feats, wins, (s0, s1))
            m = np.zeros(MAX_SYL, bool)
            m[:n] = True
            X.append(x)
            E.append(e)
            M.append(m)
            Y.append(p)
            LG.append(lid)
            meta.append((ui, s0, s1))
    if not X:
        return None
    return np.stack(X), np.stack(E), np.stack(M), np.array(Y), np.array(LG), meta


def word_logits(net, X, E, M, LG, dev, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(_forward(net, X[i:i + bs], E[i:i + bs], M[i:i + bs], LG[i:i + bs], dev).cpu())
    return torch.cat(out).numpy()


def _forward(net, X, E, M, LG, dev):
    B, S = M.shape
    x = torch.tensor(X.reshape((B * S,) + X.shape[2:]), dtype=torch.float32, device=dev)
    e = torch.tensor(E.reshape(B * S, -1), dtype=torch.float32, device=dev)
    lg = torch.tensor(np.repeat(LG, S), device=dev) if net.net.n_lang else None
    m = torch.tensor(M, device=dev)
    z = net(x, e, lg).view(B, S)
    return z.masked_fill(~m, -1e4)


_W = {}


def _init(langs):
    _W["utts"], _W["lids"] = load_split(langs, "train")


def load_split(langs, split):
    utts, lids = [], []
    for k, lang in enumerate(langs):
        ds = "fleurs_val" if split == "dev" else "fleurs"
        u = load_prepped(ds, lang, split)
        utts += u
        lids += [k] * len(u)
    return utts, lids


def _gen(args):
    seed, lo, hi = args
    rng = np.random.default_rng(seed)
    r = build_words(_W["utts"][lo:hi], _W["lids"][lo:hi], rng, aug=True)
    return r[:5] if r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="en,de,es,it,pt,ru")
    ap.add_argument("--name", default="stress_multi")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--hid", type=int, default=48)
    ap.add_argument("--no-lang-emb", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(1)
    langs = a.langs.split(",")
    cfg = {"ch": a.ch, "hid": a.hid, "n_lang": 0 if a.no_lang_emb else len(langs), "stride": 2}
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = StressNet(**cfg).to(dev)
    du, dl = load_split(langs, "dev")
    Xd, Ed, Md, yd, Ld, _ = build_words(du, dl)
    n_train = len(load_split(langs, "train")[0])
    print(f"{a.name}: {sum(p.numel() for p in net.parameters())} params, dev words {len(yd)}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    ctx = get_context("spawn")
    pool = ctx.Pool(a.workers, initializer=_init, initargs=(langs,))
    chunks = np.linspace(0, n_train, a.workers * 4 + 1).astype(int)
    submit = lambda ep: pool.map_async(_gen, [(ep * 1000 + k, chunks[k], chunks[k + 1])  # noqa: E731
                                              for k in range(len(chunks) - 1)])
    fut = submit(0)
    best, best_state, curve = -1, None, []
    total = None
    sched = None
    for ep in range(a.epochs):
        t0 = time.time()
        parts = [p for p in fut.get() if p is not None]
        if ep + 1 < a.epochs:
            fut = submit(ep + 1)
        X, E, Mk, y, LG = (np.concatenate([p[i] for p in parts]) for i in range(5))
        if sched is None:
            total = a.epochs * (len(y) // a.bs + 1)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
        perm = np.random.default_rng(ep).permutation(len(y))
        net.train()
        tl = 0.0
        nb = len(y) // a.bs
        for b in range(nb):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            z = _forward(net, X[idx], E[idx], Mk[idx], LG[idx], dev)
            loss = nn.functional.cross_entropy(z, torch.tensor(y[idx], device=dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss)
        net.eval()
        zd = word_logits(net, Xd, Ed, Md, Ld, dev)
        acc = float((zd.argmax(1) == yd).mean())
        per = {lg: float((zd.argmax(1) == yd)[Ld == k].mean()) for k, lg in enumerate(langs)}
        curve.append({"epoch": ep, "train_loss": tl / max(nb, 1), "dev_acc": acc, "dev_acc_lang": per,
                      "n_words": int(len(y)), "sec": time.time() - t0})
        print(json.dumps(curve[-1]), flush=True)
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    pool.terminate()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "cfg": cfg, "langs": langs, "curve": curve, "best_dev_acc": best},
               MODELS_DIR / f"{a.name}.pt")
    print("saved", a.name, best)


if __name__ == "__main__":
    main()
