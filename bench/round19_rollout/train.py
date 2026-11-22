"""Round 19 rollout: train a language's sound-identity head (pronunciada/identity.py; uses torch).

    python -m bench.round19_rollout.train --lang fr [--hidden 512] [--epochs 20] [--cap 8000] [--save]

Rows: FLEURS TRAIN natives and the Common Voice "train" speakers (bench.round19_rollout.collect sets
fltrain, cvtrain). Label of a sound = its reference class (identity.classes with the language's
SPEC: merges of native variants; a sound the front-end lets natives drop with prior >= exp(skip_deletable)
gives no row). Skipped classes (e.g. schwa) are still head classes, so a reduced vowel has somewhere to
go, but they are never judged. Classes with < 150 rows are not head classes. At most --cap rows per class.
Report: class accuracy on the dev controls and on the native pool (cvpool speakers, never trained on).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.round19_rollout.collect import iter_rows
from bench.round19_rollout.sources import work
from pronunciada import attr_heads as AH
from pronunciada import identity as ID

# per-language class rules (copied into data/calib/identity/<lang>.json by fit.py)
SPEC = {
    "fr": {"merge": {"ʀ": "ʁ", "ɑ": "a"}, "skip": ["ə"], "skip_deletable": -2.0},
    "es": {"merge": {"β": "b", "ð": "d", "ɣ": "g"}, "skip": [], "skip_deletable": -2.0},
    "en": {"merge": {"l̴": "l", "ə˞": "ɜ˞"}, "skip": ["ə"], "skip_deletable": -2.0},
    "it": {"merge": {}, "skip": [], "skip_deletable": -2.0},
    "pt": {"merge": {}, "skip": [], "skip_deletable": -2.0},
    "ru": {"merge": {}, "skip": [], "skip_deletable": -2.0},
    "zh": {"merge": {}, "skip": [], "skip_deletable": -2.0},
}


class _P:
    def __init__(self, ipa, alts):
        self.ipa, self.alts = ipa, alts


class Ref:
    """Minimal reference from stored rows (sound symbols and front-end alternatives)."""
    def __init__(self, ph):
        self.phones = [_P(a, [tuple(x) for x in alts]) for a, alts in ph]


def labels(row, spec, H=None, train=False):
    """Class per sound (None = no row / not judged). train=True: skipped classes still give labels."""
    sp = dict(spec, skip=[]) if train else spec
    cls, _ = ID.classes(Ref(row["ph"]), sp, H)
    return cls


def rows_xy(lang, sets, spec, keep=None, controls_only=True):
    X, y, src = [], [], []
    for name in sets:
        for r in iter_rows(lang, name):
            m = r["meta"]
            if controls_only and (m.get("label") or r.get("wrong") is not None):
                continue
            if keep is not None and not keep(r):
                continue
            cls = labels(r, spec, train=True)
            V = r["V"].astype(np.float32)
            C = AH.context_vectors(V)
            for i, c in enumerate(cls):
                if c is not None:
                    X.append(np.concatenate([V[i], C[i]]).astype(np.float16))
                    y.append(c)
                    src.append(m.get("src", name))
    return np.asarray(X, np.float16), np.asarray(y), np.asarray(src)


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


def accuracy(lang, H, sets, spec):
    X, y, src = rows_xy(lang, sets, spec)
    keep = np.array([c in set(H["classes"].tolist()) for c in y])
    X, y, src = X[keep].astype(np.float32), y[keep], src[keep]
    P = ID._mlp(H, X)
    pred = H["classes"][P.argmax(1)]
    out = {"n": int(len(y)), "acc": float(np.mean(pred == y))}
    for s in sorted(set(src.tolist())):
        out[f"acc_{s}"] = float(np.mean(pred[src == s] == y[src == s]))
    return out


def head_path(lang):
    return Path("pronunciada/data/coach/identity") / f"{lang}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    spec = SPEC[a.lang]
    X, y, _ = rows_xy(a.lang, ["fltrain", "cvtrain"], spec)
    rng = np.random.default_rng(0)
    keep = []
    for c in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == c)
        if len(idx) >= 150:
            keep.append(rng.permutation(idx)[:a.cap])
    keep = np.sort(np.concatenate(keep))
    X, y = X[keep], y[keep]
    classes = sorted(set(y.tolist()))
    print("rows", len(y), "classes", len(classes), flush=True)
    H = train(X, y, classes, a.hidden, a.epochs)
    del X
    H32 = {k: (v.astype(np.float16).astype(np.float32) if k in ("mu", "sd", "W1", "W2") else v) for k, v in H.items()}
    rep = {"lang": a.lang, "n_train": int(len(y)), "classes": classes, "hidden": a.hidden, "epochs": a.epochs,
           "cap": a.cap, "spec": spec, "dev": accuracy(a.lang, H32, ["dev"], spec),
           "pool": accuracy(a.lang, H32, ["cvpool"], spec)}
    print(json.dumps({k: v for k, v in rep.items() if k != "classes"}), flush=True)
    out = work(a.lang) / "head.npz"
    np.savez_compressed(out, **{k: (v.astype(np.float16) if k in ("mu", "sd", "W1", "W2") else v) for k, v in H.items()})
    (work(a.lang) / "head_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    if a.save:
        head_path(a.lang).parent.mkdir(parents=True, exist_ok=True)
        head_path(a.lang).write_bytes(out.read_bytes())
        Path(f"results/tables/round19_rollout_{a.lang}_head.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
