"""Round 19 (English): a small multi-task sequence head (GOPT-style) on per-sound features.

Tokens: 4 score queries (accuracy, fluency, prosodic, total) + 1 utterance-feature token + one
token per expected sound. A sound token = learned weighted sum over the ZIPA encoder stacks
(each stack: its own linear map to d; softmax weights) + a linear map of the aligner evidence
(the 48 phone-table inputs of the sound models) + an embedding of the expected sound + an
embedding of its position in the word (B/I/E/S). The utterance token = a linear map of the
utterance inputs (shipped sentence inputs SENT_NUM, delivery features, F0 summary).
A transformer encoder (pre-LN) mixes them. Outputs: 4 sentence scores (from the query tokens),
a per-sound score (0-2) and a per-sound word contribution (word score = mean over its sounds).
Targets are scaled to 0-1. Loss = MSE (sentence) + w_word * MSE (word) + w_phone * MSE (phone)
+ w_corr * (1 - Pearson) of each sentence head over the batch.

torch is a training-time dependency only; the trained module is exported to ONNX
(export_onnx) and run with onnxruntime.
"""
from __future__ import annotations

import math

import numpy as np

LAYER_DIMS = {"enc0": 192, "enc1": 256, "enc2": 384, "enc3": 512, "enc4": 384, "enc5": 256, "final": 512}


def build_module(n_tab, n_utt, n_phones, layers, d=96, n_layers=2, n_heads=4, ff=192, dropout=0.1,
                 fuse="weighted"):
    import torch
    from torch import nn

    class SeqHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = list(layers)
            self.fuse = fuse
            self.lin = nn.ModuleList([nn.Linear(LAYER_DIMS[k], d) for k in self.layers])
            self.alpha = nn.Parameter(torch.zeros(len(self.layers)))
            self.tab = nn.Linear(n_tab, d)
            self.ph = nn.Embedding(n_phones + 1, d)
            self.pos = nn.Embedding(5, d)
            self.utt = nn.Linear(n_utt, d)
            self.q = nn.Parameter(torch.randn(4, d) * 0.02)
            self.drop = nn.Dropout(dropout)
            el = nn.TransformerEncoderLayer(d, n_heads, ff, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(el, n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.head_s = nn.Linear(d, 4)
            self.head_p = nn.Linear(d, 1)
            self.head_w = nn.Linear(d, 1)

        def forward(self, xl: list, xt, pid, ppos, xu, mask):
            """xl: per layer [B, N, C_k]; xt [B, N, n_tab]; pid, ppos [B, N] long; xu [B, n_utt];
            mask [B, N] bool (True = real sound). Returns sentence [B, 4], phone [B, N], word-part [B, N]."""
            w = torch.softmax(self.alpha, 0)
            h = sum(w[i] * self.lin[i](x) for i, x in enumerate(xl))
            h = h + self.tab(xt) + self.ph(pid) + self.pos(ppos)
            B = xt.shape[0]
            u = self.utt(xu)[:, None, :]
            q = self.q[None].expand(B, -1, -1)
            z = torch.cat([q, u, self.drop(h)], 1)
            keep = torch.cat([torch.ones(B, 5, dtype=torch.bool, device=mask.device), mask], 1)
            z = self.enc(z, src_key_padding_mask=~keep)
            z = self.norm(z)
            s = torch.diagonal(self.head_s(z[:, :4]), dim1=1, dim2=2)
            hp = z[:, 5:]
            return s, self.head_p(hp)[..., 0], self.head_w(hp)[..., 0]

    return SeqHead()


def n_params(m) -> int:
    return int(sum(p.numel() for p in m.parameters()))


# ----------------------------------------------------------------------------- data

POS = {"B": 1, "I": 2, "E": 3, "S": 4}


def word_pos(phone_word):
    pw = np.asarray(phone_word)
    out = np.zeros(len(pw), np.int64)
    for i in range(len(pw)):
        first = i == 0 or pw[i - 1] != pw[i]
        last = i == len(pw) - 1 or pw[i + 1] != pw[i]
        out[i] = POS["S"] if first and last else POS["B"] if first else POS["E"] if last else POS["I"]
    return out


class Norm:
    """Per-column standardisation fitted on training rows (NaN ignored, NaN -> 0 after),
    each column clipped to +-clip standard deviations."""

    def __init__(self, X, clip=5.0):
        X = np.asarray(X, np.float64)
        self.mu = np.nan_to_num(np.nanmean(X, 0))
        sd = np.nan_to_num(np.nanstd(X, 0))
        self.sd = np.where(sd < 1e-6, 1.0, sd)
        self.clip = clip

    def __call__(self, X):
        Z = (np.asarray(X, np.float64) - self.mu) / self.sd
        return np.clip(np.nan_to_num(Z, nan=0.0), -self.clip, self.clip).astype(np.float32)

    def state(self, prefix):
        return {f"{prefix}_mu": self.mu.astype(np.float32), f"{prefix}_sd": self.sd.astype(np.float32),
                f"{prefix}_clip": np.array([self.clip], np.float32)}


def pad_batch(items, layers):
    """items: list of dicts with xl {k: [n, C]}, xt [n, F], pid [n], ppos [n], xu [F_u],
    yp [n], yw_ph [n] (word target broadcast to its sounds), ys [4]."""
    import torch

    B = len(items)
    N = max(len(it["pid"]) for it in items)
    out = {"xl": [torch.zeros(B, N, LAYER_DIMS[k]) for k in layers],
           "xt": torch.zeros(B, N, items[0]["xt"].shape[1]), "pid": torch.zeros(B, N, dtype=torch.long),
           "ppos": torch.zeros(B, N, dtype=torch.long), "mask": torch.zeros(B, N, dtype=torch.bool),
           "xu": torch.tensor(np.stack([it["xu"] for it in items])),
           "ys": torch.tensor(np.stack([it["ys"] for it in items]), dtype=torch.float32),
           "yp": torch.full((B, N), float("nan")), "yw": torch.full((B, N), float("nan")),
           "wid": torch.full((B, N), -1, dtype=torch.long)}
    for b, it in enumerate(items):
        n = len(it["pid"])
        for i, k in enumerate(layers):
            out["xl"][i][b, :n] = torch.from_numpy(it["xl"][k])
        out["xt"][b, :n] = torch.from_numpy(it["xt"])
        out["pid"][b, :n] = torch.from_numpy(it["pid"])
        out["ppos"][b, :n] = torch.from_numpy(it["ppos"])
        out["mask"][b, :n] = True
        out["yp"][b, :n] = torch.from_numpy(it["yp"].astype(np.float32))
        out["yw"][b, :n] = torch.from_numpy(it["yw_ph"].astype(np.float32))
        out["wid"][b, :n] = torch.from_numpy(it["wid"])
    return out


def word_means(pw, wid, mask):
    """Per (batch, word) mean of per-sound word contributions -> list of tensors aligned with
    the first sound of each word (returns [B, N] where each sound holds its word's mean)."""
    import torch

    B, N = pw.shape
    W = int(wid.max().item()) + 1 if wid.numel() else 1
    W = max(W, 1)
    oh = torch.zeros(B, N, W)
    ok = mask & (wid >= 0)
    oh[ok] = torch.nn.functional.one_hot(wid[ok], W).float()
    s = torch.einsum("bn,bnw->bw", pw, oh)
    c = oh.sum(1).clamp(min=1.0)
    m = s / c
    return torch.einsum("bw,bnw->bn", m, oh)


def corr_loss(p, y):
    p = p - p.mean(0)
    y = y - y.mean(0)
    r = (p * y).sum(0) / (p.norm(dim=0) * y.norm(dim=0) + 1e-6)
    return (1 - r).mean()


def train(model, items, layers, epochs=40, bs=32, lr=1e-3, wd=1e-2, w_word=0.5, w_phone=0.5, w_corr=0.5,
          seed=0, log=None):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(items) / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    for ep in range(epochs):
        order = rng.permutation(len(items))
        tot = 0.0
        for i in range(0, len(order), bs):
            b = pad_batch([items[j] for j in order[i:i + bs]], layers)
            s, p, pw = model(b["xl"], b["xt"], b["pid"], b["ppos"], b["xu"], b["mask"])
            loss = ((s - b["ys"]) ** 2).mean()
            if w_corr and len(s) > 4:
                loss = loss + w_corr * corr_loss(s, b["ys"])
            okp = b["mask"] & torch.isfinite(b["yp"])
            if w_phone and okp.any():
                loss = loss + w_phone * ((p[okp] - b["yp"][okp]) ** 2).mean()
            wm = word_means(pw, b["wid"], b["mask"])
            okw = b["mask"] & torch.isfinite(b["yw"])
            if w_word and okw.any():
                loss = loss + w_word * ((wm[okw] - b["yw"][okw]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += float(loss) * len(s)
        if log and (ep + 1) % 10 == 0:
            log(f"  epoch {ep + 1}: loss {tot / len(items):.4f}")
    model.eval()
    return model


def predict(model, items, layers, bs=64):
    import torch

    S, P, Wd = [], [], []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            b = pad_batch(chunk, layers)
            s, p, pw = model(b["xl"], b["xt"], b["pid"], b["ppos"], b["xu"], b["mask"])
            wm = word_means(pw, b["wid"], b["mask"])
            for k, it in enumerate(chunk):
                n = len(it["pid"])
                S.append(s[k].numpy())
                P.append(p[k, :n].numpy())
                Wd.append(wm[k, :n].numpy())
    return np.stack(S), P, Wd
