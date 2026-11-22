"""Train the padded-window vowel-length head (round 20, Japanese).

    python -m bench.round20_ja_length.train cv   --cfg NAME      # 4-fold speaker-grouped CV on the fit pool
    python -m bench.round20_ja_length.train fit  --cfg NAME      # all fit data -> head_<NAME>.pt / .onnx
    python -m bench.round20_ja_length.train dev  --cfg NAME      # the fitted head on the dev pool windows

Model (``Head``): per-frame linear projection of the stack's activations (C channels), two 1-D
convolutions (kernel 5) with GELU, masked attention pooling over the window, the log local mora
duration appended (``rate`` = "input"), one logit = log-odds that the AUDIO vowel is long. Loss:
binary cross-entropy against the soft label (natural vowels: the text's label; WSOLA targets: the
label implied by the stretch factor, bench.round20_ja_length.common.soft_label).

Rate options: "input" (log mora duration as an input), "none", "warp" (the window is resampled in
time so it spans the same number of morae whatever the speaking rate: +-w frames at the median
rate; implemented in ``warp_windows``, also used at runtime).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from bench.round20_ja_length.common import OUT, W

CFGS = {
    # name: layers, half-width (frames), rate handling, channels, epochs
    "e1_w8": dict(layers=["enc1"], w=8, rate="input", ch=48, epochs=12),
    "e1_w5": dict(layers=["enc1"], w=5, rate="input", ch=48, epochs=12),
    "e1_w10": dict(layers=["enc1"], w=10, rate="input", ch=48, epochs=12),
    "e5_w8": dict(layers=["enc5"], w=8, rate="input", ch=48, epochs=12),
    "e15_w8": dict(layers=["enc1", "enc5"], w=8, rate="input", ch=48, epochs=12),
    "e1_w8_norate": dict(layers=["enc1"], w=8, rate="none", ch=48, epochs=12),
    "e1_w8_warp": dict(layers=["enc1"], w=8, rate="warp", ch=48, epochs=12),
}
MED_MD = 3.9           # median local mora duration (20 ms frames) of the fit pool natives


def warp_windows(win: np.ndarray, md: np.ndarray, w: int) -> np.ndarray:
    """[N, 2W+1, D] windows -> [N, 2w+1, D], resampled (linear) so the output spans
    +-w * md / MED_MD frames of the input: the same number of morae at every rate."""
    N, L, D = win.shape
    c = (L - 1) / 2
    out = np.zeros((N, 2 * w + 1, D), np.float32)
    k = np.arange(-w, w + 1, dtype=np.float32)
    for n in range(N):
        t = c + k * float(md[n]) / MED_MD
        t = np.clip(t, 0, L - 1)
        i0 = np.floor(t).astype(int)
        i1 = np.minimum(i0 + 1, L - 1)
        a = (t - i0)[:, None]
        out[n] = (1 - a) * win[n, i0] + a * win[n, i1]
    return out


def load(name: str, layers, w: int, rate: str):
    from bench.round20_ja_length.probe import dims, load as load_raw

    X, R, lay = load_raw(name)
    sl = dims(lay)
    parts = [X[:, :, sl[L][0]:sl[L][1]] for L in layers]
    A = np.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]
    md = R["md_loc"].astype(np.float32)
    if rate == "warp":
        A = warp_windows(A, md, w).astype(np.float16)
    else:
        A = np.ascontiguousarray(A[:, W - w: W + w + 1])
    mask = np.zeros((len(A), 2 * w + 1), np.float32)
    off = np.arange(2 * w + 1) + (W - w)
    if rate != "warp":
        mask = ((off[None] >= R["v0"][:, None]) & (off[None] < R["v1"][:, None])).astype(np.float32)
    else:
        mask[:] = 1.0
    return A, mask, R


def make_head(D: int, ch: int, rate: str, w: int):
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(D)
            self.proj = nn.Linear(D, ch)
            self.c1 = nn.Conv1d(ch, ch, 5, padding=2)
            self.c2 = nn.Conv1d(ch, ch, 5, padding=2)
            self.att = nn.Linear(ch, 1)
            self.pos = nn.Parameter(torch.zeros(2 * w + 1, ch))
            self.out = nn.Linear(ch + (1 if rate == "input" else 0), 1)
            self.rate = rate

        def forward(self, x, mask, lrate):
            # x [B, T, D], mask [B, T], lrate [B]
            h = self.proj(self.norm(x)) + self.pos
            h = h * mask[..., None]
            h = h.transpose(1, 2)
            h = torch.nn.functional.gelu(self.c1(h))
            h = torch.nn.functional.gelu(self.c2(h)) + h
            h = h.transpose(1, 2)
            a = self.att(h)[..., 0] + (mask - 1.0) * 1e4
            a = torch.softmax(a, dim=1)
            z = (h * a[..., None]).sum(1)
            if self.rate == "input":
                z = torch.cat([z, lrate[:, None]], 1)
            return self.out(z)[:, 0]

    return Head()


def train_one(A, M, R, idx, cfg, seed=0, log=None, val=None):
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    D = A.shape[2]
    net = make_head(D, cfg["ch"], cfg["rate"], cfg["w"])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    y = R["y"].astype(np.float32)
    lr_ = (np.log(R["md_loc"]) - math.log(MED_MD)).astype(np.float32)
    # balance: long and short targets weigh the same in total
    wts = np.where(y > 0.5, 0.5 / max((y[idx] > 0.5).mean(), 1e-3), 0.5 / max((y[idx] <= 0.5).mean(), 1e-3))
    wts = wts.astype(np.float32)
    rng = np.random.default_rng(seed)
    E = cfg["epochs"]
    bs = 256
    steps = E * (len(idx) // bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    hist = []
    for ep in range(E):
        net.train()
        perm = rng.permutation(idx)
        tot = 0.0
        for b in range(0, len(perm), bs):
            j = np.sort(perm[b: b + bs])
            x = torch.from_numpy(A[j].astype(np.float32))
            m = torch.from_numpy(M[j])
            if cfg.get("time_mask"):
                pass
            out = net(x, m, torch.from_numpy(lr_[j]))
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(out, torch.from_numpy(y[j]), reduction="none")
                    * torch.from_numpy(wts[j])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(j)
        rec = {"epoch": ep + 1, "loss": tot / len(perm)}
        if val is not None:
            rec.update(val(net))
        hist.append(rec)
        if log:
            log(json.dumps(rec))
    return net, hist


def predict(net, A, M, R, idx=None):
    import torch

    idx = np.arange(len(A)) if idx is None else idx
    lr_ = (np.log(R["md_loc"]) - math.log(MED_MD)).astype(np.float32)
    out = np.zeros(len(idx), np.float32)
    net.eval()
    with torch.no_grad():
        for b in range(0, len(idx), 1024):
            j = idx[b: b + 1024]
            out[b: b + len(j)] = net(torch.from_numpy(A[j].astype(np.float32)), torch.from_numpy(M[j]),
                                     torch.from_numpy(lr_[j])).numpy()
    return out


def evaluate(z, R, idx):
    """AUCs on natural clean vowels (text label) and accuracy on WSOLA targets by SNR."""
    from sklearn.metrics import roc_auc_score

    out = {}
    kind, snr, f, y = R["kind"][idx], R["snr"][idx], R["factor"][idx], R["y"][idx]
    tl = R["text_long"][idx]
    nat = kind == 0
    if nat.any() and 0 < tl[nat].sum() < nat.sum():
        out["nat_auc"] = float(roc_auc_score(tl[nat], z[nat]))
    noisy_nat = (kind == 3) & (R["factor"][idx] == 1.0) & (snr < 99)
    if noisy_nat.any() and 0 < tl[noisy_nat].sum() < noisy_nat.sum():
        out["natnoise_auc"] = float(roc_auc_score(tl[noisy_nat], z[noisy_nat]))
    tgt = (kind > 0) & (np.abs(f - 1) > 0.1) & ((y > 0.9) | (y < 0.1))
    for s in sorted(set(np.round(snr[tgt]).tolist())):
        sel = tgt & (np.round(snr) == s)
        if sel.sum() >= 20:
            out[f"wsola_acc_snr{int(s)}"] = float(((z[sel] > 0) == (y[sel] > 0.5)).mean())
            out[f"wsola_n_snr{int(s)}"] = int(sel.sum())
    # mismatched pairs only: text says long, audio short (compressed long vowels) and vice versa
    for tlv, nm in ((1.0, "long2short"), (0.0, "short2long")):
        sel = tgt & (tl == tlv) & ((y > 0.5) != bool(tlv))
        if sel.sum() >= 20:
            out[f"wsola_{nm}_acc"] = float(((z[sel] > 0) == (y[sel] > 0.5)).mean())
    return out


def folds(groups, k=4, seed=0):
    from bench.round20_ja_length.probe import folds as f

    return f(groups, k, seed)


def cmd_cv(name: str) -> None:
    cfg = CFGS[name]
    A, M, R = load("win_fit", cfg["layers"], cfg["w"], cfg["rate"])
    fo = folds(R["group"])
    z = np.zeros(len(A), np.float32)
    hists = []
    for k in range(4):
        tr, te = np.flatnonzero(fo != k), np.flatnonzero(fo == k)
        net, h = train_one(A, M, R, tr, cfg, seed=k, log=lambda s: print(f"fold {k}", s, file=sys.stderr, flush=True),
                           val=lambda n: evaluate(predict(n, A, M, R, te), R, te))
        z[te] = predict(net, A, M, R, te)
        hists.append(h)
    res = {"cfg": cfg, "oof": evaluate(z, R, np.arange(len(A))), "hist": hists}
    print(name, json.dumps(res["oof"]), flush=True)
    (OUT / f"cv_{name}.json").write_text(json.dumps(res, indent=1))
    np.save(OUT / f"cv_{name}_oof.npy", z)


def cmd_fit(name: str) -> None:
    import torch

    cfg = CFGS[name]
    A, M, R = load("win_fit", cfg["layers"], cfg["w"], cfg["rate"])
    net, h = train_one(A, M, R, np.arange(len(A)), cfg, seed=0, log=lambda s: print(s, file=sys.stderr, flush=True))
    torch.save(net.state_dict(), OUT / f"head_{name}.pt")
    export_onnx(net, A.shape[2], cfg, OUT / f"head_{name}.onnx")
    (OUT / f"fit_{name}.json").write_text(json.dumps({"cfg": cfg, "hist": h}, indent=1))


def export_onnx(net, D: int, cfg: dict, path) -> None:
    import torch

    T = 2 * cfg["w"] + 1
    net.eval()
    x = torch.zeros(2, T, D)
    m = torch.ones(2, T)
    r = torch.zeros(2)
    torch.onnx.export(net, (x, m, r), str(path), input_names=["x", "mask", "lrate"], output_names=["logit"],
                      dynamic_axes={"x": {0: "n"}, "mask": {0: "n"}, "lrate": {0: "n"}, "logit": {0: "n"}},
                      opset_version=17, dynamo=False)


def cmd_dev(name: str) -> None:
    import torch

    cfg = CFGS[name]
    A, M, R = load("win_dev", cfg["layers"], cfg["w"], cfg["rate"])
    net = make_head(A.shape[2], cfg["ch"], cfg["rate"], cfg["w"])
    net.load_state_dict(torch.load(OUT / f"head_{name}.pt"))
    z = predict(net, A, M, R)
    res = evaluate(z, R, np.arange(len(A)))
    # the shipped length model on the same natural vowels, as P(long)
    from sklearn.metrics import roc_auc_score

    nat = R["kind"] == 0
    tl = R["text_long"]
    pl = np.where(tl == 1, R["lprob"], 1 - R["lprob"])
    res["shipped_nat_auc"] = float(roc_auc_score(tl[nat], pl[nat]))
    for s in ("cv", "fleurs"):
        sel = nat & (R["source"] == s)
        res[f"nat_auc_{s}"] = float(roc_auc_score(tl[sel], z[sel]))
        res[f"shipped_nat_auc_{s}"] = float(roc_auc_score(tl[sel], pl[sel]))
    # shipped model on the WSOLA mismatches: P(matches text) should be low when audio != text
    tgt = (R["kind"] > 0) & (np.abs(R["factor"] - 1) > 0.1) & ((R["y"] > 0.9) | (R["y"] < 0.1))
    mis = tgt & ((R["y"] > 0.5) != (tl > 0.5))
    for s in sorted(set(np.round(R["snr"][tgt]).tolist())):
        sel_m = mis & (np.round(R["snr"]) == s)
        sel_n = nat
        if sel_m.sum() >= 20:
            yy = np.r_[np.ones(sel_m.sum()), np.zeros(sel_n.sum())]
            # mismatch detection AUC (natives = matching): head vs shipped
            hz = np.where(tl == 1, -z, z)        # evidence AGAINST the text's length
            res[f"mismatch_auc_head_snr{int(s)}"] = float(roc_auc_score(yy, np.r_[hz[sel_m], hz[sel_n]]))
            res[f"mismatch_auc_shipped_snr{int(s)}"] = float(roc_auc_score(yy, -np.r_[R["lprob"][sel_m], R["lprob"][sel_n]]))
    print(name, json.dumps(res), flush=True)
    (OUT / f"dev_{name}.json").write_text(json.dumps(res, indent=1))
    np.save(OUT / f"dev_{name}_z.npy", z)


def load_meta(name: str) -> dict:
    rows: dict = {}
    for f in sorted((OUT / name).glob("shard_*.npz")):
        m = json.loads(str(np.load(f)["meta"]))["rows"]
        for k, v in m.items():
            rows.setdefault(k, []).extend(v)
    return {k: np.array(v) for k, v in rows.items()}


QS = (0.25, 0.5, 1, 2, 3, 5)


def cmd_knees(name: str) -> None:
    """Native quantiles (in %) of the head's out-of-fold match evidence per expected class (natural
    clean fit-pool vowels): where the lower-only step starts (pronunciada.ja_length20)."""
    R = load_meta("win_fit")
    z = np.load(OUT / f"cv_{name}_oof.npy")
    tl = R["text_long"]
    zh = np.where(tl == 1, z, -z)
    nat = (R["kind"] == 0) & (R["factor"] == 1.0)
    out = {c: {f"{q:g}": float(np.quantile(zh[nat & (tl == v)], q / 100)) for q in QS}
           for c, v in (("long", 1), ("short", 0))}
    out["n"] = {"long": int((nat & (tl == 1)).sum()), "short": int((nat & (tl == 0)).sum())}
    (OUT / f"knees_{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("cv", "fit", "dev", "knees"))
    ap.add_argument("--cfg", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    {"cv": cmd_cv, "fit": cmd_fit, "dev": cmd_dev, "knees": cmd_knees}[a.cmd](a.cfg)
    print(f"{a.cmd} {a.cfg} done in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
