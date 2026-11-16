"""Round 8: the tone-v3z recipe for word stress (en de es it pt ru) and Japanese pitch accent.

    .venv/bin/python -m bench.prosody.prosody3 train stress --name stress_v3 [--steps 6000]
    .venv/bin/python -m bench.prosody.prosody3 train pitch  --name pitch_ja_v3
    .venv/bin/python -m bench.prosody.prosody3 eval stress stress_multi stress_v3     # dev only
    .venv/bin/python -m bench.prosody.prosody3 eval pitch pitch_ja_s2 pitch_ja_v3

Inputs per 10 ms frame (as tone_zh_v3z): the 5 pitch / energy features, local-register pitch (2),
low-frequency log-mel (20), and the syllable's pooled ZIPA enc1+enc3 activations (PCA 32 + flag)
broadcast over its window, so a syllable's context window also carries its neighbours'.
Training data: the same as the shipped models (FLEURS train of each language, front-end labels).
Dev = FLEURS validation; held-out sets are not read here.

Counterfactual ("transplant") augmentation in feature space:
  stress  for a share of words, the pitch + energy frames (lf0, energy and their deltas) of the
          primary-stressed syllable and of another syllable of the word are swapped (each time-
          warped to the other's window); the label moves to the other syllable. Durations, spectra
          and ZIPA activations stay -> segment cues alone no longer predict the label.
  pitch   for a share of utterances, the pitch frames (lf0, d_lf0) of 1-3 pairs of morae with
          opposite high/low labels are swapped (time-warped); both labels flip.
Leak checks (dev, feature space, the same for old and new models):
  stress  swap: the cue swap above on every test word (fixed seed) -> share whose argmax follows
          the cues to the other syllable / stays on the dictionary syllable;
          flatten: pitch + energy of the word's syllables replaced by the word mean -> share still
          on the dictionary syllable (chance = mean 1 / n syllables).
  pitch   flatten: lf0 := 0 (the utterance mean) everywhere -> share of morae still classified as
          their label; swap: every opposite-label adjacent pair swapped -> share that flips.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from multiprocessing import get_context

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from bench.prosody.common import (MODELS_DIR, OUT, jitter_windows, load_prepped, save_json,  # noqa: E402
                                  scale_windows, stretch_track, to_track)
from pronunciada.prosody.features import (N_IN, N_LOWMEL, N_REG, N_STRESS_EXTRA, context_input,  # noqa: E402
                                          frame_features, register_features, stress_extra, syllable_windows)

STRESS_LANGS = ("en", "de", "es", "it", "pt", "ru")
N_ZE = 32
MAX_SYL = 8
BASE = {"reg": False, "lowmel": False, "zemb": False}
V3 = {"reg": True, "lowmel": True, "zemb": True}
STRESS_COLS = [0, 2, 3, 4]
PITCH_COLS = [0, 3]
_C = {}


def n_in(cfg):
    return N_IN + (N_REG if cfg["reg"] else 0) + (N_LOWMEL if cfg["lowmel"] else 0) + (N_ZE + 1 if cfg["zemb"] else 0)


# ----------------------------------------------------------------------------- feature caches

def pca(task):
    """Fixed PCA of pooled ZIPA enc1+enc3 syllable vectors on FLEURS TRAIN of the task's languages."""
    from bench.prosody.zipa_emb_cache import EMB

    p = EMB / f"pca_{task}.npz"
    if not p.exists():
        langs = STRESS_LANGS if task == "stress" else ("ja",)
        rng = np.random.default_rng(0)
        X = []
        for lg in langs:
            d = np.load(EMB / f"fleurs-{lg}-train.npz")
            ks = d.files[::max(1, len(d.files) // 300)]
            X += [d[k].astype(np.float32) for k in ks]
        X = np.concatenate(X)
        X = X[rng.permutation(len(X))[:60000]]
        mu = X.mean(0)
        _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        np.savez(p, mu=mu, W=(Vt[:N_ZE].T / (S[:N_ZE] / np.sqrt(len(X)))).astype(np.float32))
    d = np.load(p)
    return d["mu"], d["W"]


def _lowmel(u):
    if "lm" not in _C:
        from bench.prosody.lowmel_cache import LowMel

        _C["lm"] = LowMel()
    return _C["lm"](u)


def _zemb(u, wins, T0, task):
    if "ze" not in _C:
        from bench.prosody.zipa_emb_cache import ZipaEmb

        _C["ze"] = (ZipaEmb(), *pca(task))
    src, mu, W = _C["ze"]
    E, flag = src(u)
    out = np.zeros((T0, N_ZE + 1), np.float32)
    if flag:
        Z = np.clip((E - mu) @ W, -5, 5)
        for si, w in enumerate(wins):
            out[w.start:w.end, :N_ZE] = Z[si]
            out[w.start:w.end, N_ZE] = 1.0
    return out


def swap_cues(feats, wins, i, j, cols):
    """Swap frames `cols` of syllable windows i and j, each time-warped to the other's length."""
    a, b = wins[i], wins[j]
    fa, fb = feats[a.start:a.end, cols].copy(), feats[b.start:b.end, cols].copy()
    if len(fa) == 0 or len(fb) == 0:
        return feats

    def warp(x, n):
        src = np.linspace(0, len(x) - 1, n)
        return np.stack([np.interp(src, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], 1)

    feats[a.start:a.end, cols] = warp(fb, a.end - a.start)
    feats[b.start:b.end, cols] = warp(fa, b.end - b.start)
    return feats


def utt_feats(u, cfg, task, rng=None, aug=False, edit=None):
    """(feats [T, C], wins). `edit(feats, wins)` modifies the base pitch / energy frames (cue swap,
    flattening) before the register features are derived."""
    d = u["track"]
    if aug and "noisy" in u and rng.random() < 0.3:
        d = u["noisy"]
    tr = to_track(d)
    T0 = len(tr.f0)
    wins = syllable_windows(u["ref"], u["spans"], T0)
    lm = _lowmel(u) if cfg["lowmel"] else None
    if lm is not None and len(lm) != T0:
        lm = lm[:T0] if len(lm) > T0 else np.pad(lm, ((0, T0 - len(lm)), (0, 0)))
    ze = _zemb(u, wins, T0, task) if cfg["zemb"] else None
    if aug:
        if lm is not None and rng.random() < 0.3:
            lm = np.zeros_like(lm)
        if ze is not None and rng.random() < 0.3:
            ze = np.zeros_like(ze)
        r = float(rng.uniform(0.85, 1.15))
        tr = stretch_track(tr, r)
        T1 = len(tr.f0)
        wins = scale_windows(wins, T1 / T0, T1)
        ix = np.clip(np.round(np.linspace(0, T0 - 1, T1)).astype(int), 0, T0 - 1)
        lm = lm[ix] if lm is not None else None
        ze = ze[ix] if ze is not None else None
    feats = frame_features(tr)
    if aug:
        s = float(rng.uniform(0.7, 1.3))
        feats[:, 0] *= s
        feats[:, 3] *= s
        feats[:, 0] += rng.normal(0, 0.05, len(feats)).astype(np.float32)
        feats[:, 2] += float(rng.normal(0, 0.2))
        wins = jitter_windows(wins, rng, 2, len(feats))
    if edit is not None:
        feats = edit(feats, wins)
    cols = [feats]
    if cfg["reg"]:
        cols.append(register_features(feats))
    if lm is not None:
        cols.append(lm)
    if ze is not None:
        cols.append(ze)
    return np.concatenate(cols, 1), wins


# ----------------------------------------------------------------------------- stress data

def word_groups(u):
    ref = u["ref"]
    out = []
    for w in ref.words:
        s0, s1 = w.syllables
        syl = ref.syllables[s0:s1]
        if not (2 <= s1 - s0 <= MAX_SYL) or any(s.stress is None for s in syl):
            continue
        prim = [i for i, s in enumerate(syl) if s.stress == 1]
        if len(prim) == 1:
            out.append((s0, s1, prim[0]))
    return out


def stress_words(u, lid, cfg, rng=None, aug=False, tp=0.0, mode=None, seed=0):
    """Per word: (X [S, L, C] f16, E [S, 7], n, label, lid). mode: None | "swap" | "flatten"."""
    g = word_groups(u)
    if not g:
        return []
    plan = {}
    r2 = np.random.default_rng(seed)
    for s0, s1, p in g:
        n = s1 - s0
        if mode == "swap" or (aug and rng.random() < tp):
            rr = r2 if mode == "swap" else rng
            j = int(rr.choice([k for k in range(n) if k != p]))
            plan[s0] = ("swap", s0 + p, s0 + j, j)
        elif mode == "flatten":
            plan[s0] = ("flat", s0, s1, p)

    def edit(feats, wins):
        for s0, (kind, a, b, _) in plan.items():
            if kind == "swap":
                feats = swap_cues(feats, wins, a, b, STRESS_COLS)
            else:
                lo, hi = wins[a].start, wins[b - 1].end
                feats[lo:hi, STRESS_COLS] = feats[lo:hi, STRESS_COLS].mean(0, keepdims=True)
        return feats

    feats, wins = utt_feats(u, cfg, "stress", rng, aug, edit if plan else None)
    out = []
    for s0, s1, p in g:
        n = s1 - s0
        x = np.zeros((MAX_SYL, context_input(feats, wins, s0).shape[0], feats.shape[1] + 5), np.float16)
        for k in range(n):
            x[k] = context_input(feats, wins, s0 + k)
        e = np.zeros((MAX_SYL, N_STRESS_EXTRA), np.float32)
        e[:n] = stress_extra(feats, wins, (s0, s1))
        lab = plan[s0][3] if s0 in plan and plan[s0][0] == "swap" else p
        out.append((x, e, n, lab, lid, p))
    return out


# ----------------------------------------------------------------------------- pitch data

def pitch_utt(u, cfg, rng=None, aug=False, tp=0.0, mode=None, seed=0):
    """(X [S, L, C], labels [S] (1 high / 0 low / -1), original labels)."""
    labs = np.array([-1 if s.pitch_high is None else int(s.pitch_high) for s in u["ref"].syllables])
    orig = labs.copy()
    pairs = []
    if mode == "swap":
        pairs = [(i, i + 1) for i in range(0, len(labs) - 1, 2) if labs[i] >= 0 and labs[i + 1] >= 0
                 and labs[i] != labs[i + 1]]
    elif aug and rng.random() < tp:
        hi, lo = np.flatnonzero(labs == 1), np.flatnonzero(labs == 0)
        k = min(int(rng.integers(1, 4)), len(hi), len(lo))
        if k:
            pairs = list(zip(rng.choice(hi, k, replace=False), rng.choice(lo, k, replace=False)))

    def edit(feats, wins):
        if mode == "flatten":
            feats[:, PITCH_COLS] = 0.0
        for i, j in pairs:
            feats = swap_cues(feats, wins, int(i), int(j), PITCH_COLS)
        return feats

    feats, wins = utt_feats(u, cfg, "pitch", rng, aug, edit if (pairs or mode == "flatten") else None)
    for i, j in pairs:
        labs[i], labs[j] = labs[j], labs[i]
    X = np.stack([context_input(feats, wins, i) for i in range(len(wins))])
    return X, labs, orig


# ----------------------------------------------------------------------------- models

def make_net(task, cfg):
    from bench.prosody.model import ProsodyNet, SeqToneNet
    from torch import nn

    if task == "stress":
        class StressNet3(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = ProsodyNet(n_in=n_in(cfg), n_out=1, ch=64, hid=48, n_lang=len(STRESS_LANGS),
                                      n_extra=N_STRESS_EXTRA, stride=2)

            def forward(self, x, extra, lang=None):
                return self.net(x, lang, extra)

        return StressNet3()
    return SeqToneNet(n_in=n_in(cfg), n_out=1)


def load_train(task):
    if task == "stress":
        out = []
        for k, lg in enumerate(STRESS_LANGS):
            out += [(u, k) for u in load_prepped("fleurs", lg, "train")]
        return out
    return [(u, 0) for u in load_prepped("fleurs", "ja", "train")]


def load_dev(task, lang=None):
    if task == "stress":
        langs = [lang] if lang else STRESS_LANGS
        return [(u, STRESS_LANGS.index(lg)) for lg in langs for u in load_prepped("fleurs_val", lg, "dev")]
    return [(u, 0) for u in load_prepped("fleurs_val", "ja", "dev")]


_W = {}


def _init(task, cfg, tp):
    _W.update(task=task, cfg=cfg, tp=tp, data=load_train(task))


def _gen(args):
    seed, idx = args
    rng = np.random.default_rng(seed)
    out = []
    for k in idx:
        u, lid = _W["data"][k]
        if not u["ref"].syllables:
            continue
        if _W["task"] == "stress":
            out += stress_words(u, lid, _W["cfg"], rng, True, _W["tp"])
        else:
            X, y, _ = pitch_utt(u, _W["cfg"], rng, True, _W["tp"])
            if (y >= 0).any():
                out.append((X.astype(np.float16), y))
    return out


def stress_batch(net, items, dev):
    import torch

    X = np.stack([it[0] for it in items])
    E = np.stack([it[1] for it in items])
    B, S = X.shape[:2]
    m = np.zeros((B, S), bool)
    for b, it in enumerate(items):
        m[b, :it[2]] = True
    x = torch.tensor(X.reshape((B * S,) + X.shape[2:]), dtype=torch.float32, device=dev)
    e = torch.tensor(E.reshape(B * S, -1), dtype=torch.float32, device=dev)
    lg = torch.tensor(np.repeat([it[4] for it in items], S), device=dev)
    z = net(x, e, lg).view(B, S).masked_fill(~torch.tensor(m, device=dev), -1e4)
    return z


def pitch_batch(net, items, dev):
    import torch

    B = len(items)
    S = max(len(x) for x, _ in items)
    X = np.concatenate([x for x, _ in items])
    index = np.concatenate([b * S + np.arange(len(x)) for b, (x, _) in enumerate(items)])
    lengths = torch.tensor([len(x) for x, _ in items])
    z = net(torch.tensor(X, dtype=torch.float32, device=dev), torch.tensor(index, device=dev), lengths, B, S)[:, 0]
    y = np.concatenate([yy for _, yy in items])
    return z, y


def predict(task, net, data, cfg, mode=None):
    """Dev predictions. stress: list of (P [n], label, dictionary label, lid, n); pitch: list of
    (p_high [S], labels, original labels)."""
    import torch

    out = []
    with torch.no_grad():
        for ui, (u, lid) in enumerate(data):
            if not u["ref"].syllables:
                continue
            seed = zlib.crc32(u["id"].encode())
            if task == "stress":
                ws = stress_words(u, lid, cfg, mode=mode, seed=seed)
                if not ws:
                    continue
                z = stress_batch(net, ws, "cpu").numpy()
                for it, zz in zip(ws, z):
                    n = it[2]
                    p = np.exp(zz[:n] - zz[:n].max())
                    out.append((p / p.sum(), it[3], it[5], lid, n))
            else:
                X, y, orig = pitch_utt(u, cfg, mode=mode, seed=seed)
                if getattr(net, "_local", False):
                    z = net(torch.tensor(X, dtype=torch.float32))[:, 0].numpy()
                else:
                    z = net.forward_utt(torch.tensor(X, dtype=torch.float32))[:, 0].numpy()
                out.append((1 / (1 + np.exp(-z)), y, orig))
    return out


def train(a):
    import torch

    torch.set_num_threads(1)
    cfg = dict(V3 if a.v3 else BASE)
    if a.inputs is not None:
        cfg = {k: k in a.inputs.split(",") for k in BASE}
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = make_net(a.task, cfg).to(dev)
    dev_data = load_dev(a.task)
    n = len(load_train(a.task))
    print(f"{a.name}: {sum(p.numel() for p in net.parameters())} params, {n} train utts, cfg {cfg}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 10, pct_start=0.08)
    pool = get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.task, cfg, a.transplant))
    rng = np.random.default_rng(0)

    def chunks():
        ep = 0
        while True:
            perm = rng.permutation(n)
            for c in range(0, n, 100):
                yield (ep * 100000 + c, perm[c:c + 100])
            ep += 1

    gen = chunks()
    pending = [pool.apply_async(_gen, (next(gen),)) for _ in range(a.workers * 2)]
    step, best, best_state, curve, t0, tl, nb = 0, -1.0, None, [], time.time(), 0.0, 0
    while step < a.steps:
        items = pending.pop(0).get()
        pending.append(pool.apply_async(_gen, (next(gen),)))
        order = rng.permutation(len(items))
        bs = a.bs
        net.train()
        for b in range(0, len(order) - bs + 1, bs):
            it = [items[i] for i in order[b:b + bs]]
            if a.task == "stress":
                z = stress_batch(net, it, dev)
                loss = torch.nn.functional.cross_entropy(z, torch.tensor([x[3] for x in it], device=dev))
            else:
                z, y = pitch_batch(net, it, dev)
                m = torch.tensor(y >= 0, device=dev)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    z[m], torch.tensor(y, dtype=torch.float32, device=dev)[m])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tl += float(loss)
            nb += 1
            if step % a.eval_every == 0 or step >= a.steps:
                net.eval()
                net.cpu()
                acc = dev_acc(a.task, predict(a.task, net, dev_data, cfg))
                net.to(dev)
                curve.append({"step": step, "train_loss": tl / max(nb, 1), "dev_acc": acc, "sec": time.time() - t0})
                print(json.dumps(curve[-1]), flush=True)
                tl, nb = 0.0, 0
                if acc["all"] > best:
                    best = acc["all"]
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                net.train()
            if step >= a.steps:
                break
    pool.terminate()
    torch.save({"state": best_state, "task": a.task, "cfg": cfg, "transplant": a.transplant, "steps": a.steps,
                "curve": curve, "best_dev_acc": best, "langs": list(STRESS_LANGS) if a.task == "stress" else ["ja"]},
               MODELS_DIR / f"{a.name}.pt")
    print("saved", a.name, best)


def dev_acc(task, P):
    if task == "stress":
        ok = np.array([p.argmax() == lab for p, lab, *_ in P])
        lid = np.array([x[3] for x in P])
        return {"all": float(ok.mean()), **{lg: float(ok[lid == k].mean()) for k, lg in enumerate(STRESS_LANGS)
                                            if (lid == k).any()}}
    y = np.concatenate([yy for _, yy, _ in P])
    p = np.concatenate([pp for pp, _, _ in P])
    m = y >= 0
    return {"all": float(((p[m] > 0.5) == (y[m] == 1)).mean())}


# ----------------------------------------------------------------------------- evaluation

def load_any(task, name):
    """New (prosody3) or shipped-recipe checkpoints; returns (net, cfg)."""
    import torch

    ck = torch.load(MODELS_DIR / f"{name}.pt", weights_only=False)
    if isinstance(ck.get("cfg"), dict) and "zemb" in ck["cfg"]:
        net = make_net(task, ck["cfg"])
        net.load_state_dict(ck["state"])
        return net.eval(), ck["cfg"]
    if task == "stress":
        from bench.prosody.stress_train import StressNet

        net = StressNet(**ck["cfg"])
        net.load_state_dict(ck["state"])
        return net.eval(), dict(BASE)
    from bench.prosody.tone_eval import load_model

    net, _ = load_model(name)
    net._local = True
    return net.eval(), dict(BASE)


def evaluate(task, name):
    from bench import metrics as M
    from bench.prosody.tone_eval import platt_fit

    net, cfg = load_any(task, name)
    out = {"model": name, "task": task, "cfg": cfg}
    if task == "stress":
        out["platt"], out["dev"] = {}, {}
        for k, lg in enumerate(STRESS_LANGS):
            data = load_dev(task, lg)
            P = predict(task, net, data, cfg)
            lab, sc = [], []
            for p, y, *_ in P:
                for j in range(len(p)):
                    lab.append(int(j != y))
                    sc.append(p[j])
            lab, sc = np.array(lab), np.array(sc)
            out["platt"][lg] = platt_fit(sc, lab)
            Ps = predict(task, net, data, cfg, mode="swap")
            Pf = predict(task, net, data, cfg, mode="flatten")
            r = {"acc": float(np.mean([p.argmax() == y for p, y, *_ in P])), "shift_auc": M.roc_auc(lab, 1 - sc),
                 "swap_follow": float(np.mean([p.argmax() == y for p, y, *_ in Ps])),
                 "swap_stay_dictionary": float(np.mean([p.argmax() == d for p, y, d, *_ in Ps])),
                 "flatten_still_dictionary": float(np.mean([p.argmax() == d for p, y, d, *_ in Pf])),
                 "chance": float(np.mean([1 / n for *_, n in P])), "n_words": len(P)}
            out["dev"][lg] = r
            print(name, lg, {a: round(b, 4) if isinstance(b, float) else b for a, b in r.items()}, flush=True)
    else:
        data = load_dev(task)
        P = predict(task, net, data, cfg)
        y = np.concatenate([yy for _, yy, _ in P])
        p = np.concatenate([pp for pp, _, _ in P])
        m = y >= 0
        pe = np.where(y[m] == 1, p[m], 1 - p[m])                 # P(expected high/low)
        lab = np.r_[np.zeros(m.sum()), np.ones(m.sum())]
        sc = np.r_[pe, 1 - pe]                                     # flipped label = positives
        out["platt"] = platt_fit(sc, lab)
        Pf = predict(task, net, data, cfg, mode="flatten")
        Ps = predict(task, net, data, cfg, mode="swap")
        yf = np.concatenate([o for _, _, o in Pf])
        pf = np.concatenate([pp for pp, _, _ in Pf])
        ys = np.concatenate([yy for _, yy, _ in Ps])
        yo = np.concatenate([o for _, _, o in Ps])
        ps = np.concatenate([pp for pp, _, _ in Ps])
        sw = (ys != yo) & (yo >= 0)
        mf = yf >= 0
        out["dev"] = {"acc": float(((p[m] > 0.5) == (y[m] == 1)).mean()),
                      "recall_high": float((p[m][y[m] == 1] > 0.5).mean()), "recall_low": float((p[m][y[m] == 0] <= 0.5).mean()),
                      "flip_auc": M.roc_auc(lab.astype(int), 1 - sc),
                      "flatten_still_label": float(((pf[mf] > 0.5) == (yf[mf] == 1)).mean()),
                      "swap_follow": float(((ps[sw] > 0.5) == (ys[sw] == 1)).mean()), "n_swapped": int(sw.sum()),
                      "n_morae": int(m.sum()), "share_high": float((y[m] == 1).mean())}
        print(name, {a: round(b, 4) if isinstance(b, float) else b for a, b in out["dev"].items()}, flush=True)
    save_json(out, f"r8_{name}.json")
    return out


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "train":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("task", choices=["stress", "pitch"])
        ap.add_argument("--name", required=True)
        ap.add_argument("--base", dest="v3", action="store_false", help="old inputs (ablation)")
        ap.add_argument("--transplant", type=float, default=0.4)
        ap.add_argument("--inputs", default=None, help="comma list of reg,lowmel,zemb (default: all three)")
        ap.add_argument("--steps", type=int, default=6000)
        ap.add_argument("--bs", type=int, default=64)
        ap.add_argument("--lr", type=float, default=2e-3)
        ap.add_argument("--eval-every", type=int, default=1000)
        ap.add_argument("--workers", type=int, default=2)
        train(ap.parse_args())
    else:
        for nm in sys.argv[3:]:
            evaluate(sys.argv[2], nm)
    _ = OUT
