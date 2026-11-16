"""Tone model, round 3 ("tone v3"): in-domain data, local-register pitch features, optional
low-frequency log-mel patch. Builds on bench.prosody.tone2 (same SeqToneNet family, contour
transplant, evaluation code).

    .venv/bin/python -m bench.prosody.tone3 train --name t3_reg --reg --fleurs-weight 3 [--lowmel]
    .venv/bin/python -m bench.prosody.tone3 eval t3_reg [...]

Data "v3" = tone2 "new" (aishell3 train, FLEURS zh train first 2500, AISHELL-1, CV zh-TW train)
+ the remaining 742 FLEURS zh TRAIN clips (bench.prosody.fetch_fleurs_zh_rest). FLEURS train
utterances are sampled `--fleurs-weight` times as often as the others (in-domain for the product:
natural connected read speech). FLEURS validation / test are never used for training.

Frame inputs (all on the 10 ms grid, after augmentation and contour transplant):
  base   the 5 pitch / energy features of pronunciada.prosody.features (as v1 / v2)
  --reg  + lf0 minus its local register (voicing-weighted mean over +-0.75 s) and the register
         itself (register_features): pitch relative to the declining local level
  --lowmel + 20 low-frequency log-mel bins (lowmel_features; cache: bench.prosody.lowmel_cache);
         `--spec-dropout` share of training utterances get the patch zeroed
Contour transplant (tone2.transplant) only changes the pitch track: the spectral patch keeps the
original audio's segments, so on transplanted syllables segment identity predicts the WRONG tone,
which penalises reading the tone off the segments.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from bench.prosody import tone2  # noqa: E402
from bench.prosody.common import (MODELS_DIR, agree_map, jitter_windows, load_prepped,  # noqa: E402
                                  scale_windows, stretch_track, to_track, tone_targets)
from pronunciada.prosody.features import (N_IN, N_LOWMEL, N_REG, context_input, frame_features,  # noqa: E402
                                          register_features, syllable_windows)

CFG = {"reg": False, "lowmel": False, "spec_dropout": 0.0, "zemb": False, "zemb_dropout": 0.0}
_LM = None
_ZE = None
N_ZE = 32          # PCA dims of the pooled ZIPA enc1+enc3 syllable vector (+ 1 availability flag)


def n_in(cfg=None):
    c = cfg or CFG
    return (N_IN + (N_REG if c["reg"] else 0) + (N_LOWMEL if c["lowmel"] else 0)
            + (N_ZE + 1 if c.get("zemb") else 0))


def zemb_pca():
    """Fixed PCA (mean, [768, N_ZE] / sqrt(eigenvalue)) fitted on FLEURS zh TRAIN (rest) syllables."""
    from bench.prosody.zipa_emb_cache import EMB

    p = EMB / "pca.npz"
    if not p.exists():
        d = np.load(EMB / "fleurs_zh_rest-zh-train.npz")
        X = np.concatenate([d[k].astype(np.float32) for k in d.files])
        mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        W = Vt[:N_ZE].T / (S[:N_ZE] / np.sqrt(len(X)))
        np.savez(p, mu=mu, W=W.astype(np.float32))
    d = np.load(p)
    return d["mu"], d["W"]


def _zemb_frames(u, wins, T0):
    global _ZE
    if _ZE is None:
        from bench.prosody.zipa_emb_cache import ZipaEmb

        _ZE = (ZipaEmb(), *zemb_pca())
    src, mu, W = _ZE
    E, flag = src(u)
    out = np.zeros((T0, N_ZE + 1), np.float32)
    if flag:
        Z = np.clip((E - mu) @ W, -5, 5)
        for si, w in enumerate(wins):
            out[w.start:w.end, :N_ZE] = Z[si]
            out[w.start:w.end, N_ZE] = 1.0
    return out


def _lowmel(u):
    global _LM
    if _LM is None:
        from bench.prosody.lowmel_cache import LowMel

        _LM = LowMel()
    return _LM(u)


def extend_feats(feats, cfg, lm=None):
    """Append the v3 extra frame columns (shared with the runtime, pronunciada.prosody)."""
    cols = [feats]
    if cfg["reg"]:
        cols.append(register_features(feats))
    if cfg["lowmel"]:
        cols.append(lm if lm is not None else np.zeros((len(feats), N_LOWMEL), np.float32))
    return np.concatenate(cols, 1) if len(cols) > 1 else feats


def utt_example(u, rng=None, aug=False, tp_p=0.0, tp_k=(1, 3), agree=None, train_only=False,
                fixed_tp=None, jitter=0):
    """As tone2.utt_example, with the v3 frame inputs of CFG."""
    d = u["track"]
    if aug and "noisy" in u and rng.random() < 0.3:
        d = u["noisy"]
    tr = to_track(d)
    T0 = len(tr.f0)
    wins = syllable_windows(u["ref"], u["spans"], T0)
    labs, mask, accs = tone_targets(u, agree)
    if not train_only:
        mask = labs >= 0
    tps = []
    if fixed_tp is not None or (tp_p > 0 and rng.random() < tp_p):
        k = 1 if fixed_tp is not None else int(rng.integers(tp_k[0], tp_k[1] + 1))
        r = fixed_tp if fixed_tp is not None else rng
        tr, labs, tps = tone2.transplant(tr, wins, labs, accs, r, k)
        for i, _, _ in tps:
            mask[i] = True
    ze = None
    if CFG.get("zemb"):
        ze = _zemb_frames(u, wins, T0)
        if aug and rng.random() < CFG.get("zemb_dropout", 0.0):
            ze = np.zeros_like(ze)
    lm = None
    if CFG["lowmel"]:
        lm = _lowmel(u)
        if len(lm) != T0:
            lm = lm[:T0] if len(lm) > T0 else np.pad(lm, ((0, T0 - len(lm)), (0, 0)))
        if aug and rng.random() < CFG["spec_dropout"]:
            lm = np.zeros_like(lm)
    if aug:
        r = float(rng.uniform(0.85, 1.15))
        tr = stretch_track(tr, r)
        T1 = len(tr.f0)
        wins = scale_windows(wins, T1 / T0, T1)
        ix = np.clip(np.round(np.linspace(0, T0 - 1, T1)).astype(int), 0, T0 - 1)
        if lm is not None:
            lm = lm[ix]
        if ze is not None:
            ze = ze[ix]
    feats = frame_features(tr)
    if aug:
        s = float(rng.uniform(0.7, 1.3))
        feats[:, 0] *= s
        feats[:, 3] *= s
        feats[:, 0] += rng.normal(0, 0.05, len(feats)).astype(np.float32)
        feats[:, 2] += float(rng.normal(0, 0.2))
        if lm is not None:
            lm = lm + rng.normal(0, 0.05, lm.shape).astype(np.float32)
        wins = jitter_windows(wins, rng, 2, len(feats))
    elif jitter:
        wins = jitter_windows(wins, rng, jitter, len(feats))
    feats = extend_feats(feats, CFG, lm)
    if ze is not None:
        feats = np.concatenate([feats, ze], 1)
    X = np.stack([context_input(feats, wins, i) for i in range(len(wins))])
    return X, labs, mask, tps


# patch tone2 so its evaluation code (examples, transplant_test, monotone_test) uses v3 inputs
tone2.utt_example = utt_example


def load_data(which: str):
    if which == "fleurs":         # in-domain fine-tuning: FLEURS zh TRAIN only (3242 clips)
        return load_prepped("fleurs", "zh", "train") + load_prepped("fleurs_zh_rest", "zh", "train")
    utts = tone2.load_data("new")
    if which == "v3":
        utts += load_prepped("fleurs_zh_rest", "zh", "train")
    return utts


_W = {}


def _init(which, tp_p, cfg):
    CFG.update(cfg)
    _W["tp_k"] = tuple(cfg.get("tp_k", (1, 3)))
    _W["utts"] = load_data(which)
    _W["agree"] = agree_map(_W["utts"])
    _W["tp"] = tp_p


def _gen(args):
    seed, idx = args
    rng = np.random.default_rng(seed)
    out = []
    for k in idx:
        u = _W["utts"][k]
        if not u["ref"].syllables or len(u["ref"].syllables) > tone2.MAX_SYL:
            continue
        X, y, m, _ = utt_example(u, rng, aug=True, tp_p=_W["tp"], tp_k=_W["tp_k"], agree=_W["agree"],
                                 train_only=True)
        if m.any():
            out.append((X.astype(np.float16), y, m))
    return out


def make_net(cfg):
    from bench.prosody.model import SeqToneNet

    return SeqToneNet(n_in=n_in(cfg), seq_hid=cfg.get("seq_hid", 64))


_load2_v2 = tone2.load2
DEV_SETS = [("aishell3", "test"), ("zhtw", "test")]


def load3(name):
    import torch

    ck = torch.load(MODELS_DIR / f"{name}.pt", weights_only=False)
    if "cfg" not in ck:
        CFG.update(reg=False, lowmel=False, zemb=False)
        return _load2_v2(name)
    CFG.update(zemb=False, zemb_dropout=0.0)
    CFG.update(ck["cfg"])
    net = make_net(ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, "seq", ck


tone2.load2 = load3


def train(a):
    import torch

    torch.set_num_threads(1)
    cfg = {"reg": a.reg, "lowmel": a.lowmel, "spec_dropout": a.spec_dropout, "seq_hid": a.seq_hid,
           "zemb": a.zemb, "zemb_dropout": a.zemb_dropout,
           "tp_k": [int(x) for x in a.tp_k.split("-")]}
    CFG.update(cfg)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = make_net(cfg)
    if a.init:                    # fine-tune from a trained v3 checkpoint (same inputs)
        ck0 = torch.load(MODELS_DIR / f"{a.init}.pt", weights_only=False)
        assert ck0["cfg"]["reg"] == cfg["reg"] and ck0["cfg"]["lowmel"] == cfg["lowmel"]
        st = dict(ck0["state"])
        w0 = st["enc.conv1.weight"]                    # warm start with extra input channels
        w = net.state_dict()["enc.conv1.weight"].clone()
        if w0.shape != w.shape:
            w.zero_()
            w[:, :w0.shape[1]] = w0
            st["enc.conv1.weight"] = w
        net.load_state_dict(st)
    net = net.to(dev)
    dev_u = load_prepped("fleurs_val", "zh", "dev")
    dev_ex = [utt_example(u)[:3] for u in dev_u if u["ref"].syllables]
    utts = load_data(a.data)
    is_fl = np.array([u["id"].startswith("fleurs-") for u in utts])
    base_idx = np.arange(len(utts))
    reps = [base_idx] + [base_idx[is_fl]] * (a.fleurs_weight - 1)
    pool_idx = np.concatenate(reps)
    del utts
    n = len(pool_idx)
    print(f"{a.name}: {sum(p.numel() for p in net.parameters())} params, {len(base_idx)} train utts "
          f"({is_fl.sum()} FLEURS x{a.fleurs_weight}), cfg {cfg}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 10, pct_start=0.08)
    pool = get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.data, a.transplant, cfg))
    rng = np.random.default_rng(a.seed)
    CH = 300

    def chunks():
        ep = 0
        while True:
            perm = pool_idx[rng.permutation(n)]
            for c in range(0, n, CH):
                yield (a.seed * 7919 + ep * 100000 + c, perm[c:c + CH])
            ep += 1

    gen = chunks()
    pending = [pool.apply_async(_gen, (next(gen),)) for _ in range(a.workers * 2)]
    step, best, best_state, curve, t0, tl, nb, seen = 0, -1.0, None, [], time.time(), 0.0, 0, 0
    while step < a.steps:
        items = pending.pop(0).get()
        pending.append(pool.apply_async(_gen, (next(gen),)))
        order = rng.permutation(len(items))
        batches, cur, cnt = [], [], 0
        for i in order:
            cur.append(items[i])
            cnt += int(items[i][2].sum())
            if cnt >= a.bs:
                batches.append(cur)
                cur, cnt = [], 0
        net.train()
        for b in batches:
            z, y = tone2.run_batch(net, "seq", b, dev)
            loss = torch.nn.functional.cross_entropy(z, y, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += len(y)
            tl += float(loss)
            nb += 1
            if step % a.eval_every == 0 or step >= a.steps:
                net.eval()
                P = tone2.predict_utts(net.cpu(), "seq", [x for x, _, _ in dev_ex])
                net.to(dev)
                y_all = np.concatenate([yy for _, yy, _ in dev_ex])
                pred = np.concatenate(P).argmax(1)
                ok = y_all >= 0
                acc = float((pred[ok] == y_all[ok]).mean())
                curve.append({"step": step, "train_loss": tl / max(nb, 1), "dev_acc": acc,
                              "syllables_seen": seen, "sec": time.time() - t0})
                print(json.dumps(curve[-1]), flush=True)
                tl, nb = 0.0, 0
                if acc > best:
                    best = acc
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                net.train()
            if step >= a.steps:
                break
    pool.terminate()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "model": "seq", "cfg": cfg, "data": a.data, "transplant": a.transplant,
                "fleurs_weight": a.fleurs_weight, "init": a.init, "lr": a.lr, "steps": a.steps, "curve": curve, "best_dev_acc": best,
                "n_train_utts": int(len(base_idx))}, MODELS_DIR / f"{a.name}.pt")
    print("saved", a.name, best)


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "train":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--name", required=True)
        ap.add_argument("--data", default="v3")
        ap.add_argument("--reg", action="store_true")
        ap.add_argument("--lowmel", action="store_true")
        ap.add_argument("--spec-dropout", type=float, default=0.3)
        ap.add_argument("--fleurs-weight", type=int, default=3)
        ap.add_argument("--transplant", type=float, default=0.4)
        ap.add_argument("--tp-k", default="1-3", help="transplanted syllables per utterance, min-max")
        ap.add_argument("--seq-hid", type=int, default=64)
        ap.add_argument("--steps", type=int, default=14000)
        ap.add_argument("--bs", type=int, default=256)
        ap.add_argument("--lr", type=float, default=2e-3)
        ap.add_argument("--eval-every", type=int, default=1000)
        ap.add_argument("--workers", type=int, default=2)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
        ap.add_argument("--zemb", action="store_true", help="B: pooled ZIPA enc1+enc3 per syllable")
        ap.add_argument("--zemb-dropout", type=float, default=0.3)
        train(ap.parse_args())
    else:
        # selection sets only: FLEURS validation (inside evaluate) + aishell3 test + zhtw.
        # FLEURS zh TEST is the held-out product test: never used here.
        for nm in sys.argv[2:]:
            tone2.evaluate(nm, sets=DEV_SETS, quick=True)
