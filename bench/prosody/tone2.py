"""Tone model, round 2: more data, utterance-level sequence model, contour-transplant augmentation.

    .venv/bin/python -m bench.prosody.tone2 train --name t2_new_seq_tp --data new --model seq --transplant 0.4
    .venv/bin/python -m bench.prosody.tone2 eval t2_new_seq_tp [...]

Data: old = aishell3 train (agree/sandhi syllables) + FLEURS zh train; new = old + AISHELL-1
(300 train speakers) + Common Voice zh-TW train (183 speakers), front-end labels.
Models: local = stride-2 ProsodyNet (one syllable + neighbours); seq = SeqToneNet (same encoder +
BiGRU across the syllables of the utterance).
Contour transplant (feature space): for a share of utterances, the pitch + voicing track of k = 1-3
syllables is replaced by the time-warped track of a donor syllable of the same utterance with a
tone outside the target's acceptable set and a duration ratio within 0.6-1.6, shifted so that the
donor's local pitch level (median log-F0 over donor +-1 syllable) matches the target's; the label
becomes the donor's tone. Training budget: a fixed number of optimiser steps for every variant.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from bench import metrics as M  # noqa: E402
from bench.prosody.common import (MODELS_DIR, OUT, agree_map, jitter_windows, load_prepped,  # noqa: E402
                                  save_json, scale_windows, stretch_track, to_track, tone_targets)
from pronunciada.prosody.f0 import F0Track  # noqa: E402
from pronunciada.prosody.features import context_input, frame_features, syllable_windows  # noqa: E402

MAX_SYL = 96


# ----------------------------------------------------------------------------- transplant

def _local_level(lf, vo, wins, i):
    a = wins[max(0, i - 1)].start
    b = wins[min(len(wins) - 1, i + 1)].end
    v = vo[a:b]
    return float(np.median(lf[a:b][v])) if v.sum() >= 3 else None


def transplant(tr: F0Track, wins, labs, accs, rng, k: int, force_donor_diff=True):
    """Returns (new track, new labels, [(target, donor, donor_tone)])."""
    f0 = tr.f0.copy()
    vo = tr.voiced.copy()
    vp = tr.voicing.copy()
    lf = np.log(np.maximum(tr.f0, 1.0))
    labs = labs.copy()
    n = len(wins)
    order = rng.permutation(n)
    done, used = [], set()
    for i in order:
        if len(done) >= k:
            break
        if labs[i] < 0 or i in used:
            continue
        di = wins[i].end - wins[i].start
        cand = [j for j in range(n) if j != i and labs[j] >= 0 and (labs[j] + 1) not in accs[i]
                and 0.6 <= (wins[j].end - wins[j].start) / max(di, 1) <= 1.6
                and tr.voiced[wins[j].start:wins[j].end].mean() >= 0.3]
        if not cand:
            continue
        j = int(cand[rng.integers(len(cand))])
        lt, ld = _local_level(lf, tr.voiced, wins, i), _local_level(lf, tr.voiced, wins, j)
        if lt is None or ld is None:
            continue
        s, e = wins[i].start, wins[i].end
        ds, de = wins[j].start, wins[j].end
        src = np.linspace(ds, de - 1, e - s)
        near = np.clip(np.round(src).astype(int), ds, de - 1)
        dv = tr.voiced[near]
        dl = np.interp(src, np.arange(len(lf)), np.where(tr.voiced, lf, np.nan))
        dl = np.where(np.isnan(dl), lf[near], dl)
        f0[s:e] = np.where(dv, np.exp(dl + (lt - ld)), 0.0)
        vo[s:e] = dv
        vp[s:e] = np.interp(src, np.arange(len(vp)), tr.voicing)
        labs[i] = labs[j]
        used.update((i, j))
        done.append((int(i), int(j), int(labs[j]) + 1))
    return F0Track(f0.astype(np.float32), vo, vp.astype(np.float32), tr.energy_db), labs, done


# ----------------------------------------------------------------------------- inputs

def utt_example(u, rng=None, aug=False, tp_p=0.0, tp_k=(1, 3), agree=None, train_only=False,
                fixed_tp=None, jitter=0):
    """(X [S, L, C], y [S] (0-4, -1 = no label), loss mask [S], transplant list)."""
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
        tr, labs, tps = transplant(tr, wins, labs, accs, r, k)
        for i, _, _ in tps:
            mask[i] = True
    if aug:
        r = float(rng.uniform(0.85, 1.15))
        tr = stretch_track(tr, r)
        wins = scale_windows(wins, len(tr.f0) / T0, len(tr.f0))
    feats = frame_features(tr)
    if aug:
        s = float(rng.uniform(0.7, 1.3))
        feats[:, 0] *= s
        feats[:, 3] *= s
        feats[:, 0] += rng.normal(0, 0.05, len(feats)).astype(np.float32)
        feats[:, 2] += float(rng.normal(0, 0.2))
        wins = jitter_windows(wins, rng, 2, len(feats))
    elif jitter:
        wins = jitter_windows(wins, rng, jitter, len(feats))
    X = np.stack([context_input(feats, wins, i) for i in range(len(wins))])
    return X, labs, mask, tps


# ----------------------------------------------------------------------------- data

def load_data(which: str):
    utts = load_prepped("aishell3", "zh", "train") + load_prepped("fleurs", "zh", "train")
    if which == "new":
        utts += load_prepped("aishell1", "zh", "train") + load_prepped("cvtw", "zh", "train")
    return utts


_W = {}


def _init(which, tp_p):
    _W["utts"] = load_data(which)
    _W["agree"] = agree_map(_W["utts"])
    _W["tp"] = tp_p


def _gen(args):
    seed, idx = args
    rng = np.random.default_rng(seed)
    out = []
    for k in idx:
        u = _W["utts"][k]
        if not u["ref"].syllables or len(u["ref"].syllables) > MAX_SYL:
            continue
        X, y, m, _ = utt_example(u, rng, aug=True, tp_p=_W["tp"], agree=_W["agree"], train_only=True)
        if m.any():
            out.append((X.astype(np.float16), y, m))
    return out


# ----------------------------------------------------------------------------- model helpers

def make_net(kind):
    from bench.prosody.model import ProsodyNet, SeqToneNet

    return SeqToneNet() if kind == "seq" else ProsodyNet(ch=64, hid=48, stride=2)


def run_batch(net, kind, items, dev):
    """items: list of (X, y, m). Returns (logits [N, 5] over the masked syllables, targets)."""
    import torch

    if kind == "local":
        X = np.concatenate([x[m] for x, _, m in items])
        y = np.concatenate([yy[m] for _, yy, m in items])
        return net(torch.tensor(X, dtype=torch.float32, device=dev)), torch.tensor(y, device=dev)
    B = len(items)
    S = max(len(x) for x, _, _ in items)
    X = np.concatenate([x for x, _, _ in items])
    index = np.concatenate([b * S + np.arange(len(x)) for b, (x, _, _) in enumerate(items)])
    lengths = torch.tensor([len(x) for x, _, _ in items])
    z = net(torch.tensor(X, dtype=torch.float32, device=dev), torch.tensor(index, device=dev), lengths, B, S)
    m = np.concatenate([mm for _, _, mm in items])
    y = np.concatenate([yy for _, yy, _ in items])
    sel = torch.tensor(np.flatnonzero(m), device=dev)
    return z[sel], torch.tensor(y[m], device=dev)


def predict_utts(net, kind, examples):
    """examples: list of X [S, L, C] -> list of P [S, 5] (logits, untempered)."""
    import torch

    out = []
    with torch.no_grad():
        if kind == "local":
            flat = np.concatenate(examples)
            t = torch.tensor(flat, dtype=torch.float32)
            z = torch.cat([net(t[i:i + 2048]) for i in range(0, len(t), 2048)]).numpy()
            k = 0
            for x in examples:
                out.append(z[k:k + len(x)])
                k += len(x)
        else:
            for x in examples:
                out.append(net.forward_utt(torch.tensor(x, dtype=torch.float32)).numpy())
    return out


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ----------------------------------------------------------------------------- train

def train(a):
    import torch

    torch.set_num_threads(1)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = make_net(a.model).to(dev)
    dev_u = load_prepped("fleurs_val", "zh", "dev")
    dev_ex = [utt_example(u)[:3] for u in dev_u if u["ref"].syllables]
    n = len(load_data(a.data))
    print(f"{a.name}: {sum(p.numel() for p in net.parameters())} params, {n} train utts", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 10, pct_start=0.08)
    pool = get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.data, a.transplant))
    rng = np.random.default_rng(0)
    CH = 300

    def chunks():
        ep = 0
        while True:
            perm = rng.permutation(n)
            for c in range(0, n, CH):
                yield (ep * 100000 + c, perm[c:c + CH])
            ep += 1

    gen = chunks()
    pending = [pool.apply_async(_gen, (next(gen),)) for _ in range(a.workers * 2)]
    step, best, best_state, curve, t0, tl, nb = 0, -1.0, None, [], time.time(), 0.0, 0
    seen = 0
    while step < a.steps:
        items = pending.pop(0).get()
        pending.append(pool.apply_async(_gen, (next(gen),)))
        order = rng.permutation(len(items))
        if a.model == "local":
            X = np.concatenate([items[i][0][items[i][2]] for i in order])
            Y = np.concatenate([items[i][1][items[i][2]] for i in order])
            p2 = rng.permutation(len(Y))
            batches = [[(X[p2[b:b + a.bs]], Y[p2[b:b + a.bs]], np.ones(len(p2[b:b + a.bs]), bool))]
                       for b in range(0, len(Y) - a.bs + 1, a.bs)]
        else:
            batches, cur, cnt = [], [], 0
            for i in order:
                cur.append(items[i])
                cnt += int(items[i][2].sum())
                if cnt >= a.bs:
                    batches.append(cur)
                    cur, cnt = [], 0
        net.train()
        for b in batches:
            z, y = run_batch(net, a.model, b, dev)
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
                P = predict_utts(net.cpu(), a.model, [x for x, _, _ in dev_ex])
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
    torch.save({"state": best_state, "model": a.model, "data": a.data, "transplant": a.transplant,
                "steps": a.steps, "curve": curve, "best_dev_acc": best, "n_train_utts": n},
               MODELS_DIR / f"{a.name}.pt")
    print("saved", a.name, best)


# ----------------------------------------------------------------------------- evaluation

def load2(name):
    import torch

    ck = torch.load(MODELS_DIR / f"{name}.pt", weights_only=False)
    if "model" not in ck:                      # round-1 checkpoint (local ProsodyNet)
        from bench.prosody.tone_eval import load_model

        net, ck1 = load_model(name)
        return net, "local", ck1
    net = make_net(ck["model"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["model"], ck


def examples(utts, **kw):
    ex, keep = [], []
    for ui, u in enumerate(utts):
        if u["ref"].syllables:
            ex.append(utt_example(u, **kw))
            keep.append(ui)
    return ex, keep


def fit_temperature(z, y):
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.3), np.log(5), 120)):
        p = softmax(z, T)
        best = min(best, (-np.log(p[np.arange(len(y)), y] + 1e-12).mean(), T))
    return best[1]


def acc_sets(u):
    return [{s.tone, *s.tone_alts} for s in u["ref"].syllables]


def wrong_tone_rows(P, utts, keep):
    from bench.prosody.tone_eval import detection_pairs

    meta = [(ui, si) for ui in keep for si in range(len(utts[ui]["ref"].syllables))
            if utts[ui]["ref"].syllables[si].tone]
    Pf = np.concatenate([p[[si for si in range(len(utts[ui]["ref"].syllables)) if utts[ui]["ref"].syllables[si].tone]]
                         for p, ui in zip(P, keep)])
    return detection_pairs(Pf, utts, meta), Pf, meta


def hit_at_fa(y, s, fa=0.05):
    thr = np.quantile(s[y == 0], 1 - fa)
    return float((s[y == 1] > thr).mean())


def transplant_test(net, kind, T, ab, utts, seed=7):
    from bench.prosody.tone_eval import platt_apply

    ex, keep = [], []
    for ui, u in enumerate(utts):
        if not u["ref"].syllables:
            continue
        X, y, m, tps = utt_example(u, fixed_tp=np.random.default_rng(seed + ui))
        if tps:
            ex.append(X)
            keep.append((ui, tps[0]))
    P = [softmax(z, T) for z in predict_utts(net, kind, ex)]
    lab, sc, rec = [], [], []
    for p, (ui, (i, j, dt)) in zip(P, keep):
        acc = acc_sets(utts[ui])
        for si, s in enumerate(utts[ui]["ref"].syllables):
            if not s.tone:
                continue
            pa = sum(p[si, t - 1] for t in acc[si])
            lab.append(1 if si == i else 0)
            sc.append(pa)
            if si == i:
                rec.append(int(p[si].argmax()) + 1 == dt)
    lab, sc = np.array(lab), np.array(sc)
    e, _ = M.eer(lab, 1 - sc)
    return {"n_utts": len(keep), "n_pos": int(lab.sum()), "n_neg": int((lab == 0).sum()),
            "auc": M.roc_auc(lab, 1 - sc), "eer": e, "hit_at_5fa": hit_at_fa(lab, 1 - sc),
            "donor_tone_acc": float(np.mean(rec)), "ece_cal": M.ece(1 - lab, platt_apply(sc, ab))}


def monotone_test(net, kind, T):
    """Flat-pitch resynthesis (bench.prosody.leak_test make), syllables with label 2/3/4."""
    res = {}
    P = {}
    for cond in ("orig", "mono"):
        utts = load_prepped(f"aishell3{cond}", "zh", "test")
        ex, keep = examples(utts)
        P[cond] = ([softmax(z, T) for z in predict_utts(net, kind, [x for x, _, _, _ in ex])], keep, utts)
    (Po, keep, uo), (Pm, _, _) = P["orig"], P["mono"]
    pa_o, pa_m, still = [], [], []
    for po, pm, ui in zip(Po, Pm, keep):
        acc = acc_sets(uo[ui])
        for si, s in enumerate(uo[ui]["ref"].syllables):
            if s.tone in (2, 3, 4):
                pa_o.append(sum(po[si, t - 1] for t in acc[si]))
                pa_m.append(sum(pm[si, t - 1] for t in acc[si]))
                still.append(int(pm[si].argmax()) + 1 == s.tone)
    lab = np.r_[np.zeros(len(pa_o)), np.ones(len(pa_m))]
    res = {"n": len(pa_o), "still_label_mono": float(np.mean(still)),
           "mono_auc": M.roc_auc(lab, 1 - np.r_[pa_o, pa_m])}
    return res


SETS = [("aishell3", "test"), ("fleurs", "test"), ("zhtw", "test")]


def evaluate(name, sets=SETS, quick=False):
    from bench.prosody.tone_eval import det_metrics, platt_fit

    net, kind, ck = load2(name)
    dev_u = load_prepped("fleurs_val", "zh", "dev")
    ex, keep = examples(dev_u)
    Z = predict_utts(net, kind, [x for x, _, _, _ in ex])
    y = np.concatenate([yy for _, yy, _, _ in ex])
    zf = np.concatenate(Z)
    T = fit_temperature(zf[y >= 0], y[y >= 0])
    P = [softmax(z, T) for z in Z]
    rows_d, _, _ = wrong_tone_rows(P, dev_u, keep)
    ab = platt_fit(rows_d[:, 1], rows_d[:, 0])
    out = {"model": name, "kind": kind, "temperature": T, "platt": ab, "curve": ck.get("curve"),
           "data": ck.get("data", "old"), "transplant": ck.get("transplant", 0.0)}
    out["dev"] = {"acc": float((zf.argmax(1) == y)[y >= 0].mean()), **det_metrics(rows_d, ab),
                  "transplant": transplant_test(net, kind, T, ab, dev_u)}
    for ds, split in sets:
        utts = load_prepped(ds, "zh", split)
        ex, keep = examples(utts)
        P = [softmax(z, T) for z in predict_utts(net, kind, [x for x, _, _, _ in ex])]
        rows, Pf, meta = wrong_tone_rows(P, utts, keep)
        yy = np.array([utts[u]["ref"].syllables[s].tone - 1 for u, s in meta])
        pred = Pf.argmax(1)
        r = {"acc": float((pred == yy).mean()), "detection": det_metrics(rows, ab),
             "recall": {t + 1: float((pred[yy == t] == t).mean()) for t in range(5)}}
        sel_neg = (rows[:, 0] == 0) & np.isin(rows[:, 2], [2, 3])
        sel_pos = (rows[:, 0] == 1) & (((rows[:, 2] == 2) & (rows[:, 3] == 3)) | ((rows[:, 2] == 3) & (rows[:, 3] == 2)))
        s = sel_neg | sel_pos
        r["auc_2_3"] = M.roc_auc(rows[s, 0], 1 - rows[s, 1])
        r["hit_at_5fa_wrong_tone"] = hit_at_fa(rows[:, 0], 1 - rows[:, 1])
        r["transplant"] = transplant_test(net, kind, T, ab, utts)
        if ds == "aishell3":
            ext = np.array([(utts[u]["ext"][s]["tone"] if utts[u].get("ext") else None) or 0 for u, s in meta])
            r["acc_vs_aishell_labels"] = float((pred[ext > 0] + 1 == ext[ext > 0]).mean())
        if not quick:
            exj, keepj = examples(utts, rng=np.random.default_rng(0), jitter=4)
            Pj = [softmax(z, T) for z in predict_utts(net, kind, [x for x, _, _, _ in exj])]
            rj, Pjf, _ = wrong_tone_rows(Pj, utts, keepj)
            r["acc_jitter40ms"] = float((Pjf.argmax(1) == yy).mean())
            cm = np.zeros((5, 5), int)
            np.add.at(cm, (yy, pred), 1)
            r["confusion"] = cm.tolist()
        out[f"{ds}-{split}"] = r
        print(name, ds, {"acc": round(r["acc"], 4), "auc": round(r["detection"]["auc"], 4),
                         "eer": round(r["detection"]["eer"], 4), "auc23": round(r["auc_2_3"], 4),
                         "tp_auc": round(r["transplant"]["auc"], 4), "tp_hit5": round(r["transplant"]["hit_at_5fa"], 4),
                         "donor_acc": round(r["transplant"]["donor_tone_acc"], 4)}, flush=True)
    out["monotone"] = monotone_test(net, kind, T)
    print(name, "monotone", out["monotone"], flush=True)
    save_json(out, f"{name}.json")
    return out


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "train":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--name", required=True)
        ap.add_argument("--data", default="old")
        ap.add_argument("--model", default="local")
        ap.add_argument("--transplant", type=float, default=0.0)
        ap.add_argument("--steps", type=int, default=10000)
        ap.add_argument("--bs", type=int, default=256)
        ap.add_argument("--lr", type=float, default=2e-3)
        ap.add_argument("--eval-every", type=int, default=1000)
        ap.add_argument("--workers", type=int, default=3)
        train(ap.parse_args())
    else:
        for n in sys.argv[2:]:
            evaluate(n)
