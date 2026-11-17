"""Evaluate a trained tone model on held-out sets.

    .venv/bin/python -m bench.prosody.tone_eval tone_ctx [tone_mlp ...]

Sets: aishell3 test (20 unseen speakers), fleurs zh test, zhtw (Taiwan, Common Voice).
Dev (calibration only): FLEURS validation zh.
Wrong-tone detection: for every syllable, negatives = the reference tone (score P(tone or any
tone_alt)), positives = the reference tone replaced by each tone in 1-4 that is neither the tone
nor an alt (bench.corrupt `tone` replacement set; score P(replacement or its alts = none)).
Error score = 1 - P(acceptable). Calibration: temperature on dev logits, then Platt on the dev
detection pairs; ECE (10 equal-width bins) on the calibrated P(acceptable).
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")

from bench import metrics as M  # noqa: E402
from bench.prosody.common import (MODELS_DIR, agree_map, build_arrays, load_prepped,  # noqa: E402
                                  save_json)

SETS = [("aishell3", "zh", "test"), ("fleurs", "zh", "test"), ("zhtw", "zh", "test")]


def load_model(name):
    import torch

    from bench.prosody.model import ContourMLP, ProsodyNet

    from bench.prosody.common import n_extra, set_opts

    ck = torch.load(MODELS_DIR / f"{name}.pt", weights_only=False)
    set_opts(**ck.get("opts", {"extra": "", "target": "syllable"}))
    n_out = 5
    net = ProsodyNet(n_in=10 + n_extra(), ch=ck["ch"], hid=ck["hid"], n_out=n_out, arch=ck.get("arch", "gru"),
                                      stride=ck.get("stride", 1)) if ck["kind"] == "ctx" else ContourMLP(n_out=n_out)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


def logits(net, X):
    import torch

    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32)
        return torch.cat([net(t[i:i + 2048]) for i in range(0, len(t), 2048)]).numpy()


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(z, y):
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.3), np.log(5), 120)):
        p = softmax(z, T)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        best = min(best, (nll, T))
    return best[1]


def detection_pairs(P, utts, meta):
    """Rows (error_label, p_acceptable, true_tone, ref_tone_scored, utt, syl)."""
    rows = []
    for k, (ui, si) in enumerate(meta):
        s = utts[ui]["ref"].syllables[si]
        acc = {s.tone, *s.tone_alts}
        rows.append((0, sum(P[k, t - 1] for t in acc), s.tone, s.tone, ui, si))
        for t in (1, 2, 3, 4):
            if t not in acc:
                rows.append((1, P[k, t - 1], s.tone, t, ui, si))
    return np.array(rows, dtype=float)


def platt_fit(p, y):
    from sklearn.linear_model import LogisticRegression

    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))[:, None]
    lr = LogisticRegression(C=100).fit(x, 1 - y)          # predicts P(acceptable)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(p, ab):
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return 1 / (1 + np.exp(-(ab[0] * x + ab[1])))


def position(ref, si):
    s = ref.syllables[si]
    w = ref.words[s.word]
    n = w.syllables[1] - w.syllables[0]
    k = si - w.syllables[0]
    if n == 1:
        return "mono"
    return "initial" if k == 0 else ("final" if k == n - 1 else "medial")


def speaking_rate(u):
    from pronunciada.prosody.features import syllable_windows

    w = syllable_windows(u["ref"], u["spans"], len(u["track"]["f0"]))
    span = (w[-1].end - w[0].start) / 100
    return len(w) / max(span, 0.1)


def det_metrics(rows, ab):
    y, p = rows[:, 0], rows[:, 1]
    pc = platt_apply(p, ab)
    e, _ = M.eer(y, 1 - p)
    return {"n_neg": int((y == 0).sum()), "n_pos": int((y == 1).sum()), "auc": M.roc_auc(y, 1 - p),
            "eer": e, "ece_calibrated": M.ece(1 - y, pc), "ece_raw": M.ece(1 - y, p)}


def evaluate(name: str) -> dict:
    net, ck = load_model(name)
    kind = ck["kind"]
    dev_u = load_prepped("fleurs_val", "zh", "dev")
    Xd, yd, md = build_arrays(dev_u, kind)
    zd = logits(net, Xd)
    T = fit_temperature(zd, yd)
    Pd = softmax(zd, T)
    rows_d = detection_pairs(Pd, dev_u, md)
    ab = platt_fit(rows_d[:, 1], rows_d[:, 0])
    out = {"model": name, "kind": kind, "temperature": T, "platt": ab,
           "dev": {"acc": float((Pd.argmax(1) == yd).mean()), **det_metrics(rows_d, ab)},
           "curve": ck["curve"]}
    rng = np.random.default_rng(0)
    for ds, lang, split in SETS:
        utts = load_prepped(ds, lang, split)
        X, y, meta = build_arrays(utts, kind)
        P = softmax(logits(net, X), T)
        pred = P.argmax(1)
        r = {"n_syllables": int(len(y)), "n_utts": len(utts), "acc": float((pred == y).mean())}
        r["acc_ci95"] = M.bootstrap_ci(lambda a, b: float((a == b).mean()), pred, y,
                                       clusters=[m[0] for m in meta], n_boot=300)
        r["recall"] = {t + 1: float((pred[y == t] == t).mean()) for t in range(5) if (y == t).any()}
        r["support"] = {t + 1: int((y == t).sum()) for t in range(5)}
        cm = np.zeros((5, 5), int)
        np.add.at(cm, (y, pred), 1)
        r["confusion"] = cm.tolist()
        # acceptable-tone accuracy (argmax in tone or alts)
        r["acc_with_alts"] = float(np.mean([pred[k] + 1 in {utts[u]["ref"].syllables[s].tone,
                                                            *utts[u]["ref"].syllables[s].tone_alts}
                                            for k, (u, s) in enumerate(meta)]))
        rows = detection_pairs(P, utts, meta)
        r["detection"] = det_metrics(rows, ab)
        pairs = {}
        for a_, b_ in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (5, 1), (5, 2), (5, 3), (5, 4)]:
            sel_neg = (rows[:, 0] == 0) & np.isin(rows[:, 2], [a_, b_])
            sel_pos = (rows[:, 0] == 1) & (((rows[:, 2] == a_) & (rows[:, 3] == b_)) |
                                           ((rows[:, 2] == b_) & (rows[:, 3] == a_)))
            sel = sel_neg | sel_pos
            if a_ == 5:   # neutral-tone syllables scored against a full tone (only one direction)
                sel = ((rows[:, 0] == 0) & (rows[:, 2] == 5)) | ((rows[:, 0] == 1) & (rows[:, 2] == 5) & (rows[:, 3] == b_))
            if (rows[sel, 0] == 1).any() and (rows[sel, 0] == 0).any():
                pairs[f"{a_}-{b_}"] = {"auc": M.roc_auc(rows[sel, 0], 1 - rows[sel, 1]),
                                       "eer": M.eer(rows[sel, 0], 1 - rows[sel, 1])[0],
                                       "n_pos": int((rows[sel, 0] == 1).sum())}
        r["pairs"] = pairs
        pos = np.array([position(utts[int(u)]["ref"], int(s)) for u, s in rows[:, 4:6]])
        last = np.array([int(s) == len(utts[int(u)]["ref"].syllables) - 1 for u, s in rows[:, 4:6]])
        r["by_position"] = {}
        for pname, sel in [("mono", pos == "mono"), ("word_initial", pos == "initial"),
                           ("word_medial", pos == "medial"), ("word_final", pos == "final"),
                           ("utterance_final", last)]:
            if sel.any():
                ys = rows[sel, 0]
                r["by_position"][pname] = {"auc": M.roc_auc(ys, 1 - rows[sel, 1]), "n_neg": int((ys == 0).sum())}
        # boundary jitter +-40 ms (uniform per boundary)
        Xj, yj, mj = build_arrays(utts, kind, rng=rng, jitter=4)
        Pj = softmax(logits(net, Xj), T)
        r["acc_jitter40ms"] = float((Pj.argmax(1) == yj).mean())
        r["auc_jitter40ms"] = M.roc_auc(*(lambda q: (q[:, 0], 1 - q[:, 1]))(detection_pairs(Pj, utts, mj)))
        # speaking rate: per speaker (aishell3) or per utterance; slowest / fastest third
        rate = np.array([speaking_rate(u) for u in utts])
        if ds == "aishell3":
            spk = np.array([u["speaker"] for u in utts])
            sr = {s: rate[spk == s].mean() for s in set(spk)}
            rate = np.array([sr[s] for s in spk])
        q1, q2 = np.quantile(rate, [1 / 3, 2 / 3])
        ur = rate[[m[0] for m in meta]]
        corr = pred == y
        r["by_rate"] = {"slow_third": {"acc": float(corr[ur <= q1].mean()), "syl_per_s": float(rate[rate <= q1].mean())},
                        "middle_third": {"acc": float(corr[(ur > q1) & (ur <= q2)].mean())},
                        "fast_third": {"acc": float(corr[ur > q2].mean()), "syl_per_s": float(rate[rate > q2].mean())}}
        if ds == "aishell3":
            ag = agree_map(utts)
            cat = np.array([ag.get(utts[u]["id"], [None] * 999)[s] or "none" for u, s in meta])
            ext = np.array([(utts[u]["ext"][s]["tone"] if utts[u].get("ext") else None) or 0 for u, s in meta])
            sel = cat == "agree"
            r["acc_on_agreeing"] = float(corr[sel].mean())
            okx = ext > 0
            r["acc_vs_aishell_labels"] = float((pred[okx] + 1 == ext[okx]).mean())
            r["category_counts"] = {c: int((cat == c).sum()) for c in set(cat)}
        out[f"{ds}-{split}"] = r
        print(name, ds, {k: r[k] for k in ("acc", "acc_with_alts", "acc_jitter40ms")}, r["detection"],
              "2-3:", r["pairs"].get("2-3"), flush=True)
    save_json(out, f"{name}.json")
    return out


if __name__ == "__main__":
    for n in sys.argv[1:]:
        evaluate(n)
