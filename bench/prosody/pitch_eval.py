"""Japanese pitch accent: per-mora P(high) vs the front-end's pitch_high, on FLEURS ja test.

    .venv/bin/python -m bench.prosody.pitch_eval pitch_ja

Mora accuracy vs labels; pattern-flip detection (bench.corrupt kind `pitch`): for every word whose
morae all carry pitch marks, negatives = the reference pattern, positives = every other Tokyo
pattern of that length (heiban / atamadaka / nakadaka / odaka); word score = mean over morae of
log P(mora matches pattern); also mora-level flip AUC (P(expected) vs P(flipped)).
Calibration of pitch_prob: Platt on dev (FLEURS validation ja) mora pairs.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")

from bench import metrics as M  # noqa: E402
from bench.corrupt import _tokyo_patterns  # noqa: E402
from bench.prosody.common import build_arrays, load_prepped, save_json  # noqa: E402
from bench.prosody.tone_eval import load_model, logits, platt_apply, platt_fit  # noqa: E402


def ph(net, utts):
    X, y, meta = build_arrays(utts, "ctx", task="pitch")
    p = 1 / (1 + np.exp(-logits(net, X)[:, 0]))
    return p, y, meta


def mora_pairs(p, y):
    pe = np.where(y == 1, p, 1 - p)
    return np.concatenate([np.zeros(len(p)), np.ones(len(p))]), np.concatenate([pe, 1 - pe])


def word_rows(p, utts, meta):
    idx = {m: k for k, m in enumerate(meta)}
    rows = []
    for ui, u in enumerate(utts):
        ref = u["ref"]
        for w in ref.words:
            s0, s1 = w.syllables
            if s1 - s0 < 1 or any(ref.syllables[i].pitch_high is None for i in range(s0, s1)):
                continue
            q = np.array([p[idx[(ui, i)]] for i in range(s0, s1)])
            cur = tuple(bool(ref.syllables[i].pitch_high) for i in range(s0, s1))
            sc = lambda pat: float(np.mean(np.log(np.where(np.array(pat), q, 1 - q) + 1e-6)))  # noqa: E731
            rows.append((0, sc(cur), s1 - s0))
            for pat in _tokyo_patterns(s1 - s0):
                if pat != cur:
                    rows.append((1, sc(pat), s1 - s0))
    return np.array(rows)


def evaluate(name):
    net, ck = load_model(name)
    pd, yd, _ = ph(net, load_prepped("fleurs_val", "ja", "dev"))
    ym, pm = mora_pairs(pd, yd)
    ab = platt_fit(pm, ym)
    out = {"model": name, "platt": ab, "curve": ck["curve"], "dev_acc": float(((pd > 0.5) == (yd == 1)).mean())}
    utts = load_prepped("fleurs", "ja", "test")
    p, y, meta = ph(net, utts)
    pred = p > 0.5
    ym, pm = mora_pairs(p, y)
    wr = word_rows(p, utts, meta)
    multi = wr[:, 2] >= 2
    out["test"] = {
        "n_morae": int(len(y)), "share_high_labels": float(y.mean()),
        "acc": float((pred == (y == 1)).mean()),
        "acc_ci95": M.bootstrap_ci(lambda a, b: float((a == b).mean()), pred, y == 1,
                                   clusters=[m[0] for m in meta], n_boot=300),
        "recall_high": float(pred[y == 1].mean()), "recall_low": float((~pred)[y == 0].mean()),
        "mora_flip_auc": M.roc_auc(ym, 1 - pm), "mora_flip_eer": M.eer(ym, 1 - pm)[0],
        "mora_ece_calibrated": M.ece(1 - ym, platt_apply(pm, ab)),
        "word_pattern_auc": M.roc_auc(wr[:, 0], -wr[:, 1]), "word_pattern_eer": M.eer(wr[:, 0], -wr[:, 1])[0],
        "word_pattern_auc_multimora": M.roc_auc(wr[multi, 0], -wr[multi, 1]),
        "n_words": int((wr[:, 0] == 0).sum()), "n_wrong_patterns": int((wr[:, 0] == 1).sum()),
        "majority_baseline_acc": float(max(y.mean(), 1 - y.mean())),
    }
    print(out["test"])
    save_json(out, f"{name}.json")
    return out


if __name__ == "__main__":
    evaluate(sys.argv[1])
