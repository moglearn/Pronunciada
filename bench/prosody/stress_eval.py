"""Evaluate word-stress models.

    .venv/bin/python -m bench.prosody.stress_eval stress_multi [stress_es ...]

Per language on FLEURS test: accuracy of locating the primary stress (argmax over the word's
syllables == front-end's primary stress); stress-shift detection (bench.corrupt `stress`: the
primary stress moved to another syllable): negatives = P(expected syllable), positives =
P(each other syllable) for every word; AUC / EER; ECE after Platt calibration (per language) on
FLEURS validation. English also: speechocean762 test, P(stress where expected) vs the human word
`stress` score (0-10).
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch  # noqa: E402

from bench import metrics as M  # noqa: E402
from bench.prosody.common import MODELS_DIR, load_prepped, save_json  # noqa: E402
from bench.prosody.stress_train import StressNet, build_words, word_logits  # noqa: E402
from bench.prosody.tone_eval import platt_apply, platt_fit  # noqa: E402


def load(name):
    ck = torch.load(MODELS_DIR / f"{name}.pt", weights_only=False)
    net = StressNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


def probs(net, utts, lid):
    r = build_words(utts, [lid] * len(utts))
    if r is None:
        return None
    X, E, Mk, y, LG, meta = r
    z = word_logits(net, X, E, Mk, LG, torch.device("cpu"))
    z = z - z.max(1, keepdims=True)
    P = np.exp(z) * Mk
    P /= P.sum(1, keepdims=True)
    return P, y, Mk, meta


def pairs(P, y, Mk):
    lab, sc = [], []
    for k in range(len(y)):
        lab.append(0)
        sc.append(P[k, y[k]])
        for j in np.flatnonzero(Mk[k]):
            if j != y[k]:
                lab.append(1)
                sc.append(P[k, j])
    return np.array(lab), np.array(sc)


def evaluate(name):
    net, ck = load(name)
    langs = ck["langs"]
    out = {"model": name, "langs": langs, "cfg": ck["cfg"], "curve": ck["curve"], "platt": {}, "test": {}}
    for k, lang in enumerate(langs):
        lid = k if ck["cfg"].get("n_lang") else 0
        Pd, yd, Md, _ = probs(net, load_prepped("fleurs_val", lang, "dev"), lid)
        ld, sd = pairs(Pd, yd, Md)
        ab = platt_fit(sd, ld)
        out["platt"][lang] = ab
        P, y, Mk, meta = probs(net, load_prepped("fleurs", lang, "test"), lid)
        lab, sc = pairs(P, y, Mk)
        nsyl = Mk.sum(1)
        corr = P.argmax(1) == y
        r = {"n_words": int(len(y)), "acc": float(corr.mean()),
             "acc_ci95": M.bootstrap_ci(lambda a: float(a.mean()), corr, clusters=[m[0] for m in meta], n_boot=300),
             "chance_acc": float(np.mean(1 / nsyl)),
             "most_common_position_acc": _position_baseline(y, nsyl),
             "acc_by_nsyl": {int(n): float(corr[nsyl == n].mean()) for n in sorted(set(nsyl)) if (nsyl == n).sum() >= 20},
             "shift_auc": M.roc_auc(lab, 1 - sc), "shift_eer": M.eer(lab, 1 - sc)[0],
             "ece_calibrated": M.ece(1 - lab, platt_apply(sc, ab)), "n_pos": int(lab.sum()),
             "dev_acc": float((Pd.argmax(1) == yd).mean())}
        out["test"][lang] = r
        print(name, lang, {k2: v for k2, v in r.items() if k2 not in ("acc_ci95", "acc_by_nsyl")}, flush=True)
    if "en" in langs:
        out["so762"] = so762(net, langs.index("en") if ck["cfg"].get("n_lang") else 0, out["platt"]["en"])
    save_json(out, f"{name}.json")
    return out


def _position_baseline(y, nsyl):
    """Accuracy of always choosing the most frequent primary-stress position per word length
    (fitted on the same test words: an optimistic text-only reference, not a model)."""
    c = 0
    for n in set(nsyl):
        yy = y[nsyl == n]
        c += np.bincount(yy).max()
    return float(c / len(y))


def so762(net, lid, ab):
    from bench.data import load as dload

    utts = load_prepped("speechocean762", "en", "test")
    recs = {r["id"]: r for r in dload("speechocean762", split="test")}
    res = probs(net, utts, lid)
    P, y, Mk, meta = res
    p = platt_apply(P[np.arange(len(y)), y], ab)
    hum, keep = [], []
    for k, (ui, s0, s1) in enumerate(meta):
        u = utts[ui]
        wi = u["ref"].syllables[s0].word
        lw = recs[u["id"]]["labels"]["words"]
        wt = u["ref"].words[wi].text.upper().strip(".,!?;:\"'")
        # match our word to the label word by order and text
        cand = [i for i, w in enumerate(lw) if w["text"].upper() == wt]
        if len(lw) == len(u["ref"].words) and lw[wi]["text"].upper() == wt:
            hum.append(lw[wi]["stress"])
            keep.append(k)
        elif len(cand) == 1:
            hum.append(lw[cand[0]]["stress"])
            keep.append(k)
    hum = np.array(hum, float)
    p = p[keep]
    top = hum.max()
    low = (hum < top).astype(int)
    r = {"n_words": int(len(hum)), "score_counts": {int(v): int((hum == v).sum()) for v in sorted(set(hum))},
         "pearson": M.pearson(p, hum), "spearman": M.spearman(p, hum),
         "auc_below_top": M.roc_auc(low, 1 - p) if 0 < low.sum() < len(low) else None,
         "n_below_top": int(low.sum()), "top_score": float(top),
         "acc_locating_stress": float((P.argmax(1) == y)[keep].mean())}
    print("so762", r, flush=True)
    return r


if __name__ == "__main__":
    for n in sys.argv[1:]:
        evaluate(n)
