"""Round 19 (German): a German recalibration of the word model's sounds probability (logistic).

    python -m bench.round19.fit_rerank dev-text1ea [--C 0.1] [--out file.json]

The shipped word model (word.npz, pooled over languages, fitted on speechocean762 train + FLEURS
validation planted errors) gives sounds_prob. Here a logistic regression re-weights its log-odds
with a few word-level inputs it already reads plus the front-end's "unknown word" flag:
    z' = b0 + b . x,  x = [z_word, wl_min, ws_min, wl_mean, ws_mean, log n_phones, word_pos_last,
                          utt_ws_mean, oov]   (clipped, standardised)
fitted on the FIT fold of the dev set (planted edits on CV / FLEURS validation natives; changed
word = 1), evaluated on the CHECK fold (speaker-disjoint). The stress and presence factors stay
outside (WordResult.prob = sigmoid(z') x stress x present), so the stress channel is unchanged.
The intercept is then set so that native word acceptance (sounds' >= 0.5) on the fit-fold
control items equals the shipped one (the calibration of the probability is kept).
"""
from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np

from bench.round19.collect import manifest
from bench.round19.evaluate import load_dump

FEATS = ("z_word", "wl_min", "ws_min", "wl_mean", "ws_mean", "log_n")
CLIP = 12.0
_P = re.compile(r"[^\w\s-]")


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def word_feats(w, oov: bool) -> list[float]:
    W = w["W"]
    g = lambda k: float(np.clip(W[k] if W.get(k) is not None else 0.0, -CLIP, CLIP))  # noqa: E731
    return [float(np.clip(logit(w["sounds"]), -CLIP, CLIP)), g("wl_min"), g("ws_min"), g("wl_mean"),
            g("ws_mean"), math.log(max(1.0, W["n_phones"] or 1.0))]


def oov_set(rec) -> set[str]:
    return {x.split(":", 1)[1] for x in rec.get("warnings", []) if x.startswith("oov:")}


def is_oov(w, oovs) -> bool:
    return _P.sub("", w["text"]).lower() in oovs


def table(name, fold):
    M = manifest("dev")
    D = load_dump(name)
    X, y, ids = [], [], []
    for it in M:
        if fold != "all" and it["fold"] != fold:
            continue
        r = D.get(it["id"])
        if r is None or r.get("error"):
            continue
        lab = it.get("label") or {}
        oovs = oov_set(r)
        span = None
        if lab.get("word_index") is not None:
            toks = list(re.finditer(r"\S+", it["target"]))
            t = toks[lab["word_index"]]
            span = (t.start(), t.end())
        for w in r["rwords"]:
            pos = int(span is not None and w["span"][0] < span[1] and w["span"][1] > span[0])
            X.append(word_feats(w, is_oov(w, oovs)))
            y.append(pos)
            ids.append((it["id"], pos, it.get("label") is None, is_oov(w, oovs)))
    return np.array(X), np.array(y), ids


def cor_table(cor_name, fold):
    """Controlled sound errors (bench.round19.collect run_corrupt): only the corrupted words."""
    M = {m["id"]: m for m in manifest("dev")}
    D = load_dump(cor_name)
    X = []
    for k, r in D.items():
        if fold != "all" and M[k]["fold"] != fold:
            continue
        oovs = oov_set(r)
        for w, bad in zip(r["rwords"], r["wrong"]):
            if bad:
                X.append(word_feats(w, is_oov(w, oovs)))
    return np.array(X)


COR = None          # name of a devcor dump to add to the fit (None = swaps only)
COR_W = 1.0         # total weight of the controlled errors relative to the planted real-word edits


def fit(name, C=0.1):
    from sklearn.linear_model import LogisticRegression

    X, y, ids = table(name, "fit")
    n_swap_pos = int((y == 1).sum())
    if COR:
        Xc = cor_table(COR, "fit")
        X = np.vstack([X, Xc])
        y = np.r_[y, np.ones(len(Xc), int)]
        ids = ids + [("cor", 1, False, False)] * len(Xc)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    # changed words are ~4 % of words; weight them to 10 % (the prior the shipped models use)
    wpos = 0.1 / 0.9 * (y == 0).sum() / max(1, (y == 1).sum())
    sw = np.where(y == 1, wpos, 1.0)
    # length balance: the changed words are re-weighted so that their number of sounds has the
    # distribution of the correct words (a planted edit lands more often on short words, which
    # have more real-word neighbours); the length term then measures how the evidence of long
    # and short words should be read, not where edits were planted
    nb = np.clip(np.round(np.exp(X[:, FEATS.index("log_n")])), 1, 12).astype(int)
    pc = np.bincount(nb[y == 0], minlength=13) / max(1, (y == 0).sum())
    pp = np.bincount(nb[y == 1], minlength=13) / max(1, (y == 1).sum())
    ratio = np.where(pp > 0, pc / np.maximum(pp, 1e-9), 0.0)
    cor = np.array([i[0] == "cor" for i in ids])
    sw = np.where((y == 1) & ~cor, sw * ratio[nb], sw)      # controlled errors: positions are random already
    sw[(y == 1) & ~cor] *= wpos * n_swap_pos / sw[(y == 1) & ~cor].sum()
    if cor.any():                                    # the two error sets get equal total weight
        sw[cor] = COR_W * wpos * n_swap_pos / cor.sum()
    lr = LogisticRegression(C=C, max_iter=2000)
    lr.fit((X - mu) / sd, 1 - y, sample_weight=sw)       # model P(correct)
    coef = lr.coef_[0] / sd
    b0 = float(lr.intercept_[0] - (lr.coef_[0] * mu / sd).sum())
    # intercept so that native acceptance on fit-fold control words matches the shipped one
    ctl = np.array([i[2] for i in ids])
    oov = np.array([i[3] for i in ids])
    z = X @ coef + b0
    ship = (X[ctl, 0] >= 0).mean()
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ((z[ctl] + mid) >= 0).mean() < ship:
            lo = mid
        else:
            hi = mid
    b0 += hi
    # unknown words (letter-rule reference): shift their log-odds up until their native flag rate
    # (P < 0.5 on fit-fold control words) is no higher than that of lexicon words
    zc = z[ctl] + hi
    lex_rate = float((zc[~oov[ctl]] < 0).mean())
    shift = 0.0
    while shift < 20 and ((zc[oov[ctl]] + shift) < 0).mean() > lex_rate:
        shift += 0.25
    return {"features": list(FEATS), "coef": [float(c) for c in coef], "b0": b0, "clip": CLIP,
            "oov_shift": shift, "C": C, "fit_on": f"{name} fit fold", "native_accept_fit": float(ship),
            "native_flag_rate_lexicon": lex_rate,
            "native_flag_rate_oov_before": float((zc[oov[ctl]] < 0).mean()), "n_oov_native": int(oov[ctl].sum())}


def apply_fn(par):
    coef = np.array(par["coef"])

    def fn(w, c):
        oovs = oov_set(c["rec"])
        o = is_oov(w, oovs)
        z = float(np.array(word_feats(w, o)) @ coef + par["b0"]) + (par.get("oov_shift", 0.0) if o else 0.0)
        snd = 1 / (1 + math.exp(-z))
        return w["prob"] / max(w["sounds"], 1e-6) * snd
    return fn


def main(argv=None):
    from bench.round19.whatif import metrics

    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--out")
    ap.add_argument("--cor", help="devcor dump: add controlled sound errors to the fit, report them on check")
    ap.add_argument("--cor-weight", type=float, default=1.0)
    a = ap.parse_args(argv)
    global COR, COR_W
    COR, COR_W = a.cor, a.cor_weight
    par = fit(a.name, a.C)
    par["controlled_errors"] = a.cor
    par["controlled_error_weight"] = a.cor_weight
    print(json.dumps({k: v for k, v in par.items()}, indent=None))
    if a.cor:                          # controlled sound errors on the check fold vs check control words
        from bench.metrics import roc_auc

        Xc = cor_table(a.cor, "check")
        X, y, ids = table(a.name, "check")
        Xn = X[np.array([i[2] for i in ids]) & (y == 0)]
        coef = np.array(par["coef"])
        for tag, zc, zn in (("before", Xc[:, 0], Xn[:, 0]), ("after", Xc @ coef + par["b0"], Xn @ coef + par["b0"])):
            yy = np.r_[np.ones(len(zc)), np.zeros(len(zn))]
            thr = np.quantile(zn, 0.05)
            print(f"check controlled sound errors {tag}: AUC {roc_auc(yy, -np.r_[zc, zn]):.4f} "
                  f"hit@5% {np.mean(zc < thr):.4f}  (n {len(zc)})")
    for fold in ("fit", "check"):
        for src in (None, "cv"):
            m0 = metrics(a.name, None, fold=fold, source=src)
            m1 = metrics(a.name, apply_fn(par), fold=fold, source=src)
            print(f"{fold:5s} {src or 'all':4s} AUC {m0['auc']:.4f} -> {m1['auc']:.4f}   FA@80 {100 * m0['fa80']:.2f} -> {100 * m1['fa80']:.2f}")
    # length-matched check: changed items kept with a probability that gives the changed words the
    # length distribution of the correct words (10 random draws, mean)
    from bench.round19 import evaluate as E

    X, y, ids = table(a.name, "check")
    nb = np.clip(np.round(np.exp(X[:, FEATS.index("log_n")])), 1, 12).astype(int)
    pc = np.bincount(nb[y == 0], minlength=13) / (y == 0).sum()
    pp = np.bincount(nb[y == 1], minlength=13) / (y == 1).sum()
    ratio = np.where(pp > 0, pc / np.maximum(pp, 1e-9), 0)
    acc = ratio / ratio.max()
    n_of = {i[0]: n for i, n in zip(ids, nb) if i[1]}
    items = [it for it in manifest("dev") if it["fold"] == "check"]
    from bench.round19.whatif import apply as wapply

    d0, d1 = E.load_dump(a.name), wapply(a.name, apply_fn(par))
    rng = np.random.default_rng(0)
    res = []
    for _ in range(10):
        keep = [it for it in items if not it.get("label") or (it["id"] in n_of and rng.random() < acc[n_of[it["id"]]])]
        m0, m1 = E._metrics(E.units(keep, d0)), E._metrics(E.units(keep, d1))
        res.append((m0["auc"], m1["auc"], m0["fa80"], m1["fa80"]))
    r = np.mean(res, 0)
    print(f"check length-matched AUC {r[0]:.4f} -> {r[1]:.4f}   FA@80 {100 * r[2]:.2f} -> {100 * r[3]:.2f}")
    if a.out:
        from pathlib import Path

        Path(a.out).write_text(json.dumps(par, indent=1))


if __name__ == "__main__":
    main()
