"""Round 19b (German): the two-sided word step from the sound-identity head (Korean round-2 form).

    python -m bench.round19.identity_fit2 [--save]

Per sound: s = identity margin (log P(expected) - log max P(other)) minus a native threshold for its
(class, position in word) (q-quantile on the native pool; class-level fallback). Per word:
x = [min(s_min, 0), min(s_min + 4, 0), mean clip(s, -10, 10), sum clip(-s, 0, 1), 1] over the judged
sounds (no judged sound: no step). New word log-odds = old + min(b . x, cap_up) with cap_up = 3, so a
word whose sounds all clearly match is nudged up and one with a clearly different sound pushed down.
b: L2 logistic regression with the old word log-odds as a fixed offset (word changed vs not), fitted on
the dev FIT fold (planted real-word edits), the controlled sound errors of the FIT fold, and the native
pool (held-out Common Voice train speakers + FIT-fold controls), each word weight 1.
Grid: q in (0.05, 0.10) x L2 in (1, 10, 100). Guards as identity_fit.py (native pool P < 0.5 and
< 0.6 not up by more than 0.3 pt; controlled-error AUC, far-swap and word-omission AUC not lower).
Pick: lowest FIT-fold FA@80; CHECK fold reported.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from bench.round19.devset import WORK
from bench.round19.identity_collect import load
from bench.round19.identity_fit import raw_rows
from bench.round19.identity_train import pool_speaker
from pronunciada import de_identity as DI

CAP = 3.0
ONE = 1.0          # constant term of the step
POS_SHARE = 0.1


def positions(r):
    word = np.asarray(r["word"])
    out = []
    for i, w in enumerate(word):
        first = i == 0 or word[i - 1] != w
        last = i == len(word) - 1 or word[i + 1] != w
        out.append("initial" if first else "final" if last else "medial")
    return out


def thresholds(nat, q):
    by, byc = {}, {}
    for r in nat:
        for x, c, p in zip(r["s"], r["cls"], positions(r)):
            if c is not None and np.isfinite(x):
                by.setdefault(f"{c}|{p}", []).append(x)
                byc.setdefault(f"{c}|*", []).append(x)
    thr = {k: float(np.quantile(v, q)) for k, v in by.items() if len(v) >= 40}
    thr.update({k: float(np.quantile(v, q)) for k, v in byc.items() if len(v) >= 20})
    return thr


def feats(r, thr):
    s = np.array([x - thr.get(f"{c}|{p}", thr.get(f"{c}|*", 0.0)) if c is not None and np.isfinite(x) else np.nan
                  for x, c, p in zip(r["s"], r["cls"], positions(r))])
    nw = len(r["word_prob"])
    X = np.zeros((nw, 5))
    has = np.zeros(nw, bool)
    for j in range(nw):
        v = s[np.asarray(r["word"]) == j]
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        has[j] = True
        m = float(v.min())
        X[j] = [min(m, 0.0), min(m + 4.0, 0.0), float(np.clip(v, -10, 10).mean()), float(np.clip(-v, 0, 1).sum()), ONE]
    return X, has


def z_of(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_b(X, z0, y, l2):
    from scipy.optimize import minimize

    # changed / wrong words weighted to POS_SHARE of the total weight (they are ~2 % of the words)
    wpos = POS_SHARE / (1 - POS_SHARE) * (y == 0).sum() / max(1, (y == 1).sum())
    w = np.where(y == 1, wpos, 1.0)

    def f(b):
        z = z0 + np.minimum(X @ b, CAP)
        # P(correct) = sigmoid(z); y = 1 for a changed word
        lp = -np.logaddexp(0, -z)
        lq = -np.logaddexp(0, z)
        nll = -np.sum(w * np.where(y == 1, lq, lp))
        return nll + l2 * np.sum(b ** 2)
    res = minimize(f, np.zeros(X.shape[1]), method="L-BFGS-B")
    return res.x


def labels_dev(r, items):
    it = items[r["id"]]
    lab = it.get("label") or {}
    y = np.zeros(len(r["word_prob"]), int)
    if lab.get("word_index") is not None:
        import re

        t = list(re.finditer(r"\S+", it["target"]))[lab["word_index"]]
        sp = r.get("word_span") or []
        for j, (a, b) in enumerate(sp):
            if a < t.end() and b > t.start():
                y[j] = 1
    return y


def main(argv=None):
    from bench.metrics import roc_auc
    from bench.round19.collect import manifest
    from bench.round19.evaluate import _metrics, load_dump, units
    from bench.round19.whatif import rebuild

    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--dump", default="dev-r19c")
    a = ap.parse_args(argv)
    H = DI._head()
    pool = raw_rows([r for r in load("cv-train") if pool_speaker(r)], H)
    dev = raw_rows(load("dev"), H)
    cor = raw_rows(load("devcor"), H)
    items = {it["id"]: it for it in manifest("dev")}
    fit_dev = [r for r in dev if r["meta"].get("fold") == "fit"]
    nat = pool + [r for r in fit_dev if not r["meta"].get("label")]
    cor_fit = [r for r in cor if r["meta"].get("fold") == "fit"]
    cor_nat = [r for r in dev if not r["meta"].get("label") and r["meta"].get("src") == "cv" and r["meta"].get("fold") == "fit"]
    D = load_dump(a.dump)
    fit_items = [it for it in items.values() if it["fold"] == "fit"]
    chk_items = [it for it in items.values() if it["fold"] == "check"]

    def dev_metrics(delta, its):
        def fn(w, c):
            d = delta.get(c["rec"]["id"])
            if d is None or c["wi"] >= len(d):
                return w["prob"]
            return float(1 / (1 + np.exp(-(z_of(w["prob"]) + d[c["wi"]]))))
        d = {k: (rebuild(v, fn) if not v.get("error") and k in delta else v) for k, v in D.items()}
        return _metrics(units(its, d))

    base = dev_metrics({}, fit_items)
    nat_p = np.concatenate([r["word_prob"] for r in pool])
    e0 = np.concatenate([r["word_prob"][np.array(r["wrong"])] for r in cor_fit])
    n0 = np.concatenate([r["word_prob"] for r in cor_nat])
    base_cor = roc_auc(np.r_[np.ones(len(e0)), np.zeros(len(n0))], -np.r_[e0, n0])
    print(f"base fit AUC {base['auc']:.4f} FA {100 * base['fa80']:.2f}  pool <0.5 {100 * np.mean(nat_p < .5):.2f} <0.6 "
          f"{100 * np.mean(nat_p < .6):.2f}  controlled {base_cor:.4f}", flush=True)
    res = []
    for q, l2 in itertools.product((0.05, 0.10), (1.0, 10.0, 100.0)):
        thr = thresholds(nat, q)
        Xs, zs, ys = [], [], []
        for r in fit_dev:
            X, has = feats(r, thr)
            Xs.append(X[has]); zs.append(z_of(r["word_prob"])[has]); ys.append(labels_dev(r, items)[has])
        for r in pool:
            X, has = feats(r, thr)
            Xs.append(X[has]); zs.append(z_of(r["word_prob"])[has]); ys.append(np.zeros(has.sum(), int))
        for r in cor_fit:
            X, has = feats(r, thr)
            Xs.append(X[has]); zs.append(z_of(r["word_prob"])[has]); ys.append(np.array(r["wrong"], int)[has])
        b = fit_b(np.vstack(Xs), np.concatenate(zs), np.concatenate(ys), l2)

        def delta_of(r):
            X, has = feats(r, thr)
            return np.where(has, np.minimum(X @ b, CAP), 0.0)
        dl = {r["id"]: delta_of(r) for r in dev}
        new = lambda r: 1 / (1 + np.exp(-(z_of(r["word_prob"]) + delta_of(r))))  # noqa: E731
        pn = np.concatenate([new(r) for r in pool])
        d05, d06 = np.mean(pn < .5) - np.mean(nat_p < .5), np.mean(pn < .6) - np.mean(nat_p < .6)
        e1 = np.concatenate([new(r)[np.array(r["wrong"])] for r in cor_fit])
        n1 = np.concatenate([new(r) for r in cor_nat])
        cauc = roc_auc(np.r_[np.ones(len(e1)), np.zeros(len(n1))], -np.r_[e1, n1])
        m = dev_metrics(dl, fit_items)
        ok = (d05 <= 0.003 and d06 <= 0.003 and cauc >= base_cor - 1e-4
              and m["bucket_auc"]["sub_gross"] >= base["bucket_auc"]["sub_gross"] - 1e-4
              and m["bucket_auc"]["word_omission"] >= base["bucket_auc"]["word_omission"] - 1e-4)
        row = {"q": q, "l2": l2, "b": b.tolist(), "thr": thr, "d_pool_05": d05, "d_pool_06": d06, "controlled_auc": cauc,
               "fit_auc": m["auc"], "fit_fa80": m["fa80"], "fit_buckets": m["bucket_auc"], "ok": ok}
        res.append(row)
        print(f"q {q} l2 {l2}: b {np.round(b, 2).tolist()} pool {100 * d05:+.2f}/{100 * d06:+.2f} pt ctrl {cauc:.4f} "
              f"fit AUC {m['auc']:.4f} FA {100 * m['fa80']:.2f} subtle {m['bucket_auc']['sub_subtle']:.3f} "
              f"far {m['bucket_auc']['sub_gross']:.4f} om {m['bucket_auc']['word_omission']:.4f} {'OK' if ok else ''}", flush=True)
    okr = [r for r in res if r["ok"]]
    pick = min(okr, key=lambda r: (round(r["fit_fa80"], 5), -r["fit_auc"])) if okr else None
    out = {"base_fit": base, "base_controlled": base_cor, "grid": [{k: v for k, v in r.items() if k != "thr"} for r in res]}
    if pick:
        thr, b = pick["thr"], np.array(pick["b"])
        dl = {r["id"]: np.where(feats(r, thr)[1], np.minimum(feats(r, thr)[0] @ b, CAP), 0.0) for r in dev}
        mc0, mc1 = dev_metrics({}, chk_items), dev_metrics(dl, chk_items)
        print(f"PICK q {pick['q']} l2 {pick['l2']}: check AUC {mc0['auc']:.4f} -> {mc1['auc']:.4f} FA@80 "
              f"{100 * mc0['fa80']:.2f} -> {100 * mc1['fa80']:.2f} subtle {mc0['bucket_auc']['sub_subtle']:.3f} -> "
              f"{mc1['bucket_auc']['sub_subtle']:.3f}")
        out["pick"] = {k: v for k, v in pick.items() if k != "thr"}
        out["check"] = {"before": mc0, "after": mc1}
        cal = {"form": "delta", "features": ["s_min0", "s_min4", "s_mean", "n_neg", "one"], "coef": pick["b"],
               "cap_up": CAP, "thr": thr, "q": pick["q"], "l2": pick["l2"],
               "fit_on": "round-19 German dev fit fold + controlled errors + held-out Common Voice train native pool (bench/round19/identity_fit2.py)"}
        if a.save:
            Path("pronunciada/data/calib/de_identity_word.json").write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n")
            Path("results/tables/round19_de_identity_fit2.json").write_text(json.dumps(out, indent=1, default=float))
    (WORK / "identity_fit2.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
