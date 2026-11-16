"""Round 8 decision step: shipped stress model vs `stress_v3` on real learner stress marks.
speechocean762 TRAIN only (used as dev; test untouched).

    .venv/bin/python -m bench.prosody.so762_stress [--workers 3]

Per utterance: pronunciada.Scorer("en").score(audio, text) with PRONUNCIADA_STRESS_MODEL unset
(shipped `stress`) and = stress_v3 (its own calibrator en.stress_v3.json). Our words are matched
to the corpus words by position when the counts agree (else by unique text). Words with >= 2
syllables in our reference and a stress_prob.
Label: the corpus ships one word stress score per word (the aggregate of the five experts; values
10 or 5 only): stress wrong = 5.
(a) AUC of 1 - stress_prob, stress-wrong vs all other words;
(b) the same against stress-correct words with accuracy >= 8 (so the score measures stress,
    not general badness);
95 % speaker-bootstrap CIs (2000 resamples), paired bootstrap of v3 - old, two-sided p;
plus the share of stress-correct words with stress_prob < 0.5.
"""
from __future__ import annotations

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_SC = None


def _init(model):
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    if model:
        os.environ["PRONUNCIADA_STRESS_MODEL"] = model
    else:
        os.environ.pop("PRONUNCIADA_STRESS_MODEL", None)
    global _SC
    from pronunciada.scorer import Scorer

    _SC = Scorer("en", threads=1)


def _one(rec):
    try:
        res = _SC.score(rec["audio"], rec["text"])
    except ValueError:
        return rec["id"], None
    return rec["id"], [(w.text, len(w.syllables), w.stress_prob, w.prob) for w in res.words]


def collect(model, workers):
    from bench.calib.common import calib_cache
    from bench.data import load

    p = calib_cache() / f"so762_train_stress_{model or 'stress'}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    recs = load("speechocean762", split="train")
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(model,)) as ex:
        out = dict(ex.map(_one, recs, chunksize=8))
    p.write_bytes(pickle.dumps(out))
    return out


def table(out):
    from bench.data import load

    rows = []
    for r in load("speechocean762", split="train"):
        ours = out.get(r["id"])
        lw = r["labels"]["words"]
        if not ours:
            continue
        for k, (t, nsyl, sp, prob) in enumerate(ours):
            if nsyl < 2 or sp is None:
                continue
            tt = t.upper().strip(".,!?;:\"'")
            if len(ours) == len(lw) and lw[k]["text"].upper() == tt:
                lab = lw[k]
            else:
                c = [w for w in lw if w["text"].upper() == tt]
                if len(c) != 1:
                    continue
                lab = c[0]
            rows.append((r["id"], r["speaker"], k, lab["stress"], lab["accuracy"], sp))
    return rows


def main():
    from bench.metrics import roc_auc

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    T = {m: {(u, k): row for row in table(collect(m, a.workers)) for u, _, k, *_ in [row]}
         for m in (None, "stress_v3")}
    keys = sorted(set(T[None]) & set(T["stress_v3"]))
    spk = np.array([T[None][k][1] for k in keys])
    wrong = np.array([T[None][k][3] < 10 for k in keys])
    acc = np.array([T[None][k][4] for k in keys])
    s_old = np.array([T[None][k][5] for k in keys])
    s_new = np.array([T["stress_v3"][k][5] for k in keys])
    print(f"words >= 2 syllables matched: {len(keys)}; stress-wrong (score 5): {wrong.sum()}; "
          f"stress-correct with accuracy >= 8: {(~wrong & (acc >= 8)).sum()}; speakers {len(set(spk))}")
    rng = np.random.default_rng(0)
    sp = np.unique(spk)
    by = {s: np.flatnonzero(spk == s) for s in sp}
    boots = [np.concatenate([by[s] for s in rng.choice(sp, len(sp))]) for _ in range(2000)]
    for name, neg in (("(a) vs all other words", ~wrong), ("(b) vs stress-correct words with accuracy >= 8", ~wrong & (acc >= 8))):
        def auc(I, s):
            m = wrong[I] | neg[I]
            return roc_auc(wrong[I][m].astype(int), 1 - s[I][m])
        full = np.arange(len(keys))
        o, n = auc(full, s_old), auc(full, s_new)
        bo = np.array([auc(I, s_old) for I in boots])
        bn = np.array([auc(I, s_new) for I in boots])
        d = bn - bo
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        print(f"{name}: old {o:.3f} [{np.quantile(bo, .025):.3f}, {np.quantile(bo, .975):.3f}]  "
              f"v3 {n:.3f} [{np.quantile(bn, .025):.3f}, {np.quantile(bn, .975):.3f}]  "
              f"diff {n - o:+.3f} [{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}] p {p:.3f}")
    print(f"stress-correct words with stress_prob < 0.5: old {np.mean(s_old[~wrong] < 0.5):.3f}  v3 {np.mean(s_new[~wrong] < 0.5):.3f}")


if __name__ == "__main__":
    main()
