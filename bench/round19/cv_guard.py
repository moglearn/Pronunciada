"""Round 19 (German): cv-real German (bench/cv_build.py) before / after, either half.

    PRONUNCIADA_DE_R19=0 python -m bench.round19.cv_guard collect --half dev --tag off
    python -m bench.round19.cv_guard collect --half dev --tag on
    python -m bench.round19.cv_guard report --half dev off on

The protocol of bench/eval_cv.py (Test A: rejected vs accepted clips; Test B: non-native vs native
speakers): AUC of 1 - score for the primary score (SentenceResult.prob) and the secondary word scores
(mean / lowest word prob), 95 % bootstrap over speakers, paired difference with two-sided p.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from bench.round19.devset import WORK


def items(half):
    from bench.cv_build import OUT_DIR, audio_path

    out = []
    for l in (OUT_DIR / "de.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["half"] == half:
            r["wav"] = str(audio_path("de", r["path"].rsplit(".", 1)[0]))
            out.append(r)
    return out


def collect(half, tag):
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    from pronunciada import Scorer

    sc = Scorer("de", threads=2)
    rows = {}
    for r in items(half):
        try:
            res = sc.score(r["wav"], r["sentence"])
            wp = [w.prob for w in res.words]
            rows[r["id"]] = (res.prob, float(np.mean(wp)), float(np.min(wp)))
        except ValueError:
            rows[r["id"]] = (0.0, 0.0, 0.0)
    (WORK / f"cvguard_{half}_{tag}.pkl").write_bytes(pickle.dumps(rows))


def report(half, tags, n_boot=1000):
    from bench.metrics import roc_auc

    its = items(half)
    R = [pickle.loads((WORK / f"cvguard_{half}_{t}.pkl").read_bytes()) for t in tags]
    for test in ("A", "B"):
        sel = [r for r in its if r["test"] == test]
        spk = sorted({r["speaker"] for r in sel})
        by = {s: [i for i, r in enumerate(sel) if r["speaker"] == s] for s in spk}
        y = np.array([r["label"] for r in sel])
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by[spk[j]] for j in rng.integers(0, len(spk), len(spk))]) for _ in range(n_boot)]
        for k, name in enumerate(("sentence prob", "word mean", "word min")):
            S = [np.array([1 - rr[r["id"]][k] for r in sel]) for rr in R]
            aucs = [roc_auc(y, s) for s in S]
            line = f"Test {test} {name:13s} " + "  ".join(f"{t} {a:.4f}" for t, a in zip(tags, aucs))
            if len(R) == 2:
                d = np.array([roc_auc(y[ix], S[1][ix]) - roc_auc(y[ix], S[0][ix]) for ix in draws])
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                line += f"  delta {aucs[1] - aucs[0]:+.4f} [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}] p={min(1.0, p):.3f}"
            print(line)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "report"])
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--half", default="dev")
    ap.add_argument("--tag")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.half, a.tag)
    else:
        report(a.half, a.tags)


if __name__ == "__main__":
    main()
