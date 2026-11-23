"""Gate items 2 and 3 by construction: the round-20 word passes replace WordResult objects only, so every output
that the speechocean762 CV (bench/round19 extract_en -> exp_seq / cv_blend / exp_level) and the marked-sound
detection (bench/calib/api_run -> Scorer.details sound scores) read must be identical with the switches on.

    python -m bench.round20_native_fa.invariance [--n 150]

Checked on the first n TRAIN clips of speechocean762, L2-ARCTIC and EpaDB, all English switches on vs off:
Scorer.details() (sentence scores and log-odds, phone scores and log-odds, word log-odds, 0-10 word accuracy,
omission flags), res.scores and res.prob. Reports max |diff| and how many clips had a round-20 pass that changed
any WordResult.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ON = {"PRONUNCIADA_R20_UNSAID": "1", "PRONUNCIADA_R20_NUMALT": "1", "PRONUNCIADA_R20_GUESS": "d1"}


def flat(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from flat(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from flat(v)
    elif isinstance(x, np.ndarray):
        yield from (float(v) for v in np.ravel(x).astype(float))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--guess", default="d1")
    a = ap.parse_args()
    from bench.data import load
    from pronunciada import Scorer

    sc = Scorer("en", threads=2)
    out = {}
    for ds in ("speechocean762", "l2arctic", "epadb"):
        recs = load(ds, split="train")[: a.n]
        worst, changed, n = 0.0, 0, 0
        for rec in recs:
            for k in ON:
                os.environ.pop(k, None)
            try:
                r0 = sc.score(rec["audio"], rec["text"])
            except ValueError:
                continue
            d0 = list(flat({"d": sc.details(r0), "s": r0.scores, "p": r0.prob}))
            os.environ.update({**ON, "PRONUNCIADA_R20_GUESS": a.guess})
            r1 = sc.score(rec["audio"], rec["text"])
            d1 = list(flat({"d": sc.details(r1), "s": r1.scores, "p": r1.prob}))
            n += 1
            x, y = np.array(d0), np.array(d1)
            m = np.isfinite(x) & np.isfinite(y)
            worst = max(worst, float(np.max(np.abs(x[m] - y[m]))) if m.any() else 0.0)
            changed += any(w0.prob != w1.prob for w0, w1 in zip(r0.words, r1.words))
        out[ds] = {"clips": n, "max_abs_diff_details": worst, "clips_with_changed_word_prob": changed}
        print(ds, out[ds], flush=True)
    for k in ON:
        os.environ.pop(k, None)
    from bench.round20_native_fa import core as K

    K.save_json("r20_invariance.json", out)


if __name__ == "__main__":
    main()
