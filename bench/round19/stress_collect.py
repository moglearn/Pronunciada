"""Round 19b (German): per-word stress-channel rows on the dev clips, natives and planted stress errors.

    python -m bench.round19.stress_collect

For every dev clip (Common Voice dev and FLEURS validation, bench.round19.devset): the clip's own text
(native words) and the same text with 2 planted stress errors (bench.corrupt kind "stress", seed per
clip), scored through the public Scorer("de"). Per word that the stress channel judges: the scored
syllables' stress probabilities (the channel input), the shown sounds and presence factors, and the
reliability of the pitch / energy evidence: share of voiced 10 ms frames and mean voicing probability
over the word, mean frame energy, number of syllables, word duration, clip SNR (pronunciada.quality)
and speaking rate (syllables per second of speech). Rows: $CACHE/round19/german/stress_rows.pkl.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from bench.round19.collect import manifest
from bench.round19.devset import WORK


def word_rows(res, wrong, item, ref_tag):
    c = res._pronunciada  # noqa: SLF001
    trk = c.get("f0")
    aq = c.get("audio") or {}
    times = trk.times if trk is not None else np.zeros(0)
    n_syl_utt = sum(len(w.syllables) for w in res.words)
    starts = [w.start for w in res.words if w.end > w.start]
    ends = [w.end for w in res.words if w.end > w.start]
    span = (max(ends) - min(starts)) if starts else 0.0
    rate = n_syl_utt / span if span > 0 else np.nan
    out = []
    for wi, w in enumerate(res.words):
        vals = [s.stress_prob for s in w.syllables if s.stress_prob is not None]
        if not vals:
            continue
        m = (times >= w.start) & (times < w.end) if len(times) else np.zeros(0, bool)
        out.append({"item": item["id"], "src": item["source"], "fold": item["fold"], "speaker": item["speaker"],
                    "ref": ref_tag, "wrong": bool(wrong[wi]), "text": w.text,
                    "vals": [float(v) for v in vals], "p_pros": w.stress_prob, "sounds": w.sounds_prob,
                    "present": w.present_prob, "prob": w.prob, "n_syl": len(w.syllables),
                    "dur": w.end - w.start,
                    "voiced_share": float(trk.voiced[m].mean()) if m.any() else np.nan,
                    "voicing": float(trk.voicing[m].mean()) if m.any() else np.nan,
                    "energy_db": float(trk.energy_db[m].mean()) if m.any() else np.nan,
                    "snr_db": aq.get("snr_db"), "rate": rate})
    return out


def main():
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada import Scorer

    sc = Scorer("de", threads=2)
    inv = inventory("de")
    rows = []
    t0 = time.time()
    items = [it for it in manifest("dev") if not it["label"]]
    for j, it in enumerate(items):
        try:
            res = sc.score(it["audio"], it["target"])
        except ValueError:
            continue
        rows += word_rows(res, [False] * len(res.words), it, "clean")
        ref = res._pronunciada["ref"]  # noqa: SLF001
        try:
            cor = corrupt(ref, ["stress"], inv, n_errors=2, seed=stable_seed(it["id"], "stress", "r19b"), strict=False)
        except NotApplicable:
            continue
        if not cor.errors:
            continue
        r2 = sc.score(it["audio"], cor.ref)
        rows += word_rows(r2, list(cor.word_wrong), it, "stress")
        if j % 100 == 0:
            print(j, len(items), f"{time.time() - t0:.0f}s", flush=True)
    (WORK / "stress_rows.pkl").write_bytes(pickle.dumps(rows))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
