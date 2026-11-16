"""Shared helpers for the CJK front-end tests."""
from __future__ import annotations

import time


def syl_ipa(ref):
    return ["".join(p.ipa for p in ref.phones[s.phones[0]:s.phones[1]]) for s in ref.syllables]


def tones(ref):
    return [s.tone for s in ref.syllables]


def speed_ms(g, sents, reps=5):
    for s in sents:
        g(s)
    t0 = time.perf_counter()
    for _ in range(reps):
        for s in sents:
            g(s)
    return 1000 * (time.perf_counter() - t0) / (reps * len(sents))
