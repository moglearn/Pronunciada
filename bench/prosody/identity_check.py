"""Byte-identity check of Scorer outputs for languages a change must not touch.

    .venv/bin/python -m bench.prosody.identity_check dump before   [--langs zh,ja,fr,ko] [--n 5]
    .venv/bin/python -m bench.prosody.identity_check dump after
    .venv/bin/python -m bench.prosody.identity_check compare before after

Scores the first n FLEURS validation clips per language through the public Scorer and stores a
canonical view of every float (sentence prob / scores, word prob / sounds / tone / stress / pitch /
present, phone prob) as raw float64 bytes; compare reports the SHA-256 per language and whether
the two dumps are byte-identical.
"""
from __future__ import annotations

import hashlib
import pickle
import sys

import numpy as np


def view(res):
    v = [res.prob] + [res.scores.get(k) for k in sorted(res.scores or {})]
    for w in res.words:
        v += [w.prob, w.sounds_prob, w.tone_prob, w.stress_prob, w.present_prob]
        v += [s.tone_prob for s in w.syllables] + [s.stress_prob for s in w.syllables] + [s.pitch_prob for s in w.syllables]
        v += [p.prob for p in w.phones]
    return np.array([np.nan if x is None else float(x) for x in v], np.float64).tobytes()


def dump(tag, langs, n):
    from bench.calib.common import calib_cache
    from bench.prosody.prep import fleurs_validation
    from pronunciada.scorer import Scorer

    out = {}
    for lang in langs:
        sc = Scorer(lang, threads=2)
        out[lang] = []
        for k, r in zip(range(n), fleurs_validation(lang)):
            out[lang].append((r["id"], view(sc.score((r["wav"], 16000), r["text"]))))
    (calib_cache() / f"identity_{tag}.pkl").write_bytes(pickle.dumps(out))
    for lang, v in out.items():
        print(lang, len(v), hashlib.sha256(b"".join(b for _, b in v)).hexdigest()[:16])


def compare(a, b):
    from bench.calib.common import calib_cache

    A = pickle.loads((calib_cache() / f"identity_{a}.pkl").read_bytes())
    B = pickle.loads((calib_cache() / f"identity_{b}.pkl").read_bytes())
    for lang in A:
        ha = hashlib.sha256(b"".join(x for _, x in A[lang])).hexdigest()[:16]
        hb = hashlib.sha256(b"".join(x for _, x in B[lang])).hexdigest()[:16]
        print(f"{lang}: {len(A[lang])} clips, {sum(len(x) // 8 for _, x in A[lang])} values; {ha} vs {hb}: "
              f"{'IDENTICAL' if A[lang] == B[lang] else 'DIFFERENT'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "dump":
        langs = next((x.split("=")[1] for x in a if x.startswith("--langs=")), "zh,ja,fr,ko").split(",")
        n = int(next((x.split("=")[1] for x in a if x.startswith("--n=")), 5))
        dump(a[1], langs, n)
    else:
        compare(a[1], a[2])
