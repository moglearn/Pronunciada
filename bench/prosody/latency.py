"""Latency of ProsodyScorer.score on a 10 s clip (F0 + features + windows + ONNX model).

    .venv/bin/python -m bench.prosody.latency [zh|ja|en ...]
PRONUNCIADA_LAT_THREADS sets the onnxruntime threads (default 2, as in Scorer); PRONUNCIADA_TONE_MODEL
picks the tone model. Uses a FLEURS test clip of the language tiled to 10 s, its reference repeated accordingly and the
cached ZIPA posteriors for the alignment (alignment time not included).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")

from bench import postcache  # noqa: E402
from bench.data import load  # noqa: E402
from pronunciada.align import score_reference  # noqa: E402
from pronunciada.audio import load_audio  # noqa: E402
from pronunciada.prosody import ProsodyScorer  # noqa: E402
from pronunciada.prosody.f0 import track  # noqa: E402
from pronunciada.prosody.features import frame_features  # noqa: E402
from pronunciada.text import get_g2p  # noqa: E402


def run(lang: str, n: int = 30) -> dict:
    g2p = get_g2p(lang)
    cands = [x for x in load("fleurs", split="test", lang=lang) if 9.5 < x["dur"] <= 10.0]
    r = max(cands, key=lambda x: len(g2p(x["text"]).syllables))    # densest ~10 s clip
    wav = load_audio(r["audio"])[:160000]
    wav = np.pad(wav, (0, 160000 - len(wav)))
    ref = g2p(r["text"])
    ev = score_reference(postcache.get_posteriors("fleurs", r["id"], r["audio"]), ref)
    sc = ProsodyScorer(lang, threads=int(os.environ.get("PRONUNCIADA_LAT_THREADS", "2")))
    sc.score(wav, ref, ev)
    tot, f0t = [], []
    for _ in range(n):
        t = time.perf_counter()
        sc.score(wav, ref, ev)
        tot.append(time.perf_counter() - t)
        t = time.perf_counter()
        frame_features(track(wav))
        f0t.append(time.perf_counter() - t)
    res = {"lang": lang, "n_syllables": len(ref.syllables), "total_ms_min": 1e3 * min(tot),
           "total_ms_median": 1e3 * float(np.median(tot)), "f0_features_ms_min": 1e3 * min(f0t),
           "f0_features_ms_median": 1e3 * float(np.median(f0t)), "loadavg_1min": os.getloadavg()[0]}
    print(res)
    return res


if __name__ == "__main__":
    for lang in sys.argv[1:] or ["zh"]:
        run(lang)
