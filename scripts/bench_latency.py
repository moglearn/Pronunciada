"""Latency of the acoustic core: fbank, ZIPA (fp32/fp16/int8 x threads), lattice evidence.

    .venv/bin/python scripts/bench_latency.py [--threads 1 2 4] [--reps 5] [--no-exact]

Clips (~3 s, ~10 s, ~20 s) are synthesised with macOS `say` (+ ffmpeg); the reference for the
aligner is the model's own greedy phone string. Speech rate 125 wpm (~10 phones/s).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pronunciada.acoustic import PRECISION_FILES, AcousticModel, cache_root  # noqa: E402
from pronunciada.align import score_reference  # noqa: E402
from pronunciada.audio import fbank, load_audio  # noqa: E402
from pronunciada.text.ipa import segment  # noqa: E402
from pronunciada.types import RefPhone, Reference  # noqa: E402

TEXTS = {
    "3s": "I would like a cup of coffee with milk today, please.",
    "10s": ("Yesterday the committee announced a remarkable new plan for public transport "
            "across the whole region, starting next spring, with more buses and trains."),
    "20s": ("Yesterday the committee announced a remarkable new plan for public transport "
            "across the whole region, starting next spring, with more buses and trains. "
            "Local residents welcomed the decision, although several shop owners worried "
            "about the long construction period and the closure of the main street."),
}


def synth(text: str, d: str) -> np.ndarray:
    aiff, wav = os.path.join(d, "x.aiff"), os.path.join(d, "x.wav")
    subprocess.run(["say", "-r", "125", "-o", aiff, text], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", aiff, "-ar", "16000", "-ac", "1", wav],
                   check=True)
    return load_audio(wav)


def best_ms(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1000 * min(ts), 1000 * float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-exact", action="store_true")
    a = ap.parse_args()
    if not (shutil.which("say") and shutil.which("ffmpeg")):
        sys.exit("needs macOS `say` and ffmpeg")

    with tempfile.TemporaryDirectory() as d:
        clips = {k: synth(t, d) for k, t in TEXTS.items()}

    name = "zipa-small-crctc-500k"
    precisions = [p for p, f in PRECISION_FILES.items()
                  if os.path.exists(os.path.join(cache_root(), "models", name, f))]
    print(f"clips: " + ", ".join(f"{k}={len(w) / 16000:.2f}s" for k, w in clips.items()))

    print("\nmodel load (s):")
    models = {}
    for p in precisions:
        for th in a.threads:
            t = time.perf_counter()
            models[p, th] = AcousticModel(name, precision=p, threads=th)
            if th == a.threads[0]:
                print(f"  {p:5s} {time.perf_counter() - t:.2f}")

    print("\nfbank ms (min / median):")
    for k, w in clips.items():
        print(f"  {k:4s} {best_ms(lambda: fbank(w), a.reps)[0]:7.2f} / {best_ms(lambda: fbank(w), a.reps)[1]:7.2f}")

    print("\nmodel ms (min over reps; fbank excluded):")
    print(f"  {'prec':5s} {'thr':>3s} " + " ".join(f"{k:>8s}" for k in clips))
    for (p, th), m in models.items():
        row = []
        for k, w in clips.items():
            x = fbank(w)
            m.posteriors_from_features(x)  # warm-up
            row.append(best_ms(lambda: m.posteriors_from_features(x), a.reps)[0])
        print(f"  {p:5s} {th:3d} " + " ".join(f"{v:8.1f}" for v in row))

    print("\nalign ms (reference = greedy phones; fast min/median, exact single run):")
    m = models["fp32", max(a.threads)]
    for k, w in clips.items():
        lp = m.posteriors(w)
        phones = segment("".join(t for t, _, _ in m.greedy(lp)))
        ref = Reference("en", "", phones=[RefPhone(p, 0, 0) for p in phones])
        fmin, fmed = best_ms(lambda: score_reference(lp, ref), a.reps)
        line = f"  {k:4s} T={len(lp):4d} phones={len(phones):3d} fast {fmin:6.1f} / {fmed:6.1f}"
        if not a.no_exact:
            t = time.perf_counter()
            ex = score_reference(lp, ref, exact=True)
            te = 1000 * (time.perf_counter() - t)
            fa = score_reference(lp, ref)
            err = max(max(abs(x.llr_sub - y.llr_sub), abs(x.llr_del - y.llr_del))
                      for x, y in zip(fa.slots, ex.slots))
            line += f"   exact {te:8.0f}   max|fast-exact| {err:.1e}"
        print(line)


if __name__ == "__main__":
    main()
