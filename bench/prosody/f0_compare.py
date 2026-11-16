"""Pitch trackers on FLEURS-zh validation (dev): ours (pronunciada.prosody.f0, YIN + Viterbi) vs
SwiftF0 (bench.prosody.f0_swift), both against Praat autocorrelation (the offline
reference of bench/prosody/f0_eval.py), on the 10 ms grid. Caveat: our tracker's voicing gate was
tuned against Praat on FLEURS validation (5 clips per language), which favours it on VDE here.

    .venv/bin/python -m bench.prosody.f0_compare [n_clips=60]

Metrics: GPE (both voiced, |f/f_ref - 1| > 20 %), octave errors (~2x / ~0.5x), FPE (std in cents
without gross errors), VDE (voicing decision differs), voiced share; plus the same against each
other; and the speed on a 10 s clip. Also on the noisier half of the clips (by an SNR proxy:
95th - 10th percentile of frame energy).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from bench.prosody.f0_eval import praat  # noqa: E402
from bench.prosody.prep import fleurs_validation  # noqa: E402
from pronunciada.prosody import f0 as F  # noqa: E402
from bench.prosody import f0_swift as S  # noqa: E402


def cmp(f, v, ref, rv):
    both = v & rv
    ratio = f[both] / ref[both]
    gross = np.abs(ratio - 1) > 0.2
    octv = (np.abs(ratio - 2) < 0.3) | (np.abs(ratio - 0.5) < 0.08)
    return dict(both=int(both.sum()), gross=int(gross.sum()), octave=int(octv.sum()),
                vde=int((v != rv).sum()), frames=len(rv), voiced=int(v.sum()),
                cents=1200 * np.log2(ratio[~gross]))


def summ(rows):
    k = lambda key: sum(r[key] for r in rows)  # noqa: E731
    c = np.concatenate([r["cents"] for r in rows])
    return {"GPE": k("gross") / max(k("both"), 1), "octave": k("octave") / max(k("both"), 1),
            "FPE_cents": float(np.std(c)), "VDE": k("vde") / k("frames"), "voiced": k("voiced") / k("frames")}


def main(n=60):
    recs = []
    for r in fleurs_validation("zh"):
        recs.append(r)
    recs = recs[::max(1, len(recs) // n)][:n]
    res = {"ours": [], "swift": [], "swift_vs_ours": []}
    snr, pref = [], []
    for r in recs:
        w = r["wav"]
        a = F.track(w)
        b = S.track(w)
        ref = praat(w, len(a.f0))
        rv = ref > 0
        res["ours"].append(cmp(a.f0, a.voiced, ref, rv))
        res["swift"].append(cmp(b.f0, b.voiced, ref, rv))
        res["swift_vs_ours"].append(cmp(b.f0, b.voiced, a.f0, a.voiced))
        snr.append(np.percentile(a.energy_db, 95) - np.percentile(a.energy_db, 10))
        pref.append(rv.mean())
    noisy = np.argsort(snr)[:len(snr) // 2]
    print(f"{len(recs)} FLEURS-zh validation clips, {sum(len(r['wav']) for r in recs) / 16000:.0f} s; "
          f"Praat voiced share {np.mean(pref):.3f}")
    for k, rows in res.items():
        print(k, "all:", {a: round(b, 4) for a, b in summ(rows).items()},
              "| noisier half:", {a: round(b, 4) for a, b in summ([rows[i] for i in noisy]).items()})
    w = np.concatenate([r["wav"] for r in recs])[:160000]
    for name, fn in (("ours", lambda: F.track(w)), ("swift 1 thread", lambda: S.track(w, 1)),
                     ("swift 2 threads", lambda: S.track(w, 2))):
        fn()
        ts = []
        for _ in range(15):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1e3)
        print(f"10 s clip, {name}: min {min(ts):.1f} / median {np.median(ts):.1f} ms (load {os.getloadavg()[0]:.1f})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
