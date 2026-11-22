"""Round 20 (Mandarin): end-to-end latency, round-20 rule against round 19, interleaved, 4 threads.

    python -m bench.round20_mandarin.latency [--n 40] [--reps 2]

FLEURS validation zh clips of 2.5-4.5 s (round-19 dev manifest audio), each scored with
PRONUNCIADA_IDENTITY_ZH_R20=1 and =0 in alternating order after one warm-up; medians and p95 of the
SentenceResult total time. Writes results/tables/round20_mandarin_latency.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args(argv)
    from bench.round19_rollout.devset import manifest
    from pronunciada import Scorer
    from pronunciada.audio import load_audio

    items = [it for it in manifest("zh") if it["label"] is None and 2.5 <= it["dur"] <= 4.5][: a.n]
    sc = Scorer("zh", threads=4)
    T = {"new": [], "r19": []}
    for it in items:
        wav = load_audio(it["audio"])
        sc.score((wav, 16000), it["target"])
        for k in range(a.reps):
            order = ("new", "r19") if k % 2 == 0 else ("r19", "new")
            for arm in order:
                os.environ["PRONUNCIADA_IDENTITY_ZH_R20"] = "1" if arm == "new" else "0"
                t = time.perf_counter()
                sc.score((wav, 16000), it["target"])
                T[arm].append((time.perf_counter() - t) * 1e3)
    out = {arm: {"median_ms": float(np.median(v)), "p95_ms": float(np.quantile(v, 0.95)), "n": len(v)} for arm, v in T.items()}
    out["n_clips"] = len(items)
    print(out)
    Path("results/tables/round20_mandarin_latency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
