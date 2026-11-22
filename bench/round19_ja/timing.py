"""Latency of the round-19 parts: the same clips scored with round 18 and with round 19, 4 threads,
each clip scored twice and the second (warm) timing kept; median per stage.

    python -m bench.round19_ja.timing IN.jsonl [-n 60]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    import warnings

    warnings.simplefilter("ignore")
    from bench.round19_ja.dump import _wav
    from pronunciada.scorer import Scorer

    rows = [json.loads(x) for x in open(a.inp, encoding="utf-8")][:a.n]
    wavs = [(_wav(r["audio"]), r["text"]) for r in rows]
    out = {}
    for tag, val in (("round18", "0"), ("round19", "1")):
        os.environ["PRONUNCIADA_JA_ROUND19"] = val
        sc = Scorer("ja", threads=a.threads)
        T = {}
        for w, t in wavs:
            for _ in range(2):
                sc._lp_cache.clear()  # noqa: SLF001
                t0 = time.perf_counter()
                r = sc.score((w, 16000), t)
                tot = (time.perf_counter() - t0) * 1e3
            for k, v in {**r.timing_ms, "wall": tot}.items():
                T.setdefault(k, []).append(v)
        out[tag] = {k: float(np.median(v)) for k, v in T.items()}
        del sc
    for k in sorted(set(out["round18"]) | set(out["round19"])):
        print(f"{k:14s} {out['round18'].get(k, 0):8.2f} {out['round19'].get(k, 0):8.2f}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
