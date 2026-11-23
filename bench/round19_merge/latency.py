"""Round 19 merge: paired latency of two source trees (old = main, new = merged), interleaved.

    python bench/round19_merge/latency.py --old <tree> --new <tree> --lang ko --set fleurs-val \
        --precision fp32 --threads 4 --n 100 --out ko.json

Two worker processes, one per tree (each imports pronunciada from its own tree), each holding one
warm Scorer(lang, precision, threads). The parent sends every clip to both, one after the other
(the order alternates clip by clip), so only one scorer runs at any moment and slow phases of the
machine hit both sides alike. Each worker times Scorer.score(wav, text) with the waveform already
in memory (no file decoding) after 3 warm-up calls. Reports the median per side, the median of the
paired differences with a clip-bootstrap 95 % interval, and each worker's peak resident memory.
Clips: the dev / validation sets of bench/round19_merge/equiv.py (own text only).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np


def _worker(tree, env, lang, precision, threads, conn):
    os.environ.update(env)
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    sys.path.insert(0, tree)
    import resource

    import pronunciada
    from pronunciada import Scorer

    assert os.path.realpath(os.path.dirname(os.path.dirname(pronunciada.__file__))) == os.path.realpath(tree)
    sc = Scorer(lang, precision=precision, threads=threads)
    emb = sorted(getattr(sc.am, "embed_outputs", {}) or ({"round5": 1} if sc.am.embeddings else {}))
    conn.send({"ready": True, "embed_outputs": emb, "n_outputs": len(sc.am._out_names)})  # noqa: SLF001
    while True:
        msg = conn.recv()
        if msg is None:
            break
        wav, text = msg
        try:
            sc._lp_cache.clear()      # noqa: SLF001  (no posterior reuse between calls)
        except AttributeError:
            pass
        t = time.perf_counter()
        try:
            sc.score((wav, 16000), text)
            ok = True
        except ValueError:
            ok = False
        conn.send(((time.perf_counter() - t) * 1e3, ok))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    conn.send({"max_rss_mb": rss / 2**20 if sys.platform == "darwin" else rss / 2**10})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old-env", default="", help="KEY=VAL,KEY=VAL for the old side")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--set", default="fleurs-val")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.new)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from equiv import _clips

    clips = [(cid, wav, text) for cid, wav, text in _clips(a.set, a.lang, a.n) if not cid.endswith("#swap")]
    ctx = mp.get_context("spawn")
    env_old = dict(kv.split("=", 1) for kv in a.old_env.split(",") if kv)
    sides = {}
    for name, tree, env in (("old", a.old, env_old), ("new", a.new, {})):
        p_conn, c_conn = ctx.Pipe()
        pr = ctx.Process(target=_worker, args=(tree, env, a.lang, a.precision, a.threads, c_conn))
        pr.start()
        info = p_conn.recv()
        sides[name] = {"proc": pr, "conn": p_conn, "info": info}
    for _ in range(3):                       # warm-up
        for name in ("old", "new"):
            sides[name]["conn"].send((clips[0][1], clips[0][2]))
            sides[name]["conn"].recv()
    ms = {"old": [], "new": []}
    t0 = time.time()
    load0 = os.getloadavg()[0]
    for k, (_, wav, text) in enumerate(clips):
        order = ("old", "new") if k % 2 == 0 else ("new", "old")
        for name in order:
            sides[name]["conn"].send((wav, text))
            ms[name].append(sides[name]["conn"].recv())
    out = {"lang": a.lang, "set": a.set, "precision": a.precision, "threads": a.threads, "n": len(clips),
           "old_tree": a.old, "new_tree": a.new, "old_env": env_old,
           "clip_seconds_median": float(np.median([len(c[1]) / 16000 for c in clips])),
           "load_avg_1min_start_end": [load0, os.getloadavg()[0]], "wall_s": time.time() - t0}
    for name in ("old", "new"):
        sides[name]["conn"].send(None)
        out[f"{name}_max_rss_mb"] = sides[name]["conn"].recv()["max_rss_mb"]
        out[f"{name}_embed_outputs"] = sides[name]["info"]["embed_outputs"]
        sides[name]["proc"].join()
    o, n = np.array([x for x, _ in ms["old"]]), np.array([x for x, _ in ms["new"]])
    d = n - o
    rng = np.random.default_rng(0)
    boots = [float(np.median(d[rng.integers(0, len(d), len(d))])) for _ in range(2000)]
    out.update(old_median_ms=float(np.median(o)), new_median_ms=float(np.median(n)),
               paired_median_diff_ms=float(np.median(d)),
               paired_median_diff_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
               old_p90_ms=float(np.percentile(o, 90)), new_p90_ms=float(np.percentile(n, 90)))
    mboots = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(2000)]
    out.update(paired_mean_diff_ms=float(np.mean(d)),       # round 20: the mean too (rare costly clips)
               paired_mean_diff_ci=[float(np.percentile(mboots, 2.5)), float(np.percentile(mboots, 97.5))])
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if "tree" not in k}))


if __name__ == "__main__":
    main()
