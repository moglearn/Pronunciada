"""Round 19 rollout: the shipped runtime equals the offline rule, the switch restores the old scores,
and the added latency.

    python -m bench.round19_rollout.verify --lang fr [--n 80]

On the first n dev items (planted changes and controls): word P with the switch off must equal the stored
collection P; with it on, it must equal fit.new_probs(stored row) (max |dP| reported). Latency: median
end-to-end time of Scorer.score on CV dev clips of 2.5-4.5 s (4 threads), on and off, interleaved.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--n", type=int, default=80)
    a = ap.parse_args(argv)
    from bench.round19_rollout import fit as FT
    from bench.round19_rollout.devset import manifest
    from pronunciada import Scorer
    from pronunciada import identity as ID

    env = f"PRONUNCIADA_IDENTITY_{a.lang.upper()}"
    cal = ID._cal(a.lang)
    ev = {r["id"]: r for r in FT.load_ev(a.lang, "dev")}
    items = manifest(a.lang)[: a.n]
    sc = Scorer(a.lang, threads=2)
    d_off, d_on = [], []
    for it in items:
        r = ev.get(it["id"])
        if r is None:
            continue
        os.environ[env] = "0"
        p0 = np.array([w.prob for w in sc.score(it["audio"], it["target"]).words])
        os.environ[env] = "1"
        p1 = np.array([w.prob for w in sc.score(it["audio"], it["target"]).words])
        d_off.append(np.max(np.abs(p0 - r["prob"])))
        d_on.append(np.max(np.abs(p1 - FT.new_probs(r, cal, cal.get("offset", {})))))
    out = {"n": len(d_on), "max_abs_dP_off_vs_stored": float(np.max(d_off)), "max_abs_dP_on_vs_offline": float(np.max(d_on))}
    # latency, 4 threads, 2.5-4.5 s clips, interleaved on / off
    sc4 = Scorer(a.lang, threads=4, cache_size=1)
    clips = [it for it in manifest(a.lang) if it["source"] == "cv" and 2.5 <= it["dur"] <= 4.5 and not it["label"]][:40]
    t = {"0": [], "1": []}
    for rep in range(2):
        for it in clips:
            for v in ("1", "0") if rep else ("0", "1"):
                os.environ[env] = v
                sc4._lp_cache.clear()
                t0 = time.perf_counter()
                sc4.score(it["audio"], it["target"])
                t[v].append((time.perf_counter() - t0) * 1e3)
    out.update(latency_ms_off=float(np.median(t["0"])), latency_ms_on=float(np.median(t["1"])), n_latency=len(t["0"]))
    # the head alone (evidence for every sound of the clip + word rule), on stored dev vectors
    from bench.round19_rollout.collect import iter_rows
    from bench.round19_rollout.train import Ref

    ts = []
    for k, r in enumerate(iter_rows(a.lang, "dev")):
        if k >= 200:
            break
        V = r["V"].astype(np.float32)
        t0 = time.perf_counter()
        s = ID.evidence(a.lang, Ref(r["ph"]), V)
        for w in range(len(r["prob"])):
            ID.apply_word(cal, s[np.asarray(r["word"]) == w], 0.9, 0.5)
        ts.append((time.perf_counter() - t0) * 1e3)
    out.update(head_ms_median=float(np.median(ts)), head_ms_p95=float(np.quantile(ts, 0.95)))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
