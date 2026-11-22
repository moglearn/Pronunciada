"""Round 19 (Korean): the runtime factor equals the offline one, and what it costs.

    python -m bench.round19ko.verify_runtime [--n 60]

Scores n dev items (CV audio files) with the public Scorer, PRONUNCIADA_KO_IDENTITY on, and compares
the word probabilities with the offline computation from the factor-off rows (bench.round19ko.guards
after_probs). Latency: median total ms of a 3-4 s clip with the switch on vs off (4 threads, 30 runs).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    from bench.round19ko import collect as CO
    from bench.round19ko.guards import after_probs
    from pronunciada import Scorer

    R = CO.load("dev", "base")
    keys = [k for k, r in R.items() if r["meta"]["src"] == "cv" and "error" not in r][: a.n]
    sub = {k: R[k] for k in keys}
    off = after_probs(sub)
    sc = Scorer("ko", threads=2)
    items = {json.loads(x)["item"]: json.loads(x) for x in (CO.work() / "dev_items.jsonl").read_text(encoding="utf-8").splitlines()}
    d = []
    for k in keys:
        res = sc.score(items[k]["audio"], items[k]["target"])
        d.append(np.max(np.abs(np.array([w.prob for w in res.words]) - off[k])))
    rep = {"n": len(keys), "max_abs_diff": float(np.max(d))}
    # latency on one ~3.5 s clip, 4 threads
    clip = min((it for it in items.values() if it["src"] == "cv" and it.get("dur") and 3.0 <= it["dur"] <= 4.0),
               key=lambda it: abs(it["dur"] - 3.5))
    lat = {}
    for flag in ("0", "1"):
        os.environ["PRONUNCIADA_KO_IDENTITY"] = flag
        s4 = Scorer("ko", threads=4)
        s4.score(clip["audio"], clip["target"])
        ts = []
        for _ in range(30):
            s4._lp_cache.clear()
            t = time.perf_counter()
            s4.score(clip["audio"], clip["target"])
            ts.append((time.perf_counter() - t) * 1e3)
        lat[flag] = float(np.median(ts))
    rep["latency_ms_off_on"] = [lat["0"], lat["1"]]
    rep["clip_dur_s"] = clip["dur"]
    Path("results/tables/round19_ko_runtime.json").write_text(json.dumps(rep, indent=1))
    print(rep)


if __name__ == "__main__":
    main()
