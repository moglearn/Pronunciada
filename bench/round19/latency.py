"""Round 19: added latency of the English head, old and new scorers INTERLEAVED call by call (so
machine load affects both alike), after waiting for a quieter machine.

    python -m bench.round19.latency [--max-load 4] [--wait-min 45] [--n 150]

20 speechocean762 TRAIN clips closest to 3.5 s; 4 ONNX threads; fp32 and int8; median and the
median of paired differences. Writes results/tables/round19_english_latency.json.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np


def main():
    from bench.calib.common import save_json
    from bench.pipeline.common import records
    from bench.round19.runtime_check import scorer
    from pronunciada.audio import load_audio

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-load", type=float, default=4.0)
    ap.add_argument("--wait-min", type=float, default=45)
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    t0 = time.time()
    while os.getloadavg()[0] > a.max_load and time.time() - t0 < a.wait_min * 60:
        time.sleep(30)
    load0 = os.getloadavg()[0]
    recs = [r for r in records("speechocean762") if r["split"] == "train"][:200]
    items = sorted(((load_audio(r["audio"]), r["text"]) for r in recs), key=lambda x: abs(len(x[0]) / 16000 - 3.5))[:20]
    out = {"load_avg_1min_start": load0, "clip_s_median": float(np.median([len(w) / 16000 for w, _ in items]))}
    for prec in ("fp32", "int8"):
        old, new = scorer(False, precision=prec, threads=4), scorer(True, precision=prec, threads=4)
        for sc in (old, new):                         # warm-up
            for w, t in items[:3]:
                sc.score(w, t)
        to, tn, stage = [], [], []
        for k in range(a.n):
            w, t = items[k % len(items)]
            for sc, acc in ((old, to), (new, tn)) if k % 2 == 0 else ((new, tn), (old, to)):
                t1 = time.perf_counter()
                r = sc.score(w + 0.0, t)
                acc.append((time.perf_counter() - t1) * 1e3)
                if sc is new:
                    stage.append(r.timing_ms.get("sentence_head", np.nan))
        d = np.array(tn) - np.array(to)
        out[prec] = {"old_ms": float(np.median(to)), "new_ms": float(np.median(tn)),
                     "paired_diff_median_ms": float(np.median(d)), "sentence_head_stage_ms": float(np.nanmedian(stage))}
        print(prec, out[prec], flush=True)
        del old, new
    out["load_avg_1min_end"] = os.getloadavg()[0]
    print(out, flush=True)
    save_json("round19_english_latency.json", out)


if __name__ == "__main__":
    main()
