"""Round 19 (German): end-to-end latency of Scorer("de").score on short dev clips (4 threads).

    PRONUNCIADA_DE_R19=0 python -m bench.round19.latency_de      # before
    python -m bench.round19.latency_de                            # after

40 control and 40 changed dev items with 3-4.5 s of audio; 3 warm-up calls; each item scored 3
times, the median kept; prints the median over items, split by whether the explain-away pass ran.
"""
from __future__ import annotations

import statistics
import time

from bench.round19.collect import manifest


def main():
    from pronunciada import Scorer

    items = [it for it in manifest("dev") if 3.0 <= it["dur"] <= 4.5]
    ctl = [it for it in items if not it["label"]][:40]
    chg = [it for it in items if it["label"]][:40]
    sc = Scorer("de", threads=4)
    for it in ctl[:3]:
        sc.score(it["audio"], it["target"])
    out = {"control": [], "changed": []}
    passes = {"control": 0, "changed": 0}
    for name, its in (("control", ctl), ("changed", chg)):
        for it in its:
            ts = []
            for _ in range(3):
                sc._lp_cache.clear()                     # full pipeline each time
                t = time.perf_counter()
                res = sc.score(it["audio"], it["target"])
                ts.append((time.perf_counter() - t) * 1000)
            passes[name] += int(bool(res._pronunciada.get("explained")))
            out[name].append(statistics.median(ts))
    for k, v in out.items():
        print(f"{k}: median {statistics.median(v):.1f} ms, mean {statistics.mean(v):.1f} ms, "
              f"explain-away pass ran on {passes[k]}/{len(v)}")


if __name__ == "__main__":
    main()
