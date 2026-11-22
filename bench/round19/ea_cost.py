"""Round 19b (German): cost of the bounded explain-away pass.

    PRONUNCIADA_EXPLAIN_LANGS=1 PRONUNCIADA_EA_BAND=0,1 PRONUNCIADA_EA_PASSES=1 python -m bench.round19.ea_cost

Uses the pass's own timer (SentenceResult.timing_ms["explain_away"], 4 threads) on dev clips of
9-11 s (worst case) and of 3-4.5 s, changed items (a pass is forced by the band [0, 1)); reports the
median and the 95th percentile. The average added time on a realistic mix is then
(share of clips whose lowest word falls in the chosen band) x (median cost), measured on dev.
"""
from __future__ import annotations

import statistics

from bench.round19.collect import manifest


def main():
    from pronunciada import Scorer

    sc = Scorer("de", threads=4, explain_away=True)
    items = manifest("dev")
    for lo, hi in ((9.0, 11.0), (3.0, 4.5)):
        its = [it for it in items if it["label"] and lo <= it["dur"] <= hi][:40]
        ts = []
        for it in its:
            res = sc.score(it["audio"], it["target"])
            if "explain_away" in res.timing_ms:
                ts.append(res.timing_ms["explain_away"])
        ts.sort()
        if ts:
            print(f"{lo}-{hi} s: n {len(ts)} median {statistics.median(ts):.1f} ms p95 {ts[int(0.95 * (len(ts) - 1))]:.1f} ms max {ts[-1]:.1f}")


if __name__ == "__main__":
    main()
