"""Round 19: the confident-length flag threshold and the coaching thresholds of the round-19 length
model, on dev.

    python -m bench.round19_ja.lenflag [--length r19]

* coaching thresholds: the native quantiles of length_prob (dev-pool natives, every judged sound)
  at the native flag rates of the shipped "careful" / "balanced" / "eager" points (0.5 / 1.5 / 2.0 %);
* flag: a word is marked mispronounced when one of its judged sounds has length_prob <= T. For T on a
  grid: planted length units caught on dev, and the share of dev-pool native words newly flagged.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle

import numpy as np

from bench.round19_ja.cv_pool import WORK
from bench.round19_ja.stepgrid import unit_index


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", default="r19")
    a = ap.parse_args(argv)
    D = pickle.loads((WORK / f"sel2_{a.length}.pkl").read_bytes())
    nat_s = np.array([x for r in D["devnat"] for w in r["words"] for x in w["lens"]])
    p = 1 / (1 + np.exp(-nat_s))
    coach = {k: float(np.quantile(p, q)) for k, q in (("careful", 0.005), ("balanced", 0.015), ("eager", 0.02))}
    print("coach thresholds", coach)
    nat_w = np.array([min(w["lens"]) for r in D["devnat"] for w in r["words"] if w["lens"]])
    n_nat_words = sum(len(r["words"]) for r in D["devnat"])
    flat, units = unit_index(D["dev"])
    zmin = np.array([min(w["lens"]) if w["lens"] else 99.0 for w in flat])
    out = {"coach": coach, "grid": []}
    for T in (0.002, 0.005, 0.01, 0.0135, 0.02, 0.03, 0.05, 0.1):
        zt = math.log(T / (1 - T))
        caught = sum(1 for ws, lab, f, src, b in units if lab and b == "length" and any(zmin[i] <= zt for i in ws))
        n_len = sum(1 for ws, lab, f, src, b in units if lab and b == "length")
        fa = float(np.sum(nat_w <= zt) / n_nat_words)
        out["grid"].append({"T": T, "length_caught": caught, "n_length": n_len, "native_words_flagged": fa})
        print(f"T={T:<7} length units caught {caught}/{n_len}  native words flagged {fa:.4f}")
    (WORK / f"lenflag_{a.length}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
