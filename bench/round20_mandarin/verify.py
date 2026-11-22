"""Round 20 (Mandarin): the runtime rule equals the offline rule on dev items.

    python -m bench.round20_mandarin.verify [--n 80]

Scores n dev items (every 30th of the round-19 zh dev manifest) through the public Scorer with the
round-20 rule on and with PRONUNCIADA_IDENTITY_ZH_R20=0, and compares word P with the offline
computation of bench.round20_mandarin.dev (stored evidence and factors). Writes
results/tables/round20_mandarin_verify.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    a = ap.parse_args(argv)
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    from bench.round19_rollout.devset import manifest
    from bench.round19_rollout.fit import load_ev
    from bench.round20_mandarin import dev as D
    from bench.round20_mandarin.collect import load as load_extra
    from pronunciada import Scorer

    cal = D.r19_cal()
    arms = {"r19": {k: v for k, v in cal.items() if k != "round20"},
            "new": {**{k: v for k, v in cal.items() if k != "round20"}, **cal["round20"]}}
    ev = {r["id"]: r for r in D.attach(load_ev("zh", "dev"), load_extra("zh", "dev"))}
    items = [it for it in manifest("zh") if it["id"] in ev][::30][: a.n]
    sc = Scorer("zh", threads=2)
    worst = {k: 0.0 for k in arms}
    for it in items:
        r = ev[it["id"]]
        for arm, env in (("r19", "0"), ("new", "1")):
            os.environ["PRONUNCIADA_IDENTITY_ZH_R20"] = env
            p_rt = np.array([w.prob for w in sc.score(it["audio"], it["target"]).words])
            p_off = D.new_probs(r, arms[arm])
            worst[arm] = max(worst[arm], float(np.abs(p_rt - p_off).max()))
    out = {"n_items": len(items), "max_abs_diff_runtime_vs_offline": worst}
    print(out)
    Path("results/tables/round20_mandarin_verify.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
