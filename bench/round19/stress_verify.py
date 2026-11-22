"""Round 19 (German): checks of the reliability-weighted stress factor.

    python -m bench.round19.stress_verify

On 60 dev control clips, WordResult.stress_prob, every SyllableResult.stress_prob and the coaching
items that mention stress (result.coaching()) are compared with the switch on and off
(PRONUNCIADA_DE_STRESS_REL); the number of words whose WordResult.prob changed is reported.
"""
from __future__ import annotations

import os

import numpy as np

from bench.round19.collect import manifest


def run():
    import importlib

    from pronunciada import stress_rel as SR
    from pronunciada.scorer import Scorer

    items = [it for it in manifest("dev") if not it["label"]][::12][:60]
    out = {}
    for mode in ("1", "0"):
        os.environ["PRONUNCIADA_DE_STRESS_REL"] = mode
        importlib.reload(SR)
        sc = Scorer("de", threads=2)
        assert (sc.stress_rel is not None) == (mode == "1")
        rows = []
        for it in items:
            res = sc.score(it["audio"], it["target"])
            co = res.coaching()
            stress_items = [(i.get("kind") if isinstance(i, dict) else getattr(i, "kind", None), str(i))
                            for i in (co.get("items", []) if isinstance(co, dict) else co)
                            if "stress" in str(i)]
            rows.append({"word_stress": [w.stress_prob for w in res.words],
                         "syl_stress": [s.stress_prob for w in res.words for s in w.syllables],
                         "prob": [w.prob for w in res.words], "stress_items": stress_items})
        out[mode] = rows
    same_w = all(a["word_stress"] == b["word_stress"] for a, b in zip(out["1"], out["0"]))
    same_s = all(a["syl_stress"] == b["syl_stress"] for a, b in zip(out["1"], out["0"]))
    same_c = all(a["stress_items"] == b["stress_items"] for a, b in zip(out["1"], out["0"]))
    n_items = sum(len(a["stress_items"]) for a in out["0"])
    diff = sum(1 for a, b in zip(out["1"], out["0"]) for x, y in zip(a["prob"], b["prob"]) if abs(x - y) > 1e-9)
    nw = sum(len(a["prob"]) for a in out["0"])
    print(f"clips {len(items)}: word stress_prob identical {same_w}; syllable stress_prob identical {same_s}; "
          f"coaching stress items identical {same_c} ({n_items} items); word P changed on {diff} of {nw} words")


if __name__ == "__main__":
    run()
