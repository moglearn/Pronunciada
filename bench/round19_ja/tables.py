"""Write the round-19 Japanese tables (results/tables/round19_ja_*.json) from the dumps.

    python -m bench.round19_ja.tables
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bench.round19_ja.cv_pool import WORK
from bench.round19_ja.length_fit import native_word_rates
from bench.round19_ja.unit_eval import load, paired, summary, units_of

ROOT = Path(__file__).resolve().parents[2]


def block(before, after, boot=2000):
    out = {}
    A, B = load(before), load(after)
    for src in ("all", "cv", "fleurs"):
        a = [r for r in A if src == "all" or r.get("source") == src]
        b = [r for r in B if src == "all" or r.get("source") == src]
        if not a:
            continue
        ra, rb = units_of(a), units_of(b)
        out[src] = {"before": summary(ra, boot), "after": summary(rb, boot), "paired": paired(ra, rb, boot)}
    out["latency_ms_median"] = {"before": float(np.median([r["ms"] for r in A if r.get("ms")])),
                                "after": float(np.median([r["ms"] for r in B if r.get("ms")]))}
    return out


def main():
    T = ROOT / "results" / "tables"
    lx = block(WORK / "lx_before.jsonl", WORK / "lx_after.jsonl")
    lx["note"] = ("Lingolix test 1 Japanese (200 clips), benchmark unit mapping re-implemented in "
                  "bench/round19_ja/unit_eval.py (reproduces the published 0.846 / 20.0 %); clip bootstrap. "
                  "before = PRONUNCIADA_JA_ROUND19=0 (round 18), after = frozen round-19 candidate 112446b.")
    (T / "round19_ja_lingolix.json").write_text(json.dumps(lx, indent=1, default=float))
    dev = block(WORK / "devF_base.jsonl", WORK / "devF_r19.jsonl", 1000)
    dev["ablation"] = {}
    for name, f in (("merge", "dev_merge.jsonl"), ("merge+labels", "devF_nowc.jsonl"),
                    ("min(shipped, recal)", "devF_min.jsonl"), ("geomean(shipped, recal)", "devF_geo.jsonl")):
        ra, rb = units_of(load(WORK / "devF_base.jsonl")), units_of(load(WORK / f))
        if len(ra) != len(rb):
            keep = {r["id"] for r in load(WORK / f)}
            ra = units_of([r for r in load(WORK / "devF_base.jsonl") if r["id"] in keep])
        dev["ablation"][name] = {"summary": summary(rb, 300), "paired_vs_base": paired(ra, rb, 500)}
    nat = {}
    for tag in ("base", "r19"):
        R = [r for r in load(WORK / f"devnat_{tag}.jsonl") if not r.get("error")]
        nat[tag] = native_word_rates(R)
    dev["natives_dev_pool"] = nat
    (T / "round19_ja_dev.json").write_text(json.dumps(dev, indent=1, default=float))
    print(json.dumps({"lingolix": {k: (v["before"]["auc"], v["after"]["auc"], v["before"]["fa80"], v["after"]["fa80"],
                                       v["paired"]) for k, v in lx.items() if k == "all"},
                      "latency": lx["latency_ms_median"]}, default=float, indent=1))


if __name__ == "__main__":
    main()
