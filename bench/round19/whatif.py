"""Round 19 (German): offline what-if on a dump: recompute every reference word's probability with a
function of its stored chain, rebuild the benchmark word list and re-run the metric.

The benchmark word list (to_azure) = reference words in order, with Insertion entries (score 0)
interleaved; a reference word's score is round(100 x prob, 1) / 100.
"""
from __future__ import annotations

import copy

from bench.round19.evaluate import _metrics, load_dump, manifest_for, units


def rebuild(rec: dict, fn) -> dict:
    rec = copy.copy(rec)
    vw = []
    it = iter(rec["rwords"])
    ctx = {"rec": rec}
    for w in rec["words"]:
        if w.get("error_type") == "Insertion":
            vw.append(w)
            continue
        rw = next(it)
        ctx["wi"] = ctx.get("wi", -1) + 1
        p = fn(rw, ctx)
        vw.append({**w, "score": round(min(1.0, max(0.0, p)) * 100, 1) / 100})
    rec["words"] = vw
    return rec


def apply(name: str, fn) -> dict:
    d = load_dump(name)
    return {k: (rebuild(v, fn) if not v.get("error") else v) for k, v in d.items()}


def metrics(name, fn=None, fold="all", source=None):
    items = manifest_for(name)
    if not name.startswith("lingolix"):
        items = [it for it in items if (fold == "all" or it["fold"] == fold) and (source is None or it["source"] == source)]
    d = apply(name, fn) if fn else load_dump(name)
    return _metrics(units(items, d))
