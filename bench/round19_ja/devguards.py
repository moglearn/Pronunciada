"""Dev metrics and dev guards from runtime dumps (bench/round19_ja/dump.py) of the four dev sets.

    python -m bench.round19_ja.devguards TAG [TAG ...]     # files <set>_<TAG>.jsonl; first TAG = reference

Sets: dev (set_dev), devnat (dev-pool natives), devswap (FLEURS-swap dev: orig / near / far),
cvdev (cv-real ja Test A dev half). Round 18 = the *_base files (dev: devF_base).
Writes results/tables/round19_ja_devguards.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bench.round19_ja.cv_pool import WORK

FILES = {"base": {"dev": "devF_base", "devnat": "devnat_base", "devswap": "devswap_base", "cvdev": "cvdev_base"}}


def load_tag(tag):
    from bench.round19_ja.unit_eval import load

    names = FILES.get(tag) or {k: f"{k}2_{tag}" for k in ("dev", "devnat", "devswap", "cvdev")}
    D = {}
    for k, f in names.items():
        recs = [r for r in load(WORK / f"{f}.jsonl") if not r.get("error")]
        D[k] = [{**r, "words": [{"text": w["text"], "span": w["span"], "base": w["prob"], "zlen": None, "zid": None,
                                 "mm": None} for w in r["words"]]} for r in recs]
    return D


def main(argv=None):
    from bench.round19_ja.select2 import guard_ok, metrics

    tags = (argv if argv is not None else sys.argv[1:]) or ["base", "r19"]
    out = {}
    ref = None
    for t in tags:
        m = metrics(load_tag(t), {}, None)
        out[t] = {"m": m}
        if ref is None:
            ref = m
        else:
            out[t]["guards_ok_vs_" + tags[0]] = guard_ok(m, ref)
        print(t, json.dumps({k: round(v, 4) for k, v in m.items()}), out[t].get("guards_ok_vs_" + tags[0], ""))
    p = Path(__file__).resolve().parents[2] / "results" / "tables" / "round19_ja_devguards.json"
    p.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
