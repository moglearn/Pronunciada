"""Round 20 (latency): short clips cut from FLEURS validation, for timing on ~3.5 s audio.

    python bench/round20_latency/make_cut.py --lang ko [--n 100] [--target 3.5]

The clips of the fleurs-val set of bench/round19_merge/equiv.py (seed 19; own text) are scored once;
each is cut in the pause after the last word that ends by `target` + 0.3 s (cut = middle of the gap
to the next word's start), and the text is cut to the same words. Only languages that separate
words with spaces, and only clips whose reference has one word per space-separated token, are used;
cuts outside 2.5-4.5 s are dropped. Writes $PRONUNCIADA_CACHE/round20/latency/fleurs_cut_<lang>.jsonl
(id, parquet, row, cut in samples, text), read by the fleurs-val-cut set of equiv.py.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--target", type=float, default=3.5)
    a = ap.parse_args()
    from bench.pipeline.common import cache_root, records
    from pronunciada import Scorer

    rng = np.random.default_rng(19)
    recs = records("fleurs-validation", lang=a.lang)
    pick = sorted(rng.choice(len(recs), min(a.n, len(recs)), replace=False))
    recs = [recs[i] for i in pick]
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    from bench.pipeline.common import iter_parquet_audio

    sc = Scorer(a.lang, threads=2)
    out = []
    for i, wav in iter_parquet_audio(path, set(by)):
        r = by[i]
        toks = r["text"].split()
        res = sc.score((wav, 16000), r["text"])
        if len(res.words) != len(toks):
            continue
        k = max((j + 1 for j, w in enumerate(res.words) if w.end <= a.target + 0.3), default=0)
        if k < 2 or k >= len(toks):
            continue
        cut = 0.5 * (res.words[k - 1].end + res.words[k].start)
        if not 2.5 <= cut <= 4.5:
            continue
        out.append({"id": r["id"] + "#cut", "parquet": path, "row": i, "cut": int(round(cut * 16000)),
                    "text": " ".join(toks[:k])})
    dst = cache_root() / "round20" / "latency" / f"fleurs_cut_{a.lang}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    cuts = [o["cut"] / 16000 for o in out]
    print(json.dumps({"lang": a.lang, "clips": len(out), "median_s": float(np.median(cuts)) if cuts else None,
                      "out": str(dst)}))


if __name__ == "__main__":
    main()
