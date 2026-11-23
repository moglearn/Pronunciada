"""How well does the presence (omission) factor separate unsaid words from reduced native words?

    python -m bench.round20_native_fa.presence --lang en,de,ko,es,fr

Per word: the lowest WordResult.present_prob over the token's front-end words. Groups: the inserted unsaid word
of word-omission items (should be low), correct function words that are not next to a changed word, correct
content words likewise, and the neighbours of an inserted word. Reports quantiles and the share below 0.5.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from bench.round20_native_fa import core as K
from bench.round20_native_fa.rebuild import ROOT


def run(lang: str) -> dict:
    rows = [json.loads(l) for l in (ROOT / "tables" / f"rows_{lang}.jsonl").read_text(encoding="utf-8").splitlines()]
    dump = K.dump_rec(lang)
    iw = {(r["item"], r["k"]): r["iw"] for r in K.word_rows(lang)}
    thr = 1 - json.loads((ROOT / "tables" / f"taxonomy_{lang}.json").read_text())["thr_score"]

    def pres(r):
        rec = dump[r["item"]]
        ps = [rec["rwords"][j]["present"] for j in iw[(r["item"], r["k"])] if rec["rwords"][j]["present"] is not None]
        return min(ps) if ps else None

    groups = {
        "inserted unsaid word (changed)": [r for r in rows if r["label"] and r["kind"] == "word_omission"],
        "other changed words": [r for r in rows if r["label"] and r["kind"] != "word_omission"],
        "correct function words, not neighbours": [r for r in rows if not r["label"] and r["function"] and not r["neighbour"]],
        "correct content words, not neighbours": [r for r in rows if not r["label"] and not r["function"] and not r["neighbour"]],
        "within two words of an inserted word": [r for r in rows if not r["label"] and r["neighbour"] and r["kind"] == "word_omission"],
    }
    out = {}
    for name, rr in groups.items():
        v = np.array([p for p in (pres(r) for r in rr) if p is not None], float)
        if not len(v):
            continue
        out[name] = {"n": int(len(v)), "q": [round(float(x), 3) for x in np.percentile(v, [1, 5, 10, 25, 50])],
                     "share_below_0.5": float(np.mean(v < 0.5)),
                     "fa_rate_if_below_0.5": float(np.mean([r["err"] >= thr for r in rr if (pres(r) or 1) < 0.5]))
                     if any((pres(r) or 1) < 0.5 for r in rr) else None}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    a = ap.parse_args(argv)
    res = {}
    for lang in a.lang.split(","):
        res[lang] = run(lang)
        print(lang)
        for k, v in res[lang].items():
            print(f"  {k:42s} n {v['n']:5d}  q1/5/10/25/50 {v['q']}  <0.5 {100 * v['share_below_0.5']:.1f} %")
    K.save_json("presence.json", res)


if __name__ == "__main__":
    main()
