"""How often the round-20 word-level pass re-scores on natural text (FLEURS validation, original texts).

    python -m bench.round20_fa_rc.triggers OUT.json [--n 200] [--precision int8]

The same clips as bench/round19_merge/latency.py --set fleurs-val --n N (equiv._clips, seed 19, own text only),
scored with the release defaults. Per language: clips, clips where the pass re-scores at least one word, and
the re-scored words by trigger: a number token (a digit in the word) below ALT_GATE, or (de) a lexicon word
with several readings below ALT_GATE; plus the words below ALT_GATE that were looked at (the gate).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--langs", default="en,de,es,fr,ko")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "round19_merge"))
    from equiv import _clips

    from pronunciada import Scorer, r20

    out = {}
    for lang in a.langs.split(","):
        sc = Scorer(lang, precision=a.precision, threads=4)
        c = {"clips": 0, "clips_rescored": 0, "words": 0, "words_below_gate": 0, "number_tokens": 0,
             "number_tokens_below_gate": 0, "rescored_number": 0, "rescored_de_lexicon": 0, "r20_ms": []}
        for cid, wav, text in _clips("fleurs-val", lang, a.n):
            if cid.endswith("#swap"):
                continue
            res = sc.score((wav, 16000), text)
            c["clips"] += 1
            c["words"] += len(res.words)
            info = (res._pronunciada.get("r20") or {}).get("alts") or []
            ref = res._pronunciada["ref"]
            for i, w in enumerate(res.words):
                num = bool(re.search(r"\d", ref.words[i].text))
                c["number_tokens"] += num
                if w.prob < r20.ALT_GATE:
                    c["words_below_gate"] += 1
                    c["number_tokens_below_gate"] += num
            if info:
                c["clips_rescored"] += 1
            for i in info:
                if re.search(r"\d", ref.words[i].text):
                    c["rescored_number"] += 1
                else:
                    c["rescored_de_lexicon"] += 1
            c["r20_ms"].append(res.timing_ms.get("r20", 0.0))
        ms = sorted(c.pop("r20_ms"))
        c["r20_ms_mean"] = sum(ms) / len(ms)
        c["r20_ms_p95"] = ms[int(0.95 * (len(ms) - 1))]
        c["share_clips_rescored"] = c["clips_rescored"] / c["clips"]
        out[lang] = c
        print(lang, json.dumps(c), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
