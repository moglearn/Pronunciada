"""Practice items for the identity proofs of the round-20 false-alarm release check.

    python -m bench.round20_fa_rc.practice_items OUT.json [--n 100]

Per language (en, de, es, fr, ko): from the round-20 practice sets (bench.round20_native_fa, speaker-disjoint
practice data, never a held-out split), every item whose prompt has a digit (at most n / 2, so the number
readings are exercised) and random other items up to n (seed 20). Writes {lang: [[id, audio, text], ...]} for
the `items:<file>` set of bench/round19_merge/equiv.py.
"""
from __future__ import annotations

import argparse
import json
import random
import re


def main() -> None:
    from bench.round20_native_fa.core import practice_items

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    out = {}
    for lang in ("en", "de", "es", "fr", "ko"):
        its = sorted(practice_items(lang), key=lambda it: it["id"])
        rng = random.Random(20)
        dig = [it for it in its if re.search(r"\d", it["target"])]
        rng.shuffle(dig)
        dig = dig[:a.n // 2]
        ids = {it["id"] for it in dig}
        rest = [it for it in its if it["id"] not in ids]
        rng.shuffle(rest)
        pick = sorted(dig + rest[:a.n - len(dig)], key=lambda it: it["id"])
        out[lang] = [[it["id"], str(it["audio"]), it["target"]] for it in pick]
        print(lang, len(its), "items,", len(dig), "with a digit")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
