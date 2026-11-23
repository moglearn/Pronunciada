"""Check of the practice sets' labels: does the changed word's index point at the changed token?

    python -m bench.round20_native_fa.index_check --lang en,de,ko,es,fr

The round-19 builders store label.word_index as an index into the front-end units (bench.textswap.units),
which skip tokens without letters or digits; the benchmark-style metric (lingolix.collect) indexes the
whitespace tokens of the target. When a punctuation-only token (French « » ! ? : ;, Spanish ¿ ¡, a lone
dash or comma) precedes the change, the two disagree: the changed word is then counted as a correct word
and a correct word as the changed one. This counts such items (the check: the target token at word_index
is the replacement / added word).
"""
from __future__ import annotations

import argparse
import re

from bench.round20_native_fa import core as K


def key(t):
    return re.sub(r"[^\w]", "", t.lower())


def run(lang: str) -> dict:
    its = [it for it in K.practice_items(lang) if it.get("label")]
    bad = []
    for it in its:
        lab = it["label"]
        new = lab.get("replacement") or lab.get("added")
        toks = it["target"].split()
        k = lab["word_index"]
        tok = toks[k] if k < len(toks) else None
        if new is None or tok is None or key(new) != key(tok):
            bad.append(it["id"])
    return {"changed_items": len(its), "misindexed": len(bad), "examples": bad[:5]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    a = ap.parse_args(argv)
    res = {l: run(l) for l in a.lang.split(",")}
    for l, v in res.items():
        print(l, v)
    K.save_json("index_check.json", res)


if __name__ == "__main__":
    main()
