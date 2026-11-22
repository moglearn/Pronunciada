"""Round 20: compare the switch-check dumps (bench/round20_mandarin/run_equiv.sh) with tolerance 0.

    python -m bench.round20_mandarin.equiv_table <dump dir>   # -> results/tables/round20_mandarin_switch_equivalence.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bench.round19_merge.equiv import compare

LANGS = ("zh", "pt", "es", "ru", "en", "ko", "de", "ja", "fr", "it")


def main():
    d = Path(sys.argv[1])
    rows = []
    for prec in ("fp32", "int8"):
        pairs = [("new", lang, "main", "every output, defaults") for lang in LANGS if lang != "zh"]
        pairs += [("newoff", "zh", "main", "PRONUNCIADA_IDENTITY_ZH_R20=0 vs main"),
                  ("new", "zh", "cand", "branch head vs frozen candidate 8d1952e"),
                  ("new", "zh", "main", "round-20 rule on vs main (expected to differ in word P only)")]
        for a, lang, b, what in pairs:
            fa, fb = d / f"{a}_{lang}_{prec}.jsonl", d / f"{b}_{lang}_{prec}.jsonl"
            if not (fa.exists() and fb.exists()):
                continue
            r = compare([str(fa), str(fb)], 0.0, quiet=True)
            keys = sorted({e.split(" ", 1)[1].split(":")[0].split(".")[0] for e in r["examples"] if " " in e})
            rows.append({"precision": prec, "lang": lang, "a": a, "b": b, "what": what, "items": r["items_a"],
                         "values": r["values"], "max_abs_diff": r["max_abs_diff"],
                         "n_diff": r["n_numeric_over_tol"] + r["n_other_diff"], "example_keys": keys[:6],
                         "examples": r["examples"][:4]})
            print(f"{prec} {lang} {a:6s} vs {b:4s} items {r['items_a']} values {r['values']} max|diff| "
                  f"{r['max_abs_diff']:.3g} n_diff {rows[-1]['n_diff']} {keys[:4]}")
    Path("results/tables/round20_mandarin_switch_equivalence.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
