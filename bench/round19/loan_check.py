"""Round 19 (German): the loanword rule on dev: false flags on the affected words, before / after.

    python -m bench.round19.loan_check dev-r19e dev-r19f

Affected words: lexicon words whose German entry uses a foreign sound (pronunciada.text.de
FOREIGN_SOUNDS or a nasal vowel). Reports, per dump, their mean word score and how many are at or
above each dump's own FA@80 threshold, plus the changed words that are affected (should not rise).
"""
from __future__ import annotations

import sys

from bench.round19.audit import threshold, word_rows
from pronunciada.text.de import FOREIGN_SOUNDS


def foreign(r):
    return any(any("̃" in p["ipa"] or p["ipa"] in FOREIGN_SOUNDS for p in w["phones"]) for w in r["iw"])


def main():
    for name in sys.argv[1:]:
        for fold in ("fit", "check", "all"):
            rows = word_rows(name, fold)
            thr = threshold(rows)
            aff = [r for r in rows if not r["label"] and foreign(r)]
            ch = [r for r in rows if r["label"] and foreign(r)]
            fl = sum(r["err"] >= thr for r in aff)
            print(f"{name} {fold}: affected correct words {len(aff)}, flagged {fl}, mean score "
                  f"{sum(1 - r['err'] for r in aff) / max(1, len(aff)):.3f}; affected changed words {len(ch)}")


if __name__ == "__main__":
    main()
