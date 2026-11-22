"""Round 19 (Japanese word scores on Common Voice style audio): on / off switches.

OFF BY DEFAULT. The frozen round-19 candidate raised the Lingolix Japanese word AUC but failed two
held-out guards (FLEURS-swap far swaps, cv-real word AUCs; results/round19_japanese.md), so the
shipped behaviour is round 18's. PRONUNCIADA_JA_ROUND19=1 turns every part on;
PRONUNCIADA_JA_ROUND19_OFF=name[,name...] then turns single parts off (ablations):

* ``merge``   text front-end: a bare vowel mora repeating the previous mora's vowel inside a word
              is one long vowel (場合 バアイ -> b aː i, 思惟 シイ -> ʃ iː), as ー already is;
* ``labels``  length channel: judge geminates and single consonants across a word boundary inside
              a phrase (行っ|た, 変え|て) and short vowels before another vowel other than e+i / o+u;
* ``wordcal`` word probability: one recalibration of the word's pieces plus the free-decode
              mismatch (pronunciada.word_cal, data/calib/wordcal/ja.json), fitted on planted text
              edits in Common Voice ja train and FLEURS ja train audio.

Read on every call (cheap), so tests and benchmarks can flip them inside one process.
"""
from __future__ import annotations

import os


def on(part: str) -> bool:
    if os.environ.get("PRONUNCIADA_JA_ROUND19", "0") != "1":
        return False
    off = os.environ.get("PRONUNCIADA_JA_ROUND19_OFF", "")
    return part not in {x.strip() for x in off.split(",") if x.strip()}
