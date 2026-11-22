"""Round 19 (Japanese word scores on Common Voice style audio): on / off switches.

On by default (the second round-19 candidate passed every dev guard; results/round19_japanese.md).
PRONUNCIADA_JA_ROUND19=0 restores round 18 for paired before / after runs;
PRONUNCIADA_JA_ROUND19_OFF=name[,name...] turns single parts off (ablations):

* ``merge``   text front-end: a bare vowel mora repeating the previous mora's vowel inside a word
              is one long vowel (場合 バアイ -> b aː i, 思惟 シイ -> ʃ iː), as ー already is;
* ``labels``  length channel: judge geminates and single consonants across a word boundary inside
              a phrase (行っ|た, 変え|て) and short vowels before another vowel other than e+i / o+u;
* ``lenmodel`` length channel: the round-19 model (data/length/ja_r19.json; trees fitted on natives
              against planted long / short swaps of Common Voice ja train + FLEURS ja train, with
              filterbank durations, local mora rate, word duration ratio and the long / short
              counterfactual of pronunciada.length_r19), its coaching thresholds and its flag threshold;
* ``step``    word probability: a capped two-sided step on the round-18 word log-odds from the
              identity head (pronunciada.ja_identity), the length evidence and extra heard sounds
              (pronunciada.word_step, data/calib/word_step/ja.json).

(The first round-19 candidate, a learned word recalibration "wordcal", failed its held-out guards
and was removed; see results/round19_japanese.md.) The length parameters are read when a Scorer is
built; the other parts on every call.

Read on every call (cheap), so tests and benchmarks can flip them inside one process.
"""
from __future__ import annotations

import os


def on(part: str) -> bool:
    if os.environ.get("PRONUNCIADA_JA_ROUND19", "1") == "0":
        return False
    off = os.environ.get("PRONUNCIADA_JA_ROUND19_OFF", "")
    return part not in {x.strip() for x in off.split(",") if x.strip()}
