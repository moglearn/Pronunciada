"""Reverse check for fix D (guessed words), which the planted practice sets cannot give: their builders never
change an unknown word and never plant a capitalised word mid-sentence, so no changed word is a guessed word.

    python -m bench.round20_native_fa.dcheck --lang en,de,es,fr --s 0.5

For every control item (check half) with a guessed word (pronunciada.r20.guessed_in_ref), the first guessed word
gets one controlled error in its reference: its first vowel replaced by a far vowel (fixed map below), i.e. the
reference now says something the speaker did not. The word is scored with D off, D1 and D2 (factor --s); reported:
the share of these corrupted words caught at the practice FA@80 threshold of main (check half), and the share of the
uncorrupted guessed words (same items) that are false alarms, off vs on.
"""
from __future__ import annotations

import argparse
import json
import os

from bench.round20_native_fa import core as K
from bench.round20_native_fa import gate as G

FAR = {"i": "ɑ", "ɪ": "ɑ", "e": "u", "eɪ": "u", "ɛ": "u", "æ": "u", "a": "u", "ɑ": "i", "ɔ": "i", "o": "i", "oʊ": "i",
       "u": "æ", "ʊ": "æ", "ə": "i", "ʌ": "i", "aɪ": "u", "aʊ": "i", "y": "ɑ", "ø": "ɑ", "œ": "i", "ɐ": "i", "ɜ˞": "i"}


SWAP = {"a": "o", "e": "a", "i": "o", "o": "a", "u": "e"}
TEXT_LEVEL = False


def run(lang, s):
    from pronunciada import Scorer
    from pronunciada import r20
    from pronunciada.text import get_g2p

    thr = K.threshold(G.rows(lang, None, "check"))
    g2p = get_g2p(lang)
    its = [it for it in K.practice_items(lang) if not it.get("label") and it.get("fold") == "check"]
    sc = Scorer(lang, threads=2)
    res = {"n": 0, "caught": {"off": 0, "d1": 0, "d2": 0}, "fa_clean": {"off": 0, "d1": 0, "d2": 0}}
    for it in its:
        ref = g2p(it["target"])
        gs = sorted(r20.guessed_in_ref(ref))
        if not gs:
            continue
        wi = gs[0]
        if TEXT_LEVEL:
            # the change is made in the TEXT, as the practice builders do: the first vowel letter of the guessed
            # word is swapped for another vowel letter; the audio still says the original word
            s0, s1 = ref.words[wi].char_span
            w = it["target"][s0:s1]
            j = next((i for i, ch in enumerate(w) if ch.lower() in SWAP), None)
            if j is None:
                continue
            nw = w[:j] + (SWAP[w[j].lower()].upper() if w[j].isupper() else SWAP[w[j].lower()]) + w[j + 1:]
            bad = g2p(it["target"][:s0] + nw + it["target"][s1:])
            if len(bad.words) != len(ref.words):
                continue
        else:
            a, b = ref.words[wi].phones
            vi = next((i for i in range(a, b) if ref.phones[i].ipa in FAR), None)
            if vi is None:
                continue
            import copy

            bad = copy.deepcopy(ref)
            bad.phones[vi].ipa = FAR[ref.phones[vi].ipa]
            bad.phones[vi].alts = []
        res["n"] += 1
        for mode in ("off", "d1", "d2"):
            os.environ["PRONUNCIADA_R20_GUESS"] = "0" if mode == "off" else mode
            os.environ["PRONUNCIADA_R20_GUESS_S"] = str(s)
            p_bad = sc.score(it["audio"], bad).words[wi].prob
            p_ok = sc.score(it["audio"], ref).words[wi].prob
            res["caught"][mode] += (1 - round(p_bad * 100, 1) / 100) >= thr
            res["fa_clean"][mode] += (1 - round(p_ok * 100, 1) / 100) >= thr
    os.environ["PRONUNCIADA_R20_GUESS"] = "0"
    n = max(1, res["n"])
    out = {"n": res["n"], **{f"caught_{m}": 100 * v / n for m, v in res["caught"].items()},
           **{f"fa_clean_{m}": 100 * v / n for m, v in res["fa_clean"].items()}}
    print(lang, json.dumps({k: round(v, 1) for k, v in out.items()}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en,de,es,fr")
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--text", action="store_true", help="change the word's spelling instead of its reference sounds")
    a = ap.parse_args()
    global TEXT_LEVEL
    TEXT_LEVEL = a.text
    K.save_json("r20_dcheck_text.json" if a.text else "r20_dcheck.json", {l: run(l, a.s) for l in a.lang.split(",")})


if __name__ == "__main__":
    main()
