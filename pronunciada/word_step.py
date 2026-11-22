"""Round 19 (Japanese): a capped two-sided step on the round-18 word log-odds.

The round-18 word probability stays the base (sounds x present x contrast; the length channel enters
here instead of as its round-15 factor). Its log-odds move by

    step = min(cap, c0 + a_id * min(s - k_id, 0) + a_len * min(z_len - k_len, 0) - a_mm * extra
                    + a_long * min(z_long - k_long, 0) + a_ins * min(ins - k_ins, 0))

* s:     the word's lowest identity margin (pronunciada.ja_identity), each sound's margin taken
         relative to the 2 % native quantile of its class and position (first / middle / final);
* z_len: the word's lowest length logit (pronunciada.length, round-19 model);
* extra: sounds heard over the word in addition to its text (pronunciada.word_match);
* z_long: the lowest length logit over the word's expected-long vowels (round 19, part 3);
* ins:   the lowest extra-sound log-likelihood ratio inside the word or in the gap before it
         (pronunciada.align, insertion=True; the Korean round-19 stitching), part 3.

So a word whose sounds all clearly match is nudged up by at most `cap`, and a word with a clearly
different sound, a wrong length or an extra sound is pushed down without a floor. A word that was
already low stays low: the step never replaces the base. Parameters (chosen on the round-19 dev set
with its guards; bench/round19_ja/stepgrid.py): pronunciada/data/calib/word_step/ja.json.
Switch part: "step" (pronunciada.ja_round19).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=None)
def _params(lang: str) -> dict | None:
    p = resources.files("pronunciada").joinpath(f"data/calib/word_step/{lang}.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None


def params(lang: str) -> dict | None:
    from .ja_round19 import on

    if lang != "ja" or not on("step"):
        return None
    return _params(lang)


def positions(ref, w) -> list[str]:
    a, b = w.phones
    return ["final" if i == b - 1 else "first" if i == a else "mid" for i in range(a, b)]


def word_s(P: dict, ref, w, s_id, cls) -> float | None:
    off, g = P["offsets"], P["offset_global"]
    vals = []
    for i, pos in zip(range(*w.phones), positions(ref, w)):
        x = s_id[i]
        if x == x:                                   # not NaN
            vals.append(float(x) - off.get(f"{cls[i]}|{pos}", g))
    return min(vals) if vals else None


def step(P: dict, s_min: float | None, z_len: float | None, extra: int, z_long: float | None = None,
         ins: float = 0.0) -> float:
    """z_long: lowest length logit over the word's expected-long vowels; ins: lowest extra-sound
    log-likelihood ratio (align.py, insertion) inside the word or in the gap before it (<= 0)."""
    c = P["cfg"]
    u = c["c0"] - c["a_mm"] * extra
    if s_min is not None:
        u += c["a_id"] * min(s_min - c["k_id"], 0.0)
    if z_len is not None:
        u += c["a_len"] * min(z_len - c["k_len"], 0.0)
    if z_long is not None:
        u += c.get("a_long", 0.0) * min(z_long - c.get("k_long", 0.0), 0.0)
    u += c.get("a_ins", 0.0) * min(ins - c.get("k_ins", 0.0), 0.0)
    return min(u, c["cap"])


def apply(prob: float, st: float) -> float:
    p = min(max(prob, 1e-6), 1 - 1e-6)
    z = math.log(p / (1 - p)) + st
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
