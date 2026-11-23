"""Hangul syllable decomposition (Unicode arithmetic, no data files).

Shared by the Korean text front-end (pronunciada.text.ko) and the Korean scoring code
(pronunciada.ko_identity), so that scoring does not import the front-end.

A precomposed syllable U+AC00 + o (0 <= o < 11172) is lead jamo L[o // 588], vowel V[(o % 588) // 28]
and tail T[o % 28] ("" = no final consonant): Unicode 15.0, section 3.12.
"""
from __future__ import annotations

L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
T = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
     "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def decompose(ch: str) -> tuple[str, str, str] | None:
    """(lead, vowel, tail) compatibility jamo of a precomposed Hangul syllable; None otherwise."""
    o = ord(ch) - 0xAC00
    if not 0 <= o < 11172:
        return None
    return L[o // 588], V[(o % 588) // 28], T[o % 28]
