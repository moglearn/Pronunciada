"""Offline tests for the Japanese text front-end."""
from __future__ import annotations

import pytest

from pronunciada.text import get_g2p

from _cjk_helpers import speed_ms, syl_ipa, tones  # noqa: F401


pytest.importorskip("pyopenjtalk")
ja = get_g2p("ja")

JA_SENTS = [
    "今日は良い天気ですね。",
    "東京に行きました。",
    "ちょっと待ってください！",
    "学校で新聞を読んでいます。",
    "私は学生です。",
    "お父さんへ手紙を書きます。",
    "日本には約7,000の島々があります。",
    "コーヒーとケーキをください。",
    "ABCは3人で来ました。",
    "あっ、そうですか？",
]


def pitch(ref):
    return ["H" if s.pitch_high else "L" for s in ref.syllables]


@pytest.mark.parametrize("text", JA_SENTS)
def test_ja_validate(text):
    ref = ja(text)
    ref.validate()
    assert ref.syllables and all(s.pitch_high is not None for s in ref.syllables)
    for w in ref.words:
        assert w.text == text[w.char_span[0]:w.char_span[1]]


def test_ja_pitch_accent():
    assert pitch(ja("箸が")) == ["H", "L", "L"]     # 1: hashi ga
    assert pitch(ja("橋が")) == ["L", "H", "L"]     # 2
    assert pitch(ja("端が")) == ["L", "H", "H"]     # 0
    assert pitch(ja("雨が")) == ["H", "L", "L"]
    assert pitch(ja("飴が")) == ["L", "H", "H"]
    assert pitch(ja("日本が")) == ["L", "H", "L", "L"]


def test_ja_mora_units():
    r = ja("学校")
    assert [s.text for s in r.syllables] == ["ガ", "ッコー"]          # っ -> next mora, ー -> merged
    assert syl_ipa(r) == ["ga", "kkoː"]
    r = ja("新聞")
    assert syl_ipa(r) == ["ʃi", "m", "bu", "ŋ"]                          # ん assimilated, own unit
    nas = r.phones[r.syllables[1].phones[0]]
    assert nas.is_nucleus
    r = ja("です")
    su = r.phones[-1]
    assert su.ipa == "u" and ("", -0.2) in su.alts                     # devoiced vowel may be dropped
    r = ja("私は")
    assert syl_ipa(r)[-1] == "wa"                                       # particle は


def test_ja_numbers_latin():
    r = ja("3人")
    assert [s.text for s in r.syllables] == ["サ", "ン", "ニ", "ン"]
    assert r.words[0].text == "3"
    r = ja("ABCです")
    assert any("ABC" in w for w in r.warnings) and [s.text for s in r.syllables] == ["デ", "ス"]


def test_ja_speed():
    assert speed_ms(ja, JA_SENTS) < 5.0
