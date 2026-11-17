"""Offline tests for the Korean text front-end."""
from __future__ import annotations

import pytest

from pronunciada.text import get_g2p

from _cjk_helpers import speed_ms, syl_ipa, tones  # noqa: F401


ko = get_g2p("ko")

KO_SENTS = [
    "국물이 정말 맛있어요.",
    "같이 갑시다!",
    "신라의 역사는 깁니다.",
    "날씨가 좋다.",
    "책을 읽다가 잠들었다.",
    "전화를 했습니다.",
    "밥 먹었어요?",
    "2024년에 3.5% 성장했다.",
    "서울 Seoul은 큽니다.",
    "의사의 희망은 무엇입니까?",
]


@pytest.mark.parametrize("text", KO_SENTS)
def test_ko_validate(text):
    ref = ko(text)
    ref.validate()
    assert all(s.tone is None for s in ref.syllables)


@pytest.mark.parametrize("word,expected", [
    ("국물", ["kuŋ", "mul"]),        # nasalisation 궁물
    ("같이", ["ka", "tʃʰi"]),        # palatalisation 가치
    ("신라", ["ʃil", "la"]),         # liquidisation 실라
    ("좋다", ["tʃo", "tʰa"]),        # aspiration 조타
    ("읽다", ["ik", "ta"]),          # cluster + tensification 익따 (ZIPA writes tense as plain)
    ("입니다", ["im", "ni", "ta"]),  # 임니다
    ("밟다", ["pap", "ta"]),         # exception 밥따
    ("닭이", ["tal", "ki"]),         # cluster liaison 달기 (voiced g is an alt)
    ("값이", ["kap", "ʃi"]),         # 갑씨
    ("좋아", ["tʃo", "a"]),          # ㅎ deletion
    ("축하", ["tʃʰu", "kʰa"]),       # 추카
    ("종로", ["tʃuŋ", "no"]),        # 종노
    ("국립", ["kuŋ", "nip"]),        # 궁닙
    ("설날", ["sɔl", "lal"]),        # 설랄
])
def test_ko_rules(word, expected):
    r = ko(word)
    assert syl_ipa(r) == expected
    assert [s.text for s in r.syllables] == list(word)


def test_ko_cross_word_alternatives():
    r = ko("밥 먹어요")
    coda = r.phones[r.syllables[0].phones[1] - 1]
    assert coda.ipa == "p" and any(a == "m" for a, _ in coda.alts)      # 밤머거요 in fluent speech
    r = ko("할 일")
    coda = r.phones[r.syllables[0].phones[1] - 1]
    assert any(a == "ll" for a, _ in coda.alts)                         # ㄴ-insertion 할릴


def test_ko_numbers_latin():
    r = ko("123")
    assert [s.text for s in r.syllables] == list("백이십삼") and r.words[0].text == "123"
    r = ko("서울 Seoul")
    assert len(r.words) == 1 and any("Seoul" in w for w in r.warnings)


def test_ko_speed():
    assert speed_ms(ko, KO_SENTS) < 5.0
