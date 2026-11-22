"""Round 19 (Japanese): front-end merge, length labels, word recalibration and the switches."""
import pytest

pytest.importorskip("pyopenjtalk")


def _phones(text, word):
    from pronunciada.text.ja import JaG2P

    r = JaG2P()(text)
    w = next(w for w in r.words if w.text == word)
    return r, [r.phones[i].ipa for i in range(*w.phones)]


def test_same_vowel_merge(monkeypatch):
    monkeypatch.setenv("PRONUNCIADA_JA_ROUND19", "1")
    assert _phones("場合が多い", "場合")[1] == ["b", "aː", "i"]
    monkeypatch.setenv("PRONUNCIADA_JA_ROUND19", "0")
    assert _phones("場合が多い", "場合")[1] == ["b", "a", "a", "i"]


def test_cross_word_geminate_label(monkeypatch):
    from pronunciada import length as L

    monkeypatch.setenv("PRONUNCIADA_JA_ROUND19", "1")
    r, ph = _phones("学校に行った", "た")
    assert ph[0] == "tt"
    i = r.words[[w.text for w in r.words].index("た")].phones[0]
    assert L.labels(r)[i] == "long"
    monkeypatch.setenv("PRONUNCIADA_JA_ROUND19", "0")
    assert L.labels(r)[i] is None


def test_vowel_before_vowel_label(monkeypatch):
    from pronunciada import length as L

    monkeypatch.setenv("PRONUNCIADA_JA_ROUND19", "1")
    r, ph = _phones("倍になる", "倍")
    i = r.words[0].phones[0] + 1          # the a of b a i
    assert L.labels(r)[i] == "short"


def test_word_cal_switch(monkeypatch):
    from pronunciada import word_cal as WC

    monkeypatch.setenv("PRONUNCIADA_JA_ROUND19", "1")
    assert WC.params("ja") is not None
    assert WC.params("en") is None
    monkeypatch.setenv("PRONUNCIADA_JA_ROUND19_OFF", "wordcal")
    assert WC.params("ja") is None


def test_off_by_default(monkeypatch):
    from pronunciada.ja_round19 import on

    monkeypatch.delenv("PRONUNCIADA_JA_ROUND19", raising=False)
    assert not on("merge") and not on("labels") and not on("wordcal")


def test_word_match_counts():
    from pronunciada.types import RefPhone as P
    from pronunciada.word_match import _toks, align_word

    assert align_word([P("a", 0, 0, True)], _toks("ha")) == (1, 0, 0)
    w = [P("tʃ", 0, 0), P("o", 0, 0, True), P("tt", 0, 1), P("o", 0, 1, True)]
    assert align_word(w, _toks("tʃotto")) == (0, 0, 0)
