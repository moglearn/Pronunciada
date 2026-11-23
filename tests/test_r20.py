"""Round 20 word-level passes (pronunciada.r20) and the round-20 front-end switches; all off by default."""
import importlib

import numpy as np
import pytest

import pronunciada
from pronunciada import r20
from pronunciada.text import get_g2p

TEXT = "The quick brown fox jumps over the lazy dog"
SWITCHES = ("PRONUNCIADA_R20_UNSAID", "PRONUNCIADA_R20_NUMALT", "PRONUNCIADA_R20_GUESS", "PRONUNCIADA_R20_DE_LEXALT",
            "PRONUNCIADA_R20_ES_GLIDE", "PRONUNCIADA_R20_KO_NINS")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)


def test_all_off_by_default():
    for lang in ("en", "de", "ko", "es", "fr"):
        assert not r20.any_on(lang)


def test_remove_and_replace_words_keep_a_valid_reference():
    g = get_g2p("en")
    ref = g(TEXT)
    r2 = r20.remove_words(ref, [3])
    assert [w.text for w in r2.words] == [w.text for i, w in enumerate(ref.words) if i != 3]
    r3 = r20.replace_word(ref, 3, g("cat"))
    assert [p.ipa for p in r3.phones[r3.words[3].phones[0]:r3.words[3].phones[1]]] == ["k", "æ", "t"]
    assert r3.words[3].text == "fox"
    same = r20.replace_word(ref, 2, g("brown"))
    assert [p.ipa for p in same.phones] == [p.ipa for p in ref.phones]


@pytest.mark.parametrize("lang,tok,want", [
    ("en", "2010", "twenty ten"), ("en", "116", "one hundred sixteen"), ("en", "150", "a hundred and fifty"),
    ("de", "100", "hundert"), ("de", "1755", "siebzehnhundertfünfundfünfzig"),
    ("fr", "1755", "dix sept cent cinquante cinq"), ("ko", "3개가", "세개가"), ("ko", "56가지", "오십육가지")])
def test_number_alternatives(lang, tok, want):
    assert want in r20.number_alternatives(lang, tok)


def test_number_alternatives_leave_other_tokens_alone():
    assert r20.number_alternatives("en", "18th") == []
    assert r20.number_alternatives("en", "word") == []
    assert r20.number_alternatives("ko", "3") == []


def test_lts_alternatives_change_one_vowel_reading():
    g = get_g2p("en")
    base = g.lts.predict("blarvish")
    alts = r20.lts_alternatives(g, "blarvish")
    assert len(alts) <= r20.MAX_ALTS
    for a in alts:
        assert a != [p for x in base for p in x.split("_") if p]


def test_es_glide_switch(monkeypatch):
    import pronunciada.text.es as es

    ref = es.build()("muy bien")
    assert not any(a == "u" for p in ref.phones if p.ipa == "w" for a, _ in p.alts)
    monkeypatch.setenv("PRONUNCIADA_R20_ES_GLIDE", "1")
    try:
        es2 = importlib.reload(es)
        ref = es2.build()("muy bien")
        ws = [p for p in ref.phones if p.ipa == "w"]
        assert ws and all(any(a == "u" for a, _ in p.alts) for p in ws)
    finally:
        monkeypatch.delenv("PRONUNCIADA_R20_ES_GLIDE")
        importlib.reload(es)


def test_ko_n_insertion_switch(monkeypatch):
    import pronunciada.text.ko as ko

    def alts(r):
        return [a for p in r.phones for a, _ in p.alts]

    assert "nn" not in alts(ko.build()("꽃잎"))
    monkeypatch.setenv("PRONUNCIADA_R20_KO_NINS", "1")
    try:
        ko2 = importlib.reload(ko)
        assert "nn" in alts(ko2.build()("꽃잎"))
    finally:
        monkeypatch.delenv("PRONUNCIADA_R20_KO_NINS")
        importlib.reload(ko)


@pytest.mark.model
def test_unsaid_pass_restores_neighbours(tts, monkeypatch):
    wav = tts(TEXT)
    sc = pronunciada.Scorer("en", threads=2)
    text = "The quick brown fox elephants jumps over the lazy dog"
    off = sc.score(wav, text)
    monkeypatch.setenv("PRONUNCIADA_R20_UNSAID", "1")
    monkeypatch.setenv("PRONUNCIADA_R20_UNSAID_TAU", "0.5")
    on = sc.score(wav, text)
    info = on._pronunciada["r20"]["unsaid"]
    assert info is not None and 4 in info["removed"]
    assert on.words[4].prob == pytest.approx(off.words[4].prob)          # the unsaid word keeps its result
    assert on.words[4].omitted
    assert set(info["replaced"]) <= {2, 3, 5, 6}
    far = [i for i in range(len(on.words)) if i not in info["replaced"] and i != 4]
    assert [on.words[i].prob for i in far] == [off.words[i].prob for i in far]
    assert on.prob == off.prob                                           # sentence level: first pass
    assert "r20" in on.timing_ms


@pytest.mark.model
def test_unsaid_pass_does_nothing_on_a_correct_reading(tts, monkeypatch):
    wav = tts(TEXT)
    sc = pronunciada.Scorer("en", threads=2)
    off = sc.score(wav, TEXT)
    monkeypatch.setenv("PRONUNCIADA_R20_UNSAID", "1")
    on = sc.score(wav, TEXT)
    assert on._pronunciada["r20"]["unsaid"] is None
    assert [w.prob for w in on.words] == [w.prob for w in off.words]


@pytest.mark.model
def test_second_pass_word_probabilities_equal_a_full_pass(tts):
    wav = tts(TEXT)
    sc = pronunciada.Scorer("en", threads=2)
    a = sc.score(wav, TEXT)
    b = sc.score(wav, a._pronunciada["ref"], _second=True)
    assert np.allclose([w.prob for w in a.words], [w.prob for w in b.words], atol=0, rtol=0)


@pytest.mark.model
def test_number_readings_take_the_best(tts, monkeypatch):
    wav = tts("It happened in two thousand and ten")
    sc = pronunciada.Scorer("en", threads=2)
    off = sc.score(wav, "It happened in 2010")
    monkeypatch.setenv("PRONUNCIADA_R20_NUMALT", "1")
    on = sc.score(wav, "It happened in 2010")
    assert on.words[3].prob >= off.words[3].prob
    assert [w.prob for w in on.words[:3]] == [w.prob for w in off.words[:3]]
    # the replaced word keeps the canonical reading's sound list: per-sound consumers still work
    assert len(on.words[3].phones) == len(off.words[3].phones)
    from pronunciada.compat.azure import to_azure

    raw = to_azure(on, granularity="Phoneme", enable_miscue=True, phoneme_alphabet="IPA")
    assert len(raw["NBest"][0]["Words"]) == 4


@pytest.mark.model
def test_guess_discount(tts, monkeypatch):
    wav = tts(TEXT)
    sc = pronunciada.Scorer("en", threads=2)
    text = "The quick brown Zorbaxian jumps over the lazy dog"
    off = sc.score(wav, text)
    monkeypatch.setenv("PRONUNCIADA_R20_GUESS", "d2")
    monkeypatch.setenv("PRONUNCIADA_R20_GUESS_S", "0.5")
    on = sc.score(wav, text)
    p0, p1 = off.words[3].prob, on.words[3].prob
    assert 3 in r20.guessed_words(on)
    if p0 >= r20.GROSS:
        assert p1 == pytest.approx(1 - 0.5 * (1 - p0))
    else:
        assert p1 == p0
    assert [w.prob for i, w in enumerate(on.words) if i != 3] == [w.prob for i, w in enumerate(off.words) if i != 3]
