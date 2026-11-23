"""Round 20 fixes (pronunciada.r20): per-language defaults, switches, number readings, German lexicon readings and
the Spanish / Korean front-end changes."""
import numpy as np
import pytest

import pronunciada
from pronunciada import r20
from pronunciada.text import get_g2p

TEXT = "The quick brown fox jumps over the lazy dog"
SWITCHES = ("PRONUNCIADA_R20_NUMALT", "PRONUNCIADA_R20_DE_LEXALT", "PRONUNCIADA_R20_ES_GLIDE",
            "PRONUNCIADA_R20_KO_NINS")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)


def test_defaults_per_language():
    assert r20.DEFAULTS == {"en": (), "de": (), "es": ("numalt", "es_glide"), "fr": (), "ko": ()}
    assert r20.any_on("es")
    for lang in ("en", "de", "fr", "ko", "it", "pt", "ru", "ja", "zh"):
        assert not r20.any_on(lang)
        assert not any(r20.enabled(f, lang) for f in r20.ENV)
    assert r20.enabled("es_glide", "es") and r20.enabled("numalt", "es")


def test_switches_turn_each_fix_on_and_off(monkeypatch):
    for var in r20.ENV.values():
        monkeypatch.setenv(var, "1")
    assert all(r20.any_on(lang) for lang in ("en", "de", "es", "fr", "ko"))
    assert r20.enabled("de_lexalt", "de") and r20.enabled("ko_nins", "ko")
    for var in r20.ENV.values():
        monkeypatch.setenv(var, "0")
    assert not any(r20.enabled(f, lang) for f in r20.ENV for lang in r20.DEFAULTS)
    assert not any(r20.any_on(lang) for lang in r20.DEFAULTS)


def test_one_language_off_leaves_the_others(monkeypatch):
    monkeypatch.setitem(r20.DEFAULTS, "fr", ("numalt",))
    assert r20.any_on("fr") and r20.any_on("es")
    monkeypatch.setitem(r20.DEFAULTS, "es", ())
    assert not r20.any_on("es") and r20.any_on("fr")


def test_replace_word_keeps_a_valid_reference():
    g = get_g2p("en")
    ref = g(TEXT)
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


def test_es_glide_switch(monkeypatch):
    import pronunciada.text.es as es

    ref = es.build()("muy bien")
    ws = [p for p in ref.phones if p.ipa == "w"]
    assert ws and all(any(a == "u" for a, _ in p.alts) for p in ws)
    monkeypatch.setenv("PRONUNCIADA_R20_ES_GLIDE", "0")
    ref = es.build()("muy bien")
    assert not any(a == "u" for p in ref.phones if p.ipa == "w" for a, _ in p.alts)


def test_ko_n_insertion_switch(monkeypatch):
    import pronunciada.text.ko as ko

    def alts(r):
        return [a for p in r.phones for a, _ in p.alts]

    assert "nn" not in alts(ko.build()("꽃잎"))
    monkeypatch.setenv("PRONUNCIADA_R20_KO_NINS", "1")
    assert "nn" in alts(ko.build()("꽃잎"))


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
def test_switch_off_skips_the_pass(tts, monkeypatch):
    wav = tts(TEXT)
    sc = pronunciada.Scorer("en", threads=2)
    off = sc.score(wav, TEXT)
    monkeypatch.setenv("PRONUNCIADA_R20_NUMALT", "1")
    on = sc.score(wav, TEXT)
    assert "r20" in on.timing_ms
    monkeypatch.setenv("PRONUNCIADA_R20_NUMALT", "0")
    off = sc.score(wav, TEXT)
    assert "r20" not in off.timing_ms and "r20" not in off._pronunciada
    assert [w.prob for w in on.words] == [w.prob for w in off.words]
