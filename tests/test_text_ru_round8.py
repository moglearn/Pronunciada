"""Round 8: measured (reduction-aware) Russian vowel / t-d targets (on by default since round 8)."""
import math

import pytest

from pronunciada.text import get_g2p
from pronunciada.text import ru as R


@pytest.fixture
def g2p():
    g = get_g2p("ru")
    g.measured_targets = True
    yield g
    g.measured_targets = R.MEASURED_TARGETS


def _word(ref, text):
    for w in ref.words:
        if w.text == text:
            return ref.phones[w.phones[0]:w.phones[1]]
    raise KeyError(text)


def _last_vowel(ref, text):
    return [p for p in _word(ref, text) if p.is_nucleus][-1]


def test_vowel_keys():
    # документы: о(P2) у(P1) е(S) ы(F), stress on е (index 2)
    assert R.vowel_keys("документы", 2) == [("о", "P2", "C"), ("у", "P1", "C"), ("е", "S", "C"),
                                              ("ы", "F", "C")]
    assert R.vowel_keys("ее", 1) == [("е", "P1", "V"), ("е", "S", "V")]
    assert R.vowel_keys("жена", 1)[0] == ("е", "P1", "H")
    assert R.vowel_keys("на", None) == [("а", "C", "C")]
    assert R.cons_key("tʲ", None) == ("tʲ", "#")
    assert R.cons_key("t", "i", True) == ("t", "Vi")
    assert R.cons_key("t", "a", True) == ("t", "V")
    assert R.cons_key("d", "n", False) == ("d", "C")


def test_table_shipped_and_consistent(g2p):
    tab = g2p.vowel_table
    assert tab, "pronunciada/data/lex/ru_vowels.json missing"
    for key, (n, dist) in tab.items():
        assert n > 0 and dist
        assert abs(sum(dist.values()) - 1.0) < 0.02, key
    assert 0.0 < R.VOWEL_MIN < 0.5


def test_unstressed_endings_are_specific(g2p):
    """Final -а / -ы of lexicon words no longer accept each other (documented swap pairs)."""
    ref = g2p("Нет документа и документы.")
    a, y = _last_vowel(ref, "документа"), _last_vowel(ref, "документы")
    assert a.ipa == "a" and y.ipa == "ɨ"
    assert "ɨ" not in {x for x, _ in a.alts}
    assert "a" not in {x for x, _ in y.alts}


def test_alternatives_follow_measured_shares(g2p):
    ref = g2p("Он сказал.")
    for p in _word(ref, "сказал"):
        if not p.is_nucleus:
            continue
        for alt, prior in p.alts:
            assert alt != p.ipa and prior <= 0.0
        priors = [w for _, w in p.alts]
        assert priors == sorted(priors, reverse=True)
    key = ("о", "S", "C")
    n, dist = g2p.vowel_table[key]
    ref = g2p("Это большинство.")
    o = _last_vowel(ref, "большинство")
    prim = max(dist, key=dist.get)
    assert o.ipa == prim
    for alt, prior in o.alts:
        assert dist[alt] >= R.VOWEL_MIN
        assert math.isclose(prior, math.log(dist[alt] / dist[prim]), abs_tol=1e-6)
    assert {x for x, s in dist.items() if s >= R.VOWEL_MIN and x != prim} == {x for x, _ in o.alts}


def test_predicted_stress_keeps_broad_alts(g2p):
    """OOV words (stress predicted by the suffix model) keep the round-7 rule-based alternatives."""
    ref = g2p("Блорфингон пришёл.")
    alts = set()
    for p in _word(ref, "Блорфингон"):
        if p.is_nucleus:
            alts |= {x for x, _ in p.alts}
    assert alts & {"ɐ", "ə", "ɪ"}


def test_default_on_and_switchable():
    """Adopted in round 8: on by default; measured_targets = False gives the round-7 alternatives."""
    assert R.MEASURED_TARGETS is True
    g = get_g2p("ru")
    assert g.measured_targets is True
    y = _last_vowel(g("Нет документы."), "документы")
    assert "a" not in {x for x, _ in y.alts}
    try:
        g.measured_targets = False
        y = _last_vowel(g("Нет документы."), "документы")
        assert "a" in {x for x, _ in y.alts}
    finally:
        g.measured_targets = True
