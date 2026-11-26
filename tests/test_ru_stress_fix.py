"""Issue #6: common Russian words whose lexicon stress came from a rare homograph."""
import pytest

from pronunciada.text import get_g2p


def _stress(g, w):
    return [s.stress for s in g(w).syllables].index(1)


@pytest.mark.parametrize("word, idx", [
    ("какой", 1), ("какая", 1), ("она", 1), ("оно", 1), ("уже", 1), ("моя", 1), ("скажите", 1),
    ("минут", 1), ("четыре", 1), ("носки", 1), ("писал", 1), ("писала", 1), ("писали", 1),
    ("люди", 0), ("катя", 0), ("такой", 1), ("потом", 1), ("высоко", 2),
])
def test_common_words_stress(word, idx):
    assert _stress(get_g2p("ru"), word) == idx


def test_switch_and_accent_mark():
    g = get_g2p("ru")
    try:
        g.set_stress_fix(False)
        assert _stress(g, "она") == 0                  # the shipped lexicon entry
    finally:
        g.set_stress_fix(True)
    assert _stress(g, "она") == 1
    assert _stress(g, "о́на") == 0                # a typed accent mark still wins
    assert _stress(g, "доктора") == 0                  # до́ктора / доктора́: left to the lexicon
