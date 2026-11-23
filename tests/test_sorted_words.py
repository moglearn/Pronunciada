"""SortedWords (one bytes object + bisection) answers exactly as the frozenset it replaces."""
import gzip
import random

from pronunciada.text import _eu_common as C


def _packaged():
    with C.lex_path("es_words.txt.gz").open("rb") as fb, gzip.open(fb, "rb") as f:
        return f.read()


def test_small_list_membership():
    s = C.SortedWords("a\nab\nb\nzé\nñu".encode())
    assert s.is_sorted() and len(s) == 5
    for w in ("a", "ab", "b", "zé", "ñu"):
        assert w in s
    for w in ("", "aa", "abc", "c", "z", "zé ", "\n", "a\nab", "ñ", "\ud800", None, b"a"):
        assert w not in s


def test_spanish_list_identical_to_frozenset():
    raw = _packaged()
    assert b"\r" not in raw                       # text mode would have translated it
    old = frozenset(raw.decode("utf-8").split("\n"))
    new = C.SortedWords(raw)
    assert new.is_sorted() and len(new) == len(old)
    words = sorted(old)
    rng = random.Random(0)
    probe = rng.sample(words, 20000) + words[:200] + words[-200:]
    probe += [w + "s" for w in probe[:5000]] + [w[:-1] for w in probe[:5000]] + [w.upper() for w in probe[:2000]]
    probe += ["", " ", "zzzz", "aaaa", "méxico", "casa", "perro", "xilófono", "árbol", "árbol"]
    for w in probe:
        assert (w in new) == (w in old), w


def test_spanish_g2p_uses_it():
    from pronunciada.text import get_g2p

    g = get_g2p("es")
    assert isinstance(g.words_known, C.SortedWords)
    assert "casa" in g.words_known and "qwxz" not in g.words_known
