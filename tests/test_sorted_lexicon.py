"""SortedLexicon gives the same keys, values, order and metadata as the dict loader."""
import gzip
import random

import pytest

from pronunciada.text import _eu_common as C

LEXICONS = ["en", "de", "fr", "it"]          # every lexicon loaded through load_lexicon


@pytest.mark.parametrize("name", LEXICONS)
def test_packaged_lexicon_identical_to_dict(name):
    with C.lex_path(f"{name}.tsv.gz").open("rb") as fb, gzip.open(fb, "rb") as f:
        raw = f.read()
    checked = C.SortedLexicon.from_tsv(raw, check=True)
    assert checked is not None, "packaged lexicon not in the sorted layout"
    d, meta = C.load_lexicon_dict(f"{name}.tsv.gz")
    lex, meta2 = C.load_lexicon(f"{name}.tsv.gz")
    assert isinstance(lex, C.SortedLexicon)
    assert meta2 == meta and len(lex) == len(d)
    assert list(lex) == list(d)                       # same keys in the same order
    keys = list(d)
    rng = random.Random(1)
    probe = rng.sample(keys, min(len(keys), 20000)) + keys[:100] + keys[-100:]
    for w in probe:
        assert lex[w] == d[w] and lex.get(w) == d[w] and w in lex
    misses = [w + "q" for w in probe[:3000]] + [w[:-1] for w in probe[:3000]] + ["", "#onsets", "zzzzzz", "\ud800"]
    for w in misses:
        assert (w in lex) == (w in d) and lex.get(w) == d.get(w) and lex.get(w, 7) == d.get(w, 7)


def test_small_tsv_layouts():
    lex = C.SortedLexicon.from_tsv("#onsets\tp+l t+r\n#x\t\na\tA1\nab\nb\tB\tB2\nzé\tZ\n".encode(), check=True)
    assert lex.meta == {"onsets": "p+l t+r", "x": ""}
    assert dict(lex) == {"a": "A1", "ab": "", "b": "B\tB2", "zé": "Z"}
    assert "c" not in lex and lex.get("c") is None
    with pytest.raises(KeyError):
        lex["c"]
    assert C.SortedLexicon.from_tsv(b"b\tB\na\tA\n", check=True) is None          # unsorted
    assert C.SortedLexicon.from_tsv(b"a\tA\r\n") is None                           # text mode differs
    assert len(C.SortedLexicon.from_tsv(b"")) == 0


@pytest.mark.parametrize("name", [f"{n}.tsv.gz" for n in LEXICONS] + ["es_words.txt.gz", "ru.tsv.gz"])
def test_read_gz_matches_gzip(name):
    with C.lex_path(name).open("rb") as fb, gzip.open(fb, "rb") as f:
        assert C.read_gz(name) == f.read()


def _ru_dict():
    d, onsets = {}, None
    with C.lex_path("ru.tsv.gz").open("rb") as fb, gzip.open(fb, "rt", encoding="utf-8") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("\t")
            if k[:1] == "#":
                if k == "#onsets":
                    onsets = v
                continue
            d[k] = v
    return d, onsets


def test_russian_stress_lexicon_identical_to_dict():
    import bisect

    from pronunciada.text import get_g2p

    raw = C.read_gz("ru.tsv.gz")
    assert C.SortedLexicon.from_tsv(raw, check=True) is not None
    d, onsets = _ru_dict()
    g = get_g2p("ru")
    assert isinstance(g.stress, C.SortedLexicon) and len(g.stress) == len(d)
    assert g.onsets == C.parse_onsets(onsets)
    keys = sorted(d)
    assert list(g._keys) == keys and len(g._keys) == len(keys)
    rng = random.Random(2)
    for w in rng.sample(keys, 20000):
        assert g.stress[w] == d[w] and g.stress.get(w) == d.get(w)
    for stem in [w[:n] for w in rng.sample(keys, 3000) for n in (2, 3, 5)] + ["", "я", "яяяя"]:
        assert bisect.bisect_left(g._keys, stem) == bisect.bisect_left(keys, stem)
        assert (stem in g.stress) == (stem in d) and g.stress.get(stem + "ъ") == d.get(stem + "ъ")
    assert g._keys[-1] == keys[-1] and g._keys[2:5] == keys[2:5]
