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
