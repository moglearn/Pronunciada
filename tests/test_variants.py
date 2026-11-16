import json

import pytest

from pronunciada import variants as V
from pronunciada.types import RefPhone, RefSyllable, RefWord, Reference


def _summ(n, mean, std=1.0):
    return {"count": n, "mean": mean, "std": std, "q01": mean - 2, "q05": mean - 1.6,
            "q25": mean - 0.7, "q50": mean, "q75": mean + 0.7, "q95": mean + 1.6}


TABLE = {
    "lang": "xx",
    "variants_enabled": True,
    "variants": {"t": [["ɾ", -2.5, 40, 0.08], ["", -3.0, 25, 0.05]], "a": [["ə", -3.2, 30, 0.04]]},
    "stats": {
        "global": {f: _summ(1000, -1.0) for f in V.FEATURES},
        "class": {"stop": {f: _summ(500, -2.0) for f in V.FEATURES},
                  "vowel": {f: _summ(10, -9.0) for f in V.FEATURES}},
        "phone": {"t": {f: _summ(100, -3.0, 2.0) for f in V.FEATURES},
                  "k": {f: _summ(5, -7.0) for f in V.FEATURES}},
        "phone_pos": {"t": {"final": {f: _summ(40, -4.0, 0.5) for f in V.FEATURES},
                            "initial": {f: _summ(3, -8.0) for f in V.FEATURES}}},
    },
}


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setattr(V, "_table", lambda lang: TABLE if lang == "xx" else None)
    return TABLE


def _ref():
    # one word "ta": t (alt ɾ already rule-based) + a
    phones = [RefPhone("t", 0, 0, False, [("ɾ", -1.0)]), RefPhone("a", 0, 0, True, [])]
    return Reference("xx", "ta", [RefWord("ta", (0, 1), (0, 2))], [RefSyllable(0, (0, 2))], phones)


def test_backoff_levels(table):
    ns = V.NativeStats.load("xx")
    assert ns.stats("t", "final", "llr_sub")[1] == "phone_pos"
    assert ns.stats("t", "initial", "llr_sub")[1] == "phone"        # count 3 < 30
    assert ns.stats("k", "final", "llr_sub")[1] == "class"          # phone count 5 < 30 -> stop
    assert ns.stats("ɑ", None, "llr_sub")[1] == "global"            # vowel class count 10 < 30
    assert ns.z("t", "final", "llr_sub", -3.0) == pytest.approx(2.0)
    assert ns.z("t", None, "llr_sub", -1.0) == pytest.approx(1.0)


def test_apply_variants_no_duplicates(table):
    ref = _ref()
    out = V.apply_variants(ref, "xx")
    assert out.phones[0].alts == [("ɾ", -1.0), ("", -3.0)]           # rule-based ɾ kept, not doubled
    assert out.phones[1].alts == [("ə", -3.2)]
    assert ref.phones[0].alts == [("ɾ", -1.0)]                       # input untouched
    out.validate()
    again = V.apply_variants(out, "xx")
    assert again.phones[0].alts == out.phones[0].alts


def test_strip_is_inverse(table):
    ref = _ref()
    back = V.strip_variants(V.apply_variants(ref, "xx"), "xx")
    assert [p.alts for p in back.phones] == [p.alts for p in ref.phones]


def test_unknown_language_is_noop(table):
    ref = _ref()
    assert V.apply_variants(ref, "yy").phones[0].alts == ref.phones[0].alts
    assert V.NativeStats.load("yy") is None


def test_only_if_enabled(table, monkeypatch):
    t = dict(TABLE, variants_enabled=False)
    monkeypatch.setattr(V, "_table", lambda lang: t)
    assert V.apply_variants(_ref(), "xx", only_if_enabled=True).phones[1].alts == []
    assert V.apply_variants(_ref(), "xx").phones[1].alts == [("ə", -3.2)]


def test_broad_class():
    assert V.broad_class("a") == "vowel"
    assert V.broad_class("aɪ") == "vowel"
    assert V.broad_class("tʃ") == "affricate"
    assert V.broad_class("tsʰ") == "affricate"
    assert V.broad_class("pʰ") == "stop"
    assert V.broad_class("ʃ") == "fricative"
    assert V.broad_class("ŋ") == "nasal"
    assert V.broad_class("ɾ") == "liquid-glide"
    assert V.broad_class("w") == "liquid-glide"


@pytest.mark.parametrize("lang", ["en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko"])
def test_shipped_tables_are_valid(lang):
    from pronunciada.text.ipa import is_valid

    p = V.table_path(lang)
    if not p.exists():
        pytest.skip(f"no native table for {lang}")
    t = json.loads(p.read_text(encoding="utf-8"))
    assert t["lang"] == lang
    for f in V.FEATURES:
        assert t["stats"]["global"][f]["count"] > 0
    for ph, alts in t["variants"].items():
        assert is_valid(ph)
        for a, w, c, r in alts:
            assert a == "" or is_valid(a)
            assert a != ph and w <= 0 and c >= 20 and r >= 0.03
