"""Swap generators of the contrast audit (bench/audit/contrasts.py): no audio, no models, no network."""
import numpy as np

from bench.audit.contrast_audit import _alloc
from bench.audit.contrasts import CONTRASTS, Site, apply, get_contrast, sites
from pronunciada.types import RefPhone, RefSyllable, RefWord, Reference


def build(lang, words):
    """words: [(text, [(phones, nucleus_idx, {syllable attrs}), ...]), ...]; a phone is "x" or ("x", [alts])."""
    ref = Reference(lang=lang, text=" ".join(w for w, _ in words))
    for wi, (wt, syls) in enumerate(words):
        s0, p0 = len(ref.syllables), len(ref.phones)
        for phones, nuc, attrs in syls:
            si, ps = len(ref.syllables), len(ref.phones)
            for k, ph in enumerate(phones):
                alts = []
                if isinstance(ph, tuple):
                    ph, alts = ph
                    alts = [(a, -1.0) for a in alts]
                ref.phones.append(RefPhone(ph, wi, si, k == nuc, alts))
            ref.syllables.append(RefSyllable(word=wi, phones=(ps, len(ref.phones)), **attrs))
        ref.words.append(RefWord(wt, (s0, len(ref.syllables)), (p0, len(ref.phones))))
    ref.validate()
    return ref


def ipas(ref):
    return [p.ipa for p in ref.phones]


def test_every_contrast_has_two_member_names_and_known_kind():
    for lang, cs in CONTRASTS.items():
        names = [c.name for c in cs]
        assert len(names) == len(set(names)), lang
        for c in cs:
            assert len(c.members) == 2
            assert c.kind in ("sub", "final_dev", "gem_it", "nasal_pt", "tone", "ru_unstressed")
            if c.kind == "sub":
                assert c.pairs


def test_ja_length_and_gemination_substitute_one_phone():
    ref = build("ja", [("おばあさん", [(["o"], 0, {}), (["b", ("aː", ["a"])], 1, {}), (["s", "a"], 1, {})]),
                       ("きって", [(["k", "i"], 1, {}), ([("tt", ["t"]), "e"], 1, {})])])
    st = sites(ref, get_contrast("ja", "vowel_length"))
    long_ = [s for s in st if s.direction == "A>B"]
    assert [(s.src, s.dst) for s in long_] == [("aː", "a")]
    e = apply(ref, long_[0])
    assert ipas(e.ref)[2] == "a" and e.ref.phones[2].alts == [] and e.flag == [2]
    assert e.partner_in_alts                       # the front-end already accepts the short vowel
    assert ipas(ref)[2] == "aː"                    # the native reference is untouched
    short = [s for s in st if s.direction == "B>A"]
    assert {(s.src, s.dst) for s in short} == {("o", "oː"), ("a", "aː"), ("i", "iː"), ("e", "eː")}
    g = sites(ref, get_contrast("ja", "gemination"))
    gem = [s for s in g if s.src == "tt"]
    assert len(gem) == 1
    e = apply(ref, gem[0])
    assert ipas(e.ref) == ["o", "b", "aː", "s", "a", "k", "i", "t", "e"] and e.word == 1


def test_it_gemination_deletes_or_inserts_a_copy():
    ref = build("it", [("fatto", [(["f", "a", ("t", [""])], 1, {"stress": 1}), (["t", "o"], 1, {"stress": 0})]),
                       ("fato", [(["f", "a"], 1, {"stress": 1}), (["t", "o"], 1, {"stress": 0})])])
    st = sites(ref, get_contrast("it", "gemination"))
    deg = [s for s in st if s.direction == "A>B"]
    gem = [s for s in st if s.direction == "B>A"]
    assert [(s.index, s.src, s.dst) for s in deg] == [(2, "tt", "t")]
    assert [(s.index, s.src, s.dst) for s in gem] == [(7, "t", "tt")]
    e = apply(ref, deg[0])
    assert ipas(e.ref) == ["f", "a", "t", "o", "f", "a", "t", "o"] and e.flag == [2] and e.partner_in_alts
    e = apply(ref, gem[0])
    assert ipas(e.ref) == ["f", "a", "t", "t", "o", "f", "a", "t", "t", "o"]
    assert e.flag == [7, 8] and not e.partner_in_alts
    # the inserted copy closes the preceding syllable (as the front-end writes geminates)
    assert e.ref.phones[7].syllable == e.ref.phones[6].syllable


def test_zh_aspiration_retroflex_coda_and_tone():
    ref = build("zh", [("怕", [(["pʰ", "a"], 1, {"tone": 4})]), ("山", [(["ʂ", "a", ("n", [""])], 1, {"tone": 1})]),
                       ("知", [(["ʈʂ", "ɨ"], 1, {"tone": 1})])])
    asp = sites(ref, get_contrast("zh", "aspiration"))
    assert [(s.src, s.dst, s.direction) for s in asp] == [("pʰ", "p", "A>B"), ("ʈʂ", "ʈʂʰ", "B>A")]
    ret = sites(ref, get_contrast("zh", "retroflex"))
    assert {(s.src, s.dst) for s in ret} == {("ʂ", "s"), ("ʈʂ", "ts")}
    nng = sites(ref, get_contrast("zh", "n_ng"))
    assert [(s.index, s.dst) for s in nng] == [(4, "ŋ")]            # coda only
    tone = sites(ref, get_contrast("zh", "tone"))
    assert len(tone) == 3
    e = apply(ref, tone[0], np.random.default_rng(1))
    assert e.ref.syllables[0].tone != 4 and ref.syllables[0].tone == 4 and e.flag == [0]


def test_final_devoicing_needs_voiced_spelling_and_skips_clitics():
    ref = build("de", [("Rad", [(["ʀ", "a", "t"], 1, {"stress": 1})]), ("Rat", [(["ʀ", "ɑː", "t"], 1, {"stress": 1})])])
    st = sites(ref, get_contrast("de", "final_devoicing"))
    assert [(s.index, s.src, s.dst) for s in st] == [(2, "t", "d")]
    # word-final positions are excluded from the general voicing contrast
    v = sites(ref, get_contrast("de", "voicing"))
    assert all(s.index not in (2, 5) for s in v)
    ru = build("ru", [("в", [(["f"], 0, {"stress": 0})]), ("город", [(["g", "o", "r"], 1, {"stress": 1}),
                                                                     (["a", "t"], 0, {"stress": 0})])])
    st = sites(ru, get_contrast("ru", "final_devoicing"))
    assert [(s.index, s.dst) for s in st] == [(5, "d")]


def test_ru_unstressed_vowel_follows_spelling():
    ref = build("ru", [("молоко", [(["m", "a"], 1, {"stress": 0}), (["ɭ", "a"], 1, {"stress": 0}),
                                   (["k", "o"], 1, {"stress": 1})])])
    st = sites(ref, get_contrast("ru", "unstressed_vowel"))
    assert [(s.index, s.src, s.dst) for s in st] == [(1, "a", "o"), (3, "a", "o")]


def test_pt_nasal_both_directions():
    ref = build("pt", [("lã", [(["l", ("a", ["ɐ̃"]), ("n", ["", "ŋ"])], 1, {"stress": 1})]),
                       ("lá", [(["l", "a"], 1, {"stress": 1})])])
    st = sites(ref, get_contrast("pt", "nasal"))
    nas = [s for s in st if s.direction == "A>B"]
    ora = [s for s in st if s.direction == "B>A"]
    assert [s.index for s in nas] == [1] and [s.index for s in ora] == [4]
    e = apply(ref, nas[0])
    assert ipas(e.ref) == ["l", "a", "l", "a"] and e.ref.phones[1].alts == [] and e.partner_in_alts
    e = apply(ref, ora[0])
    assert ipas(e.ref) == ["l", "a", "n", "l", "a", "n"] and e.flag == [4, 5]


def test_onset_filter_and_extra_direction():
    ref = build("en", [("mill", [(["m", "ɪ", "l̴"], 1, {"stress": 1})]), ("him", [(["h", "ɪ", "m"], 1, {"stress": 1})])])
    mn = sites(ref, get_contrast("en", "m_n"))
    assert [s.index for s in mn] == [0]                              # coda m excluded
    rl = sites(ref, get_contrast("en", "r_l"))
    assert [(s.src, s.dst, s.direction) for s in rl] == [("l̴", "ɹ", "B>A")]
    e = apply(ref, rl[0])
    assert ipas(e.ref)[2] == "ɹ"


def test_site_roundtrip_and_alloc():
    s = Site("x", "A>B", 3, "a", "b", kind="sub")
    assert Site(**s.__dict__) == s
    assert _alloc({"A>B": 10, "B>A": 500}, 240) == {"A>B": 10, "B>A": 230}
    assert _alloc({"A>B": 400, "B>A": 500}, 240) == {"A>B": 120, "B>A": 120}
    assert _alloc({"A>B": 0}, 240) == {"A>B": 0}
