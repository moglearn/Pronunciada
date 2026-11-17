import pytest

pytest.importorskip("panphon")

from bench.corrupt import (KINDS, Inventory, NotApplicable, PhoneDistance,
                           applicable_kinds, corrupt)
from pronunciada.text.ipa import is_vowel
from pronunciada.types import RefPhone, RefSyllable, RefWord, Reference


def build(lang, words):
    """words: [(text, [ (phones, nucleus_idx, {syllable attrs}) ... ]) ...]"""
    ref = Reference(lang=lang, text=" ".join(w for w, _ in words))
    for wi, (wt, syls) in enumerate(words):
        s0, p0 = len(ref.syllables), len(ref.phones)
        for phones, nuc, attrs in syls:
            si, ps = len(ref.syllables), len(ref.phones)
            for k, ph in enumerate(phones):
                alts = []
                if isinstance(ph, tuple):
                    ph, alts = ph
                ref.phones.append(RefPhone(ph, wi, si, k == nuc, alts))
            ref.syllables.append(RefSyllable(word=wi, phones=(ps, len(ref.phones)), **attrs))
        ref.words.append(RefWord(wt, (s0, len(ref.syllables)), (p0, len(ref.phones))))
    ref.validate()
    return ref


def en_refs():
    a = build("en", [("the", [(["ð", "ə"], 1, {"stress": 0})]),
                     ("cat", [(["k", "æ", ("t", [("ɾ", -1.0)])], 1, {"stress": 1})]),
                     ("sat", [(["s", "æ", "t"], 1, {"stress": 1})]),
                     ("on", [(["ɑ", "n"], 0, {"stress": 1})]),
                     ("paper", [(["p", "eɪ"], 1, {"stress": 1}), (["p", "ə˞"], 1, {"stress": 0})])])
    b = build("en", [("a", [(["ə"], 0, {"stress": 0})]),
                     ("big", [(["b", "ɪ", "g"], 1, {"stress": 1})]),
                     ("dog", [(["d", "ɑ", "g"], 1, {"stress": 1})]),
                     ("fish", [(["f", "ɪ", "ʃ"], 1, {"stress": 1})]),
                     ("began", [(["b", "ɪ"], 1, {"stress": 0}), (["g", "æ", "n"], 1, {"stress": 1})]),
                     ("see", [(["s", "i"], 1, {"stress": 1})]),
                     ("much", [(["m", "ʌ", "tʃ"], 1, {"stress": 1})]),
                     ("then", [(["ð", "ɛ", "n"], 1, {"stress": 1})]),
                     ("zoo", [(["z", "u"], 1, {"stress": 1})])])
    return [a, b, b]


def zh_ref():
    return build("zh", [("你好", [(["n", "i"], 1, {"tone": 2, "tone_lexical": 3, "text": "你"}),
                                  (["x", "aʊ"], 1, {"tone": 3, "tone_lexical": 3, "text": "好"})]),
                        ("吗", [(["m", "a"], 1, {"tone": 5, "tone_alts": [1]})])])


def ja_ref():
    return build("ja", [("さくら", [(["s", "a"], 1, {}), (["k", "u"], 1, {}), (["ɾ", "a"], 1, {})])])


@pytest.fixture(scope="module")
def inv():
    return Inventory.from_references(en_refs(), min_count=1)


def test_distance_sanity(tmp_path):
    d = PhoneDistance(tmp_path / "d.json")
    assert d("p", "p") == 0
    assert d("p", "b") < d("p", "s") < d("p", "a")
    assert d("i", "ɪ") < d("i", "a")
    assert d("tʃ", "ʃ") < d("tʃ", "m")     # affricate close to its fricative
    assert d("p", "b") == d("b", "p")
    d.save()
    assert PhoneDistance(tmp_path / "d.json").table


@pytest.mark.parametrize("kind", ["phone_sub_near", "phone_sub_far", "phone_del", "phone_ins",
                                  "word_swap", "stress"])
def test_english_kinds_validate(kind, inv):
    ref = en_refs()[0]
    for seed in range(15):
        c = corrupt(ref, kind, inv, seed=seed)
        c.ref.validate()
        assert len(c.errors) == 1 and c.errors[0].kind == kind
        assert sum(c.word_wrong) == 1
        assert len(c.phone_wrong) == len(c.ref.phones)
        assert len(c.syllable_flags) == len(c.ref.syllables)
        assert ref.phones[1].ipa == "ə"   # original untouched
        wi = c.errors[0].word
        assert c.word_wrong[wi]
        for pi, wrong in enumerate(c.phone_wrong):
            if wrong:
                assert c.ref.phones[pi].word == wi
                assert c.ref.phones[pi].alts == []


def test_substitution_properties(inv):
    ref = en_refs()[0]
    for seed in range(40):
        for kind in ("phone_sub_near", "phone_sub_far"):
            c = corrupt(ref, kind, inv, seed=seed)
            (pi,) = [i for i, w in enumerate(c.phone_wrong) if w]
            old, new = ref.phones[pi], c.ref.phones[pi]
            assert new.ipa != old.ipa and new.ipa not in {a for a, _ in old.alts}
            assert is_vowel(new.ipa) == is_vowel(old.ipa)
            assert new.is_nucleus == old.is_nucleus
            if old.ipa == "t" and old.alts:
                assert new.ipa != "ɾ"


def test_near_closer_than_far(inv):
    ref = en_refs()[0]
    near = [corrupt(ref, "phone_sub_near", inv, seed=s).errors[0].detail["distance"] for s in range(30)]
    far = [corrupt(ref, "phone_sub_far", inv, seed=s).errors[0].detail["distance"] for s in range(30)]
    assert sum(near) / len(near) < sum(far) / len(far)


def test_del_and_ins_shapes(inv):
    ref = en_refs()[0]
    c = corrupt(ref, "phone_del", inv, seed=3)
    assert len(c.ref.phones) == len(ref.phones) - 1 and not any(c.phone_wrong)
    c = corrupt(ref, "phone_ins", inv, seed=3)
    assert len(c.ref.phones) == len(ref.phones) + 1 and sum(c.phone_wrong) == 1
    (pi,) = [i for i, w in enumerate(c.phone_wrong) if w]
    assert not c.ref.phones[pi].is_nucleus and not is_vowel(c.ref.phones[pi].ipa)


def test_word_swap_marks_only_differing_phones(inv):
    ref = en_refs()[0]
    for s in range(20):
        c = corrupt(ref, "word_swap", inv, seed=s)
        wi = c.errors[0].word
        w = c.ref.words[wi]
        assert w.text == ref.words[wi].text
        new = tuple(c.ref.phones[i].ipa for i in range(*w.phones))
        old = tuple(ref.phones[i].ipa for i in range(*ref.words[wi].phones))
        assert new != old
        assert any(c.phone_wrong[i] for i in range(*w.phones)) or len(new) < len(old)


def test_multiple_errors_distinct_words(inv):
    ref = en_refs()[1]
    c = corrupt(ref, list(KINDS[:5]), inv, n_errors=3, seed=7)
    assert len(c.errors) == 3 and len({e.word for e in c.errors}) == 3 and sum(c.word_wrong) == 3
    a = corrupt(ref, list(KINDS[:5]), inv, n_errors=3, seed=7)
    assert a.to_dict() == c.to_dict()   # seeded


def test_stress_move(inv):
    ref = en_refs()[0]
    c = corrupt(ref, "stress", inv, seed=0)
    w = c.ref.words[4]
    assert [c.ref.syllables[i].stress for i in range(*w.syllables)] == [0, 1]
    assert c.syllable_flags[4:6] == [["stress"], ["stress"]]
    one = build("en", [("cat", [(["k", "æ", "t"], 1, {"stress": 1})])])
    with pytest.raises(NotApplicable):
        corrupt(one, "stress", inv)


def test_tone():
    ref = zh_ref()
    for s in range(20):
        c = corrupt(ref, "tone", seed=s)
        c.ref.validate()
        (si,) = [i for i, f in enumerate(c.syllable_flags) if f]
        old = ref.syllables[si]
        new = c.ref.syllables[si]
        assert new.tone != old.tone and new.tone not in old.tone_alts and new.tone in (1, 2, 3, 4)
        assert c.syllable_flags[si] == ["tone"] and not any(c.phone_wrong)
    with pytest.raises(NotApplicable):
        corrupt(en_refs()[0], "tone")


def test_applicable_kinds():
    assert "tone" in applicable_kinds(zh_ref()) and "stress" not in applicable_kinds(zh_ref())
    assert not {"tone", "stress"} & set(applicable_kinds(ja_ref()))     # no prosodic kind for Japanese
    assert "stress" in applicable_kinds(en_refs()[0])


def test_nucleus_only_syllables_never_lose_nucleus(inv):
    ref = build("en", [("a", [(["ə"], 0, {"stress": 0})]), ("I", [(["aɪ"], 0, {"stress": 1})])])
    with pytest.raises(NotApplicable):
        corrupt(ref, "phone_del", inv)
