"""Length channel (round 15, pronunciada/length.py): labels, evidence, model walk, word factor,
coaching items. No audio, no acoustic model."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pronunciada import length as L
from pronunciada.types import PhoneResult, RefPhone, RefSyllable, RefWord, Reference, WordResult


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


def test_ja_labels_follow_the_canonical_symbol_not_the_alternatives():
    ref = build("ja", [("おばあさん", [(["o"], 0, {}), (["b", ("aː", ["a"])], 1, {}), (["s", "a"], 1, {}),
                                     ([("ŋ", [])], 0, {})]),
                       ("きって", [(["k", "i"], 1, {}), ([("tt", ["t"]), "e"], 1, {})]),
                       ("かた", [(["k", "a"], 1, {}), (["t", "a"], 1, {})]),
                       ("を", [(["o"], 0, {})])])
    lab = L.labels(ref)
    ipa = [p.ipa for p in ref.phones]
    got = dict(zip(range(len(ipa)), lab))
    assert got[2] == "long"                      # aː{a}: still expected long
    assert got[0] == "short" and got[4] == "short"
    assert got[1] == "short"                     # b between o and aː (a single consonant)
    assert got[8] == "long"                      # tt
    assert got[12] == "short"                    # t between a and a
    assert got[14] is None                       # one-mora word (particle を)
    assert got[3] == "short"                     # s between a and a (next vowel a, not devoiceable)


def test_ja_vowel_sequences_and_devoicing_contexts_are_not_judged():
    ref = build("ja", [("せい", [(["s", "e"], 1, {}), (["i"], 0, {})]),
                       ("です", [(["d", "e"], 1, {}), (["s", "u"], 1, {})])])
    lab = L.labels(ref)
    assert lab[1] is None and lab[2] is None     # e i (said as a long e)
    assert lab[5] is None                        # s before u: devoicing context
    assert lab[4] == "short"


def test_explicit_length_on_the_reference_wins():
    ref = build("ja", [("かた", [(["k", "a"], 1, {}), (["t", "a"], 1, {})])])
    ref.phones[1].length = "long"
    assert L.labels(ref)[1] == "long"


def test_it_and_de_labels():
    it = build("it", [("fatto", [(["f", "a", ("t", [""])], 1, {"stress": 1}), (["t", "o"], 1, {"stress": 0})]),
                      ("fato", [(["f", "a"], 1, {"stress": 1}), (["t", "o"], 1, {"stress": 0})])])
    lab = L.labels(it)
    assert lab[2] is None and lab[3] == "long" and lab[7] == "short"
    de = build("de", [("Miete", [(["m", "iː"], 1, {"stress": 1}), (["t", "ə"], 1, {"stress": 0})]),
                      ("Mitte", [(["m", "ɪ"], 1, {"stress": 1}), (["t", "ə"], 1, {"stress": 0})])])
    lab = L.labels(de)
    assert lab[1] == "long" and lab[5] == "short" and lab[3] is None   # unstressed ə: not judged


def _inputs(n_phones, onsets, energy=None):
    return {"onset": np.asarray(onsets, np.int64), "ok": np.ones(n_phones, bool),
            "colon": np.zeros(max(onsets) + 5), "energy": np.zeros(400) + 5.0 if energy is None else energy,
            "frame_shift": 0.02}


def test_features_measure_a_long_vowel_as_longer():
    ref = build("ja", [("かあさま", [(["k", "a"], 1, {}), (["s", "aː"], 1, {}), (["m", "a"], 1, {})]),
                       ("たかた", [(["t", "a"], 1, {}), (["k", "a"], 1, {}), (["t", "a"], 1, {})])])
    lab = L.labels(ref)
    on = [0, 2, 6, 8, 16, 18, 22, 24, 28, 30, 34, 36]
    idx, cls, rows = L.features(ref, lab, _inputs(len(ref.phones), on))
    r = dict(zip(idx, rows))
    assert r[3]["r_own"] > r[1]["r_own"]                     # aː held longer than a
    assert all(c in ("v", "c") for c in cls)
    assert set(L.V_FEATURES) <= set(r[3])


def test_pause_after_a_vowel_is_not_judged():
    ref = build("ja", [("かさ", [(["k", "a"], 1, {}), (["s", "a"], 1, {})]),
                       ("かさ", [(["k", "a"], 1, {}), (["s", "a"], 1, {})])])
    E = np.full(400, 5.0)
    E[2 * 6 + 3 + 2: 2 * 40 + 3] = -5.0                       # silence after the second a
    on = [0, 2, 4, 6, 40, 42, 44, 46]
    idx, _, _ = L.features(ref, L.labels(ref), _inputs(8, on, E))
    assert 3 not in idx and 1 in idx


def test_tree_walk_matches_a_hand_computed_tree():
    # one complete depth-3 tree on feature 0: thresholds split at 0 / -1 / 1 ...; leaves 0..7
    M = {"features": ["x"], "init": 0.5, "learning_rate": 0.1,
         "feat": [[0] * 7], "thr": [[0.0, -1.0, 1.0, -2.0, -0.5, 0.5, 2.0]], "leaf": [list(range(8))]}
    z = L._z_trees(M, np.array([[-3.0], [-0.7], [0.2], [5.0]]))
    assert np.allclose(z, 0.5 + 0.1 * np.array([0, 2, 4, 7]))
    p = L.p_long({"v": M}, "v", [{"x": 5.0}])
    assert p[0] == pytest.approx(1 / (1 + math.exp(-(0.5 + 0.7))))


def test_word_factor_is_one_above_the_knee_and_monotone():
    cal = {"word": {"knee": [-6.0, -4.0, -2.0], "width": 4.0, "floor": 0.4}}
    lg = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    assert L.word_factor(cal, [lg(3.0), lg(-3.0)], 0.5) == 1.0
    vals = [L.word_factor(cal, [lg(z)], 0.5) for z in (-4.0, -5.0, -6.0, -8.0, -20.0)]
    assert vals[0] == 1.0 and all(a >= b for a, b in zip(vals, vals[1:])) and vals[-1] == pytest.approx(0.4)
    # stricter = higher knee = lower factor for the same evidence
    assert L.word_factor(cal, [lg(-5.0)], 1.0) < L.word_factor(cal, [lg(-5.0)], 0.0)


def test_shipped_japanese_channel_is_small_and_complete():
    par = L.params("ja")
    if par is None:
        pytest.skip("no shipped Japanese length channel")
    assert set(par["model"]) == {"v", "c"}
    assert {"knee", "width", "floor"} <= set(par["word"])
    assert set(par["coach"]["thresholds"]) == {"careful", "balanced", "eager"}
    from importlib import resources

    size = len(resources.files("pronunciada").joinpath("data/length/ja.json").read_bytes())
    assert size <= 50_000
    assert L.params("es") is None and L.params("zh") is None


def test_length_coaching_items_and_tips():
    from pronunciada.coach import SCHEMA_MINOR, length_items
    from pronunciada.tips import tip_for

    par = L.params("ja")
    if par is None:
        pytest.skip("no shipped Japanese length channel")
    ph = [PhoneResult("k", 0.99, 0.0, 0.1), PhoneResult("oː", 0.98, 0.1, 0.2, length_expected="long", length_prob=0.001),
          PhoneResult("tt", 0.97, 0.2, 0.3, length_expected="long", length_prob=0.9)]
    w = WordResult("こっと", 0.5, 0.0, 0.3, ph)
    res = SimpleNamespace(lang="ja", words=[w])
    items = length_items(res, "balanced")
    assert len(items) == 1 and items[0]["kind"] == "length" and items[0]["expected"] == "long"
    assert items[0]["heard"] == "short" and items[0]["expected_ipa"] == "oː" and SCHEMA_MINOR == 4
    assert "two beats" in tip_for(items[0])["text"]
    it2 = dict(items[0], expected="long", expected_ipa="tt")
    assert "Pause briefly before /t/" in tip_for(it2)["text"]
    it3 = dict(items[0], expected="short", heard="long", expected_ipa="a")
    assert "short" in tip_for(it3)["text"]
    assert length_items(SimpleNamespace(lang="es", words=[w]), "balanced") == []


def test_no_aligned_sound_and_empty_energy_do_not_crash():
    ref = build("ja", [("かさ", [(["k", "a"], 1, {}), (["s", "a"], 1, {})])])
    inp = {"onset": np.zeros(4, np.int64), "ok": np.zeros(4, bool), "colon": np.zeros(0),
           "energy": np.zeros(0), "frame_shift": 0.02}
    assert L.features(ref, L.labels(ref), inp)[0] == []
