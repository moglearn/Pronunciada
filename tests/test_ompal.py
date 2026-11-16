"""OMPAL builder / evaluator logic (no network, no corpus needed)."""
import numpy as np
import pytest

from bench.eval_ompal import (auc_fn, boot_indices, syllable_char_spans, unit_score, unit_table,
                              word_char_spans)
from bench.ompal_build import fleiss_kappa, is_native, make_split, speaker_of, unit_label, unit_spans


def test_speaker_of():
    assert speaker_of("00200101") == "SPEAKER02001"
    assert speaker_of("00100322") == "SPEAKER01003"
    with pytest.raises(ValueError):
        speaker_of("2001")


def test_split_natives_always_test_and_disjoint():
    spk = ["SPEAKER01001", "SPEAKER01003", "SPEAKER02001", "SPEAKER02002", "SPEAKER02003"]
    sp = make_split(spk, ["SPEAKER01001", "SPEAKER02002"])
    assert sp == {"SPEAKER01001": "test", "SPEAKER01003": "test", "SPEAKER02001": "dev",
                  "SPEAKER02002": "test", "SPEAKER02003": "dev"}
    assert all(is_native(s) for s, v in sp.items() if v == "test" and s.startswith("SPEAKER01"))


def test_unit_spans_clean():
    spans, bad = unit_spans("他們都很忙", list("他們都很忙"))
    assert bad is None and spans == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def test_unit_spans_inserted_char_keeps_prompt_char():
    text = "孩子們還可以"
    spans, bad = unit_spans(text, ["孩", "子", "們", "還也", "可", "以"])
    assert bad is None and spans[3] == (3, 4) and spans[4] == (4, 5)


def test_unit_spans_extra_units_dropped_and_missing_units_ignored():
    spans, bad = unit_spans("所以這一天", list("所以這一天正月"))
    assert bad is None and spans[5:] == [None, None] and spans[:5] == [(i, i + 1) for i in range(5)]
    spans, _ = unit_spans("所以這一天正月", list("所以這一天"))
    assert spans == [(i, i + 1) for i in range(5)]


def test_unit_spans_malformed():
    spans, bad = unit_spans("正月初一", ["正人", "月們", "初", "一"])
    assert bad and spans == [None] * 4


def test_unit_label():
    lab = unit_label({"tone": 0, "phoneme_consonant": 1, "phoneme_vowel": 1})
    assert lab["tone_only"] == 1 and lab["seg_wrong"] == 0 and lab["any_wrong"] == 1 and lab["correct"] == 0
    lab = unit_label({"tone": 0, "phoneme_consonant": 0, "phoneme_vowel": 1})
    assert lab["tone_only"] == 0 and lab["seg_wrong"] == 1


def test_word_and_syllable_spans():
    assert word_char_spans("叮鈴鈴上課鈴響了", ["叮鈴鈴", "上課", "鈴響", "了"]) == [(0, 3), (3, 5), (5, 7), (7, 8)]
    assert word_char_spans("abc", ["x", "b"]) == [None, (1, 2)]
    assert syllable_char_spans((3, 5), 2) == [(3, 4), (4, 5)]
    assert syllable_char_spans((0, 3), 2) == [(0, 3), (0, 3)]


def test_unit_score_min_and_missing():
    items = [(0, 2, 0.9), (2, 3, 0.4), (2, 4, 0.7)]
    assert unit_score((1, 2), items) == 0.9
    assert unit_score((2, 3), items) == 0.4
    assert unit_score((5, 6), items) == 0.0
    assert np.isnan(unit_score((5, 6), items, missing=np.nan))
    assert unit_score((0, 1), [(0, 1, None), (0, 2, 0.5)]) == 0.5


def _rec(uid, spk, native, units, text):
    return {"id": uid, "speaker": spk, "text": text,
            "labels": {"native": native, "word_level_dropped": None, "units": units}}


def _u(span, wrong):
    lab = unit_label({"tone": 0 if wrong == "t" else 1, "phoneme_consonant": 0 if wrong == "c" else 1,
                      "phoneme_vowel": 1})
    return {"span": span, **lab}


def test_unit_table_failures_count_as_lowest():
    recs = [_rec("a", "S1", False, [_u([0, 1], "t"), _u([1, 2], None), _u(None, None)], "他們")]
    ours = {"a": {"status": "Success", "words": [[0, 2, 0.8, 0.9, 0.7]], "syllables": [[0, 1, 0.2], [1, 2, 0.95]]}}
    az = {"zh-TW": {"a": {"status": "NoMatch"}}}
    U = unit_table(recs, ours, az)
    c = U["cols"]
    assert list(c["ours_word"]) == [0.8, 0.8]
    assert list(c["ours_tone"]) == [0.2, 0.95]
    assert list(c["az_zh-TW"]) == [0.0, 0.0]
    assert U["counts"]["units_not_in_prompt"] == 1 and U["counts"]["azure_zh-TW_failed_utts"] == 1


def test_auc_fn_and_cluster_bootstrap():
    pos = np.array([1, 1, 0, 0, 0], bool)
    neg = ~pos
    good = np.array([0.1, 0.2, 0.9, 0.8, 0.7])
    assert auc_fn(pos, neg, good)(np.arange(5)) == 1.0
    B = boot_indices(np.array(["x", "x", "y", "y", "y"]), n_boot=20)
    for b in B:   # whole clusters are resampled: members of a cluster appear equally often
        assert np.sum(b == 0) == np.sum(b == 1)
        assert np.sum(b == 2) == np.sum(b == 3) == np.sum(b == 4)


def test_fleiss_kappa():
    assert fleiss_kappa([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]) == pytest.approx(1.0)
    assert fleiss_kappa([[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]]) < 0
