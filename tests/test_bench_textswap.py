"""bench.textswap (text-level planted errors) and bench.eval_swap (system-agnostic evaluation)."""
from __future__ import annotations

import numpy as np
import pytest

from bench import textswap as TS
from bench import eval_swap as ES
from pronunciada.text import get_g2p


# ----------------------------------------------------------------------------- helpers

def test_edit_distance_and_pair_rules():
    assert TS.edit_distance(("k", "a", "t"), ("k", "a", "t")) == 0
    assert TS.edit_distance(("k", "a", "t"), ("b", "a", "t")) == 1
    assert TS.edit_distance(("a",), ("x", "y", "z")) == 3
    assert TS.is_near(("k", "a", "t"), ("b", "a", "t"))
    assert not TS.is_near(("k", "a", "t"), ("k", "a", "t"))
    assert not TS.is_near(("k", "a", "t"), ("k", "a", "t", "s"))
    assert TS.is_far(("k", "a", "t"), ("d", "o", "g"))
    assert not TS.is_far(("k", "a", "t"), ("k", "a", "p"))                 # 1/3 < 60 %
    assert not TS.is_far(("a",), ("x", "y", "z", "w"))                      # length +3


def test_rng_is_deterministic():
    assert TS.rng_for("u1", "swap_far").random() == TS.rng_for("u1", "swap_far").random()
    assert TS.rng_for("u1", "swap_far").random() != TS.rng_for("u1", "swap_near").random()


# ----------------------------------------------------------------------------- units per script

def test_units_space_language_strips_punctuation():
    ref = get_g2p("en")("However, the slow (communication) channels.")
    us = TS.units(ref, "en")
    assert [u.text for u in us] == ["However", "the", "slow", "communication", "channels"]
    for u in us:
        assert ref.text[u.span[0]:u.span[1]] == u.text
        assert u.phones
    assert us[2].phones == ("s", "l", "oʊ")


def test_units_zh_are_refwords_with_per_hanzi_syllables():
    ref = get_g2p("zh")("我们明天去学校。")
    us = TS.units(ref, "zh")
    assert [u.text for u in us] == [w.text for w in ref.words]
    for u in us:
        assert ref.text[u.span[0]:u.span[1]] == u.text
        assert len(u.syl) == len(u.text)
        for (off, ph, tone, tlex), ch in zip(u.syl, u.text):
            assert ref.text[off] == ch and ph and tone in (1, 2, 3, 4, 5)


def test_units_ja_and_ko():
    ref = get_g2p("ja")("私は東京の学校に行きます。")
    us = TS.units(ref, "ja")
    assert "".join(u.text for u in us) == "私は東京の学校に行きます"
    ref = get_g2p("ko")("나는 학교에 갑니다.")
    assert [u.text for u in TS.units(ref, "ko")] == ["나는", "학교에", "갑니다"]


def test_source_filter_rejects_latin_in_cjk():
    g = get_g2p("zh")
    assert TS.source_ok("zh", "我们明天去学校。", g("我们明天去学校。"))
    assert not TS.source_ok("zh", "我们明天去NASA。", g("我们明天去NASA。"))
    assert not TS.source_ok("zh", "我们有3个学校。", g("我们有3个学校。"))


def test_jamo_variants():
    vs = [w for _, w in TS.jamo_variants("가")]
    assert "카" in vs and "까" in vs and "거" in vs
    assert all(len(w) == 1 for w in vs)


# ----------------------------------------------------------------------------- generation

EN_TEXTS = ["the cat sat on the mat today", "a big dog ran in the park", "the bat flew over the house",
            "my dog likes the park", "the house is big", "a hat on the cat", "the cat sat down"] * 2


def _check(lang, text, g2p, vs, variant, rel):
    new_text, lab = vs[variant]
    ou = TS.units(g2p(text), lang)
    nu = TS.units(g2p(new_text), lang)
    k = lab["word_index"]
    assert 0 < k < len(ou) - 1
    assert [w["text"] for w in lab["words"]] == [u.text for u in nu]
    a, b = lab["char_span_in_new_text"]
    assert new_text[a:b] == lab.get("new_char", lab["new_word"])
    for i, (x, y) in enumerate(zip(ou, nu)):
        if i != k:
            assert (x.text, x.phones, x.tones) == (y.text, y.phones, y.tones)
    assert rel(ou[k], nu[k])
    return lab


def test_en_far_and_near_swaps():
    g = get_g2p("en")
    vocab = TS.Vocab("en", EN_TEXTS, g)
    text = "Yesterday the cat sat on the mat."
    vs = TS.make_variants("t1", "en", text, g, vocab)
    assert vs["orig"][0] == text and vs["orig"][1]["word_index"] is None
    _check("en", text, g, vs, "swap_near", lambda a, b: TS.is_near(a.phones, b.phones))
    _check("en", text, g, vs, "swap_far", lambda a, b: TS.is_far(a.phones, b.phones))
    assert TS.make_variants("t1", "en", text, g, vocab) == vs          # deterministic


def test_zh_near_far_tone():
    g = get_g2p("zh")
    texts = ["我们明天去学校。", "他们今天在家里看书。", "这是一本新书。", "汤很好喝。", "老师说话很慢。",
             "学生在教室里学习。", "我买了一张票。", "妈妈做饭很好吃。", "他的爸爸是医生。", "我们去公园玩。",
             "天气很好。", "朋友来我家。", "中国很大。", "我想喝水。", "他跑得很快。"] * 3
    vocab = TS.Vocab("zh", texts, g)
    text = "我们今天去公园看书。"
    vs = TS.make_variants("z1", "zh", text, g, vocab)
    assert set(vs) >= {"orig"}
    for v, rel in (("swap_near", lambda s: TS.is_near(s["orig_syllable"], s["new_syllable"]) and s["orig_tone"] == s["new_tone"]),
                   ("swap_far", lambda s: TS.is_far(s["orig_syllable"], s["new_syllable"])),
                   ("tone", lambda s: s["orig_syllable"] == s["new_syllable"] and s["orig_tone"] != s["new_tone"])):
        if v not in vs:
            continue
        lab = _check("zh", text, g, vs, v, lambda a, b: True)
        assert rel(lab)
        assert len(lab["new_char"]) == 1 and lab["new_char"] != lab["orig_char"]
    assert "swap_far" in vs


def test_ko_near_falls_back_to_jamo():
    g = get_g2p("ko")
    vocab = TS.Vocab("ko", ["오늘 날씨가 좋다", "나는 학교에 갑니다", "친구가 집에 왔다"] * 2, g)
    text = "오늘 나는 학교에 갑니다."
    vs = TS.make_variants("k1", "ko", text, g, vocab)
    lab = _check("ko", text, g, vs, "swap_near", lambda a, b: TS.is_near(a.phones, b.phones))
    assert lab["source"] in ("fleurs", "jamo")


def test_ja_far_swap():
    g = get_g2p("ja")
    texts = ["私は東京の学校に行きます。", "明日は大阪で会議があります。", "毎日図書館で勉強します。",
             "友達と映画を見ました。", "新しい自転車を買いました。"] * 2
    vocab = TS.Vocab("ja", texts, g)
    text = "私は東京の学校に行きます。"
    vs = TS.make_variants("j1", "ja", text, g, vocab)
    if "swap_far" in vs:
        _check("ja", text, g, vs, "swap_far", lambda a, b: TS.is_far(a.phones, b.phones))


# ----------------------------------------------------------------------------- evaluation

def _w(word, acc, et="None"):
    return {"word": word, "accuracy": acc, "error_type": et}


def test_azure_map_space_language_sequential_tokens():
    text = "The cat, sadly, sat on 25 mats."
    parsed = {"words": [_w("the", 100), _w("cat", 90), _w("sadly", 0, "Omission"), _w("uh", 50, "Insertion"),
                        _w("sat", 80), _w("on", 95), _w("twenty", 70), _w("five", 60), _w("mats", 85)]}
    o = ES.azure_map(text, "en", parsed)
    units = [{"text": t, "span": [a, b]} for t, a, b in ES._text_tokens(text)]
    assert ES.unit_scores(units, o) == [1.0, 0.9, 0.0, 0.8, 0.95, 0.6, 0.85]


def test_azure_map_unmapped_is_missing_not_error():
    text = "the cat sat"
    o = ES.azure_map(text, "en", {"words": [_w("the", 100), _w("dog", 40), _w("sat", 90)]})
    units = [{"text": t, "span": [a, b]} for t, a, b in ES._text_tokens(text)]
    assert ES.unit_scores(units, o) == [1.0, None, 0.9]


def test_azure_map_cjk_characters():
    text = "我们去学校。"
    parsed = {"words": [_w("我们", 95), _w("去", 40), _w("学校", 0, "Omission")]}
    o = ES.azure_map(text, "zh", parsed)
    units = [{"text": "我们", "span": [0, 2]}, {"text": "去", "span": [2, 3]}, {"text": "学校", "span": [3, 5]}]
    assert ES.unit_scores(units, o) == [0.95, 0.4, 0.0]


def test_unit_scores_min_and_omission():
    units = [{"span": [0, 3]}, {"span": [4, 7]}, {"span": [8, 9]}]
    out = {"words": [(0, 2, 0.9), (1, 3, 0.5), (4, 7, 0.8)]}
    assert ES.unit_scores(units, out) == [0.5, 0.8, 0.0]


def test_hit_at_fa_splits_ties_exactly():
    native = np.array([1.0] * 90 + [0.5] * 10)
    # 5 % of natives flagged: half of the 0.5 ties -> a positive at 0.5 counts half
    assert ES._hit_at_fa(native, np.array([0.5]), 0.05) == pytest.approx(0.5)
    assert ES._hit_at_fa(native, np.array([0.1]), 0.05) == pytest.approx(1.0)
    assert ES._hit_at_fa(native, np.array([1.0]), 0.05) == pytest.approx(0.0)


def test_evaluate_perfect_and_chance_systems():
    recs, good, bad = [], {}, {}
    rng = np.random.default_rng(0)
    for i in range(30):
        src = f"s{i}"
        words = [{"text": f"w{j}", "span": [3 * j, 3 * j + 2]} for j in range(6)]
        recs.append({"id": f"{src}__orig", "lang": "en", "labels": {"variant": "orig", "src_id": src,
                     "word_index": None, "char_span_in_new_text": None, "words": words}})
        k = 1 + i % 4
        recs.append({"id": f"{src}__swap_far", "lang": "en", "labels": {
            "variant": "swap_far", "src_id": src, "word_index": k,
            "char_span_in_new_text": words[k]["span"], "words": words}})
        for rid, low in ((f"{src}__orig", None), (f"{src}__swap_far", k)):
            good[rid] = {"status": "Success", "words": [(w["span"][0], w["span"][1], 0.1 if j == low else 0.9 + 0.01 * j)
                                                        for j, w in enumerate(words)],
                         "sentence": {"prob": 0.5 if low else 0.9}}
            bad[rid] = {"status": "Success", "words": [(w["span"][0], w["span"][1], float(rng.random()))
                                                       for w in words]}
    r = ES.evaluate(recs, good, n_boot=50)["en"]["metrics"]
    assert r["swap_far:auc"]["value"] == 1.0
    assert r["swap_far:hit@5%FA"]["value"] == 1.0
    assert r["swap_far:right_word"]["value"] == 1.0
    assert r["swap_far:sent_auc_prob"]["value"] == 1.0
    assert r["native:fa<60"]["value"] == 0.0
    r = ES.evaluate(recs, bad, n_boot=50)["en"]["metrics"]
    assert 0.2 < r["swap_far:auc"]["value"] < 0.8


def test_sentence_result_adapter():
    from pronunciada.types import SentenceResult, WordResult

    res = SentenceResult(lang="en", text="a cat", prob=0.7, strictness=0.5,
                         words=[WordResult("a", 0.9, 0, 0.1), WordResult("cat", 0.2, 0.1, 0.4, omitted=True)])
    o = ES.sentence_result_output(res, [(0, 1), (2, 5)])
    assert o["words"] == [(0, 1, 0.9), (2, 5, 0.0)] and o["sentence"] == {"prob": 0.7}
