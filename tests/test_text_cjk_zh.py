"""Offline tests for the Mandarin text front-end."""
from __future__ import annotations

import pytest

from pronunciada.text import get_g2p

from _cjk_helpers import speed_ms, syl_ipa, tones  # noqa: F401


pytest.importorskip("pypinyin")
pytest.importorskip("jieba")
pytest.importorskip("opencc")
zh = get_g2p("zh")

ZH_SENTS = [
    "今天天气很好，我们一起去公园散步吧。",
    "你好，我很好。",
    "展览馆里有一只小老虎。",
    "他不是我的朋友，是不是？",
    "银行行长说长江很长。",
    "哪儿有一点儿水？",
    "1995年有3.5%的人去过北京，2个人买了10000本书。",
    "我喜欢喝茶ABC和咖啡。",
    "他慢慢地走了，跑得很快。",
    "重庆是一个重要的城市。",
    "她看着书，对不起，差不多了。",
]


@pytest.mark.parametrize("text", ZH_SENTS)
def test_zh_validate(text):
    ref = zh(text)
    ref.validate()
    assert ref.syllables and all(s.tone in (1, 2, 3, 4, 5) for s in ref.syllables)
    assert all(s.tone_lexical in (1, 2, 3, 4, 5) for s in ref.syllables)


def test_zh_zipa_conventions():
    ref = zh("今天天气很好我们一起去公园散步吧")
    assert "".join(syl_ipa(ref)) == "tɕintʰjɛntʰjɛntɕʰixənxɑʊwɔmənitɕʰitɕʰykʊŋɥœnsanpupa"
    # one nucleus per syllable; glide / coda separate phones
    gong = ref.syllables[11]
    assert [p.ipa for p in ref.phones[gong.phones[0]:gong.phones[1]]] == ["k", "ʊ", "ŋ"]
    assert syl_ipa(zh("是字有用"))[:4] == ["ʂɨ", "tsɯ", "yoʊ", "yʊŋ"]


def test_zh_third_tone_sandhi():
    assert tones(zh("你好")) == [2, 3]
    r = zh("我很好")
    assert tones(r) == [2, 2, 3]
    assert 3 in r.syllables[0].tone_alts and 3 in r.syllables[1].tone_alts
    assert [s.tone_lexical for s in r.syllables] == [3, 3, 3]
    assert tones(zh("展览馆")) == [2, 2, 3]          # [[AB]C]
    r = zh("小老虎")                                   # [A[BC]]: 3 2 3, alt 2 2 3
    assert tones(r) == [3, 2, 3] and 2 in r.syllables[0].tone_alts


def test_zh_yi_bu_sandhi():
    assert tones(zh("一个")) == [2, 4]
    assert tones(zh("一起")) == [4, 3]
    assert tones(zh("一天")) == [4, 1]
    assert tones(zh("第一")) == [4, 1]
    assert tones(zh("统一")) == [3, 1]
    assert tones(zh("一百"))[0] == 4
    assert tones(zh("不是")) == [2, 4]
    assert tones(zh("不好")) == [4, 3]
    assert tones(zh("看一看"))[1] == 5
    assert tones(zh("是不是"))[1] == 5
    assert [s.tone_lexical for s in zh("一个不是").syllables] == [1, 4, 4, 4]


def test_zh_neutral_tone_and_taiwan_alts():
    r = zh("我们的朋友")
    assert tones(r) == [3, 5, 5, 2, 5]
    assert 3 in r.syllables[4].tone_alts            # Taiwan: péngyǒu
    assert r.syllables[2].tone_alts == []            # particle 的: no full-tone alternative
    assert 2 in zh("星期").syllables[1].tone_alts    # Taiwan xīngqí


def test_zh_simplified_traditional_identical():
    from opencc import OpenCC

    for cfg in ("s2t", "s2tw", "s2hk"):
        cc = OpenCC(cfg)
        for s in ZH_SENTS + ["他的头发很干净，干活的时候发了短信。", "著名作家正看着书，土著居民唱着歌。",
                             "这只鸟在台风里飞，后来他后悔了。", "余下的时间里，他复习了复杂的课程。"]:
            t = cc.convert(s)
            a, b = zh(s), zh(t)
            assert syl_ipa(a) == syl_ipa(b), (cfg, t)
            assert [(x.tone, x.tone_lexical, x.tone_alts) for x in a.syllables] == \
                   [(x.tone, x.tone_lexical, x.tone_alts) for x in b.syllables], (cfg, t)
            assert [p.alts for p in a.phones] == [p.alts for p in b.phones]
    r = zh("我們的朋友")
    assert [s.text for s in r.syllables] == list("我們的朋友")      # user's script kept
    assert r.words[0].text == "我們"


def test_zh_erhua_numbers_latin():
    r = zh("哪儿")
    assert len(r.syllables) == 1 and r.syllables[0].text == "哪儿"
    assert r.phones[-1].ipa == "ɻ" and any(a for a, _ in r.phones[-1].alts)
    assert len(zh("女儿").syllables) == 2           # not erhua
    r = zh("2019年")
    assert [s.text for s in r.syllables] == list("二零一九年") and r.words[0].text == "2019"
    assert [s.text for s in zh("10000").syllables] == ["一", "万"]
    assert [s.text for s in zh("3.5%").syllables] == list("百分之三点五")
    assert [s.text for s in zh("2个").syllables] == ["两", "个"]
    assert "".join(s.text for s in zh("1,400").syllables) == "一千四百"
    r = zh("我喜欢ABC")
    assert len(r.syllables) == 3 and any("ABC" in w for w in r.warnings)


def test_zh_alternatives():
    r = zh("面包很好吃")
    n = r.phones[r.syllables[0].phones[1] - 1]       # coda n of 面 before p -> m
    assert n.ipa == "n" and ("m", -1.0) in n.alts


def test_zh_speed():
    assert speed_ms(zh, ZH_SENTS) < 5.0
