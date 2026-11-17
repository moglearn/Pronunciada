"""Round 13: Japanese accent classes, word-level pitch judgement, pitch_accent coaching items."""
from types import SimpleNamespace as NS

from pronunciada import coach, tips
from pronunciada.prosody import pitch_judge as PJ


def test_drop_after():
    assert PJ.drop_after([False, True, True], False) == 2          # odaka: falls onto the particle
    assert PJ.drop_after([False, True, True], True) is None        # heiban
    assert PJ.drop_after([True, False, False], None) == 0          # atamadaka
    assert PJ.drop_after([False, True, False], None) == 1          # nakadaka


def test_class_key_and_frontend():
    from pronunciada.text import get_g2p

    assert PJ.class_key("名詞/一般/head/c1/h") == "名詞/head/single/h"
    assert PJ.class_key(None) is None
    r = get_g2p("ja")("日本語を勉強しています。")
    assert all(s.pitch_class for s in r.syllables)
    assert r.syllables[0].pitch_class.startswith("名詞/")
    z = get_g2p("zh")("你好")
    assert all(s.pitch_class is None for s in z.syllables)


def _fake(p_high, labels):
    syl = [NS(pitch_high=h, pitch_class="名詞/一般/head/c1/n", text=t) for h, t in labels]
    ref = NS(words=[NS(syllables=(0, len(syl)))], syllables=syl)
    return ref


def test_judge_and_coach_item():
    labels = [(False, "ニ"), (True, "ホ"), (False, "ン")]          # nakadaka: falls after ホ
    ref = _fake(None, labels)
    cfg = {"classes": ["名詞/head/single/n"], "thresholds": {"careful": 3.0, "balanced": 0.5, "eager": 0.3}}
    ok = PJ.judge([0.1, 0.9, 0.1], ref, cfg)[0]
    assert ok["expected_drop"] == 1 and ok["heard_drop"] == 1 and not ok["flag"]["balanced"]
    bad = PJ.judge([0.1, 0.9, 0.9], ref, cfg)[0]                    # no fall heard
    assert bad["heard_drop"] is None and bad["flag"]["balanced"] and not bad["flag"]["careful"]
    result = NS(words=[NS(text="二本", start=0.0, end=0.5)])
    pr = [{"pitch_word": bad}] * 3
    items = coach.pitch_accent_items(result, ref, pr, set(), "balanced")
    assert len(items) == 1
    it = items[0]
    assert it["kind"] == "pitch_accent" and it["expected"] == 1 and it["heard"] == "none"
    assert it["schema_minor"] == coach.SCHEMA_MINOR >= 2
    t = tips.tip_for(it)
    assert t["tip_key"] == "pitch_accent.fall_missing" and "ホ" in t["text"]
    assert coach.pitch_accent_items(result, ref, pr, set(), "careful") == []
    assert "pitch_accent" in coach.KINDS
