"""Coaching feedback: articulatory table coverage, diff / tip logic, schema, audio gate."""
import json

import numpy as np
import pytest

from pronunciada import articulatory as A
from pronunciada import coach, quality, tips
from pronunciada.text.ipa import SPECIAL, load_tokens

SAMPLES = {
    "en": "The quick brown fox jumps over the lazy dog while the children watch.",
    "zh": "我们明天去北京看朋友，你想一起去吗？",
    "ja": "今日はとても暑いので、冷たい水を飲みたいです。",
    "es": "Se recomienda a los viajeros que se informen sobre el clima.",
    "fr": "Les enfants jouent dans le jardin pendant que leurs parents mangent.",
    "de": "Die Kinder spielen im Garten, während die Eltern Kaffee trinken.",
    "it": "I bambini giocano nel giardino mentre i genitori bevono il caffè.",
    "pt": "As crianças brincam no jardim enquanto os pais tomam café.",
    "ru": "Дети играют в саду, пока родители пьют чай на веранде.",
    "ko": "아이들은 정원에서 놀고 부모님은 차를 마십니다.",
}


def test_every_zipa_token_has_features():
    toks = [t for t in load_tokens() if t not in SPECIAL]
    missing = [t for t in toks if t not in A.symbols()]
    assert not missing
    for t in toks:
        if t not in A.MODIFIERS:
            f = A.features(t)
            assert f["type"] in ("consonant", "vowel")
            keys = A.CONSONANT_KEYS if f["type"] == "consonant" else A.VOWEL_KEYS
            assert set(keys) <= set(f)


def test_every_reference_phone_of_every_language_has_features():
    inv = coach.inventory
    for lang in SAMPLES:
        phones = inv(lang)
        assert len(phones) >= 20, lang
        bad = [p for p in phones if not A.covered(p)]
        assert not bad, (lang, bad)


@pytest.mark.parametrize("lang", sorted(SAMPLES))
def test_front_end_output_and_alternatives_covered(lang):
    from pronunciada.text import get_g2p

    try:
        ref = get_g2p(lang)(SAMPLES[lang])
    except Exception as e:  # noqa: BLE001  (optional text dependencies)
        pytest.skip(f"front-end unavailable: {e}")
    for p in ref.phones:
        assert A.covered(p.ipa), (lang, p.ipa)
        for a, _ in p.alts:
            assert not a or A.covered(a), (lang, a)


def test_composites_and_diff():
    assert A.features("tʃ")["manner"] == "affricate" and A.features("tʃ")["place"] == "postalveolar"
    assert A.features("tt")["long"] and A.features("tts")["manner"] == "affricate"
    assert A.features("aɪ")["glide"] == "ɪ" and A.features("aj")["glide"] == "i"
    assert A.features("ɑɻ")["rhotic"] and A.features("kʰ")["aspirated"]
    assert A.diff("t", "t") == []
    assert A.diff("s", "z") == ["voicing"]
    assert A.diff("tʰ", "t") == ["aspirated"]
    assert A.diff("i", "ɪ") == ["height"]
    assert A.diff("θ", "s") == ["place"]
    assert A.diff("l", "ɹ") == ["lateral"]
    assert A.diff("a", "t") == ["type"]
    assert A.diff("tʃ", "ʃ") == ["manner"]


def test_tip_keys_are_consistent():
    cases = {("z", "s"): "voicing.add", ("s", "z"): "voicing.remove", ("pʰ", "p"): "aspiration.add",
             ("θ", "s"): "place", ("s", "t"): "manner.continuant", ("t", "s"): "manner.stop",
             ("i", "ɪ"): "vowel.close", ("ɪ", "i"): "vowel.open", ("u", "i"): "vowel.back",
             ("y", "i"): "vowel.round", ("aɪ", "a"): "vowel.glide.add", ("ɔ̃", "ɔ"): "nasal.add",
             ("tʲ", "t"): "palatalisation.add", ("aː", "a"): "length.longer", ("n", "d"): "manner.nasal"}
    for (e, h), key in cases.items():
        k, _ = tips.sound_tip_key(e, h, A.diff(e, h))
        assert k == key, (e, h, k)
        assert k in tips.TEMPLATES


def _item(**kw):
    base = {"schema_version": 1, "kind": "substitution", "word": "think", "word_index": 0,
            "syllable_index": 0, "phone_index": 0, "position": "initial", "expected": "θ",
            "heard": "s", "expected_ipa": "θ", "heard_ipa": "s", "diff_keys": ["place"],
            "prob": 0.1, "confidence": 0.9, "start": 0.1, "end": 0.2, "gate": None}
    base.update(kw)
    return base


def test_tips_cap_and_one_per_sound():
    items = [_item(), _item(heard="f", diff_keys=["place"], prob=0.2),
             _item(word_index=1, word="tone", kind="tone", expected=2, heard=4, expected_ipa=None,
                   heard_ipa=None, diff_keys=["tone"], phone_index=None),
             _item(word_index=2, word="x", heard=None, heard_ipa=None, diff_keys=[], confidence=None),
             _item(word_index=3, word="gone", kind="missing_word", expected="gone", heard=None)]
    out = tips.tips(items, max_tips=3)
    assert len(out) == 3
    assert [t["item"]["word_index"] for t in out] == [0, 1, 3]     # diagnosed before flag-only
    assert out[0]["tip_key"] == "place" and "/θ/" in out[0]["text"]
    assert out[1]["tip_key"] == "tone.wrong" and "rising" in out[1]["text"] and "falling" in out[1]["text"]
    for key, tmpl in tips.TEMPLATES.items():
        assert tmpl.count(".") <= 2 and tmpl.endswith("."), key


def test_gate_file_and_overrides():
    g = coach.gate_params()
    assert set(coach.DEFAULT_GATE) <= set(g)
    assert coach.gate_params(min_log_ratio=9.0)["min_log_ratio"] == 9.0
    edges, vals = g["confidence"]["edges"], g["confidence"]["values"]
    assert len(vals) == len(edges) + 1 and all(0 <= v <= 1 for v in vals)


def test_model_file():
    m = coach.model()
    assert m["version"] == 2 and len(m["sound"]["coef"]) == len(m["sound"]["features"])
    t = m["thresholds"]
    assert t["careful"] > t["balanced"] > t["eager"]
    for lang in SAMPLES:
        for s in coach.SENSITIVITIES:
            assert 0 < coach.sound_threshold(m, lang, s, True) <= 1.01
    assert m["tone"]["thresholds"]["careful"] <= m["tone"]["thresholds"]["eager"]
    with pytest.raises(ValueError):
        coach.coaching(object(), sensitivity="loud")


def test_heard_candidate_picks_dominant_sound():
    from pronunciada.text.ipa import token_ids

    m = np.full(127, 1e-4)
    m[token_ids()["s"]] = 0.6
    m[token_ids()["θ"]] = 0.01
    h, lh, le = coach.heard_candidate(m, "θ", [], "en")
    assert h == "s" and lh > le


# ----------------------------------------------------------------------------- audio gate

def _speechy(sec=3.0, level=0.1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * 16000)) / 16000
    env = (np.sin(2 * np.pi * 3 * t) > 0).astype(float)            # 3 "syllables" per second
    x = level * env * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * rng.standard_normal(len(t)))
    x = np.r_[np.zeros(8000), x, np.zeros(8000)] + 1e-4 * rng.standard_normal(len(t) + 16000)
    return x.astype(np.float32)


def test_audio_gate_codes():
    clean = quality.check(_speechy(), n_syllables=9, lang="en")
    assert clean["issues"] == [] and clean["retry_recommended"] is False
    json.dumps(clean)
    clip = quality.check(np.clip(_speechy() * 40, -1, 1), 9, "en")
    assert "clipped" in clip["issues"]
    rng = np.random.default_rng(1)
    noisy = quality.check(_speechy() + 0.08 * rng.standard_normal(len(_speechy())).astype(np.float32), 9, "en")
    assert "too_noisy" in noisy["issues"]
    quiet = quality.check(_speechy() * 1e-3, 9, "en")
    assert "too_quiet" in quiet["issues"]
    short = quality.check(_speechy(sec=1.0), 30, "en")
    assert "too_short" in short["issues"]
    sil = quality.check(np.r_[np.zeros(160000, np.float32), _speechy(sec=0.5)], None, "en")
    assert "mostly_silence" in sil["issues"]
    assert quality.check(np.zeros(16000, np.float32))["retry_recommended"]


@pytest.mark.model
def test_coaching_on_scorer_result(tts):
    import pronunciada

    sc = pronunciada.Scorer("en", threads=2)
    wav = tts("The quick brown fox jumps over the lazy dog")
    res = sc.score((wav, 16000), "The quick brown cat jumps over the lazy dog")
    assert res.audio is not None and "issues" in res.audio
    items = res.coaching()
    json.dumps(items)
    omitted = {wi for wi, w in enumerate(res.words) if w.present_prob is not None and w.present_prob < 0.5}
    X = coach.sound_inputs(res)
    ep = coach.error_prob(X)
    thr = coach.sound_threshold(coach.model(), "en", "balanced", False)
    flagged = {(int(X["word"][i]), int(X["k"][i])) for i in range(len(ep)) if ep[i] >= thr and X["word"][i] not in omitted}
    assert {(it["word_index"], it["phone_index"]) for it in items if it["kind"] in ("substitution", "missing")} == flagged
    for it in items:
        assert it["schema_version"] == coach.SCHEMA_VERSION and it["kind"] in coach.KINDS
        if it["heard_ipa"]:
            assert it["diff_keys"] == A.diff(it["expected_ipa"], it["heard_ipa"])
            assert coach.sound_class(it["expected_ipa"]) not in coach.model()["heard"]["off_classes"]
    ranks = [coach._rank(it) for it in items]  # noqa: SLF001
    assert ranks == sorted(ranks, reverse=True)
    # sensitivities are nested: careful <= balanced <= eager
    key = lambda L: {(i["word_index"], i["phone_index"]) for i in L if i["kind"] == "substitution"}  # noqa: E731
    assert key(res.coaching(sensitivity="careful")) <= key(items) <= key(res.coaching(sensitivity="eager"))
    assert len(tips.tips(items)) <= 3
    # rescore keeps the audio report and coaching still works
    r2 = sc.rescore(res, 0.9)
    assert r2.audio == res.audio and isinstance(r2.coaching(), list)
