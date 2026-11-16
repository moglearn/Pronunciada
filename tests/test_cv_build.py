"""Selection logic of the cv-real benchmark (bench/cv_build.py, bench/eval_cv.py). No network."""
import json
import random
from collections import Counter

import numpy as np
import pytest

from bench import cv_build as CB
from bench import eval_cv as EV


def _r(stem, spk, sent, n_chars=None, dur=3.0, up=0, down=3, accents=""):
    return {"stem": stem, "client_id": spk, "sentence_id": "s" + sent, "sentence": sent,
            "norm": CB.norm_sentence(sent), "n_chars": n_chars if n_chars is not None else len(sent),
            "dur": dur, "up_votes": up, "down_votes": down, "accents": accents, "variant": "",
            "src_split": "invalidated", "src_shard": 0}


def test_vote_rules():
    assert CB.is_rejected(0, 2) and CB.is_rejected(1, 3) and CB.is_rejected(3, 5)
    assert not CB.is_rejected(1, 2) and not CB.is_rejected(0, 1) and not CB.is_rejected(2, 3)
    assert CB.is_accepted(2, 0) and CB.is_accepted(5, 0)
    assert not CB.is_accepted(1, 0) and not CB.is_accepted(3, 1)


def test_text_and_duration_rules():
    assert CB.text_rule("我们在2020年", "zh-CN") == "digits_in_prompt"
    assert CB.text_rule("我喜欢 iPhone", "zh-TW") == "latin_in_prompt"
    assert CB.text_rule("我喜欢ｉＰｈｏｎｅ", "zh-CN") == "latin_in_prompt"      # full-width Latin
    assert CB.text_rule("Привет мир", "ru") is None
    assert CB.text_rule("Привет OK", "ru") == "latin_in_prompt"
    assert CB.text_rule("Il y a 3 chats", "fr") is None                     # Latin script: not checked
    assert CB.text_rule("  ", "en") == "empty_prompt"
    assert CB.dur_rule(0.9) == "too_short" and CB.dur_rule(15.5) == "too_long"
    assert CB.dur_rule(1.0) is None and CB.dur_rule(15.0) is None and CB.dur_rule(None) == "no_duration"


def test_near_silent():
    sr = 16000
    rng = np.random.default_rng(0)
    silence = 1e-4 * rng.standard_normal(3 * sr).astype(np.float32)      # about -80 dBFS
    assert CB.near_silent(silence, sr)
    speech = silence.copy()
    t = np.arange(sr) / sr
    speech[sr:2 * sr] += 0.1 * np.sin(2 * np.pi * 200 * t)                 # 1 s at about -23 dBFS
    assert not CB.near_silent(speech, sr)
    blip = silence.copy()
    blip[sr:sr + 1600] += 0.1                                              # 0.1 s only
    assert CB.near_silent(blip, sr)


def test_speaker_half_is_deterministic_and_balanced():
    spk = [f"client{i}" for i in range(2000)]
    a = [CB.speaker_half(s) for s in spk]
    assert a == [CB.speaker_half(s) for s in spk]
    assert 0.45 < a.count("test") / len(a) < 0.55
    assert CB.speaker_half("x", seed=1) in ("test", "dev")


def test_cap_per_speaker():
    rows = [_r(f"c{i}", f"s{i % 3}", f"sent {i}") for i in range(30)]
    out = CB.cap_per_speaker(rows, 2, random.Random(0))
    assert Counter(r["client_id"] for r in out) == {"s0": 2, "s1": 2, "s2": 2}
    assert [r["stem"] for r in out] == [r["stem"] for r in CB.cap_per_speaker(rows, 2, random.Random(0))]


def test_make_disjoint_speakers_and_sentences():
    pos = [_r(f"p{i}", f"s{i}", f"sentence {i % 20}") for i in range(60)]
    neg = [_r(f"n{i}", f"s{i + 30}", f"sentence {i % 25}", up=3, down=0) for i in range(60)]
    p, n, cnt = CB.make_disjoint(pos, neg)
    assert not ({r["client_id"] for r in p} & {r["client_id"] for r in n})
    assert not ({r["norm"] for r in p} & {r["norm"] for r in n})
    assert cnt["speaker_on_both_sides"] > 0 and cnt["sentence_on_both_sides"] > 0
    # shared items are split between the sides, not all given to one
    assert p and n


def test_norm_sentence_matches_punctuation_and_case_variants():
    assert CB.norm_sentence("Hello, World!") == CB.norm_sentence("hello world")
    assert CB.norm_sentence("你好，世界。") == "你好世界"


def test_allocate_equal_quota_capped_and_proportional():
    q = CB.allocate({(0, 0): 100, (1, 0): 50, (2, 0): 50}, {(0, 0): 30, (1, 0): 500, (2, 0): 500}, 100)
    assert sum(q.values()) == 100
    assert q[(0, 0)] == 30                         # capped by the scarce side
    assert q[(1, 0)] >= 25 and q[(2, 0)] >= 25     # deficit refilled elsewhere
    q2 = CB.allocate({(0, 0): 5}, {(0, 0): 3}, 200)
    assert q2 == {(0, 0): 3}                       # never more than both sides have
    assert sum(CB.allocate({}, {(0, 0): 3}, 10).values()) == 0


def test_select_matched_equal_counts_per_stratum():
    rng = np.random.default_rng(0)
    pos = [_r(f"p{i}", f"ps{i}", f"p{i}", n_chars=int(rng.integers(10, 80)), dur=float(rng.uniform(1, 10)))
           for i in range(400)]
    # accepted clips are systematically longer: matching must remove that
    neg = [_r(f"n{i}", f"ns{i}", f"n{i}", n_chars=int(rng.integers(20, 120)), dur=float(rng.uniform(2, 14)),
              up=3, down=0) for i in range(1500)]
    q, lp, ln, edges = CB.select_matched(pos, neg, 200, seed=1, reserve=1.0)
    assert sum(q.values()) == 200
    for s, k in q.items():
        if k:
            assert len(lp[s]) == len(ln[s]) == k
    P = [r for v in lp.values() for r in v]
    N = [r for v in ln.values() for r in v]
    assert abs(np.mean([r["n_chars"] for r in P]) - np.mean([r["n_chars"] for r in N])) < 8
    assert abs(np.mean([r["dur"] for r in P]) - np.mean([r["dur"] for r in N])) < 1.0
    # reserves: longer lists, same order prefix
    q2, lp2, ln2, _ = CB.select_matched(pos, neg, 200, seed=1, reserve=1.35)
    assert q2 == q
    assert all(len(lp2[s]) >= len(lp[s]) for s in lp)


def test_finalize_stratum_keeps_sides_equal():
    p = [{"stem": f"p{i}", "ok": i % 3 != 0} for i in range(10)]
    n = [{"stem": f"n{i}", "ok": i != 1} for i in range(10)]
    a, b = CB.finalize_stratum(5, p, n, lambda r: r["ok"])
    assert len(a) == len(b) == 5 and all(r["ok"] for r in a + b)
    a, b = CB.finalize_stratum(9, p, n, lambda r: r["ok"])
    assert len(a) == len(b) == 6                  # only 6 positives pass


def test_build_test_halves_disjoint_and_capped():
    rng = np.random.default_rng(1)
    pos = [_r(f"p{i}", f"spk{i % 300}", f"pos sentence {i}", n_chars=int(rng.integers(10, 60)),
              dur=float(rng.uniform(1, 8))) for i in range(1200)]
    neg = [_r(f"n{i}", f"spk{(i % 400) + 200}", f"neg sentence {i}", n_chars=int(rng.integers(10, 60)),
              dur=float(rng.uniform(1, 8)), up=2, down=0) for i in range(1600)]
    out = CB.build_test(pos, neg, seed=5, target=50, reserve=1.0)
    spk_half = {}
    for half, h in out.items():
        P = [r for v in h["pos"].values() for r in v]
        N = [r for v in h["neg"].values() for r in v]
        assert len(P) == len(N) == 50
        assert not ({r["client_id"] for r in P} & {r["client_id"] for r in N})
        assert max(Counter(r["client_id"] for r in P + N).values()) <= CB.PER_SPEAKER
        for r in P + N:
            assert spk_half.setdefault(r["client_id"], half) == half
            assert CB.speaker_half(r["client_id"], 5) == half


def test_nativeness_and_regions():
    assert CB.nativeness("en", "German English") == "nonnative"
    assert CB.nativeness("en", "Non Native - French speaker") == "nonnative"
    assert CB.nativeness("en", "United States English") == "native"
    assert CB.nativeness("en", "India and South Asia (India, Pakistan, Sri Lanka)") == "unknown"
    assert CB.nativeness("en", "United States English,German") == "unknown"
    assert CB.nativeness("en", "cajun french") == "unknown"
    assert CB.nativeness("en", "") == "unknown"
    assert CB.nativeness("de", "Russisch Deutsch") == "nonnative"
    assert CB.nativeness("de", "Amerikanisches Deutsch") == "nonnative"
    assert CB.nativeness("de", "Österreichisches Deutsch") == "native"
    assert CB.nativeness("de", "Deutschland Deutsch,Französisch Deutsch") == "unknown"
    assert CB.nativeness("fr", "Français d’Allemagne") == "nonnative"
    assert CB.nativeness("fr", "Français de France") == "native"
    assert CB.nativeness("fr", "Français du Maroc") == "unknown"
    assert CB.es_region("México") == "latam"
    assert CB.es_region("España: Centro-Sur peninsular (Madrid, Toledo, Castilla-La Mancha)") == "spain"
    assert CB.es_region("") == "none"
    assert CB.pt_region("Paulista", "") == "brazil"
    assert CB.pt_region("", "Portuguese (Portugal)") == "portugal"


def test_bootstrap_auc_and_two_sided_p():
    rng = np.random.default_rng(0)
    n = 300
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    spk = [f"s{i // 2}" for i in range(2 * n)]
    good_a = np.where(y == 1, rng.normal(0, 1, 2 * n), rng.normal(1.5, 1, 2 * n))   # strong
    good_b = np.where(y == 1, rng.normal(0, 1, 2 * n), rng.normal(0.2, 1, 2 * n))   # weak
    r = EV.speaker_bootstrap(y, good_a, good_b, spk, n_boot=300)
    assert r["ours"] > 0.8 and r["azure"] < 0.65 and r["p_two_sided"] < 0.01
    assert r["ours_ci"][0] < r["ours"] < r["ours_ci"][1]
    r2 = EV.speaker_bootstrap(y, good_a, good_a, spk, n_boot=200)
    assert r2["delta"] == 0 and r2["p_two_sided"] == 1.0


def test_azure_failures_score_lowest(tmp_path, monkeypatch):
    monkeypatch.setenv("PRONUNCIADA_CACHE", str(tmp_path))
    d = tmp_path / "azure" / EV.DATASET
    d.mkdir(parents=True)
    ok = {"RecognitionStatus": "Success", "NBest": [{"Lexical": "hi", "AccuracyScore": 80, "FluencyScore": 90,
                                                      "CompletenessScore": 100, "PronScore": 85, "Words": []}]}
    (d / "a.json").write_text(json.dumps({"response": ok, "attempts": 2, "audio_dur": 3.0}))
    (d / "b.json").write_text(json.dumps({"response": {"RecognitionStatus": "NoMatch"}, "audio_dur": 2.0}))
    (d / "c.json").write_text(json.dumps({"response": {"RecognitionStatus": "HTTP400", "error": "x"}}))
    recs = [{"id": i, "dur": 1.0} for i in ("a", "b", "c", "d")]
    az = EV.azure_scores(recs)
    assert set(az) == {"a", "b", "c"}
    assert az["a"]["PronScore"] == 85 and az["a"]["status"] == "Success"
    assert az["b"]["PronScore"] == 0 and az["b"]["status"] == "NoMatch"
    assert az["c"]["AccuracyScore"] == 0 and az["c"]["status"] == "HTTP400"
    u = EV.azure_usage(az)
    assert u["clips"] == 3 and u["requests"] == 4


def test_evaluate_counts_failures_not_drops():
    recs, ours, az = [], {}, {}
    rng = np.random.default_rng(3)
    for i in range(40):
        lab = i % 2
        rid = f"r{i}"
        recs.append({"id": rid, "locale": "fr", "test": "A", "label": lab, "client_id": f"s{i}",
                     "dur": 3.0, "n_chars": 30})
        ours[rid] = {"status": "Success", "prob": float(rng.uniform()) - 0.3 * lab,
                     **{k: 0.5 for k in EV.OURS_SECONDARY}}
        az[rid] = {"status": "Success" if i > 3 else "NoMatch", "PronScore": 0.0 if i <= 3 else 50.0 + i,
                   **{k: 1.0 for k in EV.AZ_SECONDARY}}
    ev = EV.evaluate(recs, ours, az)["fr/A"]
    assert ev["n_scored_both"] == 40 and ev["azure_failures"] == {"NoMatch": 4}
    assert ev["azure_failures_by_side"] == {1: 2, 0: 2}
    assert "primary" in ev and "azure:FluencyScore" in ev["secondary"]


def test_weak_reject_rule():
    assert CB.is_weak_rejected(0, 1) and CB.is_weak_rejected(0, 3)
    assert not CB.is_weak_rejected(1, 1) and not CB.is_weak_rejected(0, 0) and not CB.is_weak_rejected(1, 0)
    assert "ko" in CB.WEAK_LOCALES and CB.WEAK_MIN_POS == 60
