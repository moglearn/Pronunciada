"""Public API: pronunciada.Scorer / pronunciada.score / CLI."""
import dataclasses
import json
import time

import numpy as np
import pytest

import pronunciada
from pronunciada.types import SentenceResult

TEXT = "The quick brown fox jumps over the lazy dog"
WRONG = "The quick brown cat jumps over the lazy dog"


@pytest.fixture(scope="module")
def scorer():
    return pronunciada.Scorer("en", threads=4)


@pytest.fixture(scope="module")
def fox(tts):
    return tts(TEXT)


def test_import_is_lazy_and_exports():
    assert {"Scorer", "score", "SentenceResult", "WordResult", "PhoneResult"} <= set(pronunciada.__all__)


@pytest.mark.model
def test_contract(scorer, fox):
    res = scorer.score((fox, 16000), TEXT)
    assert isinstance(res, SentenceResult)
    assert res.lang == "en" and res.strictness == 0.5 and 0 <= res.prob <= 1
    assert [w.text for w in res.words] == TEXT.split()
    assert 0 <= res.fluency <= 1 and 0 <= res.completeness <= 1
    for k in ("load", "g2p", "fbank", "model", "adapter", "align", "features", "calibrate", "total"):
        assert k in res.timing_ms
    dur = len(fox) / 16000
    prev_end = 0.0
    for w in res.words:
        assert 0 <= w.prob <= 1 and 0 <= w.sounds_prob <= 1
        assert w.start <= w.end <= dur + 0.1
        assert w.start >= prev_end - 0.05
        prev_end = w.end
        assert w.syllables and all(s.stress_expected is not None for s in w.syllables)
        for p in w.phones:
            assert 0 <= p.prob <= 1 and p.start <= p.end
            assert (p.heard is None) == (p.prob >= 0.5)
    assert res.prob > 0.5                       # correct text, clean TTS speech
    json.dumps(dataclasses.asdict(res))         # serialisable


@pytest.mark.model
def test_strictness_monotone_and_rescore(scorer, fox):
    res = scorer.score(fox, WRONG)
    out = [scorer.rescore(res, s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for a, b in zip(out, out[1:]):
        assert b.prob <= a.prob + 1e-12
        for wa, wb in zip(a.words, b.words):
            assert wb.prob <= wa.prob + 1e-12
            for pa, pb in zip(wa.phones, wb.phones):
                assert pb.prob <= pa.prob + 1e-12
    assert set(out[0].timing_ms) == {"rescore"}            # no model / alignment re-run
    assert out[0].timing_ms["rescore"] < 20
    same = scorer.rescore(res, 0.5)
    assert same.prob == pytest.approx(res.prob) and same.words[3].prob == pytest.approx(res.words[3].prob)
    direct = scorer.score(fox, WRONG, strictness=1.0)
    assert direct.prob == pytest.approx(out[-1].prob)
    assert scorer.rescore(res, 9).strictness == 1.5         # clipped


@pytest.mark.model
def test_deterministic(scorer, fox):
    a = scorer.score(fox, TEXT)
    scorer._lp_cache.clear()
    b = scorer.score(fox, TEXT)
    assert a.prob == b.prob
    assert [p.prob for w in a.words for p in w.phones] == [p.prob for w in b.words for p in w.phones]


@pytest.mark.model
def test_wrong_word_is_lowest(scorer, fox):
    good = scorer.score(fox, TEXT)
    bad = scorer.score(fox, WRONG)
    probs = [w.prob for w in bad.words]
    assert int(np.argmin(probs)) == 3, probs               # "cat" said as "fox"
    assert bad.words[3].prob < good.words[3].prob
    assert bad.prob <= good.prob


@pytest.mark.model
def test_inputs_and_edge_cases(scorer, fox, tmp_path):
    import soundfile as sf
    from scipy.signal import resample_poly

    with pytest.raises(ValueError):
        scorer.score(fox, "   ")
    with pytest.raises(ValueError):
        scorer.score(fox, "")
    # stereo 44.1 kHz file, bytes, path
    st = np.stack([resample_poly(fox, 441, 160)] * 2, axis=1).astype(np.float32)
    p = tmp_path / "stereo.wav"
    sf.write(p, st, 44100)
    r1 = scorer.score(str(p), TEXT)
    r2 = scorer.score(p.read_bytes(), TEXT)
    assert r1.prob == pytest.approx(r2.prob)
    assert abs(r1.prob - scorer.score(fox, TEXT).prob) < 0.2
    # silence, very short audio, text far too long for the audio
    sil = scorer.score(np.zeros(16000, np.float32), TEXT)
    assert any("silent" in w for w in sil.warnings) and sil.prob < 0.5
    short = scorer.score(fox[:1600], TEXT)
    assert any("short" in w for w in short.warnings) and short.prob < 0.5
    long = scorer.score(fox[:8000], " ".join([TEXT] * 6))
    assert long.prob < 0.5 and all(0 <= w.prob <= 1 for w in long.words)
    empty = scorer.score(np.zeros(0, np.float32), "dog")
    assert 0 <= empty.prob <= 1


@pytest.mark.model
def test_convenience_and_cli(fox, tmp_path, capsys):
    import soundfile as sf

    r = pronunciada.score((fox, 16000), TEXT, "en", strictness=1.0)
    assert r.strictness == 1.0
    p = tmp_path / "fox.wav"
    sf.write(p, fox, 16000)
    from pronunciada.__main__ import main

    assert main(["score", str(p), TEXT, "--lang", "en"]) == 0
    out = capsys.readouterr().out
    assert "quick" in out and "timing ms" in out
    assert main(["score", str(p), TEXT, "--lang", "en", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["lang"] == "en"
    assert main(["score", str(p), " ", "--lang", "en"]) == 2


@pytest.mark.model
def test_other_language_and_prosody_fields(tts):
    sc = pronunciada.Scorer("es", threads=4)
    res = sc.score(tts("hola"), "hola")       # English TTS voice: just the contract
    assert res.lang == "es" and len(res.words) == 1
    w = res.words[0]
    assert w.tone_prob is None
    assert all(s.tone_prob is None for s in w.syllables)


@pytest.mark.model
def test_latency_budget(scorer, fox):
    scorer.score(fox, TEXT)
    best = np.inf
    for _ in range(5):
        scorer._lp_cache.clear()
        t = time.perf_counter()
        res = scorer.score(fox, TEXT)
        best = min(best, (time.perf_counter() - t) * 1e3)
    assert res.timing_ms["calibrate"] < 15
    assert best < 400, best                   # ~2.5 s of speech, 4 threads, end to end


@pytest.mark.model
def test_sentence_scores_field(scorer, fox):
    res = scorer.score(fox, TEXT)
    assert set(res.scores) == {"accuracy", "fluency", "prosodic", "total"}
    assert all(0 <= v <= 1 for v in res.scores.values())
    assert res.scores["fluency"] == pytest.approx(res.fluency)
    d = scorer.details(res)["sentence"]
    assert res.scores["accuracy"] == pytest.approx(min(max(d["accuracy"] / 10, 0), 1))


@pytest.mark.model
def test_prosody_channel(tts, monkeypatch):
    """WordResult.prob = sounds_prob x P_pros x present_prob where the language has a fitted
    prosody channel and the word has scored syllables; sounds_prob unchanged; monotone in
    strictness; omitted = present_prob < 0.5. (The round-19 sound-identity step, which also moves
    the Mandarin word P, is switched off here; tests/test_round19_rollout.py covers it.)"""
    monkeypatch.setenv("PRONUNCIADA_IDENTITY", "0")
    for lang, text in (("en", TEXT), ("zh", "你好")):
        sc = pronunciada.Scorer(lang, threads=4)
        res = sc.score(tts("ni hao" if lang == "zh" else TEXT), text)
        for s in (0.0, 0.5, 1.0):
            r = sc.rescore(res, s)
            for w in r.words:
                pp = w.tone_prob if lang == "zh" else w.stress_prob
                assert 0 <= w.present_prob <= 1 and w.omitted == (w.present_prob < 0.5)
                if sc.pros_cal is None or pp is None:
                    assert w.prob == pytest.approx(w.sounds_prob * w.present_prob)
                else:
                    assert w.prob == pytest.approx(w.sounds_prob * pp * w.present_prob)
                    assert 0 <= pp <= 1
        strict = sc.rescore(res, 1.0)
        assert all(b.prob <= a.prob + 1e-12 for a, b in zip(res.words, strict.words))


@pytest.mark.model
def test_explain_away(fox):
    on = pronunciada.Scorer("en", threads=4, explain_away=True)
    off = pronunciada.Scorer("en", threads=4, explain_away=False)
    a, b = off.score(fox, WRONG), on.score(fox, WRONG)
    assert int(np.argmin([w.sounds_prob for w in b.words])) == 3
    assert b.words[3].sounds_prob == pytest.approx(a.words[3].sounds_prob)   # kept from pass 1
    assert b.prob == pytest.approx(a.prob)                                     # sentence: pass 1
    c = b._pronunciada
    if c.get("explained"):
        assert 3 in c["explained"] and "explain_away" in b.timing_ms
    # correct text: no word below the threshold -> nothing changes
    g1, g2 = off.score(fox, TEXT), on.score(fox, TEXT)
    if not g2._pronunciada.get("explained"):
        assert [w.prob for w in g1.words] == pytest.approx([w.prob for w in g2.words])
    assert on.rescore(b, 1.0).words[3].sounds_prob <= b.words[3].sounds_prob + 1e-12


@pytest.mark.model
def test_no_prosody_channel_for_japanese():
    """Japanese pitch accent is not scored (removed in round 15): no prosody model, no channel."""
    sc = pronunciada.Scorer("ja", threads=2)
    assert sc.prosody is None and sc.pros_cal is None


@pytest.mark.model
def test_omission_channel(fox):
    """A word that is not in the audio gets a low present_prob; the spoken ones keep a high one."""
    sc = pronunciada.Scorer("en", threads=4)
    good = sc.score(fox, TEXT)
    assert all(w.present_prob > 0.5 and not w.omitted for w in good.words)
    extra = sc.score(fox, TEXT.rstrip(".") + " tomorrow afternoon.")
    assert extra.words[-1].present_prob < 0.5 and extra.words[-1].omitted
    assert extra.completeness < 1.0 == good.completeness
