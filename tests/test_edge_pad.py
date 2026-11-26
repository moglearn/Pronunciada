"""Silence added around tightly trimmed clips before the speech model (issue #3)."""
import numpy as np
import pytest

from pronunciada.audio import EDGE_FRAME, SAMPLE_RATE, edge_pad


def _speech(seconds, amp=0.3, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(seconds * SAMPLE_RATE)) * amp).astype(np.float32)


def _sil(seconds, db=None, seed=1):
    n = int(seconds * SAMPLE_RATE)
    if db is None:
        return np.zeros(n, np.float32)
    return (np.random.default_rng(seed).standard_normal(n) * 10 ** (db / 20)).astype(np.float32)


def test_enough_silence_gets_nothing():
    assert edge_pad(np.concatenate([_sil(0.5), _speech(0.6), _sil(0.5)])) == (0, 0)
    # background noise before and after the speech is silence too
    assert edge_pad(np.concatenate([_sil(0.5, -55), _speech(0.6), _sil(0.5, -55)])) == (0, 0)


def test_tight_clip_is_topped_up():
    want = round(0.3 * SAMPLE_RATE / EDGE_FRAME)
    assert edge_pad(_speech(0.6)) == (want, want)
    k, m = edge_pad(np.concatenate([_sil(0.04), _speech(0.6), _sil(0.5)]))
    assert k == want - 2 and m == 0


def test_normal_margin_is_left_alone():
    # 0.1 s before the speech is short of the 0.3 s target but not tightly cut: nothing changes
    assert edge_pad(np.concatenate([_sil(0.1), _speech(0.6), _sil(0.1)])) == (0, 0)


def test_silent_or_off():
    assert edge_pad(np.zeros(SAMPLE_RATE, np.float32)) == (0, 0)
    assert edge_pad(np.zeros(10, np.float32)) == (0, 0)
    assert edge_pad(_speech(0.6), target_s=0.0) == (0, 0)


@pytest.mark.model
def test_padding_keeps_the_clip_timeline(tts, monkeypatch):
    import pronunciada
    from pronunciada.audio import SAMPLE_RATE as SR

    text = "The quick brown fox jumps over the lazy dog"
    wav = tts(text)
    a = np.flatnonzero(np.abs(wav) > 0.02)
    tight = np.ascontiguousarray(wav[max(a[0] - 160, 0): a[-1] + 160])
    assert edge_pad(tight) != (0, 0)
    sc = pronunciada.Scorer("en", threads=4)
    lp, fb, emb = sc._posteriors(tight, {})
    assert len(lp) == sc.am.num_output_frames(len(fb))
    assert all(len(v) == len(lp) for v in (emb or {}).values())
    res = sc.score((tight, SR), text)
    dur = len(tight) / SR
    assert all(0.0 <= w.start <= w.end <= dur + 0.05 for w in res.words)


@pytest.mark.model
def test_clip_with_silence_is_unchanged(tts, monkeypatch):
    import pronunciada

    wav = np.concatenate([_sil(0.4), tts("The quick brown fox"), _sil(0.4)])
    assert edge_pad(wav) == (0, 0)
    sc = pronunciada.Scorer("en", threads=4)
    on = sc._posteriors(wav, {})[0].copy()
    sc._lp_cache.clear()
    monkeypatch.setenv("PRONUNCIADA_EDGE_PAD", "0")
    off = sc._posteriors(wav, {})[0]
    assert np.array_equal(on, off)
