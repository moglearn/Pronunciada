"""Pitch tracker on synthetic signals: known pitch, glides, octave traps, silence, speed."""
import time

import numpy as np
import pytest

from pronunciada.prosody.f0 import n_frames, track

SR = 16000


def harmonic(f0, dur=None, amps=None, sr=SR, seed=0):
    """Glottal-like harmonic signal for a per-sample f0 array (or constant)."""
    if np.isscalar(f0):
        f0 = np.full(int(dur * sr), float(f0))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    amps = amps or [1.0 / k for k in range(1, 11)]
    x = sum(a * np.sin(k * ph) for k, a in enumerate(amps, start=1))
    x += 0.003 * np.random.default_rng(seed).standard_normal(len(x))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


@pytest.mark.parametrize("f", [60, 85, 120, 180, 250, 340, 450, 560])
def test_constant_pitch(f):
    tr = track(harmonic(f, 1.0))
    mid = slice(10, -10)
    assert tr.voiced[mid].mean() > 0.95
    err = np.abs(tr.f0[mid][tr.voiced[mid]] / f - 1)
    assert np.median(err) < 0.01
    assert (err > 0.2).mean() < 0.02


def test_frame_grid():
    for n in (0, 1, 159, 160, 161, 16000, 16123):
        assert len(track(np.zeros(n, np.float32)).f0) == n_frames(n) == (n + 80) // 160


def test_glides():
    t = np.arange(int(1.5 * SR)) / SR
    for f in (100 * 2 ** (t / 1.5), 320 * 2 ** (-1.3 * t / 1.5), 150 + 60 * np.sin(2 * np.pi * 2 * t)):
        tr = track(harmonic(f))
        ref = np.interp(tr.times, t, f)
        ok = tr.voiced & (tr.times > 0.05) & (tr.times < 1.45)
        assert ok.mean() > 0.85
        assert np.median(np.abs(tr.f0[ok] / ref[ok] - 1)) < 0.015
        assert (np.abs(tr.f0[ok] / ref[ok] - 1) > 0.2).mean() < 0.03


def test_octave_traps():
    # weak fundamental (missing F0 energy), strong even harmonics -> must still report F0
    amps = [0.05, 1.0, 0.3, 0.8, 0.2, 0.5, 0.1, 0.3]
    tr = track(harmonic(130, 1.0, amps=amps))
    v = tr.voiced[10:-10]
    assert v.mean() > 0.9
    assert np.median(tr.f0[10:-10][v]) == pytest.approx(130, rel=0.02)
    # alternating-period amplitude (+-10 %, mild vocal-fry like): the dip at 2T is deeper than at
    # T; we must still report 200 Hz (Praat ac reports 100 Hz here; at >= 15 % both say 100 Hz)
    x = harmonic(200, 1.0)
    t = np.arange(len(x)) / SR
    x2 = (x * (1 + 0.10 * np.sign(np.sin(np.pi * 200 * t + 0.1)))).astype(np.float32)
    tr = track(x2)
    assert np.median(tr.f0[tr.voiced]) == pytest.approx(200, rel=0.02)


def test_silence_and_noise_unvoiced():
    rng = np.random.default_rng(1)
    sil = np.zeros(SR, np.float32)
    noise = (0.05 * rng.standard_normal(SR)).astype(np.float32)
    assert track(sil).voiced.mean() < 0.02
    assert track(noise).voiced.mean() < 0.1
    # voiced island inside silence
    x = np.concatenate([sil[:8000], harmonic(200, 0.5), sil[:8000]])
    tr = track(x)
    assert tr.voiced[60:90].mean() > 0.95 and tr.voiced[:40].mean() < 0.05 and tr.voiced[-40:].mean() < 0.05


def test_resampled_input():
    x = harmonic(150, 1.0, sr=8000)
    tr = track(x, sr=8000)
    assert len(tr.f0) == n_frames(16000)
    assert np.median(tr.f0[tr.voiced]) == pytest.approx(150, rel=0.02)


def test_speed_10s():
    x = np.tile(harmonic(140 + 40 * np.sin(np.arange(SR) / SR * 6)), 10)
    track(x)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        track(x)
        ts.append(time.perf_counter() - t)
    assert min(ts) < 0.05           # budget 15 ms; loose bound for loaded CI machines
