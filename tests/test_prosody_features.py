"""Prosodic frame features, syllable windows and network inputs: shapes and invariants."""
import numpy as np

from pronunciada.prosody.f0 import F0Track, track
from pronunciada.prosody.features import (CTX_LEN, N_FEAT, N_IN, N_STRESS_EXTRA, context_input,
                                      contour_vector, frame_features, stress_extra,
                                      syllable_windows, zipa_to_10ms)
from pronunciada.types import RefPhone, RefSyllable, RefWord, Reference

SR = 16000


def _tone_wav(segments, gap=0.05):
    """Concatenate harmonic 'syllables' with given (f_start, f_end, dur) and short silences."""
    out = []
    for f0, f1, d in segments:
        n = int(d * SR)
        f = np.linspace(f0, f1, n)
        ph = 2 * np.pi * np.cumsum(f) / SR
        x = sum(np.sin(k * ph) / k for k in range(1, 8))
        out += [0.2 * x / np.abs(x).max(), np.zeros(int(gap * SR))]
    return np.concatenate(out).astype(np.float32)


def _ref(n_syl):
    phones, syls = [], []
    for i in range(n_syl):
        phones += [RefPhone("t", 0, i), RefPhone("a", 0, i, is_nucleus=True)]
        syls.append(RefSyllable(0, (2 * i, 2 * i + 2), text="a", tone=1 + i % 4))
    return Reference("zh", "x", [RefWord("x", (0, n_syl), (0, 2 * n_syl))], syls, phones)


def test_frame_features_shape_and_normalisation():
    wav = _tone_wav([(120, 180, 0.3), (200, 120, 0.3), (150, 150, 0.3)])
    tr = track(wav)
    f = frame_features(tr)
    assert f.shape == (len(tr.f0), N_FEAT) and f.dtype == np.float32
    v = tr.voiced
    assert abs(f[v, 0].mean()) < 0.05 and abs(f[v, 0].std() - 1) < 0.05
    assert np.isfinite(f).all()
    # interpolation: unvoiced frames between voiced ones lie between their neighbours
    assert (np.abs(np.diff(f[:, 0])) < 1.0).all()
    # rising first syllable -> positive slope; falling second -> negative
    a = f[5:25, 3].mean()
    b = f[40:60, 3].mean()
    assert a > 0 > b


def test_frame_features_unvoiced_utterance():
    tr = F0Track(np.zeros(50, np.float32), np.zeros(50, bool), np.zeros(50, np.float32),
                 np.full(50, -60.0, np.float32))
    f = frame_features(tr)
    assert f.shape == (50, N_FEAT) and np.isfinite(f).all() and (f[:, 0] == 0).all()


def test_syllable_windows_and_inputs():
    ref = _ref(4)
    # phone spans in ZIPA frames (20 ms): onset + nucleus per syllable; syllable 2's onset dropped
    spans = [(2, 4), (4, 9), (12, 13), (13, 20), (22, 22), (23, 30), (32, 33), (33, 41)]
    T = 2 * 45 + 7
    w = syllable_windows(ref, spans, T)
    assert len(w) == 4
    assert w[0].tok_start == zipa_to_10ms(2) == 7 and w[0].tok_end == zipa_to_10ms(9)
    for a, b in zip(w, w[1:]):
        assert a.end <= b.start
    for x in w:
        assert 0 <= x.start <= x.nuc_start < x.nuc_end <= x.end <= T
    assert (w[2].nuc_start, w[2].nuc_end) == (zipa_to_10ms(23), zipa_to_10ms(30))
    feats = np.random.default_rng(0).standard_normal((T, N_FEAT)).astype(np.float32)
    for i in range(4):
        x = context_input(feats, w, i)
        assert x.shape == (CTX_LEN, N_IN)
        assert x[:, N_FEAT].sum() == w[i].end - w[i].start
        assert x[:, N_FEAT + 1].sum() == w[i].nuc_end - w[i].nuc_start
        assert (x[:, N_FEAT + 2].sum() > 0) == (i > 0)
        assert (x[:, N_FEAT + 3].sum() > 0) == (i < 3)
        # features copied verbatim where valid
        n = int(x[:, -1].sum())
        start = w[i - 1].start if i > 0 else max(0, w[i].start - 15)
        np.testing.assert_array_equal(x[:n, :N_FEAT], feats[start:start + n])
    assert contour_vector(feats, w, 0).shape == (42,)
    e = stress_extra(feats, w, (0, 4))
    assert e.shape == (4, N_STRESS_EXTRA) and np.allclose(e[:, :5].mean(0), 0, atol=1e-5)


def test_dropped_syllable_gets_a_window():
    ref = _ref(3)
    spans = [(2, 4), (4, 9), (9, 9), (9, 9), (12, 13), (13, 20)]
    w = syllable_windows(ref, spans, 60)
    assert w[1].end > w[1].start and w[0].end <= w[1].start + 1 and w[1].end <= w[2].start + 1


def test_long_context_is_cropped_around_target():
    ref = _ref(3)
    spans = [(0, 2), (2, 60), (60, 62), (62, 140), (140, 142), (142, 200)]
    T = 2 * 200 + 7
    w = syllable_windows(ref, spans, T)
    feats = np.zeros((T, N_FEAT), np.float32)
    x = context_input(feats, w, 1)
    assert x.shape == (CTX_LEN, N_IN) and x[:, N_FEAT].sum() > 0 and x[:, -1].sum() == CTX_LEN
