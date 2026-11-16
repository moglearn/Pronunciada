import io

import numpy as np
import pytest
import soundfile as sf

from pronunciada.audio import fbank, load_audio, num_fbank_frames


def _sine(sr, sec=1.0, f=440.0, amp=0.5):
    t = np.arange(int(sr * sec)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _peak_freq(wav, sr=16000):
    spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
    return np.argmax(spec) * sr / len(wav)


def test_load_sources_agree(tmp_path):
    wav = _sine(16000)
    p = tmp_path / "x.wav"
    sf.write(p, wav, 16000, subtype="FLOAT")
    a = load_audio(str(p))
    b = load_audio(p.read_bytes())
    c = load_audio((wav, 16000))
    d = load_audio(wav)
    for x in (a, b, c, d):
        assert x.dtype == np.float32 and x.ndim == 1
        np.testing.assert_allclose(x, wav, atol=1e-6)


def test_resample_stereo_int16():
    sr = 44100
    st = np.stack([_sine(sr, f=1000.0), _sine(sr, f=1000.0)], axis=1)
    pcm = (st * 32767).astype(np.int16)
    x = load_audio((pcm, sr))
    assert x.dtype == np.float32 and x.ndim == 1
    assert abs(len(x) - 16000) <= 1
    assert abs(_peak_freq(x) - 1000.0) < 5
    assert 0.45 < np.abs(x).max() <= 0.51
    buf = io.BytesIO()
    sf.write(buf, st, 48000, format="WAV")
    y = load_audio(buf.getvalue())
    assert abs(len(y) - int(round(len(st) * 16000 / 48000))) <= 1


def _kaldi_fbank_reference(wav, sr=16000, nbins=80, low=20.0, high=-400.0):
    """Plain numpy transcription of Kaldi/lhotse fbank (snip_edges=False, povey, preemph 0.97,
    remove DC, power spectrum, log floor = float eps)."""
    wav = wav.astype(np.float64)
    n = len(wav)
    L, shift, nfft = 400, 160, 512
    T = (n + shift // 2) // shift
    out = np.empty((T, nbins))
    i = np.arange(L)
    win = (0.5 - 0.5 * np.cos(2 * np.pi * i / (L - 1))) ** 0.85
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
    hi = sr / 2 + high if high <= 0 else high
    ml, mh = mel(low), mel(hi)
    delta = (mh - ml) / (nbins + 1)
    fm = mel(np.arange(nfft // 2) * sr / nfft)
    banks = np.zeros((nbins, nfft // 2))
    for b in range(nbins):
        l, c, r = ml + b * delta, ml + (b + 1) * delta, ml + (b + 2) * delta
        up = (fm > l) & (fm <= c)
        dn = (fm > c) & (fm < r)
        banks[b, up] = (fm[up] - l) / (c - l)
        banks[b, dn] = (r - fm[dn]) / (r - c)
    for f in range(T):
        s = f * shift + shift // 2 - L // 2 + i
        s = np.where(s < 0, -s - 1, s)
        s = np.where(s >= n, 2 * n - 1 - s, s)
        x = wav[s].copy()
        x -= x.mean()
        x[1:] -= 0.97 * x[:-1].copy()
        x[0] -= 0.97 * x[0]
        x *= win
        p = np.abs(np.fft.rfft(x, nfft)) ** 2
        out[f] = np.log(np.maximum(banks @ p[: nfft // 2], np.finfo(np.float32).eps))
    return out


@pytest.mark.parametrize("n", [400, 4321, 16000, 16080])
def test_fbank_matches_kaldi_formula(n):
    rng = np.random.default_rng(n)
    wav = (rng.standard_normal(n) * 0.1 + 0.3 * np.sin(np.arange(n) / 7.0)).astype(np.float32)
    got = fbank(wav)
    assert got.shape == (num_fbank_frames(n), 80) and got.dtype == np.float32
    want = _kaldi_fbank_reference(wav)
    assert np.abs(got - want).max() < 2e-3


def test_fbank_silence_and_short():
    x = fbank(np.zeros(1600, dtype=np.float32))
    assert x.shape == (10, 80)
    np.testing.assert_allclose(x, np.log(np.finfo(np.float32).eps), atol=1e-4)
    assert fbank(np.zeros(0, dtype=np.float32)).shape == (0, 80)
    assert fbank(np.zeros(100, dtype=np.float32)).shape == (1, 80)
