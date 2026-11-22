"""Audio loading, resampling and Kaldi fbank features (lhotse-compatible, no torch)."""
from __future__ import annotations

import io
import os
from fractions import Fraction

import numpy as np

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
FRAME_SHIFT = 0.01          # fbank frame shift, seconds


def _to_mono_float(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav)
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / float(np.iinfo(wav.dtype).max + 1)
    if wav.ndim == 2:
        # soundfile gives [n, ch]; accept [ch, n] too when channels is the short side.
        wav = wav.mean(axis=1) if wav.shape[0] >= wav.shape[1] else wav.mean(axis=0)
    elif wav.ndim != 1:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {wav.shape}")
    return wav.astype(np.float32, copy=False)


def resample(wav: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    if sr_in == sr_out:
        return wav.astype(np.float32, copy=False)
    from ._resample import resample_poly     # = scipy.signal.resample_poly, without importing scipy

    f = Fraction(int(sr_out), int(sr_in)).limit_denominator(10000)
    return resample_poly(wav.astype(np.float64), f.numerator, f.denominator).astype(np.float32)


def load_audio(src, sr: int | None = None) -> np.ndarray:
    """Return float32 mono 16 kHz audio in [-1, 1].

    `src` may be a path, raw file bytes, a (array, sample_rate) tuple, or a bare array
    (then `sr` gives its rate; default 16 kHz).
    """
    if isinstance(src, tuple) and len(src) == 2:
        wav, sr_in = src
        wav = _to_mono_float(wav)
    elif isinstance(src, np.ndarray):
        wav, sr_in = _to_mono_float(src), (sr or SAMPLE_RATE)
    else:
        import soundfile as sf

        if isinstance(src, (bytes, bytearray, memoryview)):
            src = io.BytesIO(bytes(src))
        elif isinstance(src, (str, os.PathLike)):
            src = os.fspath(src)
        wav, sr_in = sf.read(src, dtype="float32", always_2d=False)
        wav = _to_mono_float(wav)
    wav = resample(wav, int(sr_in), SAMPLE_RATE)
    return np.clip(wav, -1.0, 1.0)


def _fbank_options():
    import kaldi_native_fbank as knf

    o = knf.FbankOptions()
    fo = o.frame_opts
    fo.samp_freq = SAMPLE_RATE
    fo.frame_shift_ms = 10.0
    fo.frame_length_ms = 25.0
    fo.dither = 0.0
    fo.preemph_coeff = 0.97
    fo.remove_dc_offset = True
    fo.window_type = "povey"
    fo.round_to_power_of_two = True
    fo.snip_edges = False
    mo = o.mel_opts
    mo.num_bins = NUM_MEL_BINS
    mo.low_freq = 20.0
    mo.high_freq = -400.0
    o.use_energy = False
    o.use_power = True
    o.use_log_fbank = True
    return o


_OPTS = None


def vtln_warp(fb: np.ndarray, alpha: float) -> np.ndarray:
    """Vocal-tract-length warp in the log-mel domain (round 19 experiment, off by default):
    the value of bin m becomes the (linearly interpolated) value of the original spectrum at
    frequency f_m / alpha, so alpha < 1 moves formants DOWN by the factor alpha. Bins that would
    read above the top band keep the top band's value."""
    if alpha == 1.0:
        return fb
    lo, hi = 20.0, SAMPLE_RATE / 2 - 400.0

    def mel(f):
        return 1127.0 * np.log1p(np.asarray(f, np.float64) / 700.0)

    edges = np.linspace(mel(lo), mel(hi), NUM_MEL_BINS + 2)
    centres_mel = edges[1:-1]
    centres_hz = 700.0 * np.expm1(centres_mel / 1127.0)
    src = mel(np.clip(centres_hz / alpha, centres_hz[0], centres_hz[-1]))
    pos = np.interp(src, centres_mel, np.arange(NUM_MEL_BINS))
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, NUM_MEL_BINS - 1)
    w = (pos - i0).astype(np.float32)
    return (fb[:, i0] * (1 - w) + fb[:, i1] * w).astype(np.float32)


def num_fbank_frames(num_samples: int) -> int:
    """Kaldi frame count with snip_edges=False."""
    return (int(num_samples) + 80) // 160


def fbank(wav: np.ndarray) -> np.ndarray:
    """80-bin log-mel fbank [T, 80] float32 matching lhotse Fbank(FbankConfig(num_filters=80,
    dither=0.0, snip_edges=False)) on a [-1, 1] waveform."""
    import kaldi_native_fbank as knf

    global _OPTS
    if _OPTS is None:
        _OPTS = _fbank_options()
    wav = np.ascontiguousarray(wav, dtype=np.float32).reshape(-1)
    f = knf.OnlineFbank(_OPTS)
    f.accept_waveform(SAMPLE_RATE, wav)
    f.input_finished()
    n = f.num_frames_ready
    out = np.empty((n, NUM_MEL_BINS), dtype=np.float32)
    for i in range(n):
        out[i] = f.get_frame(i)
    return out
