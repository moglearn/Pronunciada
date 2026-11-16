"""Optional spectral frame features for the tone-ceiling diagnosis (10 ms grid, same frames as F0):
  mfcc    13 cepstra of the 80-bin Kaldi fbank (pronunciada.audio.fbank), utterance mean removed, /10
  lowspec 16 bands of a 64 ms 8 kHz log-magnitude spectrum, 31-531 Hz (2 FFT bins per band, i.e.
          31 Hz resolution: harmonics of F0 are resolved), utterance mean removed, /3

    .venv/bin/python -m bench.prosody.extra_feats aishell3 train zh
Output: ~/.cache/pronunciada/prosody/extra/<dataset>-<lang>-<split>.npz {id: float16 [T, 29]}
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.fft import dct  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from bench.prosody.prep import OUT, records  # noqa: E402
from pronunciada.audio import fbank, load_audio  # noqa: E402
from pronunciada.prosody.f0 import n_frames  # noqa: E402

EXTRA = OUT / "extra"
N_MFCC, N_LOW = 13, 16


def extra(wav: np.ndarray) -> np.ndarray:
    T = n_frames(len(wav))
    fb = fbank(wav)[:T]
    m = dct(fb, type=2, norm="ortho", axis=1)[:, :N_MFCC]
    m = (m - m.mean(0)) / 10.0
    x = resample_poly(wav, 1, 2).astype(np.float32)
    W = 512
    pad = np.zeros(80 * (T - 1) + W + 1, np.float32)
    left = W // 2 - 40
    n = min(len(x), len(pad) - left)
    pad[left:left + n] = x[:n]
    fr = np.lib.stride_tricks.sliding_window_view(pad, W)[::80][:T] * np.hanning(W).astype(np.float32)
    mag = np.abs(np.fft.rfft(fr, W))[:, 2:2 + 2 * N_LOW].reshape(T, N_LOW, 2).mean(2)
    lg = np.log(mag + 1e-4)
    lg = (lg - lg.mean(0)) / 3.0
    out = np.zeros((T, N_MFCC + N_LOW), np.float32)
    out[:len(m), :N_MFCC] = m
    out[:, N_MFCC:] = lg
    return np.clip(out, -6, 6).astype(np.float16)


def main(dataset, split, lang):
    EXTRA.mkdir(parents=True, exist_ok=True)
    d = {}
    for r in records(dataset, split, lang):
        wav = r["wav"] if "wav" in r else load_audio(r["audio"])
        d[r["id"]] = extra(wav)
    np.savez(EXTRA / f"{dataset}-{lang}-{split}.npz", **d)
    print("wrote", dataset, split, lang, len(d))


if __name__ == "__main__":
    main(*sys.argv[1:4])
