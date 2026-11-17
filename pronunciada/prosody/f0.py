"""Pitch tracker: YIN-style candidates + Viterbi smoothing, numpy only.

Frames: 10 ms hop; frame i is centred at ``i * 0.01 + 0.005`` s and there are
``(n_samples + 80) // 160`` frames at 16 kHz, i.e. the same grid as the Kaldi fbank
(snip_edges=False) used by pronunciada.audio.  A ZIPA output frame j (start 0.02 j + 0.03 s)
therefore starts at 10 ms frame 2 j + 3.

Algorithm
1. Downsample to 8 kHz (polyphase FIR), remove DC.
2. For every frame, the YIN difference function d(tau) over a 40 ms window for lags up to
   1/fmin, computed for all frames at once with one batched FFT cross-correlation.  Cumulative
   mean normalised difference d'(tau) (YIN step 3).
3. Candidates: the K deepest local minima of d' inside [fs/fmax, fs/fmin] with parabolic
   interpolation.  Each candidate gets a local cost d' plus a small octave cost that favours
   higher frequencies (as in Praat) so sub-harmonic dips do not win ties.
4. Voicing probability per frame from the best d' and the frame energy (logistic).
5. Viterbi over {K candidates, unvoiced} with a transition cost proportional to the jump in
   octaves (octave jumps are strongly penalised) and a voicing switch cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._resample import resample_poly     # = scipy.signal.resample_poly, without importing scipy

SR = 16000
HOP = 0.01
FS = 8000                 # internal rate
WIN = 320                 # 40 ms at 8 kHz
NFFT = 512
K = 4                     # candidates per frame

# tuned on FLEURS validation (dev) against Praat; see bench/prosody/f0_eval.py
OCTAVE_COST = 0.03        # per octave below fmax, added to d'
VOICE_D0 = 0.60           # d' at which P(voiced) = 0.5 (energy at median)
VOICE_DS = 0.07
ENERGY_DB0 = -56.0        # frame energy (dB rel. utterance 95th pct) at which P(voiced) halves
ENERGY_DS = 4.0
JUMP_COST = 12.0          # per octave of F0 change between consecutive frames
SWITCH_COST = 2.0         # voiced <-> unvoiced
LOCAL_SCALE = 25.0        # weight of d' in the emission cost


@dataclass
class F0Track:
    f0: np.ndarray            # [T] Hz, 0 where unvoiced
    voiced: np.ndarray        # [T] bool (Viterbi decision)
    voicing: np.ndarray       # [T] float voicing probability in [0, 1]
    energy_db: np.ndarray     # [T] frame log energy (dB, utterance-relative: 0 = 95th percentile)

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.f0)) * HOP + HOP / 2


def n_frames(n_samples: int) -> int:
    return (int(n_samples) + 80) // 160


def _frames(x: np.ndarray, T: int, length: int) -> np.ndarray:
    """[T, length] frames of 8 kHz signal x; frame i starts at 80 i - WIN/2 + 40 (centred on
    the 10 ms grid centre 80 i + 40)."""
    hop = 80
    left = WIN // 2 - 40
    need = (T - 1) * hop + length
    pad = np.zeros(need + left + 1, dtype=np.float32)
    n = min(len(x), need)
    pad[left:left + n] = x[:n]
    fr = np.lib.stride_tricks.sliding_window_view(pad, length)[::hop][:T]
    return fr, pad, np.arange(T) * hop


def track(wav: np.ndarray, sr: int = SR, fmin: float = 50.0, fmax: float = 600.0) -> F0Track:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    T = n_frames(len(wav) * SR // sr) if sr != SR else n_frames(len(wav))
    if T == 0:
        z = np.zeros(0, dtype=np.float32)
        return F0Track(z, z.astype(bool), z, z)
    if sr != FS:
        g = np.gcd(int(sr), FS)
        x = resample_poly(wav, FS // g, int(sr) // g).astype(np.float32)
    else:
        x = wav
    x = x - x.mean()

    lmin = max(2, int(np.floor(FS / fmax)))
    lmax = int(np.ceil(FS / fmin))
    L = lmax + 2
    fr, pad, starts = _frames(x, T, WIN + L)          # [T, WIN+L]
    Fa = np.fft.rfft(fr[:, :WIN], NFFT)
    Fb = np.fft.rfft(fr, NFFT)
    r = np.fft.irfft(np.conj(Fa) * Fb, NFFT)[:, :L + 1]          # r[tau] = sum_j a_j b_{j+tau}
    C = np.concatenate([[0.0], np.cumsum(pad.astype(np.float64) ** 2)])
    pos = starts[:, None] + np.arange(L + 1)[None, :]
    et = C[pos + WIN] - C[pos]                                   # energy of b[tau:tau+WIN]
    e0 = et[:, 0].copy()
    d = np.maximum(e0[:, None] + et - 2.0 * r, 0.0)
    d[:, 0] = 0.0
    cs = np.cumsum(d[:, 1:], axis=1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] * np.arange(1, L + 1) / np.maximum(cs, 1e-12)

    # local minima inside the lag range
    mid = dn[:, 1:-1]
    is_min = (mid <= dn[:, :-2]) & (mid < dn[:, 2:])
    lag = np.arange(1, L)
    is_min &= (lag >= lmin) & (lag <= lmax)
    # parabolic interpolation
    y0, y1, y2 = dn[:, :-2], mid, dn[:, 2:]
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    off = np.clip(off, -0.5, 0.5)
    val = y1 - 0.25 * (y0 - y2) * off
    tau_f = lag + off
    cost = np.where(is_min, np.clip(val, 0, 2) + OCTAVE_COST * np.log2(np.maximum(tau_f, 1) / (FS / fmax)), np.inf)
    idx = np.argpartition(cost, K, axis=1)[:, :K] if cost.shape[1] > K else np.argsort(cost, 1)[:, :K]
    ccost = np.take_along_axis(cost, idx, 1)                    # [T, K]
    cval = np.take_along_axis(val, idx, 1)
    cfreq = FS / np.take_along_axis(tau_f, idx, 1)
    valid = np.isfinite(ccost)
    cval = np.where(valid, cval, 1.0)
    cfreq = np.where(valid, cfreq, 100.0)

    # energy & voicing probability
    edb = 10 * np.log10(e0 / WIN + 1e-10)
    ref = np.percentile(edb, 95)
    edb = edb - ref
    best = cval.min(axis=1)
    z = (VOICE_D0 - best) / VOICE_DS + (edb - ENERGY_DB0) / ENERGY_DS
    z = np.minimum(z, (edb - ENERGY_DB0) / ENERGY_DS + 3.0)      # very quiet frames stay unvoiced
    pv = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    # Viterbi: states 0..K-1 = candidates, K = unvoiced
    em = np.empty((T, K + 1))
    em[:, :K] = np.where(valid, LOCAL_SCALE * np.where(valid, ccost, 0.0), 1e4) - np.log(pv + 1e-6)[:, None]
    em[:, K] = -np.log(1 - pv + 1e-6)
    lf = np.log2(cfreq)
    path = _viterbi(em, lf)
    voiced = path < K
    f0 = np.zeros(T, dtype=np.float32)
    vi = np.flatnonzero(voiced)
    f0[vi] = cfreq[vi, path[vi]]
    return F0Track(f0, voiced, pv.astype(np.float32), edb.astype(np.float32))


def _viterbi(em: np.ndarray, lf: np.ndarray) -> np.ndarray:
    T, S = em.shape
    Kc = S - 1
    TR = np.full((T, S, S), SWITCH_COST)
    TR[:, Kc, Kc] = 0.0
    TR[1:, :Kc, :Kc] = JUMP_COST * np.abs(lf[:-1, :, None] - lf[1:, None, :])
    back = np.empty((T, S), dtype=np.int64)
    acc = em[0].copy()
    for t in range(1, T):
        tot = TR[t] + acc[:, None]
        back[t] = b = tot.argmin(0)
        acc = tot.min(0) + em[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(acc.argmin())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
