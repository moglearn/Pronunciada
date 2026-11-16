"""Frame-level prosodic features and syllable windows (shared by training and runtime).

Frame grid: 10 ms, frame i centred at 0.01 i + 0.005 s (see pronunciada.prosody.f0).  A ZIPA frame j
(20 ms, start 0.02 j + 0.03 s) starts at 10 ms frame 2 j + 3.

Frame features (FEAT_NAMES):
  lf0     log-F0 z-scored over the voiced frames of the utterance, linearly interpolated through
          unvoiced stretches (held flat before the first / after the last voiced frame)
  voicing voicing probability of the tracker
  energy  log-energy, z-scored over the utterance's frames above -50 dB (rel. 95th percentile)
  d_lf0, d_energy   regression deltas (+-2 frames)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .f0 import F0Track, track

FEAT_NAMES = ("lf0", "voicing", "energy", "d_lf0", "d_energy")
N_FEAT = len(FEAT_NAMES)
PAD = 6                 # a syllable window may extend this many 10 ms frames beyond its tokens


def _delta(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, 2, mode="edge")
    return ((p[3:-1] - p[1:-3]) + 2 * (p[4:] - p[:-4])) / 10.0


def _interp_voiced(lf: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(voiced)
    if len(idx) == 0:
        return np.zeros_like(lf)
    return np.interp(np.arange(len(lf)), idx, lf[idx]).astype(np.float32)


def frame_features(tr: F0Track) -> np.ndarray:
    """[T, N_FEAT] float32 features from a pitch track."""
    T = len(tr.f0)
    out = np.zeros((T, N_FEAT), dtype=np.float32)
    if T == 0:
        return out
    v = tr.voiced & (tr.f0 > 0)
    if v.sum() >= 5:
        lf = np.log(np.maximum(tr.f0, 1.0))
        mu, sd = lf[v].mean(), lf[v].std()
        z = np.zeros(T, dtype=np.float32)
        z[v] = (lf[v] - mu) / max(sd, 0.05)
        out[:, 0] = np.clip(_interp_voiced(z, v), -5, 5)
    out[:, 1] = tr.voicing
    e = tr.energy_db.astype(np.float32)
    sp = e > -50
    if sp.sum() >= 5:
        mu, sd = e[sp].mean(), e[sp].std()
    else:
        mu, sd = e.mean(), e.std()
    out[:, 2] = np.clip((e - mu) / max(sd, 1.0), -5, 5)
    out[:, 3] = _delta(out[:, 0]) * 5.0
    out[:, 4] = _delta(out[:, 2]) * 5.0
    return out


def features(wav: np.ndarray) -> tuple[F0Track, np.ndarray]:
    tr = track(wav)
    return tr, frame_features(tr)


@dataclass
class SylWindow:
    start: int          # window [start, end) in 10 ms frames
    end: int
    nuc_start: int      # nucleus span (falls back to the window)
    nuc_end: int
    tok_start: int      # span of the syllable's aligned tokens (may be empty)
    tok_end: int


def zipa_to_10ms(j):
    return 2 * np.asarray(j) + 3


def syllable_windows(ref, phone_spans, T: int) -> list[SylWindow]:
    """phone_spans: per RefPhone (start_frame, end_frame) in ZIPA frames (end exclusive; empty =
    dropped), e.g. [(s.start_frame, s.end_frame) for s in evidence.slots]."""
    n = len(ref.syllables)
    ts = np.full(n, -1)
    te = np.full(n, -1)
    ns = np.full(n, -1)
    ne = np.full(n, -1)
    for si, s in enumerate(ref.syllables):
        for pi in range(*s.phones):
            a, b = phone_spans[pi]
            if b <= a:
                continue
            a, b = int(zipa_to_10ms(a)), int(zipa_to_10ms(b))
            ts[si] = a if ts[si] < 0 else min(ts[si], a)
            te[si] = max(te[si], b)
            if ref.phones[pi].is_nucleus:
                ns[si], ne[si] = a, b
    # syllables without any aligned token: place them between their neighbours
    have = np.flatnonzero(ts >= 0)
    for si in range(n):
        if ts[si] >= 0:
            continue
        prv = have[have < si]
        nxt = have[have > si]
        a = te[prv[-1]] if len(prv) else (ts[nxt[0]] if len(nxt) else 0)
        b = ts[nxt[0]] if len(nxt) else a
        ts[si] = te[si] = int(min(a, b) if b >= a else a)
    ts = np.clip(ts, 0, T)
    te = np.clip(te, 0, T)
    out = []
    for si in range(n):
        lo = ts[si] - PAD
        hi = te[si] + PAD
        if si > 0:
            lo = max(lo, (te[si - 1] + ts[si] + 1) // 2)
        if si + 1 < n:
            hi = min(hi, (te[si] + ts[si + 1]) // 2)
        lo = int(np.clip(lo, 0, T))
        hi = int(np.clip(max(hi, lo + 1), 0, T))
        if hi <= lo:
            lo = max(0, hi - 1)
        a, b = (int(ns[si]), int(ne[si])) if ns[si] >= 0 else (lo, hi)
        a, b = max(a, lo), min(max(b, a + 1), hi)
        if b <= a:
            a, b = lo, hi
        out.append(SylWindow(lo, hi, a, b, int(ts[si]), int(te[si])))
    return out


# ----------------------------------------------------------------------------- model inputs

CTX_LEN = 160           # frames fed to the tone/pitch network
N_IN = N_FEAT + 5       # + target, nucleus, previous, next, valid masks


def context_input(feats: np.ndarray, wins: list[SylWindow], i: int, L: int = CTX_LEN,
                  target: str = "syllable") -> np.ndarray:
    """[L, N_IN] network input for syllable i: frames from the previous syllable's start to the
    next syllable's end, left-aligned and zero padded, cropped around the target if too long.
    Columns of `feats` beyond N_FEAT (optional extra frame features) are appended after the masks.
    target="rhyme": the target mask covers nucleus start .. syllable end instead of the syllable."""
    T = len(feats)
    w = wins[i]
    a = wins[i - 1].start if i > 0 else max(0, w.start - 15)
    b = wins[i + 1].end if i + 1 < len(wins) else min(T, w.end + 15)
    a, b = min(a, w.start), max(b, w.end)
    if b - a > L:
        c = (w.start + w.end) // 2
        a2 = max(a, min(c - L // 2, b - L))
        a, b = a2, a2 + L
    F = feats.shape[1]
    x = np.zeros((L, N_IN + F - N_FEAT), dtype=np.float32)
    n = max(0, min(b, T) - a)
    x[:n, :N_FEAT] = feats[a:a + n, :N_FEAT]
    if F > N_FEAT:
        x[:n, N_IN:] = feats[a:a + n, N_FEAT:]
    rel = lambda s, e: (max(0, s - a), max(0, min(e, a + n) - a))  # noqa: E731
    s, e = rel(w.nuc_start if target == "rhyme" else w.start, w.end)
    x[s:e, N_FEAT] = 1
    s, e = rel(w.nuc_start, w.nuc_end)
    x[s:e, N_FEAT + 1] = 1
    if i > 0:
        s, e = rel(wins[i - 1].start, wins[i - 1].end)
        x[s:e, N_FEAT + 2] = 1
    if i + 1 < len(wins):
        s, e = rel(wins[i + 1].start, wins[i + 1].end)
        x[s:e, N_FEAT + 3] = 1
    x[:n, N_FEAT + 4] = 1
    return x


N_PTS = 10


def contour_vector(feats: np.ndarray, wins: list[SylWindow], i: int) -> np.ndarray:
    """Baseline input: lf0 resampled to N_PTS points over the previous, target and next
    syllable (+ their mean voicing / energy / log duration)."""
    out = []
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(wins):
            w = wins[j]
            seg = feats[w.start:w.end]
            pts = np.interp(np.linspace(0, len(seg) - 1, N_PTS), np.arange(len(seg)), seg[:, 0])
            out += [pts, [seg[:, 1].mean(), seg[:, 2].mean(), np.log((w.end - w.start) / 20.0), 1.0]]
        else:
            out += [np.zeros(N_PTS), [0, 0, 0, 0.0]]
    return np.concatenate([np.asarray(o, dtype=np.float32) for o in out])


# ----------------------------------------------------------------------------- word stress

N_STRESS_EXTRA = 7


def stress_extra(feats: np.ndarray, wins: list[SylWindow], syl_range: tuple[int, int]) -> np.ndarray:
    """[n, N_STRESS_EXTRA] word-relative cues for the syllables of one word: log duration and
    log nucleus duration, nucleus energy, nucleus mean / syllable max log-F0 (each minus the word
    mean), nucleus F0 slope, nucleus voicing.  No position information (on purpose: the score
    must come from the audio, not from the language's default stress position)."""
    rows = []
    for i in range(*syl_range):
        w = wins[i]
        nuc = feats[w.nuc_start:w.nuc_end]
        syl = feats[w.start:w.end]
        if len(nuc) == 0:
            nuc = syl
        rows.append([np.log(max(w.end - w.start, 1)), np.log(max(w.nuc_end - w.nuc_start, 1)),
                     nuc[:, 2].mean(), nuc[:, 0].mean(), syl[:, 0].max(), nuc[-1, 0] - nuc[0, 0],
                     nuc[:, 1].mean()])
    x = np.asarray(rows, dtype=np.float32)
    x[:, :5] -= x[:, :5].mean(0, keepdims=True)
    return x


# ----------------------------------------------------------------------------- tone v3 extras

REG_HALF = 75           # local pitch register: voicing-weighted mean of lf0 over +-0.75 s
N_REG = 2               # lf0 minus the local register, the register itself
N_LOWMEL = 20           # lowest 20 of the 80 Kaldi mel bins (~20-650 Hz)


def register_features(feats: np.ndarray, half: int = REG_HALF) -> np.ndarray:
    """[T, N_REG]: (lf0 - local register, local register), the register being the voicing-weighted
    moving mean of the utterance-normalised lf0 over +-half frames. Removes slow declination so a
    syllable's contour is seen relative to its neighbourhood. Pitch-only (derived from feats)."""
    T = len(feats)
    out = np.zeros((T, N_REG), dtype=np.float32)
    if T == 0:
        return out
    w = np.clip(feats[:, 1], 0, 1).astype(np.float64)
    def box(x):                     # centred moving sum over +-half frames (any T)
        c = np.concatenate([[0.0], np.cumsum(x)])
        i = np.arange(T)
        return c[np.minimum(i + half + 1, T)] - c[np.maximum(i - half, 0)]

    num = box(feats[:, 0] * w)
    den = box(w)
    reg = np.where(den > 1.0, num / np.maximum(den, 1e-6), 0.0)
    out[:, 0] = np.clip(feats[:, 0] - reg, -5, 5)
    out[:, 1] = np.clip(reg, -5, 5)
    return out


_MEL_CACHE: dict = {}


def _low_mel_bank(n_fft=512, sr=16000, n_mel=80, lo=20.0, hi=8000.0, keep=N_LOWMEL):
    """Triangular Kaldi-style mel filters (80 between 20 Hz and 8 kHz); the lowest `keep`."""
    key = (n_fft, keep)
    if key not in _MEL_CACHE:
        mel = lambda f: 1127.0 * np.log(1 + np.asarray(f) / 700.0)  # noqa: E731
        m = np.linspace(mel(lo), mel(hi), n_mel + 2)[:keep + 2]
        fm = mel(np.arange(n_fft // 2 + 1) * sr / n_fft)
        W = np.zeros((keep, n_fft // 2 + 1), np.float32)
        for b in range(keep):
            l, c, r = m[b], m[b + 1], m[b + 2]
            W[b] = np.clip(np.minimum((fm - l) / (c - l), (r - fm) / (r - c)), 0, None)
        hi_bin = int(np.flatnonzero(W.sum(0) > 0).max()) + 1
        _MEL_CACHE[key] = (W[:, :hi_bin].T.copy(), hi_bin)
    return _MEL_CACHE[key]


def lowmel_features(wav: np.ndarray, T: int) -> np.ndarray:
    """[T, N_LOWMEL] low-frequency log-mel frames on the 10 ms pitch grid: frame i is the 25 ms
    window centred at 0.01 i + 0.005 s (DC removed, pre-emphasis 0.97, Povey window, 512-point
    FFT), power through the lowest 20 of 80 Kaldi-style mel filters (~20-650 Hz), log; then the
    per-bin mean over the utterance's louder frames is removed and the result scaled by 1/4.
    numpy only (vectorised; ~1-2 ms for 10 s)."""
    W, hb = _low_mel_bank()
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    if T == 0:
        return np.zeros((0, N_LOWMEL), np.float32)
    pad = np.zeros(160 * (T - 1) + 400, np.float32)
    left = 200 - 80
    seg = x[:max(0, min(len(x), len(pad) - left))]
    pad[left:left + len(seg)] = seg
    fr = np.lib.stride_tricks.sliding_window_view(pad, 400)[::160][:T].astype(np.float32)
    fr = fr - fr.mean(1, keepdims=True)
    fr = np.concatenate([fr[:, :1] * 0.03, fr[:, 1:] - 0.97 * fr[:, :-1]], 1)
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85
    spec = np.fft.rfft(fr * win.astype(np.float32), 512)[:, :hb]
    pw = spec.real ** 2 + spec.imag ** 2
    lm = np.log(np.maximum(pw.astype(np.float32) @ W, 1e-10))
    e = lm.mean(1)
    sp = e > np.quantile(e, 0.3)
    mu = lm[sp].mean(0) if sp.sum() >= 5 else lm.mean(0)
    return np.clip((lm - mu) / 4.0, -5, 5).astype(np.float32)


def pool_syllables(ref, spans, emb: np.ndarray) -> np.ndarray:
    """[S, C]: mean of emb [T', C] (ZIPA 20 ms frames) over the frames of each syllable's aligned
    phones (spans per RefPhone, end exclusive); a syllable without aligned frames gets the frame
    at its first phone's start (clipped)."""
    T = len(emb)
    out = np.zeros((len(ref.syllables), emb.shape[1]), np.float32)
    for si, s in enumerate(ref.syllables):
        fr = [j for pi in range(*s.phones) for j in range(*spans[pi]) if 0 <= j < T]
        if not fr:
            a = [spans[pi][0] for pi in range(*s.phones)]
            fr = [int(np.clip(a[0] if a else 0, 0, max(T - 1, 0)))]
        out[si] = emb[fr].mean(0)
    return out
