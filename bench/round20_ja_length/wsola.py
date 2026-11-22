"""Waveform-similarity overlap-add (WSOLA) time stretching in numpy, and a helper that stretches one
segment of a clip while leaving the rest untouched (round 20, Japanese vowel-length training pairs).

    y = stretch(x, 2.0)                       # twice as long, same pitch
    y = stretch_segment(x, a, b, 0.5)         # samples a..b compressed to half, the rest unchanged

Training-only code (bench); the runtime never imports it.
"""
from __future__ import annotations

import numpy as np

FRAME = 320        # 20 ms analysis frame at 16 kHz
HOP = 160          # synthesis hop (10 ms)
TOL = 80           # +- 5 ms search for the best-matching analysis frame


def stretch(x: np.ndarray, factor: float, frame: int = FRAME, hop: int = HOP, tol: int = TOL) -> np.ndarray:
    """Time-stretch x by `factor` (> 1 longer, < 1 shorter) with WSOLA (Verhelst & Roelands 1993).
    Output length is round(len(x) * factor)."""
    x = np.asarray(x, np.float64)
    n_out = int(round(len(x) * factor))
    if len(x) < frame + 2 * tol or n_out < frame:
        # too short for overlap-add: plain resampling of the time axis (pitch changes slightly)
        if n_out <= 1 or len(x) <= 1:
            return x[:max(n_out, 0)].copy()
        return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)
    win = np.hanning(frame)
    xp = np.concatenate([np.zeros(tol), x, np.zeros(frame + 2 * tol)])
    n_frames = int(np.ceil((n_out - frame) / hop)) + 1
    out = np.zeros(n_out + frame + hop)
    norm = np.zeros_like(out)
    prev = tol                         # position (in xp) of the previous analysis frame
    for k in range(n_frames):
        ideal = tol + int(round(k * hop / factor))
        if k == 0:
            pos = ideal
        else:
            # natural continuation of the previous frame, compared with candidates around `ideal`
            nat = xp[prev + hop: prev + hop + frame]
            lo, hi = max(ideal - tol, 0), min(ideal + tol, len(xp) - frame)
            if hi <= lo:
                pos = min(max(ideal, 0), len(xp) - frame)
            else:
                seg = xp[lo: hi + frame]
                # cross-correlation of `nat` with every candidate start in [lo, hi]
                c = np.correlate(seg, nat, mode="valid")
                pos = lo + int(np.argmax(c))
        s = k * hop
        out[s: s + frame] += xp[pos: pos + frame] * win
        norm[s: s + frame] += win
        prev = pos
    norm[norm < 1e-3] = 1.0
    return (out / norm)[:n_out]


def stretch_segment(x: np.ndarray, a: int, b: int, factor: float, fade: int = 80) -> np.ndarray:
    """x with samples [a, b) time-stretched by `factor`; short linear cross-fades at both joins."""
    x = np.asarray(x, np.float32)
    a, b = max(int(a), 0), min(int(b), len(x))
    if b - a < 2 or abs(factor - 1.0) < 1e-6:
        return x.copy()
    core = stretch(x[a:b].astype(np.float64), factor).astype(np.float32)
    left = x[:a].copy()
    right = x[b:].copy()
    f =min(fade, len(core) // 4, len(left), len(right))
    if f > 4:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        # blend the first / last f samples of the stretched core with the original continuation
        core[:f] = core[:f] * ramp + x[a: a + f] * (1 - ramp)
        core[-f:] = core[-f:] * ramp[::-1] + x[b - f: b] * (1 - ramp[::-1])
    return np.concatenate([left, core, right]).astype(np.float32)
