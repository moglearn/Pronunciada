"""Polyphase resampling with numpy only: a drop-in for ``scipy.signal.resample_poly(x, up, down)``
with its defaults (Kaiser window, beta 5.0, constant zero padding) on 1-D input.

Importing scipy.signal loads ~650 modules and costs ~90 MB of resident memory, more than the
int8 acoustic model's weights, for this one function. Same filter design (firwin, 20 * max(up,
down) + 1 taps), same output alignment and length; the result matches scipy to floating-point
rounding (sums are taken in another order; float32 input is accumulated in float64 here).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

KAISER_BETA = 5.0


@lru_cache(maxsize=16)
def lowpass(up: int, down: int) -> tuple[np.ndarray, int]:
    """(scipy.signal.firwin(2 * half_len + 1, 1 / max(up, down), window=("kaiser", 5.0)) * up,
    half_len) as built inside resample_poly."""
    max_rate = max(up, down)
    f_c = 1.0 / max_rate
    half_len = 10 * max_rate
    numtaps = 2 * half_len + 1
    alpha = 0.5 * (numtaps - 1)
    m = np.arange(0, numtaps, dtype=np.float64) - alpha
    h = f_c * np.sinc(f_c * m)
    win = np.kaiser(numtaps, KAISER_BETA)
    h *= win
    h /= np.sum(h * np.cos(np.pi * m * 0.0))
    h.flags.writeable = False                         # cached: shared between calls
    return h, half_len


def resample_poly(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """1-D x resampled by up / down; float64 output for float64 input, else x's float dtype."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("resample_poly: 1-D input only")
    out_dtype = x.dtype if x.dtype.kind == "f" else np.float64
    up, down = int(up), int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1")
    g = math.gcd(up, down)
    up //= g
    down //= g
    if up == down == 1:
        return x.copy()
    n_in = len(x)
    n_out = -(-n_in * up // down)
    h, half_len = lowpass(up, down)
    h = h.astype(out_dtype).astype(np.float64) * up     # scipy casts the filter to x's dtype
    # zero-pad the filter so the output samples sit at its centre (as scipy does)
    n_pre_pad = down - half_len % down
    n_pre_remove = (half_len + n_pre_pad) // down
    hp = np.concatenate([np.zeros(n_pre_pad), h])
    # y[i] = sum_j hp[j] * u[m_i - j], u = x upsampled by `up` (zeros between samples),
    # m_i = (i + n_pre_remove) * down. Only j = r_i + k * up (r_i = m_i % up) hit samples of x,
    # namely x[m_i // up - k]. Outputs with the same phase r_i are every up-th one and read x at
    # a stride of `down`, so each phase is one strided matrix-vector product.
    K = -(-len(hp) // up)                                # taps per phase
    hp = np.concatenate([hp, np.zeros(K * up - len(hp))])
    # xs[K + t] = x[t] (zeros around it); x[b - K + 1 .. b] is xs[b + 1 .. b + K]
    b_max = ((n_out - 1 + n_pre_remove) * down) // up
    xs = np.zeros(max(K + n_in, b_max + K + 1), dtype=np.float64)
    xs[K:K + n_in] = x
    y = np.empty(n_out, dtype=np.float64)
    for i0 in range(min(up, n_out)):
        m0 = (i0 + n_pre_remove) * down
        r, b0 = m0 % up, m0 // up
        taps = hp[r::up][:K][::-1]                       # weight of x[b0 - K + 1 + q], q = 0..K-1
        cnt = len(range(i0, n_out, up))
        win = np.lib.stride_tricks.as_strided(xs[b0 + 1:], shape=(cnt, K),
                                              strides=(xs.strides[0] * down, xs.strides[0]),
                                              writeable=False)
        y[i0::up] = win @ taps
    return y.astype(out_dtype, copy=False)
