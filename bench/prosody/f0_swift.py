"""Alternative pitch source: SwiftF0 (github.com/lars76/swift-f0, MIT; a ~14 k-parameter ONNX
model, 16 kHz input, 16 ms frames) resampled onto the 10 ms grid of pronunciada.prosody.f0, and
returned as the same F0Track, so frame_features / the prosody models can use it unchanged.

Experimental: the ONNX file is read from the installed `swift_f0` package (not vendored); nothing
here changes pronunciada.prosody.f0. Only the model file is used (the package's Python code is
not imported); its inference step is reproduced here: one session.run on the whole clip for clips
up to 30 s (the package's own window is 1875 frames = 30 s), frames whose audio peaks below 1e-3
get confidence 0 (the package's digital-silence rule).

Differences from the package defaults, both for robustness on quiet recordings: the clip is
peak-normalised to 0.9 first (pitch does not depend on gain; the package advises scaling very
quiet audio), and the search range is 50-600 Hz like our tracker.

energy_db is computed as in pronunciada.prosody.f0 (40 ms frame energy at 8 kHz, dB relative to
the utterance's 95th percentile), so the energy features do not change with the pitch source.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from pronunciada.prosody.f0 import F0Track, n_frames

SR = 16000
S_HOP = 256                  # SwiftF0 frame period (samples)
FMIN, FMAX = 50.0, 600.0
_SESS: dict = {}


def model_path() -> Path:
    env = os.environ.get("PRONUNCIADA_SWIFTF0_ONNX")
    if env:
        return Path(env)
    vend = Path(__file__).resolve().parent.parent / "data" / "prosody" / "swiftf0.onnx"
    if vend.exists():
        return vend
    spec = importlib.util.find_spec("swift_f0")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("SwiftF0 model not found (install swift-f0 or vendor swiftf0.onnx)")
    return Path(list(spec.submodule_search_locations)[0]) / "model.onnx"


def _session(threads: int):
    if threads not in _SESS:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        _SESS[threads] = ort.InferenceSession(str(model_path()), so, providers=["CPUExecutionProvider"])
    return _SESS[threads]


def _energy_db(wav: np.ndarray, T: int) -> np.ndarray:
    """As pronunciada.prosody.f0.track: 40 ms frames at 8 kHz centred on the 10 ms grid."""
    x = resample_poly(wav, 1, 2).astype(np.float32)
    x = x - x.mean()
    W, hop, left = 320, 80, 320 // 2 - 40
    pad = np.zeros((T - 1) * hop + W + left + 1, np.float32)
    n = min(len(x), len(pad) - left)
    pad[left:left + n] = x[:n]
    C = np.concatenate([[0.0], np.cumsum(pad.astype(np.float64) ** 2)])
    s = np.arange(T) * hop
    e0 = C[s + W] - C[s]
    edb = 10 * np.log10(e0 / W + 1e-10)
    return (edb - np.percentile(edb, 95)).astype(np.float32)


def raw(wav: np.ndarray, threads: int = 1):
    """SwiftF0 frames: (pitch Hz, confidence), frame k centred at k * 16 ms."""
    x = np.asarray(wav, np.float32).reshape(-1)
    pk = float(np.abs(x).max()) if len(x) else 0.0
    if pk > 0:
        x = x * (0.9 / pk)
    sess = _session(threads)
    n = max(1, len(x) // S_HOP)
    P, Cf = [], []
    step = 1875
    for a in range(0, n, step):           # windows with context, as the package does (> 30 s only)
        b = min(a + step, n)
        lo = max(0, a - 11)
        seg = x[lo * S_HOP:(b + 10) * S_HOP] if b < n else x[lo * S_HOP:]
        p, c = sess.run(["pitch", "confidence"], {"audio": seg[None, :], "fmin": np.asarray(FMIN, np.float32),
                                                   "fmax": np.asarray(FMAX, np.float32)})
        p, c = np.asarray(p[0], np.float64), np.asarray(c[0], np.float64)
        m = len(c)
        hops = seg[:m * S_HOP].reshape(m, S_HOP) if len(seg) >= S_HOP else seg[None, :]
        c[np.abs(hops).max(1) < 1e-3] = 0.0
        P.append(p[a - lo:b - lo])
        Cf.append(c[a - lo:b - lo])
    return np.concatenate(P), np.concatenate(Cf)


def track(wav: np.ndarray, threads: int = 1) -> F0Track:
    """F0Track on the 10 ms grid (frame i centred at 0.01 i + 0.005 s) from SwiftF0."""
    wav = np.asarray(wav, np.float32).reshape(-1)
    T = n_frames(len(wav))
    if T == 0:
        z = np.zeros(0, np.float32)
        return F0Track(z, z.astype(bool), z, z)
    p, c = raw(wav, threads)
    ts = np.arange(len(p)) * (S_HOP / SR)
    g = np.arange(T) * 0.01 + 0.005
    conf = np.interp(g, ts, c).astype(np.float32)
    voiced = conf >= 0.5
    v = c >= 0.5
    f0 = np.zeros(T, np.float32)
    if v.any():
        # log-F0 interpolated between voiced SwiftF0 frames (nearest voiced value at the edges)
        f0[:] = np.exp(np.interp(g, ts[v], np.log(np.maximum(p[v], 1.0))))
    f0[~voiced] = 0.0
    return F0Track(f0, voiced, conf, _energy_db(wav, T))
