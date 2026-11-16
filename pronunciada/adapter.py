"""Per-language output adapter for ZIPA log-posteriors (pure numpy at runtime).

    lp' = log_softmax(lp + f(x)),   x = max(lp, floor) / scale

f is zero at initialisation (identity adapter) and was trained (bench/pipeline/train_adapter.py,
torch) with CTC loss on native FLEURS-train speech towards the canonical reference token
sequences, blank = logaddexp(lp'[:,0], lp'[:,3]) exactly as pronunciada.align.emission_table
merges "▁" into blank. Kinds:

* ``linear``: f[t] = x[t] W + b                                   (W: [V, V])
* ``ctx``:    f[t] = sum_{k=-c..c} x[t+k] W_k + b                 (edge-replicated, W: [2c+1, V, V])
* ``conv``:   h1 = relu(conv_k(x)), h2 = relu(conv_k(h1)), f = h2 Wo + bo (edge-replicated convs)

Weights: pronunciada/data/adapters/<lang>.npz. A language without a file has no adapter
(``Adapter.load`` returns None): the raw posteriors are used.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np


def _log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    z = z - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _windows(x: np.ndarray, k: int) -> np.ndarray:
    """[T, C] -> [T, k*C]: frames t-c..t+c concatenated (edge-replicated)."""
    c = k // 2
    T = len(x)
    idx = np.clip(np.arange(T)[:, None] + np.arange(-c, c + 1)[None, :], 0, T - 1)
    return x[idx].reshape(T, -1)


class Adapter:
    def __init__(self, params: dict, lang: str = ""):
        self.lang = lang
        self.kind = str(params["kind"])
        self.floor = float(params["floor"])
        self.scale = float(params["scale"])
        self.p = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in params.items()
                  if k not in ("kind", "floor", "scale", "meta")}
        if self.kind == "ctx":
            W = self.p["W"]                                   # [K, V, V]
            self.p["Wflat"] = W.reshape(-1, W.shape[-1])      # [K*V, V]
        if self.kind == "conv":
            for n in ("W1", "W2"):
                W = self.p[n]                                 # [K, Cin, Cout]
                self.p[n + "flat"] = W.reshape(-1, W.shape[-1])

    # ------------------------------------------------------------------ io
    @staticmethod
    def path(lang: str) -> Path:
        return Path(str(resources.files("pronunciada").joinpath(f"data/adapters/{lang}.npz")))

    @classmethod
    def load(cls, lang: str) -> "Adapter | None":
        p = cls.path(lang)
        if not p.exists():
            return None
        return cls.from_file(p, lang)

    @classmethod
    def from_file(cls, path, lang: str = "") -> "Adapter":
        with np.load(path, allow_pickle=False) as z:
            params = {k: z[k] for k in z.files}
        params["kind"] = str(params["kind"])
        return cls(params, lang)

    # ------------------------------------------------------------------ apply
    def delta(self, lp: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(lp, dtype=np.float32), self.floor) / self.scale
        p = self.p
        if self.kind == "linear":
            return x @ p["W"] + p["b"]
        if self.kind == "ctx":
            K = p["W"].shape[0]
            return _windows(x, K) @ p["Wflat"] + p["b"]
        if self.kind == "conv":
            K = p["W1"].shape[0]
            h = np.maximum(_windows(x, K) @ p["W1flat"] + p["b1"], 0.0)
            h = np.maximum(_windows(h, K) @ p["W2flat"] + p["b2"], 0.0)
            return h @ p["Wo"] + p["bo"]
        raise ValueError(f"unknown adapter kind {self.kind!r}")

    def __call__(self, lp: np.ndarray) -> np.ndarray:
        """[T, V] float32 log-probs -> adapted [T, V] float32 log-probs."""
        lp = np.asarray(lp, dtype=np.float32)
        if len(lp) == 0:
            return lp.copy()
        z = lp + self.delta(lp)
        return _log_softmax(z).astype(np.float32, copy=False)


def apply(lp: np.ndarray, lang: str) -> np.ndarray:
    """Adapted log-probs for `lang` (unchanged copy if the language ships no adapter)."""
    a = _cached(lang)
    return np.asarray(lp, dtype=np.float32) if a is None else a(lp)


_CACHE: dict[str, "Adapter | None"] = {}


def _cached(lang: str):
    if lang not in _CACHE:
        _CACHE[lang] = Adapter.load(lang)
    return _CACHE[lang]
