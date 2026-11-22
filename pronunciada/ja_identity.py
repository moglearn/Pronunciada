"""Round 19 (Japanese): which sound was said, from a head on the acoustic model's internal layers.

The same design as the Korean head (round 19, pronunciada/ko_identity.py in that branch): a small
network on the pooled enc1|enc3|enc5 vector of a sound plus its neighbours' mean (the round-17 head
input) predicts the Japanese sound class that was said, fitted on native Common Voice ja train and
FLEURS ja train speech with the front-end's own reference as the label.

Classes: the five vowels short / long (u and ɯ together), the moraic nasal "N", the glide j, and the
consonants by their base symbol, a geminate written with "ː" (kː = っk).

Evidence per sound: s = log P(expected class) - log max P(other class). It enters the word score
only as a factor <= 1 (pronunciada/word_factors.py), never on its own.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np

from . import attr_heads as AH
from .text.ipa import is_vowel

_CMAP = {"ɯ": "u", "dz": "z", "ʒ": "dʒ", "ɾ": "r", "ç": "h", "ɸ": "f"}


def phone_class(p) -> str | None:
    ip = p.ipa
    if p.is_nucleus and is_vowel(ip[0]):
        base = _CMAP.get(ip[0], ip[0])
        if base not in "aiueo" or len(ip.replace("ː", "")) != 1:
            return None
        return base + ("ː" if ip.endswith("ː") else "")
    if p.is_nucleus:
        return "N"
    gem = False
    if len(ip) >= 2 and ip[0] == ip[1]:
        ip, gem = ip[1:], True
    elif ip in ("ttʃ", "tts"):
        ip, gem = ip[1:], True
    base = _CMAP.get(ip, ip)
    return base + ("ː" if gem else "")


def classes(ref) -> list[str | None]:
    return [phone_class(p) for p in ref.phones]


@lru_cache(maxsize=1)
def head():
    p = resources.files("pronunciada").joinpath("data/coach/ja_identity.npz")
    try:
        with p.open("rb") as f:
            z = dict(np.load(f, allow_pickle=False))
    except (FileNotFoundError, OSError):
        return None
    H = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in z.items()}
    H["index"] = {str(c): j for j, c in enumerate(H["classes"].tolist())}
    return H


def _mlp(H, X: np.ndarray) -> np.ndarray:
    X = (X - H["mu"]) / H["sd"]
    if "W1" in H:
        A = X @ H["W1"] + H["b1"]
        A = 0.5 * A * (1.0 + np.tanh(0.7978845608 * (A + 0.044715 * A ** 3)))
        Z = A @ H["W2"] + H["b2"]
    else:
        Z = X @ H["W"] + H["b"]
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def evidence(ref, V: np.ndarray, H=None) -> np.ndarray:
    """s per sound (NaN where not judged)."""
    H = H or head()
    cls = classes(ref)
    s = np.full(len(cls), np.nan)
    if H is None:
        return s
    idx = [i for i, c in enumerate(cls) if c is not None and c in H["index"]]
    if not idx:
        return s
    V = np.asarray(V, np.float32)
    X = np.concatenate([V, AH.context_vectors(V)], axis=1)[idx]
    lp = np.log(np.clip(_mlp(H, X), 1e-9, 1.0))
    for r, i in enumerate(idx):
        j = H["index"][cls[i]]
        s[i] = lp[r, j] - np.delete(lp[r], j).max()
    return s
