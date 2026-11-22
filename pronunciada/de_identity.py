"""Round 19b (German): sound-identity evidence from a head on the acoustic model's internal layers.

Why: a one-sound change inside a word (ihn / in, Grab / Grad, Westen / besten, uns / ums) moves the
frame-posterior evidence of the word too little, so the changed word scores almost as high as correct
words. The same idea as the Korean head of this round (pronunciada/ko_identity.py).

What: a one-hidden-layer MLP (``data/coach/de_identity.npz``) on the pooled enc1|enc3|enc5 vector of a
sound plus its neighbours' mean predicts the German sound class that was said. Classes are the
front-end's sounds with the legitimate native variants folded into the canonical class: r-coloured
long vowels count as the long vowel; z / s, ç / x, and word-final voiced / voiceless obstruents
(final devoicing) are one class each. Not judged: r, schwa, ɐ, the glottal stop, and an unstressed
word-final e (said as schwa). Fitted on native German read speech (FLEURS de train, Common Voice de
train; labels = the front-end's reference of the prompt).

Evidence per sound: s = log P(expected class) - log max P(other class) - offset(class), the offset a
low native quantile of that class. Word factor (``data/calib/de_identity_word.json``): 1 while the
word's lowest s is above a knee (a native quantile of word minima, by strictness), then linearly
down to ``floor`` over ``width`` (the round-18 form). Only WordResult.prob changes.
Switch: PRONUNCIADA_DE_IDENTITY=0 (or PRONUNCIADA_DE_R19=0).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources

import numpy as np

from . import attr_heads as AH

_R_VOWEL = {"eːɐ": "eː", "iːɐ": "iː", "uːɐ": "uː", "oːɐ": "oː", "yːɐ": "yː", "øːɐ": "øː", "ɛːɐ": "ɛː",
            "aːɐ": "ɑː", "ɑːɐ": "ɑː"}
_SKIP = {"ʀ", "ʁ", "r", "ɾ", "ə", "ɐ", "ʔ"}
_MERGE = {"z": "s", "x": "ç", "aɪ": "ai", "aʊ": "au", "aː": "ɑː"}
_DEVOICE = {"b": "p", "d": "t", "g": "k", "v": "f", "z": "s", "ʒ": "ʃ"}


def classes(ref) -> list[str | None]:
    """German sound class of every reference sound (None = not judged)."""
    out: list[str | None] = []
    for i, p in enumerate(ref.phones):
        ipa = p.ipa
        w = ref.words[p.word]
        last = i == w.phones[1] - 1
        if ipa in _SKIP:
            out.append(None)
            continue
        if p.is_nucleus:
            ipa = _R_VOWEL.get(ipa, ipa)
            if ipa.endswith("ɐ"):
                out.append(None)
                continue
            if last and ipa in ("eː", "e") and ref.syllables[p.syllable].stress != 1:
                out.append(None)                   # unstressed final -e: a schwa
                continue
            out.append(_MERGE.get(ipa, ipa))
            continue
        if last:
            ipa = _DEVOICE.get(ipa, ipa)
        out.append(_MERGE.get(ipa, ipa))
    return out


@lru_cache(maxsize=1)
def _head():
    p = resources.files("pronunciada").joinpath("data/coach/de_identity.npz")
    try:
        with p.open("rb") as f:
            z = dict(np.load(f, allow_pickle=False))
    except (FileNotFoundError, OSError):
        return None
    H = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in z.items()}
    H["index"] = {str(c): j for j, c in enumerate(H["classes"].tolist())}
    return H


@lru_cache(maxsize=1)
def _word_params() -> dict | None:
    p = resources.files("pronunciada").joinpath("data/calib/de_identity_word.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None


def enabled() -> bool:
    if os.environ.get("PRONUNCIADA_DE_IDENTITY", "1") == "0" or os.environ.get("PRONUNCIADA_DE_R19", "1") == "0":
        return False
    return _head() is not None and _word_params() is not None


def _mlp(H, X: np.ndarray) -> np.ndarray:
    X = (X - H["mu"]) / H["sd"]
    A = X @ H["W1"] + H["b1"]
    A = 0.5 * A * (1.0 + np.tanh(0.7978845608 * (A + 0.044715 * A ** 3)))
    Z = A @ H["W2"] + H["b2"]
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def raw_evidence(ref, V: np.ndarray, H=None) -> np.ndarray:
    """s per sound before the class offset (NaN where not judged)."""
    H = H or _head()
    cls = classes(ref)
    s = np.full(len(cls), np.nan)
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


def evidence(ref, V: np.ndarray, H=None, cal: dict | None = None) -> np.ndarray:
    cal = cal if cal is not None else (_word_params() or {})
    s = raw_evidence(ref, V, H)
    off = cal.get("offset", {})
    if off:
        cls = classes(ref)
        s = np.array([x - float(off.get(c, 0.0)) if c is not None and np.isfinite(x) else x for x, c in zip(s, cls)])
    return s


def knee_at(cal: dict, s: float) -> float:
    k = cal["knee"]
    s = min(max(float(s), 0.0), 1.0)
    return k[0] + (k[1] - k[0]) * 2 * s if s <= 0.5 else k[1] + (k[2] - k[1]) * (2 * s - 1)


def word_factor(cal: dict, s_word, strictness: float) -> float | None:
    v = np.asarray(s_word, np.float64)
    v = v[np.isfinite(v)]
    if cal is None or not len(v):
        return None
    t = min(max((knee_at(cal, strictness) - float(v.min())) / cal["width"], 0.0), 1.0)
    return 1.0 - (1.0 - cal["floor"]) * t
