"""Round 19 rollout: "which sound was said" evidence for the word probability, any language.

The generic form of the German and Korean heads of round 19 (pronunciada/de_identity.py,
ko_identity.py; those stay as they are). A one-hidden-layer MLP (``data/coach/identity/<lang>.npz``)
on the pooled enc1|enc3|enc5 vector of a sound plus its neighbours' mean predicts which of the
language's sound classes was said. It was fitted on native read speech only (FLEURS train, Common
Voice natives; bench/round19_rollout/train.py) with the front-end's reference of the prompt as label.

Classes and the word rule are in ``data/calib/identity/<lang>.json``:
  merge      symbol -> class (native variants folded into one class, e.g. English dark l -> l)
  skip       classes that are never judged (e.g. schwa); they are still classes of the head
  skip_deletable  a sound whose front-end entry allows dropping it with log prior >= this is not judged
  offset     per class: a low native quantile of the evidence (subtracted)
  form       "knee": one-sided factor on WordResult.prob (1 while the word's lowest evidence is above a
             knee, then linearly down to ``floor`` over ``width``; the round-18 / German form);
             "step": a bounded log-odds step on WordResult.prob (Korean form) with ``no_raise_below``.
Evidence per judged sound: s = log P(expected class) - log max P(class that is neither the expected
class nor one of the sound's allowed variants in the front-end) - offset(class).
Only WordResult.prob changes. Switch: PRONUNCIADA_IDENTITY_<LANG>=0 (or PRONUNCIADA_IDENTITY=0).
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from importlib import resources

import numpy as np

from . import attr_heads as AH


@lru_cache(maxsize=None)
def _head(lang: str):
    p = resources.files("pronunciada").joinpath(f"data/coach/identity/{lang}.npz")
    try:
        with p.open("rb") as f:
            z = dict(np.load(f, allow_pickle=False))
    except (FileNotFoundError, OSError):
        return None
    H = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in z.items()}
    H["index"] = {str(c): j for j, c in enumerate(H["classes"].tolist())}
    return H


@lru_cache(maxsize=None)
def _cal(lang: str) -> dict | None:
    p = resources.files("pronunciada").joinpath(f"data/calib/identity/{lang}.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None


def enabled(lang: str) -> bool:
    if os.environ.get("PRONUNCIADA_IDENTITY", "1") == "0":
        return False
    if os.environ.get(f"PRONUNCIADA_IDENTITY_{lang.upper()}", "1") == "0":
        return False
    return _head(lang) is not None and _cal(lang) is not None


def _cls(spec: dict, ipa: str) -> str:
    return spec.get("merge", {}).get(ipa, ipa)


def classes(ref, spec: dict, H=None) -> tuple[list, list]:
    """(class of every reference sound or None when not judged, allowed classes of every sound)."""
    skip = set(spec.get("skip", ()))
    dmin = spec.get("skip_deletable")
    index = H["index"] if H is not None else None
    out, allow = [], []
    for p in ref.phones:
        c = _cls(spec, p.ipa)
        alts = list(getattr(p, "alts", ()) or ())
        ok = c not in skip and (index is None or c in index)
        if ok and dmin is not None and any((a == "" and lp >= dmin) for a, lp in alts):
            ok = False
        out.append(c if ok else None)
        al = {c}
        for a, _lp in alts:
            if a:
                al.add(_cls(spec, a))
        allow.append(al)
    return out, allow


def _mlp(H, X: np.ndarray) -> np.ndarray:
    X = (X - H["mu"]) / H["sd"]
    A = X @ H["W1"] + H["b1"]
    A = 0.5 * A * (1.0 + np.tanh(0.7978845608 * (A + 0.044715 * A ** 3)))
    Z = A @ H["W2"] + H["b2"]
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def raw_evidence(ref, V: np.ndarray, spec: dict, H) -> tuple[np.ndarray, list]:
    """s per sound before the class offset (NaN where not judged), and the class list."""
    cls, allow = classes(ref, spec, H)
    s = np.full(len(cls), np.nan)
    idx = [i for i, c in enumerate(cls) if c is not None]
    if not idx:
        return s, cls
    V = np.asarray(V, np.float32)
    X = np.concatenate([V, AH.context_vectors(V)], axis=1)[idx]
    lp = np.log(np.clip(_mlp(H, X), 1e-9, 1.0))
    for r, i in enumerate(idx):
        j = H["index"][cls[i]]
        mask = np.ones(lp.shape[1], bool)
        for a in allow[i]:
            k = H["index"].get(a)
            if k is not None:
                mask[k] = False
        s[i] = lp[r, j] - (lp[r, mask].max() if mask.any() else -20.0)
    return s, cls


def evidence(lang: str, ref, V: np.ndarray) -> np.ndarray:
    H, cal = _head(lang), _cal(lang)
    s, cls = raw_evidence(ref, V, cal, H)
    off = cal.get("offset", {})
    return np.array([x - float(off.get(c, 0.0)) if c is not None and np.isfinite(x) else x for x, c in zip(s, cls)])


def _interp(k, s: float) -> float:
    s = min(max(float(s), 0.0), 1.0)
    return k[0] + (k[1] - k[0]) * 2 * s if s <= 0.5 else k[1] + (k[2] - k[1]) * (2 * s - 1)


def step_features(v: np.ndarray) -> np.ndarray:
    """Word features of the "step" form, from the word's judged evidence values v (finite, non-empty)."""
    m = float(v.min())
    return np.array([min(m, 0.0), min(m + 4.0, 0.0), float(np.mean(np.minimum(v, 4.0))),
                     float(np.sum(1.0 / (1.0 + np.exp(2.0 * v)))), 1.0])


def apply_word(cal: dict, s_word, prob: float, strictness: float) -> float:
    """New WordResult.prob from the word's evidence values."""
    v = np.asarray(s_word, np.float64)
    v = v[np.isfinite(v)]
    if cal is None or not len(v):
        return prob
    if cal.get("form", "knee") == "knee":
        t = min(max((_interp(cal["knee"], strictness) - float(v.min())) / cal["width"], 0.0), 1.0)
        return prob * (1.0 - (1.0 - cal["floor"]) * t)
    d = float(np.dot(np.asarray(cal["coef"], np.float64), step_features(v)))
    d = min(d, float(cal.get("cap", 3.0)))
    if cal.get("no_raise_below") is not None and float(v.min()) < cal["no_raise_below"]:
        d = min(d, 0.0)
    p = min(max(prob, 1e-6), 1 - 1e-6)
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) + d)))
