"""Round 17: head-based contrast evidence for contrasts the shown sound probability was blind to
(Korean and Mandarin aspiration, Russian soft / hard), from the coaching attribute heads
(pronunciada/attr_heads.py). numpy only.

For a sound whose expected symbol is a member of the contrast, s = logit P(head class of the
expected symbol) and z = slope * (s - s0), where s0 is a low native quantile of s (FLEURS
validation; bench/round17/fit.py). sigmoid(z) is "the feature was realised as expected".
* where the contrast is adopted as a cap, the scorer composes z with the shown sound log-odds
  by the minimum (with the round-16 caps: min, not product); word / sentence inputs and the
  coaching score are unchanged;
* coaching reads z for the aspiration / palatalisation feature tips (pronunciada.coach).
Parameters: pronunciada/data/coach/contrast_heads.json. Switch: PRONUNCIADA_HEAD_CAPS=0.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources

import numpy as np

from . import attr_heads as AH


@lru_cache(maxsize=1)
def _config() -> dict:
    p = resources.files("pronunciada").joinpath("data/coach/contrast_heads.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return {}


def entries(lang: str) -> list[dict]:
    if os.environ.get("PRONUNCIADA_HEAD_CAPS", "1") == "0":
        return []
    return list(_config().get("langs", {}).get(lang, []))


@lru_cache(maxsize=1)
def _tense_head():
    """Round 18: Korean lax / tense head (data/coach/ko_tense.npz), or None."""
    p = resources.files("pronunciada").joinpath("data/coach/ko_tense.npz")
    try:
        with p.open("rb") as f:
            z = dict(np.load(f, allow_pickle=False))
    except (FileNotFoundError, OSError):
        return None
    return {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in z.items()}


def tense_probs(V: np.ndarray, members) -> np.ndarray:
    """P(tense) per sound from the round-18 head (own vector + neighbours' mean); NaN where not a
    member."""
    H = _tense_head()
    out = np.full(len(V), np.nan)
    idx = [i for i, m in enumerate(members) if m]
    if H is None or not idx:
        return out
    C = AH.context_vectors(V)
    X = np.concatenate([V[idx], C[idx]], axis=1)
    Z = ((X - H["mu"]) / H["sd"]) @ H["W"] + H["b"]
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z)
    P /= P.sum(1, keepdims=True)
    out[idx] = P[:, 1]
    return out


def evidence(lang: str, ipa: list[str], V: np.ndarray, tense=None) -> dict:
    """{contrast: z per sound (NaN where the sound is not a member)} for the language's entries.
    tense: per sound RefPhone.tense (Korean; the "tense" entry reads it)."""
    E = entries(lang)
    heads = AH.shipped() if E else None
    if not E or heads is None:
        return {}
    C = AH.context_vectors(V)
    out = {}
    for e in E:
        if e["contrast"] == "tense":
            z = np.full(len(ipa), np.nan)
            if tense is not None:
                pt = tense_probs(V, [t is not None for t in tense])
                for i, t in enumerate(tense):
                    if t is not None and np.isfinite(pt[i]):
                        q = min(max(pt[i] if t else 1.0 - pt[i], 1e-6), 1 - 1e-6)
                        z[i] = e["slope"] * (np.log(q / (1 - q)) - e["s0"])
            out["tense"] = z
            continue
        mem = set(e["members"])
        idx = [i for i, p in enumerate(ipa) if p in mem]
        z = np.full(len(ipa), np.nan)
        if idx:
            P = heads.predict(V, idx, C)
            for i, p in zip(idx, P):
                cls = AH.labels(ipa[i]).get(e["head"])
                if cls is None or e["head"] not in p:
                    continue
                q = min(max(p[e["head"]].get(cls, 0.0), 1e-6), 1 - 1e-6)
                z[i] = e["slope"] * (np.log(q / (1 - q)) - e["s0"])
        out[e["contrast"]] = z
    return out


def cap_column(lang: str, ev: dict) -> np.ndarray | None:
    """Minimum z over the adopted caps (NaN where none applies), or None."""
    caps = [ev[e["contrast"]] for e in entries(lang) if e.get("cap") and e["contrast"] in ev]
    if not caps:
        return None
    Z = np.vstack(caps)
    with np.errstate(all="ignore"):
        out = np.nanmin(np.where(np.isfinite(Z), Z, np.inf), axis=0)
    return np.where(np.isfinite(out), out, np.nan)


def length_evidence(lang: str, par: dict, ipa: list[str], V: np.ndarray) -> np.ndarray | None:
    """Round 17 (Japanese): s = logit P(head "long" = the expected value) for the length channel's
    member sounds (par["head"]["members"]), NaN elsewhere; None when not configured."""
    H = (par or {}).get("head")
    if not H or os.environ.get("PRONUNCIADA_HEAD_CAPS", "1") == "0":
        return None
    heads = AH.shipped()
    if heads is None:
        return None
    mem = set(H["members"])
    idx = [i for i, p in enumerate(ipa) if p in mem]
    s = np.full(len(ipa), np.nan)
    if idx:
        P = heads.predict(V, idx, AH.context_vectors(V))
        for i, p in zip(idx, P):
            cls = AH.labels(ipa[i]).get("long")
            if cls is not None and "long" in p:
                q = min(max(p["long"].get(cls, 0.0), 1e-6), 1 - 1e-6)
                s[i] = np.log(q / (1 - q))
    return s
