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


def evidence(lang: str, ipa: list[str], V: np.ndarray) -> dict:
    """{contrast: z per sound (NaN where the sound is not a member)} for the language's entries."""
    E = entries(lang)
    heads = AH.shipped() if E else None
    if not E or heads is None:
        return {}
    C = AH.context_vectors(V)
    out = {}
    for e in E:
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
