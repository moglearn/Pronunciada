"""Round 18: word-level contrast factor.

Rounds 16-17 give, for sounds of specific contrasts (aspiration, voicing, palatalisation, θ / ð,
Korean tense), a contrast log-odds z ("the contrast was realised as expected": round-16 cap
logits and round-17 head evidence; Scorer cache "contrast_z"). The word factor uses the lowest
z of the word's judged sounds, like the length channel (pronunciada.length.word_factor):
1 while it is above a knee placed at a low native quantile (by strictness), then falling
linearly in log-odds to `floor` over `width`. Natives are almost never touched.
Parameters: pronunciada/data/calib/contrast_word/<lang>.json (bench/round18/fit.py).
Switch: PRONUNCIADA_WORD_CONTRAST=0.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources

import numpy as np


@lru_cache(maxsize=None)
def _params(lang: str) -> dict | None:
    p = resources.files("pronunciada").joinpath(f"data/calib/contrast_word/{lang}.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None


def params(lang: str) -> dict | None:
    if os.environ.get("PRONUNCIADA_WORD_CONTRAST", "1") == "0":
        return None
    return _params(lang)


def knee_at(cal: dict, s: float) -> float:
    k = cal["knee"]
    s = min(max(float(s), 0.0), 1.0)
    return k[0] + (k[1] - k[0]) * 2 * s if s <= 0.5 else k[1] + (k[2] - k[1]) * (2 * s - 1)


def word_factor(cal: dict, z, s: float) -> float | None:
    """Factor for one word from its sounds' contrast log-odds (NaN = not judged); None when no
    sound of the word is judged."""
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if not len(z):
        return None
    t = min(max((knee_at(cal, s) - float(z.min())) / cal["width"], 0.0), 1.0)
    return 1.0 - (1.0 - cal["floor"]) * t
