"""Round 19: per-language recalibration of the word model's sounds probability (German).

The pooled word model (data/calib/word.npz) gives each word lenient / strict log-odds (w_zl, w_zs).
For a language with a file data/calib/word_lang/<lang>.json, the SHOWN word probability reads
    z' = a * z + sum_k b_k * x_k + b0 (+ oov_shift for a word the lexicon does not know)
for z in (w_zl, w_zs), x = the word's lowest / mean lenient and strict sound log-odds and the log of
its number of sounds (inputs the word model already reads), then blends z' by strictness as before.
Since a > 0 the strict <= lenient order is kept, and at strictness 0.5 this is exactly the fitted
logistic. Fitted on planted edits on Common Voice / FLEURS-validation German natives that are not
in any benchmark (bench/round19/fit_rerank.py); the intercept keeps native word acceptance.
The sentence level, the 0-10 word score and the stress / presence factors are unchanged.
Switch: PRONUNCIADA_WORD_LANG=0 (or PRONUNCIADA_DE_R19=0 for every round-19 German change).
"""
from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from importlib import resources

import numpy as np

_P = re.compile(r"[^\w\s-]")


@lru_cache(maxsize=None)
def _params(lang: str) -> dict | None:
    p = resources.files("pronunciada").joinpath(f"data/calib/word_lang/{lang}.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None


def params(lang: str) -> dict | None:
    if os.environ.get("PRONUNCIADA_WORD_LANG", "1") == "0":
        return None
    if lang == "de" and os.environ.get("PRONUNCIADA_DE_R19", "1") == "0":
        return None
    return _params(lang)


def oov_words(ref) -> np.ndarray:
    """Per reference word: True when the front-end read it with letter rules (an "oov:" note)."""
    oov = {w.split(":", 1)[1] for w in ref.warnings if w.startswith("oov:")}
    return np.array([_P.sub("", w.text).lower() in oov or w.text.lower() in oov for w in ref.words], bool)


def transform(cal: dict, W: dict, zl, zs, oov) -> tuple[np.ndarray, np.ndarray]:
    """(z_len', z_str') for every word. W: the word table (pronunciada.features.word_table)."""
    clip = cal.get("clip", 12.0)
    a, *b = cal["coef"]
    names = cal["features"][1:]
    rest = np.full(len(zl), float(cal["b0"]))
    for k, name in zip(b, names):
        if name == "log_n":
            x = np.log(np.maximum(1.0, np.asarray(W["n_phones"], np.float64)))
        else:
            x = np.clip(np.nan_to_num(np.asarray(W[name], np.float64), nan=0.0), -clip, clip)
        rest = rest + k * x
    rest = rest + np.where(np.asarray(oov, bool), float(cal.get("oov_shift", 0.0)), 0.0)
    zl = np.clip(np.asarray(zl, np.float64), -clip, clip)
    zs = np.clip(np.minimum(np.asarray(zs, np.float64), zl), -clip, clip)
    return a * zl + rest, a * zs + rest


def _check(cal: dict) -> None:
    assert cal["features"][0] == "z_word" and cal["coef"][0] > 0 and math.isfinite(cal["b0"])
