"""Round 19 (German): a reliability-weighted stress factor for the word probability.

WordResult.prob answers "were this word's sounds acceptable". The word-stress judgement is less
reliable on casual recordings (little voicing in the word, short words, noise, fast speech), and
there it sank native words. For a language with data/calib/stress_rel/<lang>.json, the factor that
multiplies the word probability is recalibrated: a logistic regression of the stress evidence (lowest
and second-lowest syllable stress logit, number of scored syllables) with its interactions with the
reliability of the pitch / energy evidence for that word (voiced share and mean voicing probability
over the word, word energy, log number of syllables, clip SNR, speaking rate), then moved towards
acceptance by a fixed log-odds shift. Fitted on round-19 German dev natives and planted stress errors
(bench/round19/stress_fit.py); the shift was chosen on the dev trade-off curve.

Only the factor in WordResult.prob changes. WordResult.stress_prob, SyllableResult.stress_prob and
the coaching stress items keep the shipped stress channel. Switch: PRONUNCIADA_DE_STRESS_REL=0 (or
PRONUNCIADA_DE_R19=0).
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from importlib import resources

import numpy as np


@lru_cache(maxsize=None)
def _params(lang: str) -> dict | None:
    p = resources.files("pronunciada").joinpath(f"data/calib/stress_rel/{lang}.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None


def params(lang: str) -> dict | None:
    if os.environ.get("PRONUNCIADA_DE_STRESS_REL", "1") == "0":
        return None
    if lang == "de" and os.environ.get("PRONUNCIADA_DE_R19", "1") == "0":
        return None
    return _params(lang)


def utterance_context(words_span: list[tuple[float, float]], n_syl: int, trk, audio: dict | None) -> dict:
    sp = [(a, b) for a, b in words_span if b > a]
    span = (max(b for _, b in sp) - min(a for a, _ in sp)) if sp else 0.0
    return {"rate": n_syl / span if span > 0 else np.nan, "snr_db": (audio or {}).get("snr_db"),
            "trk": trk}


def reliability(ctx: dict, start: float, end: float, n_syl: int) -> list[float]:
    trk = ctx["trk"]
    if trk is not None and len(trk.f0):
        t = trk.times
        m = (t >= start) & (t < end)
    else:
        m = None
    ok = m is not None and m.any()
    return [float(trk.voiced[m].mean()) if ok else np.nan, float(trk.voicing[m].mean()) if ok else np.nan,
            float(trk.energy_db[m].mean()) if ok else np.nan, math.log(max(1, n_syl)),
            np.nan if ctx["snr_db"] is None else float(ctx["snr_db"]), ctx["rate"]]


def factor(cal: dict, pros_cal: dict, stress_probs: list[float], rel: list[float], s: float) -> float:
    """The word's stress factor at strictness s."""
    q = np.sort(np.log(np.clip(stress_probs, 1e-6, 1 - 1e-6) / (1 - np.clip(stress_probs, 1e-6, 1 - 1e-6))))
    F = np.array([q[0], q[1] if len(q) > 1 else q[0], float(len(q))])
    R = np.array([v if v is not None and np.isfinite(v) else m for v, m in zip(rel, cal["rel_median"])], np.float64)
    Rz = (R - np.array(cal["rel_mu"])) / np.array(cal["rel_sd"])
    X = np.concatenate([F, F[0] * Rz, Rz])
    z = float(X @ np.array(cal["coef"]) + cal["b0"])
    # strictness: the shipped channel's own strictness spread around s = 0.5
    b = pros_cal["coef"]
    z0 = b[0] + b[1] * F[0] + b[2] * F[1] + b[3] * F[2]
    zl, zs = z0 + pros_cal["c_len"], min(z0 + pros_cal["c_str"], z0 + pros_cal["c_len"])
    s = min(max(float(s), 0.0), 1.0)
    z += ((1 - s) * zl + s * zs) - (0.5 * zl + 0.5 * zs)
    return float(1.0 / (1.0 + math.exp(-(z + cal["shift"]))))
