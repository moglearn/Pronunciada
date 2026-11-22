"""Round 19 (Japanese): word recalibration.

The shipped word probability multiplies separate pieces (the word model of the sound evidence,
present, the length and contrast factors), each tuned on its own. On Common Voice style audio that
product separates a word whose text was changed from a correct one poorly (results/round19_japanese.md).
This module combines the same pieces, plus word-table statistics and the free-decode mismatch of
pronunciada.word_match, with boosted depth-3 trees fitted on planted one-word text edits in native
Common Voice ja train and FLEURS ja train audio (bench/round19_ja/wordcal.py). The trees' logit is
mapped by quantiles onto the shipped word logit's distribution on untouched FLEURS train words, so
native word rates stay where they were. Strictness moves the result as it moves the sounds logit.

Parameters: pronunciada/data/calib/wordcal/<lang>.json. Switch: PRONUNCIADA_JA_ROUND19=0 (or
PRONUNCIADA_JA_ROUND19_OFF=wordcal).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources

import numpy as np


@lru_cache(maxsize=None)
def _params(lang: str) -> dict | None:
    p = resources.files("pronunciada").joinpath(f"data/calib/wordcal/{lang}.json")
    try:
        M = json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None
    M["_np"] = (np.asarray(M["feat"], dtype=np.int64), np.asarray(M["thr"], dtype=np.float64),
                np.asarray(M["leaf"], dtype=np.float64))
    M["_q"] = (np.asarray(M["q_new"], dtype=np.float64), np.asarray(M["q_orig"], dtype=np.float64))
    return M


def params(lang: str) -> dict | None:
    from .ja_round19 import on

    if lang != "ja" or not on("wordcal"):
        return None
    return _params(lang)


def _logit(p: float, lo: float = 1e-6) -> float:
    p = min(max(float(p), lo), 1 - lo)
    return math.log(p / (1 - p))


def rows(sounds, len_min, con, present, mm, W) -> np.ndarray:
    """Feature matrix (one row per word) in the order of the fitted model's FEATURES."""
    out = []
    for k in range(len(sounds)):
        m = mm[k]
        n = max(m["n"], 1)
        pres = present[k]
        lz = len_min[k]
        npk = float(W["n_phones"][k])
        out.append({
            "z_sounds": _logit(sounds[k]), "z_len": 8.0 if lz is None else min(lz, 8.0),
            "has_len": float(lz is not None), "log_con": math.log(max(1.0 if con[k] is None else con[k], 1e-6)),
            "z_present": 8.0 if pres is None or not np.isfinite(pres) else _logit(pres),
            "mm_ins": m["ins"], "mm_del": m["del"], "mm_sub": m["sub"], "mm_rate": (m["ins"] + m["del"] + m["sub"]) / n,
            "ws_min": W["ws_min"][k], "wl_min": W["wl_min"][k], "ws_mean": W["ws_mean"][k],
            "word_min_llr_sub": max(float(W["word_min_llr_sub"][k]), -50.0),
            "word_mean_llr_sub": max(float(W["word_mean_llr_sub"][k]), -50.0),
            "word_min_llr_del": float(np.clip(W["word_min_llr_del"][k], -50, 50)), "n_phones": npk,
            "frames_per_phone": float(W["word_frames"][k]) / max(npk, 1.0), "utt_ws_mean": W["utt_ws_mean"][k],
            "word_pos_first": W["word_pos_first"][k], "word_pos_last": W["word_pos_last"][k]})
    return out


def probs(M: dict, feats: list[dict]) -> np.ndarray:
    """Recalibrated word probabilities."""
    if not feats:
        return np.zeros(0)
    X = np.array([[r[f] for f in M["features"]] for r in feats], dtype=np.float64)
    feat, thr, leaf = M["_np"]
    n, K = len(X), len(feat)
    node = np.zeros((n, K), dtype=np.int64)
    kk = np.arange(K)[None, :]
    for _ in range(M.get("depth", 3)):
        f = feat[kk, node]
        right = np.take_along_axis(X, f, axis=1) > thr[kk, node]
        node = 2 * node + 1 + right
    z_err = M["init"] + M["learning_rate"] * leaf[kk, node - (len(feat[0]))].sum(1)
    z = -z_err
    qn, qo = M["_q"]
    zm = np.interp(z, qn, qo)
    zm = np.where(z < qn[0], qo[0] + (z - qn[0]), zm)
    zm = np.where(z > qn[-1], qo[-1] + (z - qn[-1]), zm) + M.get("offset", 0.0)
    return 1.0 / (1.0 + np.exp(-np.clip(zm, -30, 30)))
