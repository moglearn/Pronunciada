"""Mandarin consonant / vowel coaching score with per-sound ZIPA-layer inputs (round 12b).

Per sound: the logistic of bench/calib/ompal_sound_emb.py (fitted on OMPAL DEV learners, speaker-
grouped CV) on [logit of the general coaching score, margin of the pooled activations to the
native FLEURS-zh prototype of the expected sound, ridge projection of the pooled activations],
separately for syllable-initial consonants and finals. Weights: pronunciada/data/coach/zh_sound.npz
(numpy only). The vectors come from pronunciada.sound_emb.phone_vectors (Scorer cache
"phone_vec"); without them the general score is used.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np


@lru_cache(maxsize=1)
def params():
    p = resources.files("pronunciada").joinpath("data/coach/zh_sound.npz")
    if not p.is_file():
        return None
    with np.load(str(p), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["idx"] = {p: i for i, p in enumerate(d["proto_phones"].tolist())}
    d["proto_P"] = d["proto_P"].astype(np.float32)
    return d


def margins(V: np.ndarray, expected, alts) -> np.ndarray:
    P = params()
    Z = (V - P["proto_mu"]) / P["proto_sd"]
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    C = Z @ P["proto_P"].T
    out = np.full(len(V), np.nan)
    for i, (e, al) in enumerate(zip(expected, alts)):
        acc = [P["idx"][x] for x in [e] + list(al) if x in P["idx"]]
        if not acc:
            continue
        m = np.ones(C.shape[1], bool)
        m[acc] = False
        out[i] = C[i, acc].max() - (C[i, m].max() if m.any() else -1.0)
    return out


def error_prob(X: dict, ep: np.ndarray) -> np.ndarray:
    """Per-sound Mandarin coaching score (initial / final models); ep = the general score."""
    P = params()
    V = X.get("phone_vec")
    if P is None or V is None or len(V) != len(ep):
        return ep
    V = np.asarray(V, np.float32)
    mg = margins(V, X["expected"], X["alts"])
    lg = np.log(np.clip(ep, 1e-6, 1 - 1e-6) / (1 - np.clip(ep, 1e-6, 1 - 1e-6)))
    out = np.empty(len(ep))
    ini = np.asarray(X["initial"], bool)
    for part, msk in (("initial", ini), ("final", ~ini)):
        if not msk.any():
            continue
        g = lambda k: P[f"{part}_{k}"]  # noqa: E731
        pj = ((V[msk] - g("mu")) / g("sd")) @ g("W") + float(g("b")[0])
        f = np.column_stack([lg[msk], np.where(np.isfinite(mg[msk]), mg[msk], float(g("margin_fill")[0])), pj])
        z = f @ g("coef").astype(np.float64) + float(g("intercept")[0])
        out[msk] = 1.0 / (1.0 + np.exp(-z))
    return out
