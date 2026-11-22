"""Round 19 (Korean): offline evidence on collected rows (no re-scoring).

vowel_s(rows, H): per item, per word the lowest vowel log-odds s = logit P(expected class) over
the word's judged nuclei (NaN when none). contrast(rows): per word the lowest stored contrast z
(round 16-18 aspiration / tense evidence).
"""
from __future__ import annotations

import json

import numpy as np

from bench.round19ko.vowel_head import CLASSES, predict
from pronunciada import attr_heads as AH


def load_head(path="pronunciada/data/coach/ko_vowel.npz"):
    z = np.load(path)
    return {k: (z[k].astype(np.float32) if z[k].dtype.kind == "f" else z[k]) for k in z.files}


def per_phone_s(r, H):
    ph = r["phones"]
    n = len(ph["ipa"])
    s = np.full(n, np.nan)
    idx = [i for i, v in enumerate(ph["vclass"]) if v in CLASSES]
    if idx and ph["V"] is not None:
        V = ph["V"].astype(np.float32)
        C = AH.context_vectors(V)
        P = predict(H, np.concatenate([V[idx], C[idx]], 1))
        for k, i in enumerate(idx):
            q = float(np.clip(P[k, CLASSES.index(ph["vclass"][i])], 1e-6, 1 - 1e-6))
            s[i] = np.log(q / (1 - q))
    return s


def word_min(r, s):
    w = r["phones"]["word"]
    nw = len(r["words"]["prob"])
    out = np.full(nw, np.nan)
    for j in range(nw):
        v = s[w == j]
        v = v[np.isfinite(v)]
        if len(v):
            out[j] = v.min()
    return out


def vowel_s(rows, H):
    return {k: word_min(r, per_phone_s(r, H)) for k, r in rows.items() if "error" not in r}


def contrast(rows):
    out = {}
    for k, r in rows.items():
        if "error" in r:
            continue
        cz = r["phones"]["contrast_z"]
        out[k] = word_min(r, cz.astype(np.float64)) if cz is not None else np.full(len(r["words"]["prob"]), np.nan)
    return out


def keep_items(rows, asr_path=None, max_cer=0.1):
    """Dev items kept by the read-the-prompt rule (CV only; FLEURS validation always kept)."""
    if asr_path is None:
        return rows
    ok = {}
    for line in open(asr_path, encoding="utf-8"):
        d = json.loads(line)
        ok[d["id"]] = d["cer"] <= max_cer
    return {k: r for k, r in rows.items() if r["meta"]["src"] != "cv" or ok.get(r["meta"]["id"], False)}
