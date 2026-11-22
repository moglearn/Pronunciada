"""Round 19 (English): load the feature cache of bench/round19/extract_en.py, build CV folds.

Folds: 5-fold speaker-grouped, children (<= 12) and the others dealt separately into the folds
so every fold has about the same share of children; three seeds.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

HEADS = ("accuracy", "fluency", "prosodic", "total")
LAYERS = ("enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "final")
SEEDS = (0, 1, 2)


def feat_dir(split="train", precision="fp32", dataset="speechocean762") -> Path:
    from bench.data.common import cache_root

    return cache_root() / "round19" / "english" / "feats" / f"{dataset}_{split}_{precision}"


def load(split="train", precision="fp32", keep=None) -> list[dict]:
    """Utterances in shard order (errors dropped). keep: keys to keep (saves memory)."""
    out = []
    for p in sorted(feat_dir(split, precision).glob("shard*.pkl")):
        d = pickle.loads(p.read_bytes())
        for uid, v in d.items():
            if "error" in v:
                continue
            if keep is not None:
                v = {k: v[k] for k in keep if k in v}
            v["id"] = uid
            out.append(v)
    return out


def folds(spk, age, seed, k=5):
    """[(train idx, test idx)] speaker-grouped, children / others balanced."""
    spk = np.asarray(spk)
    age = np.asarray(age)
    sa = {}
    for s, a in zip(spk, age):
        sa[s] = a
    rng = np.random.default_rng(seed)
    f = {}
    off = 0
    for grp in (sorted(s for s, a in sa.items() if a <= 12), sorted(s for s, a in sa.items() if a > 12)):
        for i, s in enumerate(rng.permutation(grp)):
            f[s] = (i + off) % k
        off += len(grp)
    fid = np.array([f[s] for s in spk])
    return [(np.flatnonzero(fid != j), np.flatnonzero(fid == j)) for j in range(k)]


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def pooled_part(v, key: str) -> np.ndarray:
    """Part of an utterance-pooled vector (mean ++ std): "enc1" = all, "enc1.mean" = the mean
    half, "enc1.std" = the std half."""
    v = np.asarray(v, np.float64)
    if key.endswith(".mean"):
        return v[..., :v.shape[-1] // 2]
    if key.endswith(".std"):
        return v[..., v.shape[-1] // 2:]
    return v


def layer_of(key: str) -> str:
    return key.split(".")[0]
