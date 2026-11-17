"""Articulatory attribute heads (coaching round 4): per feature, a small linear softmax on the
acoustic model's pooled internal activations of one sound, predicting that feature of the sound
that was actually produced (voicing, aspiration, manner, place, palatalisation, laterality;
vowel height, backness, rounding, nasality, length). Trained on native FLEURS train speech of
all ten languages pooled, labels from the reference sounds through our own alignment
(bench/coach/fit_r4.py). numpy only; weights in pronunciada/data/coach/attr_heads.npz.

    heads = AttrHeads.load()
    P = heads.predict(V, i, ctx)        # {head: class probabilities} for sound i
    labels(ipa)                         # {head: class} of a reference sound (None: n/a)
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np

from . import articulatory as A

# head -> (applies to, classes, articulatory diff key it speaks for)
HEADS = {
    "voicing": ("consonant", ("voiced", "voiceless"), "voicing"),
    "aspirated": ("stop", ("yes", "no"), "aspirated"),
    "manner": ("consonant", ("stop", "affricate", "fricative", "nasal", "approximant", "rhotic"), "manner"),
    "place": ("consonant", ("labial", "front", "post", "dorsal", "laryngeal"), "place"),
    "palatalised": ("consonant", ("yes", "no"), "palatalised"),
    "lateral": ("consonant", ("yes", "no"), "lateral"),
    "height": ("vowel", ("close", "mid", "open"), "height"),
    "backness": ("vowel", ("front", "central", "back"), "backness"),
    "rounded": ("vowel", ("yes", "no"), "rounded"),
    "nasal": ("vowel", ("yes", "no"), "nasal"),
    "long": ("any", ("yes", "no"), "long"),
}
PLACE = {"bilabial": "labial", "labiodental": "labial", "dental": "front", "alveolar": "front",
         "postalveolar": "post", "retroflex": "post", "alveolopalatal": "post", "palatal": "post",
         "labial-palatal": "post", "velar": "dorsal", "uvular": "dorsal", "labial-velar": "dorsal",
         "postalveolar-velar": "dorsal", "glottal": "laryngeal", "pharyngeal": "laryngeal",
         "epiglottal": "laryngeal"}
MANNER = {"stop": "stop", "implosive": "stop", "click": "stop", "affricate": "affricate", "fricative": "fricative",
          "nasal": "nasal", "approximant": "approximant", "trill": "rhotic", "flap": "rhotic"}
HEIGHT = {"close": "close", "near-close": "close", "close-mid": "mid", "mid": "mid", "open-mid": "mid",
          "near-open": "open", "open": "open"}
YN = {True: "yes", False: "no"}


@lru_cache(maxsize=4096)
def _labels(ipa: str) -> tuple:
    try:
        f = A.features(ipa)
    except KeyError:
        return ()
    out = {}
    if f["type"] == "consonant":
        out["voicing"] = f["voicing"]
        if f["manner"] in ("stop", "affricate"):
            out["aspirated"] = YN[bool(f["aspirated"])]
        out["manner"] = MANNER.get(f["manner"])
        out["place"] = PLACE.get(f["place"])
        out["palatalised"] = YN[bool(f["palatalised"])]
        out["lateral"] = YN[bool(f["lateral"])]
    else:
        out["height"] = HEIGHT[f["height"]]
        out["backness"] = f["backness"]
        out["rounded"] = YN[bool(f["rounded"])]
        out["nasal"] = YN[bool(f["nasal"])]
    out["long"] = YN[bool(f["long"])]
    return tuple(sorted((k, v) for k, v in out.items() if v is not None))


def labels(ipa: str) -> dict:
    """{head: class} for a sound; heads that do not apply are absent."""
    return dict(_labels(ipa))


def context_vectors(V: np.ndarray) -> np.ndarray:
    """Mean of the previous and next sound's vectors (the sound's own vector at the ends)."""
    prv = np.vstack([V[:1], V[:-1]])
    nxt = np.vstack([V[1:], V[-1:]])
    return 0.5 * (prv + nxt)


class AttrHeads:
    def __init__(self, z: dict):
        self.heads = {}
        for h in [str(x) for x in z["heads"]]:
            self.heads[h] = {"cols": z[f"{h}/cols"].astype(np.int64), "ctx": bool(z[f"{h}/ctx"][0]),
                             "mu": z[f"{h}/mu"].astype(np.float32), "sd": z[f"{h}/sd"].astype(np.float32),
                             "W": z[f"{h}/W"].astype(np.float32), "b": z[f"{h}/b"].astype(np.float32),
                             "classes": [str(c) for c in z[f"{h}/classes"]]}

    @classmethod
    def load(cls, path=None):
        from pathlib import Path

        p = Path(path) if path is not None else resources.files("pronunciada").joinpath("data/coach/attr_heads.npz")
        try:
            with p.open("rb") as f:
                z = dict(np.load(f, allow_pickle=False))
        except (FileNotFoundError, OSError):
            return None
        return cls(z)

    def predict(self, V: np.ndarray, idx, C: np.ndarray | None = None) -> list[dict]:
        """Class probabilities of every head for the sounds `idx` (rows of V; C = context rows)."""
        idx = np.asarray(list(idx), dtype=np.int64)
        out = [dict() for _ in idx]
        if not len(idx):
            return out
        for h, m in self.heads.items():
            X = V[idx][:, m["cols"]]
            if m["ctx"]:
                X = np.concatenate([X, (C if C is not None else context_vectors(V))[idx][:, m["cols"]]], axis=1)
            Z = ((X - m["mu"]) / m["sd"]) @ m["W"] + m["b"]
            Z -= Z.max(1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(1, keepdims=True)
            for r in range(len(idx)):
                out[r][h] = dict(zip(m["classes"], P[r].tolist()))
        return out


@lru_cache(maxsize=1)
def shipped() -> AttrHeads | None:
    return AttrHeads.load()
