"""Round 19 (German): re-run the shipped word model offline on the word inputs stored in a dump.

sounds(W) = sigmoid(blend(z_len, min(z_str, z_len), 0.5)) of pronunciada/data/calib/word.npz, so a
change of the word inputs (e.g. a feature replaced) can be evaluated without re-scoring audio.
"""
from __future__ import annotations

import numpy as np

_M = None


def model():
    global _M
    if _M is None:
        from pronunciada import calibrate as C

        _M = C.load_level("word")
    return _M


def sounds(rows_W: list[dict], s: float = 0.5) -> np.ndarray:
    from pronunciada import calibrate as C

    m = model()
    X = np.array([[np.nan if W[c] is None else W[c] for c in m.num] for W in rows_W], dtype=np.float64)
    w = m.logits(X=X)
    return C.sigmoid(C.blend_logit(w["z_len"], np.minimum(w["z_str"], w["z_len"]), s))
