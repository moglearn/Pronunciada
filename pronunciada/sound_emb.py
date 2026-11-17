"""Per-sound inputs from the acoustic model's internal layers (round 12b, English sound heads).

For every expected sound: the mean of the enc1 / enc3 / enc5 activations over its aligned frames,
widened by one frame on each side (empty span: the 2 frames around its position), concatenated
(1024-d). Two kinds of inputs are derived and added to the phone table before the sound
calibrator runs:

* ``emb_proj``: a ridge projection of that vector to the expert 0-2 phone score (fitted on
  speechocean762 train, bench/calib/train_sound_emb.py);
* ``proto_cos`` / ``proto_other`` / ``proto_margin`` / ``proto_has``: cosine of the language-
  standardised vector to the native prototype of the expected sound (best over its accepted
  alternatives), best cosine to any other phone, their margin, and whether a prototype exists
  (prototypes: FLEURS train natives).

Weights: pronunciada/data/calib/<sound model>_emb.npz. numpy only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LAYERS = ("enc1", "enc3", "enc5")


def phone_vectors(emb: dict, ev, n_phones: int) -> np.ndarray:
    T = len(emb[LAYERS[0]])
    X = np.concatenate([np.asarray(emb[k], np.float32) for k in LAYERS], axis=1)
    out = np.empty((n_phones, X.shape[1]), np.float32)
    for i, sl in enumerate(ev.slots[:n_phones]):
        a, b = int(sl.start_frame), int(sl.end_frame)
        if b <= a:
            a, b = a - 1, a + 1
        a, b = max(a - 1, 0), min(b + 1, T)
        if b <= a:
            a = max(min(a, T - 1), 0)
            b = a + 1
        out[i] = X[a:b].mean(0)
    return out


class SoundEmb:
    COLS = ("emb_proj", "proto_cos", "proto_other", "proto_margin", "proto_has")

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.pmu, self.psd = z["proj_mu"].astype(np.float32), z["proj_sd"].astype(np.float32)
            self.pW, self.pb = z["proj_W"].astype(np.float32), float(z["proj_b"][0])
            self.mu, self.sd = z["proto_mu"].astype(np.float32), z["proto_sd"].astype(np.float32)
            self.P = z["proto_P"].astype(np.float32)
            self.idx = {p: i for i, p in enumerate(z["proto_phones"].tolist())}

    def features(self, V: np.ndarray, ref) -> dict:
        n = len(V)
        proj = ((V - self.pmu) / self.psd) @ self.pW + self.pb
        Z = (V - self.mu) / self.sd
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        C = Z @ self.P.T
        f = np.full((n, 3), np.nan)
        for i, p in enumerate(ref.phones[:n]):
            acc = [self.idx[x] for x in [p.ipa] + [a for a, _ in p.alts] if x in self.idx]
            if not acc:
                continue
            ce = C[i, acc].max()
            m = np.ones(C.shape[1], bool)
            m[acc] = False
            bo = C[i, m].max() if m.any() else -1.0
            f[i] = (ce, bo, ce - bo)
        return {"emb_proj": proj.astype(np.float64), "proto_cos": f[:, 0], "proto_other": f[:, 1],
                "proto_margin": f[:, 2], "proto_has": np.isfinite(f[:, 0]).astype(float)}

    def augment(self, tab: dict, ref, ev, emb) -> None:
        """Add the columns to the phone table in place (NaN without activations)."""
        n = len(ref.phones)
        if not emb or n == 0:
            for c in self.COLS:
                tab[c] = np.full(n, np.nan)
            return
        tab.update(self.features(phone_vectors(emb, ev, n), ref))


def load(sound_model: str, calib_dir: str | None = None) -> SoundEmb | None:
    from .calibrate import data_dir

    d = Path(calib_dir) if calib_dir else data_dir()
    p = d / f"{sound_model}_emb.npz"
    return SoundEmb(p) if p.exists() else None
