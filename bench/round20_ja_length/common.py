"""Shared pieces of the round-20 Japanese vowel-length work (bench only)."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada") / "round20" / "jalength"
W = 10                       # stored window: +-10 output frames (+-200 ms) around the vowel's CTC peak

# DEMAND (Thiemann, Ito & Vincent 2013; Zenodo record 1227121, CC BY 4.0), 16 kHz, channel 1.
# Six environments train the head; two are held back for the dev WSOLA pairs.
TRAIN_ENVS = ("DKITCHEN", "DLIVING", "OOFFICE", "PSTATION", "TBUS", "NPARK")
DEV_ENVS = ("PCAFETER", "STRAFFIC")
NOISE_DIR = OUT / "noise"


def soft_label(text_long: float, factor: float) -> float:
    """P(the audio vowel is long) after a WSOLA factor on a natural vowel of the given text length.
    A natural short vowel becomes long around x1.5 of its stable part; a natural long vowel (about
    twice a short one) becomes short around x0.67. Factor 1 gives the natural label."""
    if abs(factor - 1.0) <= 0.1 + 1e-9:
        return float(text_long)
    mid = math.log(0.67) if text_long else math.log(1.5)
    return float(1.0 / (1.0 + math.exp(-(math.log(factor) - mid) / 0.1)))


def noise_bank(envs) -> dict[str, np.ndarray]:
    import soundfile as sf

    out = {}
    for e in envs:
        p = NOISE_DIR / f"{e}_ch01.wav"
        if p.exists():
            x, sr = sf.read(str(p), dtype="float32")
            assert sr == 16000
            out[e] = x
    return out


def active_power(x: np.ndarray) -> float:
    """Mean power over the 20 ms frames within 30 dB of the loudest frames (speech, not pauses)."""
    n = len(x) // 320
    if n < 1:
        return float(np.mean(x ** 2) + 1e-12)
    f = (x[: n * 320].reshape(n, 320) ** 2).mean(1) + 1e-12
    db = 10 * np.log10(f)
    act = f[db >= np.percentile(db, 95) - 30]
    return float(act.mean())


def add_noise(x: np.ndarray, bank: dict, snr_db: float, rng) -> np.ndarray:
    if not bank:
        raise RuntimeError("no noise files under " + str(NOISE_DIR))
    env = sorted(bank)[int(rng.integers(len(bank)))]
    nz = bank[env]
    if len(nz) <= len(x):
        nz = np.tile(nz, len(x) // len(nz) + 1)
    s = int(rng.integers(0, len(nz) - len(x) + 1))
    seg = nz[s: s + len(x)].astype(np.float32)
    ps, pn = active_power(x), float(np.mean(seg ** 2) + 1e-12)
    g = math.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    y = x + g * seg
    m = float(np.max(np.abs(y)))
    return (y / m * 0.99).astype(np.float32) if m > 0.99 else y.astype(np.float32)
