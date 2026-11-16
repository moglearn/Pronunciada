"""Shared on-disk cache of ZIPA frame log-posteriors, so every experiment reuses one model pass.

Layout: $PRONUNCIADA_CACHE/post/<model>-<precision>/<dataset>/<utt id>.npy   float16 [T, 127]
Raw model output only (never adapted). Safe for concurrent writers (atomic rename).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pronunciada.acoustic import AcousticModel, cache_root
from pronunciada.audio import load_audio

_MODELS: dict[tuple[str, str], AcousticModel] = {}


def model(name: str = "zipa-small-crctc-500k", precision: str = "fp32", threads: int = 4) -> AcousticModel:
    key = (name, precision)
    if key not in _MODELS:
        _MODELS[key] = AcousticModel(name, precision=precision, threads=threads)
    return _MODELS[key]


def post_path(dataset: str, utt_id: str, name: str = "zipa-small-crctc-500k", precision: str = "fp32") -> Path:
    safe = utt_id.replace("/", "_")
    return Path(cache_root()) / "post" / f"{name}-{precision}" / dataset / f"{safe}.npy"


def get_posteriors(dataset: str, utt_id: str, audio, name: str = "zipa-small-crctc-500k",
                   precision: str = "fp32") -> np.ndarray:
    """Load cached log-posteriors as float32 [T,127], computing and saving them if missing.
    `audio` is a path or a float32 16 kHz waveform."""
    p = post_path(dataset, utt_id, name, precision)
    if p.exists():
        return np.load(p).astype(np.float32)
    wav = load_audio(audio) if isinstance(audio, (str, os.PathLike)) else audio
    lp = model(name, precision).posteriors(wav)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, np.maximum(lp, -60.0).astype(np.float16))
    os.replace(tmp, p)
    return lp


def warm(dataset: str, records: list[dict], **kw) -> int:
    """Make sure every manifest record of `dataset` is cached. Returns how many were computed."""
    n = 0
    for r in records:
        if not post_path(dataset, r["id"], **kw).exists():
            get_posteriors(dataset, r["id"], r["audio"], **kw)
            n += 1
    return n
