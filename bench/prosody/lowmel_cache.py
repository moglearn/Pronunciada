"""Low-frequency log-mel frames for the tone v3 spectral variant (same function as the runtime:
pronunciada.prosody.features.lowmel_features on the 80-bin Kaldi fbank of pronunciada.audio).

    .venv/bin/python -m bench.prosody.lowmel_cache aishell3:train fleurs:train ... [--workers 2]

Output per prepared set: ~/.cache/pronunciada/prosody/lowmel/<dataset>-<lang>-<split>.npy
(float16 [sum T, N_LOWMEL], memory-mappable, shared by training workers) + .json {id: [offset, T]}
with T = the utterance's pitch-track length.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from bench.prosody.common import OUT, load_prepped  # noqa: E402

LOWMEL = OUT / "lowmel"


def audio_index(dataset: str, split: str, lang: str = "zh"):
    """id -> audio path or waveform source for every prepared set."""
    from bench.data import load
    from bench.prosody.leak_test import MONO

    if dataset == "aishell3mono":
        return lambda uid: str(MONO / f"{uid}.flac")
    if dataset == "aishell3orig":
        dataset, split = "aishell3", "test"
    if dataset == "fleurs_val":
        from bench.prosody.prep import fleurs_validation

        wavs = {r["id"]: r["wav"] for r in fleurs_validation(lang)}
        return lambda uid: wavs[uid]
    recs = {r["id"]: r["audio"] for r in load(dataset, split=split, lang=None if dataset != "fleurs" else lang)}
    return lambda uid: recs[uid]


def _one(key):
    from pronunciada.audio import load_audio
    from pronunciada.prosody.features import lowmel_features

    dataset, split, *rest = key.split(":")       # dataset:split[:lang] (default zh)
    lang = rest[0] if rest else "zh"
    utts = load_prepped(dataset, lang, split)
    src = audio_index(dataset, split, lang)
    index, chunks, off = {}, [], 0
    for u in utts:
        a = src(u["id"])
        wav = a if isinstance(a, np.ndarray) else load_audio(a)
        T = len(u["track"]["f0"])
        x = lowmel_features(wav, T).astype(np.float16)
        index[u["id"]] = [off, T]
        chunks.append(x)
        off += T
    LOWMEL.mkdir(parents=True, exist_ok=True)
    name = f"{dataset}-{lang}-{split}"
    tmp = LOWMEL / f"{name}.tmp.npy"
    np.save(tmp, np.concatenate(chunks))
    os.replace(tmp, LOWMEL / f"{name}.npy")
    (LOWMEL / f"{name}.json").write_text(json.dumps(index))
    return name, len(utts), off


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    w = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--workers=")), 2))
    todo = [k for k in args if not (LOWMEL / f"{k.split(':')[0]}-{(k.split(':') + ['zh'])[2]}-{k.split(':')[1]}.npy").exists()]
    with ProcessPoolExecutor(w) as ex:
        for name, n, T in ex.map(_one, todo):
            print("wrote", name, n, "utts", T, "frames", flush=True)


class LowMel:
    """Lazy memory-mapped access: LowMel()(u) -> float32 [T, N_LOWMEL] for a prepared utterance."""

    def __init__(self):
        self.sets = {}

    def __call__(self, u):
        ds, lang, split = u["_src"]
        key = f"{ds}-{lang}-{split}"
        if key not in self.sets:
            self.sets[key] = (np.load(LOWMEL / f"{key}.npy", mmap_mode="r"),
                              json.loads((LOWMEL / f"{key}.json").read_text()))
        arr, idx = self.sets[key]
        o, T = idx[u["id"]]
        return np.asarray(arr[o:o + T], dtype=np.float32)


if __name__ == "__main__":
    _ = Path
    main()
