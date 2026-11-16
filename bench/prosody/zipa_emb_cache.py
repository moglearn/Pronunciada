"""Per-syllable pooled ZIPA encoder activations (enc1 256-d + enc3 512-d, pronunciada.acoustic
EMBED_OUTPUTS) for the tone-v3 "B" experiment: mean over the frames of the syllable's aligned
phones (prep spans, ZIPA 20 ms frames); syllables without aligned frames get the nearest frame.

    .venv/bin/python -m bench.prosody.zipa_emb_cache aishell3:train fleurs:train ... [--workers=2]

Output: ~/.cache/pronunciada/prosody/zipa_emb/<dataset>-zh-<split>.npz {id: float16 [S, 768]}.
AISHELL-1 is not cached (time budget): those training utterances get a zero vector + flag 0.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

from bench.prosody.common import OUT, load_prepped  # noqa: E402

EMB = OUT / "zipa_emb"
LAYERS = ("enc1", "enc3")
DIM = 768


def pool(ref, spans, emb):
    T = len(emb)
    out = np.zeros((len(ref.syllables), emb.shape[1]), np.float32)
    for si, s in enumerate(ref.syllables):
        fr = [j for pi in range(*s.phones) for j in range(*spans[pi])]
        fr = [j for j in fr if 0 <= j < T]
        if not fr:
            a = [spans[pi][0] for pi in range(*s.phones)]
            fr = [int(np.clip(a[0] if a else 0, 0, T - 1))]
        out[si] = emb[fr].mean(0)
    return out


def _one(key):
    from bench.prosody.lowmel_cache import audio_index
    from pronunciada.acoustic import AcousticModel
    from pronunciada.audio import load_audio

    dataset, split, *rest = key.split(":")       # dataset:split[:lang] (default zh)
    lang = rest[0] if rest else "zh"
    m = AcousticModel(threads=2, embeddings=True)
    assert m.embeddings
    utts = load_prepped(dataset, lang, split)
    src = audio_index(dataset, split, lang)
    res = {}
    for u in utts:
        a = src(u["id"])
        wav = a if isinstance(a, np.ndarray) else load_audio(a)
        _, e = m.posteriors_from_features(m._features(wav), True)
        emb = np.concatenate([np.asarray(e[k]).reshape(-1, np.asarray(e[k]).shape[-1]) for k in LAYERS], 1)
        res[u["id"]] = pool(u["ref"], u["spans"], emb).astype(np.float16)
    EMB.mkdir(parents=True, exist_ok=True)
    name = f"{dataset}-{lang}-{split}"
    np.savez(EMB / f"{name}.tmp.npz", **res)
    os.replace(EMB / f"{name}.tmp.npz", EMB / f"{name}.npz")
    return name, len(res)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    w = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--workers=")), 2))
    todo = [k for k in args if not (EMB / f"{k.split(':')[0]}-{(k.split(':') + ['zh'])[2]}-{k.split(':')[1]}.npz").exists()]
    with ProcessPoolExecutor(w) as ex:
        for name, n in ex.map(_one, todo):
            print("wrote", name, n, flush=True)


class ZipaEmb:
    """ZipaEmb()(u) -> (float32 [S, DIM], available flag)."""

    def __init__(self):
        self.sets = {}

    def __call__(self, u):
        ds, lang, split = u["_src"]
        key = f"{ds}-{lang}-{split}"
        if key not in self.sets:
            p = EMB / f"{key}.npz"
            self.sets[key] = dict(np.load(p)) if p.exists() else {}
        e = self.sets[key].get(u["id"])
        if e is None:
            return np.zeros((len(u["ref"].syllables), DIM), np.float32), 0.0
        return e.astype(np.float32), 1.0


if __name__ == "__main__":
    main()
