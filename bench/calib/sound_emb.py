"""Round 12, candidate d: per-sound pooled ZIPA activations (enc1 / enc3 / enc5).

    python -m bench.calib.sound_emb protos --lang all [--n 150]       # native phone prototypes
    python -m bench.calib.sound_emb so762                            # speechocean762 TRAIN vectors

Per expected sound: the mean of each layer's activations over the sound's aligned frames widened
by one frame on each side (empty span: the 2 frames around its position), concatenated (1024-d).
Prototypes: per language, z-scored vectors (language mean / sd over native sounds) averaged per
expected phone over FLEURS TRAIN native clips (n per language). Reference-aware features of a
slot: cos to the prototype of the expected sound (max over its accepted alternatives), best cos
to any other phone of the language, and their margin -- a nearest-centroid second opinion that
knows what should have been said. Bench-time memory: streamed, 2 workers.
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

LAYERS = ("enc1", "enc3", "enc5")


def phone_vectors(emb: dict, ev, n_phones: int) -> np.ndarray:
    T = len(emb[LAYERS[0]])
    X = np.concatenate([emb[k] for k in LAYERS], axis=1)
    out = np.empty((n_phones, X.shape[1]), np.float32)
    for i, sl in enumerate(ev.slots[:n_phones]):
        a, b = int(sl.start_frame), int(sl.end_frame)
        if b <= a:
            a, b = a - 1, a + 1
        a, b = max(a - 1, 0), min(b + 1, T)
        if b <= a:
            a, b = max(min(a, T - 1), 0), max(min(a, T - 1), 0) + 1
        out[i] = X[a:b].mean(0)
    return out


def last_emb(sc):
    return next(reversed(sc._lp_cache.values()))[2]  # noqa: SLF001


def proto_path(lang):
    from bench.calib.common import calib_cache

    d = calib_cache() / "sound2"
    d.mkdir(exist_ok=True)
    return d / f"protos_{lang}.npz"


_SC = None


def _init(lang):
    import os

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "2"
    global _SC
    from pronunciada.scorer import Scorer

    _SC = Scorer(lang, threads=2, explain_away=False, prosody=False)
    assert _SC.am.embeddings, "embedding model missing"


def _proto_clip(rec):
    try:
        res = _SC.score(rec["audio"], rec["text"])
    except ValueError:
        return None
    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    V = phone_vectors(last_emb(_SC), c["ev"], len(ref.phones))
    return [p.ipa for p in ref.phones], V


def protos(lang, n, workers=2):
    from concurrent.futures import ProcessPoolExecutor

    from bench.pipeline.common import records

    recs = records("fleurs", split="train", lang=lang)
    rng = np.random.default_rng(12)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]
    t0 = time.time()
    S, S2, N = None, None, 0
    sums, cnt = {}, {}
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(lang,)) as ex:
        for r in ex.map(_proto_clip, recs, chunksize=2):
            if r is None:
                continue
            ipa, V = r
            rows.append((ipa, V))
            S = V.sum(0) if S is None else S + V.sum(0)
            S2 = (V.astype(np.float64) ** 2).sum(0) if S2 is None else S2 + (V.astype(np.float64) ** 2).sum(0)
            N += len(V)
    mu = S / N
    sd = np.sqrt(np.maximum(S2 / N - mu ** 2, 1e-8))
    for ipa, V in rows:
        Z = (V - mu) / sd
        for p, z in zip(ipa, Z):
            sums[p] = sums.get(p, 0) + z
            cnt[p] = cnt.get(p, 0) + 1
    phones = sorted(p for p in sums if cnt[p] >= 5)
    P = np.stack([sums[p] / cnt[p] for p in phones])
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    np.savez(proto_path(lang), mu=mu.astype(np.float32), sd=sd.astype(np.float32), P=P.astype(np.float32),
             phones=np.array(phones), counts=np.array([cnt[p] for p in phones]))
    print(f"{lang}: {len(rows)} clips, {N} sounds, {len(phones)} prototypes ({time.time() - t0:.0f}s)", flush=True)


class Protos:
    def __init__(self, lang):
        z = np.load(proto_path(lang))
        self.mu, self.sd, self.P = z["mu"], z["sd"], z["P"]
        self.idx = {p: i for i, p in enumerate(z["phones"].tolist())}

    def features(self, V, ref) -> np.ndarray:
        """[n, 3]: cos to the expected sound (best over accepted alternatives), best cos to any
        other phone, margin. NaN when the expected sound has no prototype."""
        Z = (V - self.mu) / self.sd
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        C = Z @ self.P.T                                      # [n, n_protos]
        out = np.full((len(V), 3), np.nan, np.float32)
        for i, p in enumerate(ref.phones[:len(V)]):
            acc = [self.idx[x] for x in [p.ipa] + [a for a, _ in p.alts] if x in self.idx]
            if not acc:
                continue
            ce = C[i, acc].max()
            m = np.ones(C.shape[1], bool)
            m[acc] = False
            bo = C[i, m].max() if m.any() else -1.0
            out[i] = (ce, bo, ce - bo)
        return out


# ----------------------------------------------------------------------------- speechocean762

def _so_init():
    from bench.calib import sound2

    sound2._init("en")                 # shipped English scorer + prototypes (sound2._item adds them)
    global _SC
    _SC = sound2._SC


def _so_clip(rec):
    from bench.calib import sound2

    try:
        res = _SC.score(rec["audio"], rec["text"])
    except ValueError:
        return rec["id"], None
    c = res._pronunciada  # noqa: SLF001
    it = sound2._item(res)
    it["V"] = phone_vectors(last_emb(_SC), c["ev"], len(c["ref"].phones)).astype(np.float16)
    return rec["id"], it


def so762(workers=2):
    from concurrent.futures import ProcessPoolExecutor

    from bench.pipeline.common import records

    recs = [r for r in records("speechocean762") if r["split"] == "train"]
    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_so_init) as ex:
        for k, (uid, v) in enumerate(ex.map(_so_clip, recs, chunksize=4)):
            if v is not None:
                out[uid] = v
            if (k + 1) % 500 == 0:
                print(f"so762 {k + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    (proto_path("en").parent / "so762_train_vectors.pkl").write_bytes(pickle.dumps(out))
    print(f"so762: {len(out)} utts ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["protos", "so762"])
    ap.add_argument("--lang", default="all")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.step == "protos":
        from bench.calib.sound2 import LANGS

        for lang in (LANGS if a.lang == "all" else a.lang.split(",")):
            protos(lang, a.n, a.workers)
    else:
        so762(a.workers)


if __name__ == "__main__":
    main()
