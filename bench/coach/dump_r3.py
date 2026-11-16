"""Round 3 inputs: reading miscues and delivery on FLEURS validation natives (+ a word removed
from the prompt, so the audio holds one "extra" word) and speechocean762 train (expert fluency /
prosody scores). Public Scorer only; nothing fitted here.

    python -m bench.coach.dump_r3 [--workers 2] [--per-lang 250]

Writes $PRONUNCIADA_CACHE/bench/coach/r3-fleurs.pkl and r3-so762.pkl.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import deque

from bench.coach.gather import out_dir
from bench.pipeline.common import stable_seed

_SC = {}


def _init():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "2"


def _scorer(lang):
    from pronunciada.scorer import Scorer

    if lang not in _SC:
        _SC.clear()
        _SC[lang] = Scorer(lang, threads=2)
    return _SC[lang]


def remove_word(ref, k):
    a, b = ref.words[k].char_span
    t = ref.text
    if ref.lang in ("zh", "ja"):
        return t[:a] + t[b:]
    return (t[:a].rstrip() + " " + t[b:].lstrip()).strip()


def _one(job):
    from pronunciada.delivery_coach import delivery_inputs, extra_segments

    kind, lang, rid, audio, text, labels = job
    sc = _scorer(lang)
    try:
        res = sc.score(audio, text)
    except ValueError as e:
        return rid, {"error": str(e)}
    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    out = {"lang": lang, "n_words": len(ref.words), "delivery": delivery_inputs(c),
           "extra": extra_segments(c), "dur": c["dur"], "labels": labels}
    if kind == "fleurs" and len(ref.words) >= 4:
        import numpy as np

        rng = np.random.default_rng(stable_seed(rid, "remove-word"))
        cand = [k for k in range(1, len(ref.words) - 1) if ref.words[k].phones[1] - ref.words[k].phones[0] >= 2]
        if cand:
            k = int(cand[int(rng.integers(len(cand)))])
            w = res.words[k]
            t2 = remove_word(ref, k)
            try:
                r2 = sc.score(audio, t2)
                c2 = r2._pronunciada  # noqa: SLF001
                out["removed"] = {"k": k, "word": w.text, "start": w.start, "end": w.end,
                                  "n_words": len(c2["ref"].words), "extra": extra_segments(c2)}
            except ValueError:
                pass
    return rid, out


def jobs(per_lang):
    from bench.data import load
    from bench.pipeline.common import LANGS, fleurs_validation_records, iter_parquet_audio

    for r in load("speechocean762", split="train"):
        yield ("so762", "en", r["id"], r["audio"], r["text"], r["labels"]["sentence"])
    for lang in LANGS:
        recs = fleurs_validation_records(lang)[:per_lang]
        by_row = {int(r["audio"].rsplit(":", 1)[1]): r for r in recs}
        path = recs[0]["audio"].split(":", 1)[1].rsplit(":", 1)[0]
        for row, wav in iter_parquet_audio(path, set(by_row)):
            r = by_row[row]
            yield ("fleurs", lang, r["id"], (wav, 16000), r["text"], None)


def main():
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--per-lang", type=int, default=250)
    a = ap.parse_args()
    out = {"so762": {}, "fleurs": {}}
    t0 = time.time()
    with ProcessPoolExecutor(min(a.workers, 2), initializer=_init) as ex:
        q: deque = deque()
        n = 0
        for j in jobs(a.per_lang):
            q.append((j[0], ex.submit(_one, j)))
            while len(q) >= 16:
                kind, f = q.popleft()
                rid, o = f.result()
                out[kind][rid] = o
                n += 1
                if n % 500 == 0:
                    print(n, f"{time.time() - t0:.0f}s", flush=True)
        while q:
            kind, f = q.popleft()
            rid, o = f.result()
            out[kind][rid] = o
    (out_dir() / "r3.pkl").write_bytes(pickle.dumps(out))
    print("done", {k: len(v) for k, v in out.items()}, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
