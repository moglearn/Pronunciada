"""Per-sound coaching inputs for whole datasets through the public Scorer (round 2).

    python -m bench.coach.dump --sets so762-train,ompal-dev,fleurs-val [--workers 2]

One pickle per set under $PRONUNCIADA_CACHE/bench/coach/dump-<set>.pkl: per utterance the
coaching inputs of EVERY reference sound (pronunciada.coach.sound_inputs: calibrated logits,
raw evidence, the heard candidate), per syllable the tone / stress outputs, and per word the
probabilities. Streaming: workers return one utterance at a time; nothing is fitted here.
Sets: so762-train, so762-test, ompal-dev, ompal-test, l2arctic-test, epadb-test,
fleurs-val (250 clean clips per language, all ten languages).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

from bench.coach.gather import out_dir

SETS = {"so762-train": ("speechocean762", "train", "en"), "so762-test": ("speechocean762", "test", "en"),
        "ompal-dev": ("ompal", "dev", "zh"), "ompal-test": ("ompal", "test", "zh"),
        "l2arctic-test": ("l2arctic", "test", "en"), "epadb-test": ("epadb", "test", "en")}
_SC = {}


def _init():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "2"


def _scorer(lang):
    from pronunciada.scorer import Scorer

    if lang not in _SC:
        _SC.clear()                                  # one language at a time per worker
        _SC[lang] = Scorer(lang, threads=2)
    return _SC[lang]


def utt_record(res) -> dict:
    from pronunciada.coach import sound_inputs

    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    pr = c.get("prosody") or []
    X = sound_inputs(res, heard=True)
    X = {k: (v.astype("float32") if hasattr(v, "dtype") and v.dtype.kind == "f" else v) for k, v in X.items()}
    syl = []
    for wi, w in enumerate(res.words):
        s0 = ref.words[wi].syllables[0]
        for k, sy in enumerate(w.syllables):
            d = pr[s0 + k] if s0 + k < len(pr) else {}
            syl.append({"word": wi, "k": k, "tone_expected": sy.tone_expected, "tone_prob": sy.tone_prob,
                        "tone_probs": d.get("tone_probs"), "tone_alts": list(ref.syllables[s0 + k].tone_alts),
                        "stress_prob": sy.stress_prob, "stress_dist": d.get("stress_dist")})
    return {"sounds": X, "syllables": syl,
            "words": [{"text": w.text, "prob": w.prob, "sounds_prob": w.sounds_prob, "tone_prob": w.tone_prob,
                       "stress_prob": w.stress_prob, "present_prob": w.present_prob,
                       "phones": [p.ipa for p in w.phones], "n_syl": len(w.syllables)} for w in res.words]}


def _one(a):
    lang, rid, audio, text = a
    try:
        res = _scorer(lang).score(audio, text)
    except ValueError as e:
        return rid, {"error": str(e)}
    return rid, utt_record(res)


def _fleurs_jobs(per_lang=250):
    from bench.pipeline.common import LANGS, fleurs_validation_records, iter_parquet_audio

    for lang in LANGS:
        recs = fleurs_validation_records(lang)[:per_lang]
        by_row = {int(r["audio"].rsplit(":", 1)[1]): r for r in recs}
        path = recs[0]["audio"].split(":", 1)[1].rsplit(":", 1)[0]
        for row, wav in iter_parquet_audio(path, set(by_row)):
            r = by_row[row]
            yield (lang, r["id"], (wav, 16000), r["text"])


def run(name, workers):
    from concurrent.futures import ProcessPoolExecutor

    from bench.data import load

    if name == "fleurs-val":
        jobs = _fleurs_jobs()
    else:
        ds, split, lang = SETS[name]
        jobs = ((lang, r["id"], r["audio"], r["text"]) for r in load(ds, split=split))
    out = {}
    t0 = time.time()
    from collections import deque

    with ProcessPoolExecutor(min(workers, 2), initializer=_init) as ex:
        q: deque = deque()
        k = 0
        for job in jobs:                       # bounded window: never more than 16 clips queued
            q.append(ex.submit(_one, job))
            while len(q) >= 16:
                rid, o = q.popleft().result()
                out[rid] = o
                k += 1
                if k % 500 == 0:
                    print(name, k, f"{time.time() - t0:.0f}s", flush=True)
        while q:
            rid, o = q.popleft().result()
            out[rid] = o
    (out_dir() / f"dump-{name}.pkl").write_bytes(pickle.dumps(out))
    print(name, len(out), f"{time.time() - t0:.0f}s", flush=True)


def load_dump(name):
    return pickle.loads((out_dir() / f"dump-{name}.pkl").read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="so762-train,ompal-dev,fleurs-val")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for s in a.sets.split(","):
        run(s, a.workers)


if __name__ == "__main__":
    main()
