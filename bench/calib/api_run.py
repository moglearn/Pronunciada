"""Run the PUBLIC API (pronunciada.Scorer) over a dataset split and keep compact per-utterance outputs.

Used by bench/eval_english.py and bench/eval_languages.py. Every utterance goes through
Scorer.score(audio_path, text) exactly like a user call (audio decoded from the file, fbank,
ZIPA ONNX, adapter, G2P, alignment, features, calibrators). Worker processes each build one
Scorer (1 ONNX thread each). Outputs are cached as pickles under $PRONUNCIADA_CACHE/calib/api/.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

_SC = None
_EXTRA = None


def _init(lang, precision, threads):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    global _SC, _EXTRA
    from bench.calib.common import calib_cache
    from pronunciada.calibrate import LevelModel
    from pronunciada.scorer import Scorer

    _SC = Scorer(lang, precision=precision, threads=threads)
    _EXTRA = {}
    for name in ("sound_A", "sound_C", "sound_W", "sound_R"):
        p = calib_cache() / f"{name}.npz"
        if p.exists():
            _EXTRA[name[-1]] = LevelModel(p)


def summarize(sc, res, extra: dict) -> dict:
    """Compact, picklable view of a SentenceResult + the side-by-side sound models."""
    from pronunciada.calibrate import logit

    d = sc.details(res)
    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    out = {"text": res.text, "prob": res.prob, "fluency": res.fluency, "completeness": res.completeness,
           "warnings": res.warnings, "timing_ms": res.timing_ms, "sentence": d["sentence"],
           "sentence_z": (d["sentence_z_len"], d["sentence_z_str"]),
           "phones": [p.ipa for p in ref.phones], "phone_word": [p.word for p in ref.phones],
           "words": [w.text for w in ref.words],
           "word_char_span": [tuple(w.char_span) for w in ref.words],
           "word_z": (np.asarray(d["word_z_len"]), np.asarray(d["word_z_str"])),
           "word_reg": np.asarray(d["word_accuracy"]), "omitted": np.asarray(d["omitted"]),
           "word_prob": np.array([w.prob for w in res.words]),
           "sound": {sc.sound_name[-1]: {"reg": np.asarray(d["phone_score"]),
                                         "z_len": np.asarray(d["phone_z_len"]),
                                         "z_str": np.asarray(d["phone_z_str"])}},
           "prosody_active": sc.prosody is not None}
    if sc.sound_name != sc.universal_name:
        o = sc.sound_outputs(res, sc.universal)
        out["sound"]["U"] = {"reg": o["reg"], "z_len": logit(o["len"]) + sc.universal_bias,
                             "z_str": np.minimum(logit(o["str"]), logit(o["len"])) + sc.universal_bias}
    for k, m in extra.items():
        o = sc.sound_outputs(res, m)
        out["sound"][k] = {"reg": o["reg"], "z_len": logit(o["len"]),
                           "z_str": np.minimum(logit(o["str"]), logit(o["len"]))}
    out["explained"] = list(c.get("explained", []))
    # the same outputs WITHOUT the explain-away pass (first pass), for before / after tables
    f = c.get("first_pass")
    if f is None:
        out["first"] = {k: out[k] for k in ("word_z", "word_reg", "sound")}
    else:
        sound = {}
        for name, m in [(sc.sound_name[-1], sc.sound), ("U", sc.universal)] + list(extra.items()):
            o = m.predict(f["tab"])
            sound[name] = {"reg": o["reg"], "z_len": logit(o["len"]) + (sc.universal_bias if m is sc.universal else 0.0),
                           "z_str": np.minimum(logit(o["str"]), logit(o["len"])) + (sc.universal_bias if m is sc.universal else 0.0)}
        out["first"] = {"word_z": (np.asarray(f["w_zl"]), np.asarray(f["w_zs"])), "word_reg": np.asarray(f["w_reg"]),
                        "sound": sound}
    return out


def _one(rec):
    try:
        res = _SC.score(rec["audio"], rec["text"])
    except ValueError as e:
        return rec["id"], {"error": str(e)}
    return rec["id"], summarize(_SC, res, _EXTRA)


def run(recs, lang, precision="fp32", workers=5, threads=1, tag=None, force=False):
    """{utt id: summary} for every record (cached)."""
    from concurrent.futures import ProcessPoolExecutor

    from bench.calib.common import calib_cache

    from pathlib import Path

    # PRONUNCIADA_API_DIR: keep a run's outputs apart from the shared cache (paired before / after
    # runs while other work uses calib/api)
    d = Path(os.environ["PRONUNCIADA_API_DIR"]) if os.environ.get("PRONUNCIADA_API_DIR") else calib_cache() / "api"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{tag}.pkl" if tag else None
    if path is not None and path.exists() and not force:
        out = pickle.loads(path.read_bytes())
        if all(r["id"] in out for r in recs):
            return out
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(lang, precision, threads)) as ex:
        for k, (uid, s) in enumerate(ex.map(_one, recs, chunksize=4)):
            out[uid] = s
            if (k + 1) % 250 == 0:
                print(f"{tag}: {k + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    if path is not None:
        path.write_bytes(pickle.dumps(out))
    print(f"{tag}: {len(out)} utts in {time.time() - t0:.0f}s", flush=True)
    return out
