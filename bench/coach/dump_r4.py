"""Round 4 inputs: per-sound pooled ZIPA activations (enc1 | enc3 | enc5, the round-12b
pronunciada.sound_emb.phone_vectors pooling) through the public Scorer.

    python -m bench.coach.dump_r4 --sets fleurs-train,audit,so762-train,so762-test,ompal-dev,ompal-test,l2arctic-test,epadb-test [--workers 3]

fleurs-train  FLEURS TRAIN natives, 200 clips per language (head training); per sound: vector,
              reference symbol, language
audit         FLEURS VALIDATION, the contrast audit's clips and planted swaps
              (bench/audit/contrast_audit.make_plan, bench/audit/contrasts.apply): unedited
              natives (vector per sound) and, per swap, the vector of the edited sound under the
              EDITED reference's alignment, with the audio's true symbol and the reference's
other sets    utt_record of bench/coach/dump.py (coaching inputs) + one vector per sound
Vectors are float16. One pickle per set: $PRONUNCIADA_CACHE/bench/coach/dump-r4-<set>.pkl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
import zlib
from collections import deque

import numpy as np

from bench.coach.gather import out_dir

SETS = {"so762-train": ("speechocean762", "train", "en"), "so762-test": ("speechocean762", "test", "en"),
        "ompal-dev": ("ompal", "dev", "zh"), "ompal-test": ("ompal", "test", "zh"),
        "l2arctic-test": ("l2arctic", "test", "en"), "epadb-test": ("epadb", "test", "en")}
LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
_SC = {}


def _init():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


def _scorer(lang):
    from pronunciada.scorer import Scorer

    if lang not in _SC:
        _SC.clear()
        _SC[lang] = Scorer(lang, threads=1)
    return _SC[lang]


def vectors(sc, res, wav=None):
    from pronunciada.audio import load_audio
    from pronunciada.sound_emb import phone_vectors

    c = res._pronunciada  # noqa: SLF001
    emb = c.get("emb")
    if emb is None:                                   # older scorer: from the posterior cache
        w = load_audio(wav)
        emb = sc._lp_cache[hashlib.sha1(w.tobytes()).hexdigest()][2]  # noqa: SLF001
    return phone_vectors(emb, c["ev"], len(c["ref"].phones)).astype(np.float16)


def _one(job):
    kind, lang, rid, audio, text, extra = job
    sc = _scorer(lang)
    try:
        if kind == "audit":
            from pronunciada.text import get_g2p

            ref = get_g2p(lang)(text)
            res = sc.score(audio, ref)
        else:
            res = sc.score(audio, text)
    except Exception as e:  # noqa: BLE001
        return rid, {"error": str(e)[:200]}
    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    V = vectors(sc, res, audio)
    if kind == "fleurs-train":
        return rid, {"lang": lang, "ipa": [p.ipa for p in ref.phones], "vec": V,
                     "nucleus": [p.is_nucleus for p in ref.phones]}
    if kind == "audit":
        from bench.audit.contrasts import Site, apply

        out = {"lang": lang, "ipa": [p.ipa for p in ref.phones], "vec": V, "alts": [[a for a, _ in p.alts] for p in ref.phones],
               "prob": [p.prob for w in res.words for p in w.phones], "edits": []}
        for sd in extra:
            site = Site(**sd)
            if site.kind == "tone":
                continue
            rng = np.random.default_rng(zlib.crc32(f"{rid}/{sd}".encode()))
            try:
                e = apply(ref, site, rng)
                r2 = sc.score(audio, e.ref)
            except Exception:  # noqa: BLE001
                continue
            V2 = vectors(sc, r2, audio)
            probs = [p.prob for w in r2.words for p in w.phones]
            out["edits"].append({"contrast": site.contrast, "kind": site.kind, "direction": site.direction,
                                 "src": site.src, "dst": site.dst, "index": site.index, "flag": list(e.flag),
                                 "alt": bool(e.partner_in_alts),
                                 "ref_ipa": [e.ref.phones[i].ipa for i in e.flag],
                                 "vec": V2[e.flag], "prob": [probs[i] for i in e.flag]})
        return rid, out
    from bench.coach.dump import utt_record

    o = utt_record(res)
    o["vec"] = V
    return rid, o


def jobs(name, per_lang=200):
    from bench.pipeline.common import fleurs_validation_records, iter_parquet_audio, records

    if name == "fleurs-train":
        for lang in LANGS:
            from pathlib import Path

            recs = records("fleurs", split="train", lang=lang)
            rng = np.random.default_rng(zlib.crc32(f"r4/{lang}".encode()))
            pick = [recs[i] for i in sorted(rng.choice(len(recs), min(per_lang, len(recs)), replace=False))]
            for r in pick:
                a = r["audio"]
                if str(a).startswith("parquet:"):
                    continue
                if Path(a).exists():
                    yield ("fleurs-train", lang, r["id"], a, r["text"], None)
        return
    if name == "audit":
        from bench.audit.contrast_audit import cache_dir, choose

        for lang in LANGS:
            recs = choose(lang, 250)
            plan = json.loads((cache_dir(lang) / "plan_250.json").read_text())["plan"]
            by = {int(r["audio"].split(":")[2]): r for r in recs}
            paths = {r["audio"].split(":")[1] for r in recs}
            for path in sorted(paths):
                rows = {i for i, r in by.items() if r["audio"].split(":")[1] == path}
                for i, wav in iter_parquet_audio(path, rows):
                    r = by[i]
                    yield ("audit", lang, r["id"], (wav, 16000), r["text"], plan.get(r["id"], []))
        return
    if name == "fleurs-val":                      # natives, coaching inputs (wrong claims per clip)
        for lang in LANGS:
            recs = fleurs_validation_records(lang)[:250]
            by_row = {int(r["audio"].rsplit(":", 1)[1]): r for r in recs}
            path = recs[0]["audio"].split(":", 1)[1].rsplit(":", 1)[0]
            for row, wav in iter_parquet_audio(path, set(by_row)):
                r = by_row[row]
                yield ("fleurs-val", lang, r["id"], (wav, 16000), r["text"], None)
        return
    from bench.data import load

    ds, split, lang = SETS[name]
    for r in load(ds, split=split):
        yield (name, lang, r["id"], r["audio"], r["text"], None)


def run(name, workers):
    from concurrent.futures import ProcessPoolExecutor

    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(min(workers, 3), initializer=_init) as ex:
        q: deque = deque()
        k = 0
        for j in jobs(name):
            q.append(ex.submit(_one, j))
            while len(q) >= 12:
                rid, o = q.popleft().result()
                out[rid] = o
                k += 1
                if k % 500 == 0:
                    print(name, k, f"{time.time() - t0:.0f}s", flush=True)
        while q:
            rid, o = q.popleft().result()
            out[rid] = o
    (out_dir() / f"dump-r4-{name}.pkl").write_bytes(pickle.dumps(out))
    print(name, "done", len(out), f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="fleurs-train,audit")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    for s in a.sets.split(","):
        run(s, a.workers)


if __name__ == "__main__":
    main()
