"""Round 18 A: word probabilities and per-word contrast log-odds through the public Scorer, with
the word-level contrast factor OFF (PRONUNCIADA_WORD_CONTRAST=0). The factor is a deterministic
function of (word probability, the word's lowest contrast log-odds, strictness), so "after" for
any candidate setting is computed offline from these rows, paired with "before" exactly.

    python -m bench.round18.collect --sets fleurs-val --lang zh,en,ru,ja,de,ko [--workers 3]

Sets (rows per scored reference: per word P at strictness 0 / 0.5 / 1 and min contrast z):
  fleurs-val    FLEURS VALIDATION: the contrast audit's clips (contrast swaps of the language's
                capped / head contrasts) and the dev swap sources (bench/calib/bias_grid
                dev_swap: orig / near / far texts); on every clip the controlled-error kinds
                (bench.eval_languages KINDS, 2 errors, seed "features-v1").
  fleurs-test   FLEURS TEST: the round-17 test clip choice and the audit plan's contrast swaps.
  lang10        FLEURS TEST: bench.eval_languages._utt rows (the ten-language benchmark) + z.
  swapbench     bench.eval_swap fleurs-swap test records of the language + z.
  so762-train, so762-test, ompal-dev, ompal-test   learner sets (en / zh).
Pickles: $PRONUNCIADA_CACHE/bench/round18/<set>-<lang>.pkl.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np

CONTRASTS = {"zh": ("aspiration",), "en": ("th_s", "dh_z"), "ru": ("hard_soft", "final_devoicing", "voicing"),
             "ja": ("voicing",), "de": ("voicing", "final_devoicing"), "ko": ("aspiration",)}
S_LIST = (0.0, 0.5, 1.0)


def root():
    base = os.environ.get("PRONUNCIADA_CACHE")
    r = Path(base).expanduser() if base else Path.home() / ".cache" / "pronunciada"
    d = r / "bench" / "round18"
    d.mkdir(parents=True, exist_ok=True)
    return d


_SC = None


def _init(lang):
    os.environ["PRONUNCIADA_WORD_CONTRAST"] = "0"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    from bench.calib import api_run

    api_run._init(lang, "fp32", 1)
    global _SC
    _SC = api_run._SC


def word_rows(res) -> dict:
    """{wp: {s: [P per word]}, z: [min contrast z per word, NaN if none], text, span}."""
    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    cz = c.get("contrast_z")
    z = []
    for w in ref.words:
        v = np.asarray(cz[w.phones[0]:w.phones[1]], np.float64) if cz is not None else np.array([np.nan])
        v = v[np.isfinite(v)]
        z.append(float(v.min()) if len(v) else np.nan)
    wp = {0.5: [float(w.prob) for w in res.words]}
    for s in (0.0, 1.0):
        wp[s] = [float(w.prob) for w in _SC.rescore(res, s).words]
    return {"wp": wp, "z": z, "text": [w.text for w in ref.words], "span": [tuple(w.char_span) for w in ref.words]}


def _fleurs_clip(args):
    rec, audio, plan, swaps = args
    from bench.audit.contrasts import Site, apply
    from bench.corrupt import NotApplicable, corrupt
    from bench.eval_languages import KINDS
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory

    sc = _SC
    try:
        ref = sc.g2p(rec["text"])
        res = sc.score(audio, ref)
    except Exception:  # noqa: BLE001
        return rec["id"], None
    out = {"clean": word_rows(res), "edits": [], "cor": {}, "swap": {}}
    for sd in plan:
        if sd["contrast"] not in CONTRASTS.get(sc.lang, ()):
            continue
        site = Site(**sd)
        try:
            e = apply(ref, site, np.random.default_rng(zlib.crc32(f"{rec['id']}/{sd}".encode())))
            r2 = sc.score(audio, e.ref)
        except Exception:  # noqa: BLE001
            continue
        out["edits"].append({"contrast": site.contrast, "direction": site.direction, "word": e.word, **word_rows(r2)})
    if swaps is not None:
        inv = inventory(sc.lang)
        for k in KINDS:
            try:
                cor = corrupt(ref, [k], inv, n_errors=2, seed=stable_seed(rec["id"], k, "features-v1"), strict=False)
            except NotApplicable:
                continue
            if not cor.errors:
                continue
            r2 = sc.score(audio, cor.ref)
            out["cor"][k] = {**word_rows(r2), "word_wrong": list(cor.word_wrong)}
        for s in swaps:
            try:
                r3 = sc.score(audio, s["text"])
            except ValueError:
                continue
            out["swap"][s["id"]] = {**word_rows(r3), "labels": s["labels"]}
    if sc.lang == "ko":                              # round 18 B: tense flags for offline tense swaps
        out["tense"] = [getattr(p, "tense", None) for p in ref.phones]
        out["word_of"] = [p.word for p in ref.phones]
        c = res._pronunciada  # noqa: SLF001
        out["contrast_z"] = list(np.asarray(c.get("contrast_z")))
        ev = c.get("contrast_ev") or {}
        out["tense_z"] = list(np.asarray(ev["tense"])) if "tense" in ev else None
    return rec["id"], out


def _lang10(rec):
    """bench.eval_languages._utt plus the per-word z of every reference."""
    from bench.corrupt import NotApplicable, corrupt
    from bench.eval_languages import KINDS, PROS_KIND, _pack
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory

    sc = _SC
    try:
        res = sc.score(rec["audio"], rec["text"])
    except ValueError as e:
        return rec["id"], {"error": str(e)}
    out = {"clean": _pack(sc, res), "dur": rec["dur"], "warnings": res.warnings, "cor": {},
           "latency_ms": float(res.timing_ms.get("total", 0.0)), "timing": res.timing_ms}
    out["clean"]["z"] = word_rows(res)["z"]
    ref = res._pronunciada["ref"]  # noqa: SLF001
    inv = inventory(sc.lang)
    kinds = KINDS + ((PROS_KIND[sc.lang],) if sc.lang in PROS_KIND else ())
    for kind in kinds:
        try:
            cor = corrupt(ref, [kind], inv, n_errors=2, seed=stable_seed(rec["id"], kind, "features-v1"), strict=False)
        except NotApplicable:
            continue
        if not cor.errors:
            continue
        r2 = sc.score(rec["audio"], cor.ref)
        d = _pack(sc, r2)
        d["phone_wrong"] = np.asarray(cor.phone_wrong, bool)
        d["word_wrong"] = np.asarray(cor.word_wrong, bool)
        d["z"] = word_rows(r2)["z"]
        out["cor"]["prosody" if kind == PROS_KIND.get(sc.lang) else kind] = d
    return rec["id"], out


def _swapbench(rec):
    from bench.eval_languages import _swap_one

    rid, o = _swap_one(rec)
    try:
        res = _SC.score(rec["audio"], rec["text"])
        o["z"] = word_rows(res)["z"]
    except ValueError:
        o["z"] = []
    return rid, o


def _learner(rec):
    try:
        res = _SC.score(rec["audio"], rec["text"])
    except ValueError as e:
        return rec["id"], {"error": str(e)}
    o = word_rows(res)
    o["omitted"] = [w.omitted for w in res.words]
    return rec["id"], o


def jobs(name, lang):
    from bench.data import load
    from pronunciada.text import get_g2p

    if name in ("fleurs-val", "fleurs-test"):
        import json

        from bench.audit.contrast_audit import cache_dir
        from bench.pipeline.common import iter_parquet_audio
        from bench.round17.collect import clips, plan_for

        split = "validation" if name == "fleurs-val" else "test"
        recs = clips(lang, split)
        plan = plan_for(lang, split, recs, get_g2p(lang)) if split == "test" else \
            json.loads((cache_dir(lang) / "plan_250.json").read_text())["plan"]
        swaps = {}
        if split == "validation":
            from bench.calib.bias_grid import dev_swap
            from bench.pipeline.common import records

            for s in dev_swap(lang):
                swaps.setdefault(s["src"], []).append(s)
            ids = {r["id"] for r in recs}
            recs = recs + [r for r in records("fleurs-validation", lang=lang) if r["id"] in swaps and r["id"] not in ids]
        if str(recs[0]["audio"]).startswith("parquet:"):
            by = {int(r["audio"].split(":")[2]): r for r in recs}
            path = recs[0]["audio"].split(":")[1]
            for i, wav in iter_parquet_audio(path, set(by)):
                r = by[i]
                yield _fleurs_clip, (r, (wav, 16000), plan.get(r["id"], []), swaps.get(r["id"], []) if split == "validation" else None)
        else:
            for r in recs:
                yield _fleurs_clip, (r, r["audio"], plan.get(r["id"], []), None)
        return
    if name == "lang10":
        for r in load("fleurs", split="test", lang=lang):
            yield _lang10, r
        return
    if name == "swapbench":
        for r in load("fleurs-swap", split="test"):
            if r["lang"] == lang:
                yield _swapbench, r
        return
    ds, split = {"so762-train": ("speechocean762", "train"), "so762-test": ("speechocean762", "test"),
                 "ompal-dev": ("ompal", "dev"), "ompal-test": ("ompal", "test")}[name]
    for r in load(ds, split=split):
        yield _learner, r


def _call(fa):
    fn, a = fa
    return fn(a)


def run(name, lang, workers):
    from concurrent.futures import ProcessPoolExecutor

    p = root() / f"{name}-{lang}.pkl"
    if p.exists():
        print("cached", p, flush=True)
        return
    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(min(workers, 3), initializer=_init, initargs=(lang,)) as ex:
        q: deque = deque()
        for j in jobs(name, lang):
            q.append(ex.submit(_call, j))
            while len(q) >= 12:
                rid, o = q.popleft().result()
                out[rid] = o
        while q:
            rid, o = q.popleft().result()
            out[rid] = o
    p.write_bytes(pickle.dumps(out))
    print(name, lang, len(out), f"{time.time() - t0:.0f}s", flush=True)


def load_set(name, lang):
    return pickle.loads((root() / f"{name}-{lang}.pkl").read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="fleurs-val")
    ap.add_argument("--lang", default="zh,en,ru,ja,de,ko")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    for name in a.sets.split(","):
        langs = a.lang.split(",")
        if name.startswith("so762"):
            langs = ["en"]
        if name.startswith("ompal"):
            langs = ["zh"]
        for lang in langs:
            run(name, lang, a.workers)


if __name__ == "__main__":
    main()
