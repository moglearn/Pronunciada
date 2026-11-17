"""Japanese length channel, held-out, measured through the SHIPPED runtime (FLEURS test).

    python -m bench.length.heldout_runtime [--lang ja] [--workers 3]   # -> results/tables/length_runtime_<lang>.json

It replaces the round-15 script bench/length/heldout.py (removed 2026-11-17), which recomputed the
channel offline with the round-15 arithmetic and so missed the round-17 head input
(pronunciada.length.score, head_s; its own runtime parity check differed by up to 0.166). This
script reads `PhoneResult.length_prob` and `WordResult.prob` straight from `Scorer(lang).score(...)`,
on the same clips and planted swaps (bench/audit/contrasts.py generators, PER_DIR per direction,
seeds "length-v1"):
* sound level: threshold = 5 % quantile of natives' judged length probabilities; share of swapped
  sounds (min over the edited sounds) at or below it, per contrast and direction;
* word level: threshold = 5 % quantile of native word probabilities; share of swapped words below.
Nothing is fitted; no Azure call.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

_SC = None


def _init(lang):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    global _SC
    import warnings

    warnings.simplefilter("ignore")
    from pronunciada.scorer import Scorer

    _SC = Scorer(lang, threads=1)


def _one(args):
    rec, lang = args
    from bench.audit.contrasts import apply, get_contrast, sites
    from bench.length.fit_length import CONTRASTS, PER_DIR
    from bench.pipeline.common import stable_seed
    from pronunciada.audio import load_audio

    try:
        wav = load_audio(rec["audio"])
        res = _SC.score((wav, 16000), rec["text"])
    except ValueError:
        return None
    ref = res._pronunciada["ref"]  # noqa: SLF001
    out = {"nat_len": [p.length_prob for w in res.words for p in w.phones if p.length_prob is not None],
           "nat_word": [w.prob for w in res.words], "edits": []}
    for cn in CONTRASTS[lang]:
        st = sites(ref, get_contrast(lang, cn))
        for dr in ("A>B", "B>A"):
            pool = [s for s in st if s.direction == dr]
            rng = np.random.default_rng(stable_seed(rec["id"], cn, dr, "length-v1"))
            for j in rng.permutation(len(pool))[:PER_DIR]:
                e = apply(ref, pool[j], rng)
                r2 = _SC.score((wav, 16000), e.ref)
                ph = [p for w in r2.words for p in w.phones]
                v = [ph[i].length_prob for i in e.flag if ph[i].length_prob is not None]
                out["edits"].append({"contrast": cn, "direction": dr, "len": min(v) if v else None,
                                     "word": float(r2.words[e.word].prob)})
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor

    from bench.pipeline.common import records

    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    recs = sorted(records("fleurs", split="test", lang=a.lang), key=lambda r: r["id"])
    with ProcessPoolExecutor(min(a.workers, 3), initializer=_init, initargs=(a.lang,)) as ex:
        rows = [r for r in ex.map(_one, [(r, a.lang) for r in recs], chunksize=4) if r is not None]
    nat = np.array([x for r in rows for x in r["nat_len"]], float)
    natw = np.array([x for r in rows for x in r["nat_word"]], float)
    t5, tw5 = float(np.quantile(nat, 0.05)), float(np.quantile(natw, 0.05))
    E = [e for r in rows for e in r["edits"]]
    out = {"lang": a.lang, "data": "FLEURS test, shipped Scorer defaults", "n_clips": len(rows),
           "n_native_judged_sounds": int(len(nat)), "sound_threshold_5": t5, "word_threshold_5": tw5,
           "native_word_accept_0.5": float(np.mean(natw >= 0.5)), "sound": {}, "word": {}}
    keys = sorted({(e["contrast"], e["direction"]) for e in E}) + [("all", "")]
    for c, d in keys:
        sel = [e for e in E if c == "all" or (e["contrast"] == c and e["direction"] == d)]
        lv = np.array([e["len"] for e in sel if e["len"] is not None], float)
        wv = np.array([e["word"] for e in sel], float)
        k = c if c == "all" else f"{c}:{d}"
        out["sound"][k] = {"n": int(len(lv)), "hit5": float(np.mean(lv <= t5)) if len(lv) else None}
        out["word"][k] = {"n": int(len(wv)), "hit5": float(np.mean(wv <= tw5)) if len(wv) else None}
    p = Path("results/tables") / f"length_runtime_{a.lang}.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
