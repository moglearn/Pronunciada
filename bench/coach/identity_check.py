"""Proof that coaching / the audio gate change no score: every probability byte-identical.

    python -m bench.coach.identity_check --dump OUT.json [--coach] [--per-lang 3]
    python -m bench.coach.identity_check --compare A.json B.json

--dump scores the first N FLEURS validation clips per language with whatever `pronunciada`
is first on sys.path (run it once with a copy of the package whose scorer.py / types.py are
the pre-coaching versions from git, once with the current tree and --coach, which also calls
result.coaching() and pronunciada.tips.tips() before reading the numbers) and writes every
sound / word / syllable / sentence probability and score as repr() strings. --compare
reports any difference. Also times coaching() (median over clips, ms per 10 s of audio).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def _r(x):
    return None if x is None else repr(float(x))


def dump(out, use_coach, per_lang, langs=None):
    import pronunciada
    from bench.pipeline.common import LANGS, fleurs_validation_records, iter_parquet_audio

    rows, lat = {}, []
    for lang in (langs or LANGS):
        sc = pronunciada.Scorer(lang, threads=1)
        recs = fleurs_validation_records(lang)[:per_lang]
        by_row = {int(r["audio"].rsplit(":", 1)[1]): r for r in recs}
        path = recs[0]["audio"].split(":", 1)[1].rsplit(":", 1)[0]
        for row, wav in iter_parquet_audio(path, set(by_row)):
            rec = by_row[row]
            res = sc.score(wav, rec["text"])
            if use_coach:
                from pronunciada.tips import tips

                res.coaching()                       # warm caches
                t = time.perf_counter()
                for _ in range(5):
                    items = res.coaching()
                    tips(items)
                lat.append((time.perf_counter() - t) / 5 * 1e3 / (len(wav) / 16000 / 10))
            d = {"prob": _r(res.prob), "fluency": _r(res.fluency), "completeness": _r(res.completeness),
                 "match": _r(res.match_prob), "scores": {k: _r(v) for k, v in res.scores.items()},
                 "words": []}
            for w in res.words:
                d["words"].append({"prob": _r(w.prob), "sounds": _r(w.sounds_prob), "stress": _r(w.stress_prob),
                                   "tone": _r(w.tone_prob), "present": _r(w.present_prob), "omitted": w.omitted,
                                   "phones": [[p.ipa, _r(p.prob), p.heard, p.start, p.end] for p in w.phones],
                                   "syl": [[_r(s.tone_prob), s.tone_heard, _r(s.stress_prob)]
                                           for s in w.syllables]})
            rows[rec["id"]] = d
        print(lang, len(rows), flush=True)
    meta = {"coach_ms_per_10s_median": float(np.median(lat)) if lat else None,
            "coach_ms_per_10s_p90": float(np.quantile(lat, 0.9)) if lat else None}
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False)
    print(meta)


def compare(a, b):
    A, B = json.load(open(a, encoding="utf-8")), json.load(open(b, encoding="utf-8"))
    ra, rb = A["rows"], B["rows"]
    diff = [k for k in ra if ra[k] != rb.get(k)]
    n_vals = sum(len(json.dumps(v)) for v in ra.values())
    print(f"clips {len(ra)} / {len(rb)}; differing clips: {len(diff)}; json chars compared {n_vals}")
    print("meta", A["meta"], B["meta"])
    return len(diff) == 0 and set(ra) == set(rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--coach", action="store_true")
    ap.add_argument("--per-lang", type=int, default=3)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--langs", default="")
    a = ap.parse_args()
    if a.dump:
        dump(a.dump, a.coach, a.per_lang, a.langs.split(",") if a.langs else None)
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)


if __name__ == "__main__":
    main()
