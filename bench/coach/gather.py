"""Coaching-gate calibration data: planted errors on FLEURS VALIDATION, all ten languages.

    python -m bench.coach.gather [--lang all] [--workers 3] [--max-clips N]

Per validation clip: the public Scorer on the correct prompt ("clean"), then on references with
planted errors (bench.corrupt; 2 errors per reference, seed = stable_seed(id, kind, "coach-v1")):
  phone_sub_near / phone_sub_far  the reference says Y where the audio has X  -> truth heard = X
  phone_ins                       the reference has an extra consonant         -> truth = missing
  tone (zh) / stress (en de es it pt ru)                                       -> truth = original
For every planted phone and every flagged clean phone (prob < 0.5) one row is kept with the gate
inputs (span posteriors of the ADAPTED model output, calibrated probabilities, duration ...).
Rows go to $PRONUNCIADA_CACHE/bench/coach/gather-<lang>.pkl. Nothing is fitted here.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import time
from pathlib import Path

import numpy as np

PHONE_KINDS = ("phone_sub_near", "phone_sub_far", "phone_ins")
PROS = {"zh": "tone", "en": "stress", "de": "stress", "es": "stress", "it": "stress",
        "pt": "stress", "ru": "stress"}
TAB_COLS = ("comp_tok", "comp_mean", "can_mean", "can_max", "comp_max", "log_ratio_comp",
            "last_comp_tok", "last_can_mean", "last_comp_mean", "n_frames", "llr_sub", "llr_del",
            "peak_post", "heard", "blank_mean", "z_llr_sub", "cls")


def out_dir() -> Path:
    base = os.environ.get("PRONUNCIADA_CACHE")
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "pronunciada"
    d = root / "bench" / "coach"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _span_vectors(lpa, s, e):
    T = len(lpa)
    a, b = (max(s - 1, 0), min(s + 1, T)) if e <= s else (max(s - 1, 0), min(e + 1, T))
    if b <= a:
        a, b = max(T - 1, 0), T
    P = np.exp(lpa[a:b].astype(np.float64))
    return P.mean(0).astype(np.float16), P.max(0).astype(np.float16)


def phone_rows(sc, res, wav, idx, truth, kind, uid):
    c = res._pronunciada  # noqa: SLF001
    key = hashlib.sha1(wav.tobytes()).hexdigest()
    lpa = sc._lp_cache[key][0]  # noqa: SLF001
    ref, tab, ev = c["ref"], c["tab"], c["ev"]
    flat = [p for w in res.words for p in w.phones]
    rows = []
    for i, t in zip(idx, truth):
        sl = ev.slots[i]
        m, mx = _span_vectors(lpa, sl.start_frame, sl.end_frame)
        r = {"utt": uid, "kind": kind, "i": i, "expected": ref.phones[i].ipa,
             "alts": [a for a, _ in ref.phones[i].alts], "truth": t, "prob": flat[i].prob,
             "p_len": flat[i].features["p_lenient"], "p_str": flat[i].features["p_strict"],
             "heard_api": flat[i].heard, "m": m, "mx": mx, "word": ref.phones[i].word,
             "nucleus": bool(ref.phones[i].is_nucleus), "slot_heard": sl.heard,
             "start": sl.start_frame, "end": sl.end_frame}
        for k in TAB_COLS:
            v = tab[k][i]
            r[k] = v if isinstance(v, (str, type(None))) else float(v)
        rows.append(r)
    return rows


def _pros_rows(res, kind, cor=None):
    c = res._pronunciada  # noqa: SLF001
    ref, pr = c["ref"], c.get("prosody") or []
    out = []
    if kind == "tone":
        flags = [("tone" in f) for f in cor.syllable_flags] if cor else [False] * len(ref.syllables)
        for j, sy in enumerate(ref.syllables):
            d = pr[j] if j < len(pr) else {}
            if sy.tone is None or d.get("tone_probs") is None:
                continue
            if cor is not None and not flags[j]:
                continue
            truth = cor.original.syllables[j].tone if cor is not None else sy.tone
            out.append({"syl": j, "expected": sy.tone, "alts": list(sy.tone_alts), "truth": truth,
                        "planted": cor is not None, "tone_probs": d["tone_probs"],
                        "tone_prob": d.get("tone_prob")})
    else:
        for wi, w in enumerate(ref.words):
            s0, s1 = w.syllables
            j = next((k for k in range(s0, s1) if ref.syllables[k].stress == 1), None)
            if j is None or j >= len(pr) or pr[j].get("stress_dist") is None:
                continue
            if cor is not None:
                if not any("stress" in cor.syllable_flags[k] for k in range(s0, s1)):
                    continue
                truth = next(k for k in range(s0, s1) if cor.original.syllables[k].stress == 1) - s0
            else:
                truth = j - s0
            out.append({"word": wi, "expected": j - s0, "truth": truth, "planted": cor is not None,
                        "n_syl": s1 - s0, "dist": pr[j]["stress_dist"], "stress_prob": pr[j]["stress_prob"]})
    return out


def run_lang(args):
    lang, max_clips = args
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import fleurs_validation_records, iter_parquet_audio, stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada.scorer import Scorer

    sc = Scorer(lang, threads=1)
    inv = inventory(lang)
    recs = fleurs_validation_records(lang)[:max_clips or None]
    by_row = {int(r["audio"].rsplit(":", 1)[1]): r for r in recs}
    path = recs[0]["audio"].split(":", 1)[1].rsplit(":", 1)[0]
    phones, pros, clips = [], [], []
    t0 = time.time()
    for row, wav in iter_parquet_audio(path, set(by_row)):
        rec = by_row[row]
        try:
            res = sc.score(wav, rec["text"])
        except ValueError:
            continue
        ref = res._pronunciada["ref"]  # noqa: SLF001
        flat = [p for w in res.words for p in w.phones]
        bad = [i for i, p in enumerate(flat) if p.prob < 0.5]
        phones += phone_rows(sc, res, wav, bad, [ref.phones[i].ipa for i in bad], "clean", rec["id"])
        clips.append({"utt": rec["id"], "n_phones": len(flat), "n_flagged": len(bad),
                      "n_words": len(res.words), "omitted": [w.omitted for w in res.words],
                      "present": [w.present_prob for w in res.words]})
        if lang in PROS:
            pros += [dict(r, utt=rec["id"], kind=PROS[lang]) for r in _pros_rows(res, PROS[lang])]
        for kind in PHONE_KINDS + ((PROS[lang],) if lang in PROS else ()):
            try:
                cor = corrupt(ref, [kind], inv, n_errors=2, seed=stable_seed(rec["id"], kind, "coach-v1"),
                              strict=False)
            except NotApplicable:
                continue
            if not cor.errors:
                continue
            r2 = sc.score(wav, cor.ref)
            if kind in PHONE_KINDS:
                idx = [i for i, w in enumerate(cor.phone_wrong) if w]
                truth = []
                for i in idx:
                    p = cor.ref.phones[i]
                    e = next((e for e in cor.errors if e.word == p.word), None)
                    truth.append("" if kind == "phone_ins" else (e.detail.get("from") if e else None))
                phones += phone_rows(sc, r2, wav, idx, truth, kind, rec["id"])
            else:
                pros += [dict(r, utt=rec["id"], kind=kind) for r in _pros_rows(r2, kind, cor)]
    out = {"lang": lang, "phones": phones, "pros": pros, "clips": clips}
    (out_dir() / f"gather-{lang}.pkl").write_bytes(pickle.dumps(out))
    return lang, len(clips), len(phones), len(pros), time.time() - t0


def main():
    from concurrent.futures import ProcessPoolExecutor

    from bench.pipeline.common import LANGS

    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-clips", type=int, default=0)
    a = ap.parse_args()
    langs = list(LANGS) if a.lang == "all" else a.lang.split(",")
    with ProcessPoolExecutor(min(a.workers, 2)) as ex:
        for lang, n, npn, npr, dt in ex.map(run_lang, [(l, a.max_clips) for l in langs]):
            print(f"{lang}: {n} clips, {npn} phone rows, {npr} prosody rows, {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
