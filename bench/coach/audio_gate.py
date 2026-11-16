"""Audio-quality gate thresholds (pronunciada.quality), set on data.

    python -m bench.coach.audio_gate --steps measure,fit [--workers 2]

measure: pronunciada.quality.measure on
  * FLEURS VALIDATION, all ten languages (normal audio), and synthetic degradations of every
    4th clip: white noise at 20 / 10 / 5 / 0 dB SNR (relative to the speech-frame power),
    hard clipping after x3 / x10 gain, attenuation by 35 / 50 dB, truncation to the first 40 %,
    and the clip embedded in 4x its length of near-silence (mostly silence);
  * Common Voice "cv-real" DEV half, test A (accepted vs rejected by CV reviewers). The TEST
    half is never read.
fit: per language median speech seconds per syllable (quality.SEC_PER_SYL) and one threshold
per measurement at a low quantile of FLEURS validation (union flag rate < 2 %), then the flag
rates on every degradation and on cv-real dev. Writes results/tables/audio_gate.json.
Streaming: one clip in memory at a time per worker.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from bench.coach.gather import out_dir

DEGR = ("noise20", "noise10", "noise5", "noise0", "clip3", "clip10", "att35", "att50", "trunc40", "pad4")
BAD = {"noise5", "noise0", "clip10", "att50", "trunc40", "pad4"}     # must be caught
EXPECT = {"noise20": None, "noise10": "too_noisy", "noise5": "too_noisy", "noise0": "too_noisy",
          "clip3": "clipped", "clip10": "clipped", "att35": "too_quiet", "att50": "too_quiet",
          "trunc40": "too_short", "pad4": "mostly_silence"}


def degrade(wav: np.ndarray, kind: str, rng) -> np.ndarray:
    from pronunciada.quality import FRAME, _frame_db

    w = wav.astype(np.float64)
    if kind.startswith("noise"):
        snr = float(kind[5:])
        db = _frame_db(wav)
        act = db >= np.percentile(db, 50) if len(db) else None
        n = len(db) * FRAME
        p = float((w[:n].reshape(-1, FRAME)[act] ** 2).mean()) if act is not None and act.any() else float((w ** 2).mean())
        return np.clip(w + rng.standard_normal(len(w)) * np.sqrt(p / 10 ** (snr / 10)), -1, 1).astype(np.float32)
    if kind.startswith("clip"):
        g = float(kind[4:])
        return np.clip(w * g / max(np.abs(w).max(), 1e-6) * 0.9, -1, 1).astype(np.float32)
    if kind.startswith("att"):
        return (w * 10 ** (-float(kind[3:]) / 20)).astype(np.float32)
    if kind == "trunc40":
        return wav[: int(len(wav) * 0.4)].copy()
    if kind == "pad4":
        n = len(wav)
        out = rng.standard_normal(4 * n) * 10 ** (-70 / 20)
        out[n * 2: n * 3] += w
        return np.clip(out, -1, 1).astype(np.float32)
    raise ValueError(kind)


def _fleurs(lang):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from bench.pipeline.common import build_ref, fleurs_validation_records, iter_parquet_audio, stable_seed
    from pronunciada.quality import measure

    recs = fleurs_validation_records(lang)
    by_row = {int(r["audio"].rsplit(":", 1)[1]): r for r in recs}
    path = recs[0]["audio"].split(":", 1)[1].rsplit(":", 1)[0]
    rows = []
    for k, (row, wav) in enumerate(iter_parquet_audio(path, set(by_row))):
        rec = by_row[row]
        ref = build_ref(lang, rec["text"])
        ns = len(ref.syllables) if ref else None
        rows.append({"set": "fleurs", "lang": lang, "id": rec["id"], "n_syl": ns, "kind": "clean",
                     **measure(wav, ns, lang)})
        if k % 4 == 0:
            rng = np.random.default_rng(stable_seed(rec["id"], "audio-gate"))
            for d in DEGR:
                rows.append({"set": "fleurs", "lang": lang, "id": rec["id"], "n_syl": ns, "kind": d,
                             **measure(degrade(wav, d, rng), ns, lang)})
    return rows


def _cv(lang):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from bench.cv_build import LOCALES, load_manifest
    from bench.pipeline.common import build_ref
    from pronunciada.audio import load_audio
    from pronunciada.quality import measure

    rows = []
    for r in load_manifest([l for l in LOCALES if l.split("-")[0] == lang], half="dev", test="A"):
        if not Path(r["audio"]).exists():
            continue
        ref = build_ref(lang, r["sentence"])
        ns = len(ref.syllables) if ref else None
        rows.append({"set": "cv", "lang": lang, "id": r["id"], "n_syl": ns, "kind": r["side"],
                     **measure(load_audio(r["audio"]), ns, lang)})
    return rows


def _job(a):
    which, lang = a
    return (_fleurs if which == "fleurs" else _cv)(lang)


def step_measure(workers):
    from concurrent.futures import ProcessPoolExecutor

    from bench.pipeline.common import LANGS

    rows = []
    jobs = [(w, l) for l in LANGS for w in ("fleurs", "cv")]
    with ProcessPoolExecutor(min(workers, 2)) as ex:
        for (w, l), r in zip(jobs, ex.map(_job, jobs)):
            rows += r
            print(w, l, len(r), flush=True)
    (out_dir() / "audio_gate.pkl").write_bytes(pickle.dumps(rows))


def apply(rows, spl, thr):
    from pronunciada.quality import issues

    out = []
    for r in rows:
        m = dict(r)
        exp = r["n_syl"] * spl[r["lang"]] if r["n_syl"] else None
        m["speech_ratio"] = r["speech_s"] / exp if exp else None
        out.append(issues(m, thr))
    return out


def step_fit():
    rows = pickle.loads((out_dir() / "audio_gate.pkl").read_bytes())
    fl = [r for r in rows if r["set"] == "fleurs" and r["kind"] == "clean"]
    langs = sorted({r["lang"] for r in fl})
    spl = {l: round(float(np.median([r["speech_s"] / r["n_syl"] for r in fl if r["lang"] == l and r["n_syl"]])), 3)
           for l in langs}

    def q(key, p, ratio=False):
        v = []
        for r in fl:
            if ratio:
                if r["n_syl"]:
                    v.append(r["speech_s"] / (r["n_syl"] * spl[r["lang"]]))
            elif r[key] is not None:
                v.append(r[key])
        return float(np.quantile(v, p))

    # one-sided quantiles of normal audio; 0.4 % each keeps the union under 2 %
    thr = {"clipped_share": max(round(q("clipped_share", 0.996), 4), 0.002),
           "snr_db": round(q("snr_db", 0.004), 1),
           "speech_db": round(q("speech_db", 0.004), 1),
           "speech_ratio": round(q(None, 0.004, ratio=True), 3),
           "speech_share": round(q("speech_share", 0.004), 3),
           "min_speech_s": 0.3}
    res = {"sec_per_syl": spl, "thresholds": thr, "rates": {}}
    iss = apply(rows, spl, thr)
    groups: dict = {}
    for r, i in zip(rows, iss):
        groups.setdefault((r["set"], r["kind"]), []).append((r["lang"], i))
    for (s, k), v in sorted(groups.items()):
        codes = {}
        for _, i in v:
            for c in i:
                codes[c] = codes.get(c, 0) + 1
        res["rates"][f"{s}/{k}"] = {"n": len(v), "flagged": float(np.mean([bool(i) for _, i in v])),
                                    "codes": {c: n / len(v) for c, n in sorted(codes.items())},
                                    "by_lang": {l: float(np.mean([bool(i) for ll, i in v if ll == l]))
                                                for l in langs if any(ll == l for ll, _ in v)}}
        if k in EXPECT and EXPECT[k]:
            res["rates"][f"{s}/{k}"]["expected_code"] = float(np.mean([EXPECT[k] in i for _, i in v]))
    p = Path("results/tables/audio_gate.json")
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: (v["n"], round(v["flagged"], 3), v.get("expected_code")) for k, v in res["rates"].items()}, indent=0))
    print(thr, spl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="measure,fit")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if "measure" in a.steps:
        step_measure(a.workers)
    if "fit" in a.steps:
        step_fit()


if __name__ == "__main__":
    main()
