"""Shared helpers for the phase-2 evidence pipeline (bench only; may use anything).

Datasets: every name of bench.data.DATASETS plus "fleurs-validation" (the FLEURS validation
parquets = DEV set; records are built here, audio is read from the parquet on demand).
FLEURS validation ids: the FLEURS `id` column is a *sentence* id shared by several speakers,
so the utterance id is `fleurs-<lang>-validation-<wav stem>` (same scheme as the train/test
manifest); the FLEURS id is kept in labels.fleurs_id.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench.data import DATASETS
from bench.data import load as load_manifest
from bench.data.common import cache_root

LANGS = ("en", "zh", "es", "it", "ja", "ko", "fr", "de", "pt", "ru")
FLEURS_CFG = {"en": "en_us", "zh": "cmn_hans_cn", "ja": "ja_jp", "es": "es_419", "fr": "fr_fr",
              "de": "de_de", "it": "it_it", "pt": "pt_br", "ru": "ru_ru", "ko": "ko_kr"}
ALL_DATASETS = tuple(DATASETS) + ("fleurs-validation",)
MAX_DUR = 30.0
MODEL = "zipa-small-crctc-500k"
PRECISION = "fp32"


def pipe_dir(*parts) -> Path:
    p = cache_root().joinpath("pipeline", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def feat_dir() -> Path:
    p = cache_root() / "features"
    p.mkdir(parents=True, exist_ok=True)
    return p


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_langs(arg: str) -> list[str]:
    if arg == "all":
        return list(LANGS)
    out = [a.strip() for a in arg.split(",") if a.strip()]
    for a in out:
        if a not in LANGS:
            raise SystemExit(f"unknown lang {a}")
    return out


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- records

def _val_parquet(lang: str) -> Path:
    return cache_root() / "data" / "fleurs" / FLEURS_CFG[lang] / "validation-0.parquet"


def fleurs_validation_records(lang: str) -> list[dict]:
    """Records of the FLEURS validation split (clips > 30 s skipped, as for train/test)."""
    cache = pipe_dir("manifests") / f"fleurs-validation-{lang}.jsonl"
    if cache.exists():
        return [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    import pyarrow.parquet as pq

    path = _val_parquet(lang)
    t = pq.read_table(path, columns=["id", "path", "num_samples", "raw_transcription",
                                     "transcription", "gender"]).to_pylist()
    recs = []
    for row_i, row in enumerate(t):
        dur = row["num_samples"] / 16000
        if dur > MAX_DUR:
            continue
        stem = Path(row["path"]).stem
        recs.append({"id": f"fleurs-{lang}-validation-{stem}", "audio": f"parquet:{path}:{row_i}",
                     "lang": lang, "text": row["raw_transcription"], "split": "validation",
                     "speaker": None, "dur": round(dur, 3),
                     "labels": {"fleurs_id": row["id"], "transcription": row["transcription"],
                                "gender": row["gender"], "config": FLEURS_CFG[lang]}})
    tmp = cache.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    os.replace(tmp, cache)
    return recs


def records(dataset: str, split: str | None = None, lang: str | None = None) -> list[dict]:
    if dataset == "fleurs-validation":
        langs = [lang] if lang else list(LANGS)
        return [r for l in langs for r in fleurs_validation_records(l)]
    return load_manifest(dataset, split=split, lang=lang)


def iter_parquet_audio(path: str, rows: set[int] | None = None):
    """Yield (row index, float32 16 kHz waveform) from a FLEURS parquet."""
    import pyarrow.parquet as pq
    import soundfile as sf

    pf = pq.ParquetFile(path)
    i = 0
    for batch in pf.iter_batches(batch_size=16, columns=["audio"]):
        for row in batch.column(0).to_pylist():
            if rows is None or i in rows:
                wav, sr = sf.read(io.BytesIO(row["bytes"]), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(1)
                assert sr == 16000, sr
                yield i, wav
            i += 1


# ----------------------------------------------------------------------------- posteriors

def lp_path(dataset: str, utt_id: str) -> Path:
    from bench.postcache import post_path

    return post_path(dataset, utt_id, MODEL, PRECISION)


def load_lp(dataset: str, rec: dict) -> np.ndarray | None:
    """Cached raw log-posteriors float32 [T,127]; computes them if missing (never for parquet
    records outside warm)."""
    p = lp_path(dataset, rec["id"])
    if p.exists():
        return np.load(p).astype(np.float32)
    if str(rec["audio"]).startswith("parquet:"):
        return None
    from bench.postcache import get_posteriors

    return get_posteriors(dataset, rec["id"], rec["audio"], MODEL, PRECISION)


def near_silent(lp: np.ndarray, dur: float) -> bool:
    """True if the greedy decode has < 1 speech token per second (e.g. 177 FLEURS es_419 train
    clips are digital silence, peak |x| < 1e-4). Such clips are excluded from fitting / stats."""
    from pronunciada.acoustic import greedy_tokens

    return len(greedy_tokens(lp)) < max(dur, 1.0)


# ----------------------------------------------------------------------------- references

_G2P = {}


def build_ref(lang: str, text: str):
    """Reference from our front-end, or None if it fails / is empty / invalid."""
    from pronunciada.text import get_g2p

    if lang not in _G2P:
        _G2P[lang] = get_g2p(lang)
    try:
        ref = _G2P[lang](text)
        ref.validate()
    except Exception:  # noqa: BLE001
        return None
    return ref if ref.phones else None


def g2p_version(lang: str) -> str:
    """Hash of the front-end source files: caches keyed on it are invalidated when the text
    agent edits the front-end."""
    d = repo_root() / "pronunciada" / "text"
    h = hashlib.sha256()
    names = [f"{lang}.py", "ipa.py", "base.py"]
    names += ["_eu_base.py", "_eu_common.py", "_eu_lts.py"] if lang in ("en", "es", "it", "pt", "ru", "de", "fr") else ["_cjk_common.py"]
    for n in names:
        p = d / n
        if p.exists():
            h.update(p.read_bytes())
    lex = repo_root() / "pronunciada" / "data" / "lex"
    for p in sorted(lex.glob(f"{lang}*")):
        h.update(str(p.stat().st_size).encode())
        h.update(str(int(p.stat().st_mtime)).encode())
    return h.hexdigest()[:12]


# ----------------------------------------------------------------------------- phone classes

from pronunciada.variants import CLASSES, broad_class, phone_position as position  # noqa: E402,F401


# ----------------------------------------------------------------------------- evidence

EV_FIELDS = ("llr_sub", "llr_del", "n_frames", "peak_post", "alt_index", "start_frame", "end_frame")


def evidence_arrays(lp: np.ndarray, ref) -> dict:
    from pronunciada.align import score_reference

    ev = score_reference(lp, ref)
    out = {k: np.array([getattr(s, k) for s in ev.slots]) for k in EV_FIELDS}
    out["heard"] = [s.heard for s in ev.slots]
    out["L_can"] = ev.L_can
    out["_lp"] = lp
    out["T"] = ev.n_frames
    return out


# ----------------------------------------------------------------------------- parallel

def pmap(fn, items, workers: int = 6, chunksize: int = 4, desc: str = ""):
    """Process-parallel map preserving order; each worker single-threaded BLAS."""
    items = list(items)
    if workers <= 1 or len(items) < 8:
        return [fn(x) for x in items]
    out = []
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        for k, r in enumerate(ex.map(fn, items, chunksize=chunksize)):
            out.append(r)
            if desc and (k + 1) % 500 == 0:
                log(f"{desc}: {k + 1}/{len(items)} ({time.time() - t0:.0f}s)")
    return out


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except Exception:  # noqa: BLE001
        pass


def base_parser(desc: str, lang_default: str = "all") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--lang", default=lang_default)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    return ap


def set_threads(n: int):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, str(n))


if __name__ == "__main__":
    sys.exit("library module")
