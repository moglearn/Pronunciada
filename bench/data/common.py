"""Shared helpers for the benchmark data fetchers.

Layout: ``$PRONUNCIADA_CACHE/data/<name>/`` (default ``~/.cache/pronunciada/data/<name>/``)
    manifest.jsonl          final manifest, one JSON object per utterance
    parts/<key>.jsonl       per-chunk partial manifests (resumability); a chunk is done when
                            parts/<key>.done exists
    audio/<split>/...       16 kHz mono FLAC (PCM_16)
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
SR = 16000
MANIFEST_KEYS = ("id", "audio", "lang", "text", "split", "speaker", "dur", "labels")


def cache_root() -> Path:
    base = os.environ.get("PRONUNCIADA_CACHE")
    return Path(base).expanduser() if base else Path.home() / ".cache" / "pronunciada"


def data_dir(name: str) -> Path:
    return cache_root() / "data" / name


def free_gb(path: Path | None = None) -> float:
    p = path or cache_root()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def check_disk(min_free_gb: float = 6.0) -> None:
    f = free_gb()
    if f < min_free_gb:
        sys.exit(f"refusing to continue: only {f:.1f} GB free (< {min_free_gb} GB)")


def dir_size_gb(path: Path) -> float:
    tot = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                tot += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return tot / 1e9


# ----------------------------------------------------------------------------- audio

def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode any soundfile-readable byte string -> (float32 mono, sr)."""
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def resample(x: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def write_flac(path: Path, x: np.ndarray, sr: int = SR) -> float:
    """Write 16 kHz mono FLAC (PCM_16) atomically; returns duration in seconds."""
    import soundfile as sf

    x = resample(np.asarray(x, dtype=np.float32), sr, SR)
    x = np.clip(x, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.flac")
    sf.write(tmp, x, SR, format="FLAC", subtype="PCM_16")
    os.replace(tmp, path)
    return len(x) / SR


def flac_duration(path: Path) -> float:
    import soundfile as sf

    return sf.info(str(path)).duration


# ----------------------------------------------------------------------------- manifests

def make_record(id: str, audio: Path, lang: str, text: str, split: str, speaker, dur: float,
                labels: dict) -> dict:
    assert lang in LANGS, lang
    return {"id": id, "audio": str(Path(audio).resolve()), "lang": lang, "text": text,
            "split": split, "speaker": speaker, "dur": round(float(dur), 3), "labels": labels}


class PartWriter:
    """Append-only partial manifest for one chunk (e.g. fleurs/ja/train)."""

    def __init__(self, root: Path, key: str):
        self.dir = root / "parts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{key}.jsonl"
        self.done_flag = self.dir / f"{key}.done"
        self.records: list[dict] = list(read_jsonl(self.path)) if self.path.exists() else []
        self.ids = {r["id"] for r in self.records}

    @property
    def done(self) -> bool:
        return self.done_flag.exists()

    def add(self, rec: dict) -> None:
        if rec["id"] in self.ids:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.records.append(rec)
        self.ids.add(rec["id"])

    def finish(self) -> None:
        self.done_flag.touch()


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_manifest(root: Path, records: Iterable[dict]) -> Path:
    path = root / "manifest.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            assert tuple(r.keys()) == MANIFEST_KEYS, r.keys()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    print(f"wrote {n} records -> {path}")
    return path


def merge_parts(root: Path, order: Iterable[str] | None = None) -> Path:
    parts = root / "parts"
    keys = list(order) if order is not None else sorted(p.stem for p in parts.glob("*.jsonl"))
    recs: list[dict] = []
    for k in keys:
        p = parts / f"{k}.jsonl"
        if p.exists():
            recs.extend(read_jsonl(p))
    return write_manifest(root, recs)


# ----------------------------------------------------------------------------- remote parquet

def resolve_redirects(url: str, max_hops: int = 5) -> str:
    """Follow HF redirects once and return the final (signed CDN) URL, so later range requests
    do not each count against HF's per-IP "resolve" rate limit (3000 / 5 min)."""
    import time

    import requests

    for _ in range(max_hops):
        for k in range(8):
            r = requests.head(url, allow_redirects=False, timeout=30)
            if r.status_code != 429:
                break
            time.sleep(min(30 * (k + 1), 300))
        if r.status_code in (301, 302, 303, 307, 308):
            url = requests.compat.urljoin(url, r.headers["Location"])
            continue
        return url
    return url


def open_remote_parquet(url: str, resolve: bool = True):
    """Open a remote parquet file over HTTP range requests (only touched byte ranges are fetched)."""
    import fsspec
    import pyarrow.parquet as pq

    fs = fsspec.filesystem("http")
    f = fs.open(resolve_redirects(url) if resolve else url, block_size=8 << 20, cache_type="none")
    return pq.ParquetFile(f, pre_buffer=False)


def summarize(records: list[dict]) -> dict:
    from collections import defaultdict

    out: dict = defaultdict(lambda: {"n": 0, "hours": 0.0})
    for r in records:
        k = f"{r['lang']}/{r['split']}"
        out[k]["n"] += 1
        out[k]["hours"] += r["dur"] / 3600
    return {k: {"n": v["n"], "hours": round(v["hours"], 2)} for k, v in sorted(out.items())}
