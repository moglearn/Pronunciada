"""Learner-speech evaluation suite (round 19): held-out tests on real learner recordings.

Every corpus has a fetch + build script ``python -m bench.learners.<corpus>`` that writes, under
``$PRONUNCIADA_CACHE/data/learners/<corpus>/`` (never in the repo):

    manifest.jsonl   every usable read-speech item (16 kHz mono FLAC under audio/)
    test.jsonl       the fixed test selection (seeded; see the module docstring for the rule)
    dev.jsonl        the speaker-disjoint rest, set aside (never used by this suite)
    build.json       counts, exclusions with reasons, licence, source checksums

Records use the benchmark manifest layout (docs/benchmarks_data.md): ``id, audio, lang, text,
split, speaker, dur, labels`` with ``labels.native`` (bool), ``labels.l1`` and any corpus label.
``python -m bench.learners.suite`` merges every test selection (plus the existing expert-scored
learner tests speechocean762 test and OMPAL test, referenced in place) into
``data/learners/suite_test.jsonl``; ``python -m bench.learners.evaluate`` scores and reports it.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from bench.data.common import cache_root, make_record, read_jsonl  # noqa: F401  (re-exported)

SUITE = "learners"


def root(corpus: str | None = None) -> Path:
    d = cache_root() / "data" / SUITE
    return d / corpus if corpus else d


def downloads(corpus: str) -> Path:
    """Raw downloads live outside the manifests, in their own folder (untrusted data)."""
    return cache_root() / "round19" / "evalsuite" / "downloads" / corpus


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, size: int | None = None, sha: str | None = None, tries: int = 5) -> Path:
    """Resumable download with curl (-C -). Verifies the size and/or sha256 when given; a file that
    already verifies is not fetched again."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size is None or dest.stat().st_size == size) and (sha is None or sha256(dest) == sha):
        if size is not None or sha is not None:
            return dest
    part = dest.with_name(dest.name + ".part")
    for k in range(tries):
        r = subprocess.run(["curl", "-sSL", "--fail", "-C", "-", "-o", str(part), url])
        if r.returncode in (0, 33) and part.exists():  # 33: server refused range on a complete file
            break
        time.sleep(5 * (k + 1))
    else:
        sys.exit(f"download failed: {url}")
    if size is not None and part.stat().st_size != size:
        sys.exit(f"size mismatch for {url}: {part.stat().st_size} != {size}")
    if sha is not None and sha256(part) != sha:
        part.unlink()
        sys.exit(f"sha256 mismatch for {url}")
    os.replace(part, dest)
    return dest


def write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def load(corpus: str, part: str = "test") -> list[dict]:
    p = root(corpus) / f"{part}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `python -m bench.learners.{corpus}` first")
    return list(read_jsonl(p))


def hours(recs) -> float:
    return round(sum(r["dur"] for r in recs) / 3600, 3)
