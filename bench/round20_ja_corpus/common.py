"""Shared paths and helpers for the round-20 Japanese length corpus (bench/round20_ja_corpus)."""
from __future__ import annotations

import csv
import os
from pathlib import Path

CACHE = Path(os.environ.get("PRONUNCIADA_CACHE", Path.home() / ".cache" / "pronunciada"))
WORK = CACHE / "round20" / "jacorpus"
RAW = WORK / "raw"
CVMETA = RAW / "cvmeta"
AUDIO = WORK / "audio"            # 16 kHz mono FLAC, one copy, <source>/<id>.flac
ALIGN = WORK / "align"            # aligner inputs / outputs
CORPUS = WORK / "corpus"          # final manifests + README

CV_REPO = "fsicoli/common_voice_22_0"
CV_REV = "ae911de250fe9375d08e2daa2105b671d659d446"
CV_URL = f"https://huggingface.co/datasets/{CV_REPO}/resolve/{CV_REV}"


def read_tsv(p: Path):
    csv.field_size_limit(1 << 30)
    with open(p, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def cv_durations() -> dict[str, float]:
    out = {}
    with open(CVMETA / "clip_durations.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1].strip().isdigit():
                out[p[0]] = int(p[1]) / 1000.0
    return out
