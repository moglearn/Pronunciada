"""Benchmark datasets. Each dataset has a fetch script ``python -m bench.data.<name>`` that writes
``~/.cache/pronunciada/data/<name>/manifest.jsonl`` (see docs/benchmarks_data.md for schemas)."""
from __future__ import annotations

from .common import data_dir, read_jsonl

DATASETS = ("speechocean762", "fleurs", "l2arctic", "epadb", "aishell3", "zhtw")


def load(name: str, split: str | None = None, lang: str | None = None) -> list[dict]:
    """Return the manifest records of dataset `name`, optionally filtered by split / lang."""
    path = data_dir(name) / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m bench.data.{name}` first")
    return [r for r in read_jsonl(path)
            if (split is None or r["split"] == split) and (lang is None or r["lang"] == lang)]
