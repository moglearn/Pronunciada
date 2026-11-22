"""Build a cache overlay for a frozen benchmark run that must not read or overwrite earlier outputs.

    python bench/round19_merge/overlay.py <real cache> <overlay dir>

The overlay mirrors the real $PRONUNCIADA_CACHE with symbolic links (data, models, posteriors,
cached Azure responses, fitted inputs), except for the folders and files the frozen run writes:
those are new, empty real folders (or simply absent files), so every Pronunciada output of the run
is computed by the code under test and nothing that already exists is touched. The contrast-audit
edit plans are copied (not linked) so the run uses the same edits as the 2026-11-17 / 18 runs.
Run the jobs with PRONUNCIADA_CACHE=<overlay dir>.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

FRESH = ["calib/api", "audit/contrast_v1_test", "third_party/lingolix/responses", "cv-real/ours",
         "ompal/ours", "round19/evalsuite/ours"]
ABSENT = ["bench/coach/dump-so762-test.pkl"] + [
    f"bench/coach/dump-r4-{s}.pkl" for s in ("so762-test", "l2arctic-test", "epadb-test", "ompal-test")]
SKIP_TOP = {"heavy.lock", "heavy-cpu.lock", "heavy-train.lock"}
COPY_AUDIT = ("plan_250.json", "asp_250.pkl")


def build(real: Path, ov: Path) -> None:
    keep = {Path(p) for p in FRESH + ABSENT}
    real_dirs = {Path(".")}
    for p in keep:
        real_dirs.update(p.parents)
    real_dirs.discard(Path("."))
    real_dirs = {Path(".")} | real_dirs
    for d in sorted(real_dirs, key=lambda x: len(x.parts)):
        (ov / d).mkdir(parents=True, exist_ok=True)
        src = real / d
        if not src.is_dir():
            continue
        for child in src.iterdir():
            rel = d / child.name
            if rel in keep or rel in real_dirs or (d == Path(".") and child.name in SKIP_TOP):
                continue
            link = ov / rel
            if not link.exists() and not link.is_symlink():
                link.symlink_to(child)
    for p in FRESH:
        (ov / p).mkdir(parents=True, exist_ok=True)
    src = real / "audit" / "contrast_v1_test"
    for lang_dir in sorted(src.iterdir()) if src.is_dir() else []:
        for name in COPY_AUDIT:
            f = lang_dir / name
            if f.exists():
                (ov / "audit" / "contrast_v1_test" / lang_dir.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, ov / "audit" / "contrast_v1_test" / lang_dir.name / name)
    print(f"overlay ready: {ov}")


if __name__ == "__main__":
    build(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser())
