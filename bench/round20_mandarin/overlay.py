"""Cache overlay for one arm of a round-20 run (bench/round19_merge/overlay.py, plus the validation audit).

    python -m bench.round20_mandarin.overlay <overlay dir>

Like the round-19 overlay: symbolic links to the real $PRONUNCIADA_CACHE except the output folders,
which start empty. Both contrast-audit folders (validation and test) start empty, with the existing
edit plans and aspiration files copied, so every arm uses the same edits as earlier runs.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from bench.round19_merge import overlay as O
from bench.round19_rollout.sources import CACHE


def build(ov: Path) -> None:
    O.FRESH = O.FRESH + ["audit/contrast_v1"]
    O.build(CACHE, ov)
    src = CACHE / "audit" / "contrast_v1"
    for lang_dir in sorted(src.iterdir()) if src.is_dir() else []:
        for name in O.COPY_AUDIT:
            f = lang_dir / name
            if f.exists():
                (ov / "audit" / "contrast_v1" / lang_dir.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, ov / "audit" / "contrast_v1" / lang_dir.name / name)


if __name__ == "__main__":
    build(Path(sys.argv[1]).expanduser())
