"""Round 20 release check, step 2: seed the per-clip output caches that are keyed by the package
fingerprint (cv-real, learner suite) for the release code, in a COPY of the round-19 frozen-run
overlay, with the round-19 frozen run's outputs of every language except Mandarin.

    PRONUNCIADA_CACHE=<new overlay> python -m bench.round20_refresh.seed <round-19 fingerprint>

Step 1 (bench/round20_refresh/run_equiv.sh) proves that the release code gives outputs identical to
the frozen run's code for those nine languages, so their outputs are reused, as the round-19
frozen run 2 reused the outputs of the languages its last merges did not change. Mandarin clips
are not copied: the `ours` steps score them fresh with the release code. Nothing outside the new
overlay is written.
"""
from __future__ import annotations

import json
import shutil
import sys

from bench.pipeline.common import cache_root


def main() -> None:
    old_fp = sys.argv[1]
    from bench import eval_cv
    from bench.learners import evaluate as L

    new_fp = eval_cv.fingerprint()
    assert new_fp == L.fingerprint(), (new_fp, L.fingerprint())
    root = cache_root()
    assert "round20" in str(root), f"refusing to seed outside a round-20 overlay: {root}"
    out = {"old_fingerprint": old_fp, "new_fingerprint": new_fp}

    # cv-real: files are cv22-<lang>-<id>.json
    src, dst = eval_cv.ours_root() / old_fp, eval_cv.ours_root() / new_fp
    dst.mkdir(parents=True, exist_ok=True)
    n = {"copied": 0, "zh_left_out": 0}
    for p in sorted(src.glob("*.json")):
        if p.name == "_meta.json":
            continue
        lang = p.name.split("-")[1]
        if lang == "zh":
            n["zh_left_out"] += 1
            continue
        shutil.copy2(p, dst / p.name)
        n["copied"] += 1
    out["cv_real"] = n

    # learner suite: language from the suite manifest
    lang_of = {r["id"]: r["lang"] for r in L.suite()}
    src, dst = L.work_dir() / "ours" / old_fp, L.work_dir() / "ours" / new_fp
    dst.mkdir(parents=True, exist_ok=True)
    n = {"copied": 0, "zh_left_out": 0}
    for p in sorted(src.glob("*.json")):
        if p.name == "_meta.json":
            continue
        lang = lang_of[p.name[:-5]]
        if lang == "zh":
            n["zh_left_out"] += 1
            continue
        shutil.copy2(p, dst / p.name)
        n["copied"] += 1
    out["learners"] = n
    out["learner_langs"] = sorted(set(lang_of.values()))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
