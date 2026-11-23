"""Round 20 false-alarm release check, step 3: the overlay for the one held-out run.

    python -m bench.round20_fa_rc.overlay <main overlay> <new overlay> <moved-aside dir> <rc fingerprint> <main fingerprint>

<main overlay> is the overlay that holds main's held-out outputs (the round-20 release check's copy of the
round-19 frozen-run overlay). It is COPIED (symbolic links stay links); nothing in it is changed. In the copy:
  * the outputs the release candidate can change are moved aside (to <moved-aside dir>), so the run computes
    them fresh: everything English, German, Spanish, French and Korean (the FLEURS test / swap pickles, the
    English learner-set pickles, the coaching dumps of the English learner sets, the contrast-audit chunks of
    the five languages with their edit plans kept, Pronunciada's answers on Lingolix's test 1 for the five
    languages and on test 2);
  * the per-clip caches keyed by the package fingerprint (cv-real, learner suite, OMPAL) are seeded under the
    release candidate's fingerprint with main's outputs of it, pt, ru, ja and zh only (their outputs are
    identical: bench/round20_fa_rc/run_equiv.sh); the five languages are scored fresh.
Japanese, Italian, Portuguese, Russian and Mandarin outputs stay as they are, and so do the steps that only
have rows for them (OMPAL scoring, the Japanese length table, Pronunciada's answers on Lingolix's Japanese test).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

FIVE = ("en", "de", "es", "fr", "ko")


def main() -> None:
    src, dst, aside, new_fp, old_fp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
    assert "round20" in str(dst) and "farc" in str(dst), dst
    assert not dst.exists(), f"{dst} exists"
    subprocess.run(["cp", "-a", str(src), str(dst)], check=True)          # -a: links stay links
    aside.mkdir(parents=True, exist_ok=True)
    moved = []

    def move(rel: str) -> None:
        p = dst / rel
        if p.is_symlink():
            p.unlink()
            moved.append(rel + " (link removed)")
            return
        if p.exists():
            q = aside / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(q))
            moved.append(rel)

    for lang in FIVE:
        move(f"calib/api/fleurs-test-{lang}.pkl")
        move(f"calib/api/fleurs-swap-{lang}.pkl")
        for p in sorted((dst / "audit" / "contrast_v1_test" / lang).glob("chunk_*.pkl")):
            move(str(p.relative_to(dst)))
        move(f"third_party/lingolix/responses/pronunciada/tier1_{lang}.jsonl")
    for rel in ("calib/api/speechocean762-test-fp32.pkl", "calib/api/speechocean762-test-int8.pkl",
                "calib/api/l2arctic-test-fp32.pkl", "calib/api/epadb-test-fp32.pkl",
                "third_party/lingolix/responses/pronunciada/tier2.jsonl",
                "bench/coach/dump-so762-test.pkl", "bench/coach/dump-r4-so762-test.pkl",
                "bench/coach/dump-r4-l2arctic-test.pkl", "bench/coach/dump-r4-epadb-test.pkl"):
        move(rel)

    seeded = {}
    # cv-real: cv22-<lang>-<id>.json
    s, d = dst / "cv-real" / "ours" / old_fp, dst / "cv-real" / "ours" / new_fp
    d.mkdir(parents=True, exist_ok=True)
    n = {"copied": 0, "left_out": 0}
    for p in sorted(s.glob("*.json")):
        if p.name == "_meta.json":
            continue
        if p.name.split("-")[1] in FIVE:
            n["left_out"] += 1
            continue
        shutil.copy2(p, d / p.name)
        n["copied"] += 1
    seeded["cv_real"] = n
    # learner suite: language from the suite manifest (read from the repository's bench code)
    from bench.learners import evaluate as L

    lang_of = {r["id"]: r["lang"] for r in L.suite()}
    s, d = dst / "round19" / "evalsuite" / "ours" / old_fp, dst / "round19" / "evalsuite" / "ours" / new_fp
    d.mkdir(parents=True, exist_ok=True)
    n = {"copied": 0, "left_out": 0}
    for p in sorted(s.glob("*.json")):
        if p.name == "_meta.json":
            continue
        if lang_of[p.name[:-5]] in FIVE:
            n["left_out"] += 1
            continue
        shutil.copy2(p, d / p.name)
        n["copied"] += 1
    seeded["learners"] = n
    seeded["learner_langs"] = sorted(set(lang_of.values()))
    # OMPAL (Mandarin only): every clip
    for s in sorted((dst / "ompal" / "ours").glob(f"*__{old_fp}")):
        d = s.parent / s.name.replace(old_fp, new_fp)
        if not d.exists():
            shutil.copytree(s, d)
        seeded["ompal"] = {"dir": d.name, "files": len(list(d.glob("*")))}
    out = {"from": src.name, "moved_aside": moved, "seeded": seeded, "old_fingerprint": old_fp,
           "new_fingerprint": new_fp}
    (aside / "overlay.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (len(v) if k == "moved_aside" else v) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
