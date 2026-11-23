"""Round 20 false-alarm release check, step 4: the overlay of the final per-language table (no scoring).

    python -m bench.round20_fa_rc.final_overlay <rc overlay> <new overlay> <main outputs moved aside> \
        <main tier2x answers> <final fingerprint> <rc fingerprint> <main fingerprint> <languages kept on>

The final table switches off every language of the held-out run that failed the rule; their outputs are
then main's (identity: bench/round20_fa_rc/run_equiv_final.sh). This copies the release candidate's overlay
and puts back main's per-clip outputs of every language that is off (the files moved aside before the run,
and main's answers on all 2,500 speechocean762 test clips when English is off). The fingerprint-keyed caches
are seeded under the final fingerprint: the languages kept on from the release candidate's cache, every
other language from main's. Nothing is scored; the report steps then rebuild the tables from these outputs.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

FIVE = ("en", "de", "es", "fr", "ko")


def main() -> None:
    src, dst, aside, t2x = (Path(x) for x in sys.argv[1:5])
    fin_fp, rc_fp, main_fp = sys.argv[5:8]
    keep = tuple(x for x in sys.argv[8].split(",") if x)
    off = [x for x in FIVE if x not in keep]
    assert "farc" in str(dst) and not dst.exists(), dst
    subprocess.run(["cp", "-a", str(src), str(dst)], check=True)
    log = json.loads((aside / "overlay.json").read_text())
    restored = []
    for rel in log["moved_aside"]:
        if rel.endswith("(link removed)"):
            continue
        p = Path(rel)
        name = p.name
        lang = None
        for x in FIVE:
            if (f"-{x}.pkl" in name or f"_{x}.jsonl" in name or f"/{x}/" in f"/{rel}"):
                lang = x
        en_only = lang is None                        # English learner sets, test 2, English coaching dumps
        if (lang in off) or (en_only and "en" in off):
            q = dst / rel
            if q.exists() or q.is_symlink():
                q.unlink()
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(aside / rel, q)
            restored.append(rel)
    if "en" in off:
        q = dst / "third_party/lingolix/responses/pronunciada/tier2x.jsonl"
        q.unlink()
        shutil.copy2(t2x, q)
        restored.append("third_party/lingolix/responses/pronunciada/tier2x.jsonl (main)")
    seeded = {}
    from bench.learners import evaluate as L

    lang_of = {r["id"]: r["lang"] for r in L.suite()}
    for name, base, lang_fn in (
            ("cv_real", dst / "cv-real" / "ours", lambda f: f.split("-")[1]),
            ("learners", dst / "round19" / "evalsuite" / "ours", lambda f: lang_of[f[:-5]])):
        d = base / fin_fp
        d.mkdir(parents=True, exist_ok=True)
        n = {"from_rc": 0, "from_main": 0}
        for fp, key in ((rc_fp, "from_rc"), (main_fp, "from_main")):
            for p in sorted((base / fp).glob("*.json")):
                if p.name == "_meta.json":
                    continue
                lg = lang_fn(p.name)
                if (lg in keep) == (key == "from_rc"):
                    shutil.copy2(p, d / p.name)
                    n[key] += 1
        seeded[name] = n
    for s in sorted((dst / "ompal" / "ours").glob(f"*__{main_fp}")):
        d = s.parent / s.name.replace(main_fp, fin_fp)
        if not d.exists():
            shutil.copytree(s, d)
        seeded["ompal"] = d.name
    out = {"kept_on": keep, "off": off, "restored_from_main": restored, "seeded": seeded,
           "fingerprints": {"final": fin_fp, "rc": rc_fp, "main": main_fp}}
    (dst / "final_overlay.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({**out, "restored_from_main": len(restored)}, indent=1))


if __name__ == "__main__":
    main()
