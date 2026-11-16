"""Prepared default switch: word stress `stress` -> `stress_v3` (round 8; decided on
speechocean762 TRAIN, results/round8_prosody.md section 5). NOT run yet -- run it once the Russian
job that depends on the current default has finished:

    .venv/bin/python -m bench.prosody.switch_stress_v3 [--dry-run]

Touches exactly:
  pronunciada/prosody/__init__.py                       STRESS_MODEL = "stress" -> "stress_v3"
  pronunciada/data/calib/prosody/<lang>.json            -> copied to <lang>.stress.json (old model keeps its calibrator)
  pronunciada/data/calib/prosody/<lang>.stress_v3.json  -> becomes <lang>.json (file removed)
for lang in en de es it pt ru. stress.onnx stays (selectable: PRONUNCIADA_STRESS_MODEL=stress).
If ru.json was re-fitted after ru.stress_v3.json (Russian front-end change), re-fit the latter
first: PRONUNCIADA_STRESS_MODEL=stress_v3 python -m bench.calib.fit_prosody --lang ru.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANGS = ("en", "de", "es", "it", "pt", "ru")


def main(dry=False):
    init = ROOT / "pronunciada" / "prosody" / "__init__.py"
    s = init.read_text()
    assert s.count('STRESS_MODEL = "stress"\n') == 1, "default already switched?"
    cal = ROOT / "pronunciada" / "data" / "calib" / "prosody"
    plan = []
    for lg in LANGS:
        cur, new = cal / f"{lg}.json", cal / f"{lg}.stress_v3.json"
        old = json.loads(cur.read_text())
        nj = json.loads(new.read_text())
        assert nj.get("prosody_model") == "stress_v3" and nj["prosody_model_meta"]["model"] == "stress_v3"
        assert "prosody_model" not in old and old["prosody_model_meta"]["model"] == "stress_multi"
        plan.append((lg, cur, new, old, nj))
    print("will edit", init.relative_to(ROOT))
    for lg, cur, new, old, nj in plan:
        print(f"  {cur.name} -> {lg}.stress.json; {new.name} -> {cur.name}")
    if dry:
        return
    for lg, cur, new, old, nj in plan:
        (cal / f"{lg}.stress.json").write_text(json.dumps({**old, "prosody_model": "stress"}, indent=1))
        nj.pop("prosody_model")
        cur.write_text(json.dumps(nj, indent=1))
        new.unlink()
    init.write_text(s.replace('STRESS_MODEL = "stress"\n', 'STRESS_MODEL = "stress_v3"\n'))
    print("switched")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
