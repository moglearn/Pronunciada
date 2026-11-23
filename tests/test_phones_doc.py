"""docs/phones.json and docs/PHONES.md are what scripts/export_phones.py generates now."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_phone_docs_are_current():
    for mod in ("jieba", "pypinyin", "pyopenjtalk"):
        pytest.importorskip(mod)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PRONUNCIADA_R20_")}
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run([sys.executable, str(ROOT / "scripts/export_phones.py"), "--check"], capture_output=True,
                         text=True, env=env, cwd=ROOT, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr[-2000:]
