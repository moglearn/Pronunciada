"""scripts/build_language_bundles.py: a language bundle alone is enough for score_phones.

Builds the bundle for two languages from one practice clip + one synthetic clip each, then scores
the same clips with a copy of the package code whose data folder holds ONLY that bundle (fresh
process); the outputs must equal the full installation's. Skipped without the model or the FLEURS
practice clips."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.model
@pytest.mark.parametrize("lang", ["en", "ko"])
def test_bundle_alone_scores_identically(lang, tmp_path):
    try:
        from bench.opt.common import practice_clips

        practice_clips([lang], n=1)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"practice clips not cached: {e}")
    table = tmp_path / "t.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build_language_bundles.py"), "--langs", lang, "--n", "1",
                        "--out", str(tmp_path / "b"), "--check", "--table", str(table)],
                       capture_output=True, text=True, timeout=1800, cwd=ROOT,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    row = json.loads(table.read_text())["languages"][0]
    assert row["check_identical"] and not row["unlisted_licence"]
    man = json.loads((tmp_path / "b" / lang / "manifest.json").read_text())
    names = [f["file"] for f in man["files"]]
    assert f"adapters/{lang}.npz" in names and "zipa_tokens.txt" in names
    assert not [n for n in names if n.startswith(("lex/", "cjk/"))]
