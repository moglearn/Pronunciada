"""Core requirements match what the package imports."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _imported_top_level() -> set[str]:
    mods = set()
    for p in (ROOT / "pronunciada").rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    return mods


def test_scipy_not_imported_and_not_a_core_requirement():
    tomllib = pytest.importorskip("tomllib")
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {d.split(">")[0].split("=")[0].split("<")[0].strip().lower() for d in deps}
    assert "scipy" not in _imported_top_level()
    assert "scipy" not in names
    # every core requirement is imported somewhere in the package
    imported = {m.lower().replace("_", "-") for m in _imported_top_level()}
    assert names <= imported, names - imported


def test_scoring_import_does_not_load_the_http_stack():
    import subprocess
    import sys

    code = ("import sys, pronunciada.scorer, pronunciada.acoustic; "
            "print('urllib.request' in sys.modules or 'http.client' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert out.stdout.strip() == "False", out.stderr
