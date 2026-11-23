"""scripts/export_sherpa_onnx.py and scripts/verify_sherpa_onnx.py.

The sherpa-onnx check runs only when an interpreter with sherpa-onnx is available
($PRONUNCIADA_SHERPA_PYTHON, or the default packaging virtualenv) and the export folder exists;
otherwise it is skipped (sherpa-onnx is not a dependency of this package)."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_metadata_entries_roundtrip(tmp_path):
    ex = _mod("export_sherpa_onnx")
    p = tmp_path / "m.onnx"
    p.write_bytes(ex.metadata_entry("model_type", "zipformer2_ctc") + ex.metadata_entry("license", "CC-BY-4.0"))
    assert ex.read_metadata(p) == {"model_type": "zipformer2_ctc", "license": "CC-BY-4.0"}


def _sherpa_python():
    p = os.environ.get("PRONUNCIADA_SHERPA_PYTHON") or os.path.expanduser(
        "~/.cache/pronunciada/opt/venv_mobile/bin/python")
    if not os.path.exists(p):
        return None
    r = subprocess.run([p, "-I", "-c", "import sherpa_onnx"], capture_output=True)
    return p if r.returncode == 0 else None


@pytest.mark.model
def test_exported_folder_decodes_like_the_repo(tmp_path):
    ex = _mod("export_sherpa_onnx")
    d = Path(os.environ.get("PRONUNCIADA_SHERPA_DIR", ex.DEFAULT_OUT))
    py = _sherpa_python()
    if py is None or not (d / "model.int8.onnx").exists():
        pytest.skip("sherpa-onnx interpreter or export folder not available")
    try:
        from bench.opt.common import practice_clips

        practice_clips(["en"], n=1)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"practice clips not cached: {e}")
    meta = ex.read_metadata(d / "model.int8.onnx")
    assert meta["model_type"] == "zipformer2_ctc" and meta["license"] == "CC-BY-4.0"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/verify_sherpa_onnx.py"), "--sherpa-python", py,
                        "--dir", str(d), "--langs", "en,ko,zh", "--n", "1", "--models", "model.int8.onnx"],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "all identical: True" in r.stdout
