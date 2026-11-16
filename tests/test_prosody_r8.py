"""Round 8: selectable stress / pitch-accent candidates load through the public Scorer with their
own channel calibrator and score a clip; defaults are unchanged."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pronunciada.prosody import DATA

ZIPA = Path(os.environ.get("PRONUNCIADA_CACHE", os.path.expanduser("~/.cache/pronunciada"))) / "models" / \
    "zipa-small-crctc-500k" / "model.onnx"

CASES = [  # env var, model stem, language, `say` voice, text
    ("PRONUNCIADA_STRESS_MODEL", "stress_v3", "en", "Samantha",
     "The government announced a remarkable economic recovery yesterday."),
    ("PRONUNCIADA_PITCH_MODEL", "pitch_ja_v3", "ja", "Kyoko", "今日はとても良い天気ですね。"),
    ("PRONUNCIADA_PITCH_MODEL", "pitch_ja_v3r", "ja", "Kyoko", "今日はとても良い天気ですね。"),
]


def test_defaults_unchanged():
    from pronunciada import prosody

    assert prosody.STRESS_MODEL == "stress" and prosody.PITCH_MODEL == "pitch_ja"
    assert prosody.selected_model("en") == "stress" or os.environ.get("PRONUNCIADA_STRESS_MODEL")


@pytest.mark.parametrize("env,stem,lang,voice,text", CASES)
def test_round8_candidate_through_scorer(tmp_path, monkeypatch, env, stem, lang, voice, text):
    if not (DATA / f"{stem}.onnx").exists():
        pytest.skip(f"{stem} not exported")
    if not ZIPA.exists() or shutil.which("say") is None:
        pytest.skip("needs the cached ZIPA model and macOS `say`")
    out = tmp_path / "say.wav"
    r = subprocess.run(["say", "-v", voice, "-o", str(out), "--data-format=LEI16@16000", text], capture_output=True)
    if r.returncode != 0:
        pytest.skip("voice unavailable")
    monkeypatch.setenv(env, stem)
    from pronunciada.scorer import Scorer

    sc = Scorer(lang, threads=2, experimental=True)
    assert sc.prosody is not None and sc.prosody.meta["model"]
    if sc.prosody.inputs.get("zemb"):
        assert sc.am.embeddings
    assert sc.pros_cal and sc.pros_cal.get("prosody_model") == stem, "candidate without its own calibrator"
    res = sc.score(str(out), text)
    assert not any("prosody failed" in w for w in res.warnings)
    field = "stress_prob" if lang != "ja" else "pitch_prob"
    vals = [getattr(s, field) for w in res.words for s in w.syllables if getattr(s, field) is not None]
    assert vals and all(0 <= v <= 1 for v in vals)
    assert all(0 <= w.prob <= 1 for w in res.words)
