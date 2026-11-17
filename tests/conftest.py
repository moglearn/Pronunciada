import os
import shutil
import subprocess

import numpy as np
import pytest

MODEL_NAME = "zipa-small-crctc-500k"

# Timing budgets are set on a laptop; shared CI runners are several times slower and noisy.
TIME_SCALE = 5.0 if os.environ.get("CI") else 1.0


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs the cached ZIPA model (skipped otherwise)")
    config.addinivalue_line("markers", "network: needs network access")


def _model_dir():
    root = os.environ.get("PRONUNCIADA_CACHE") or os.path.expanduser("~/.cache/pronunciada")
    return os.path.join(root, "models", MODEL_NAME)


def model_available() -> bool:
    d = _model_dir()
    return all(os.path.exists(os.path.join(d, f)) for f in ("model.onnx", "tokens.txt"))


def pytest_collection_modifyitems(config, items):
    if model_available():
        return
    skip = pytest.mark.skip(reason=f"ZIPA model not cached at {_model_dir()}")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def acoustic_model():
    from pronunciada.acoustic import AcousticModel

    return AcousticModel(MODEL_NAME, threads=4)


@pytest.fixture(scope="session")
def tts(tmp_path_factory):
    """tts(text) -> float32 16 kHz waveform synthesised with macOS `say` + ffmpeg."""
    if not (shutil.which("say") and shutil.which("ffmpeg")):
        pytest.skip("needs macOS `say` and ffmpeg")
    from pronunciada.audio import load_audio

    d = tmp_path_factory.mktemp("tts")
    cache: dict[str, np.ndarray] = {}

    def make(text: str) -> np.ndarray:
        if text not in cache:
            k = len(cache)
            aiff, wav = d / f"u{k}.aiff", d / f"u{k}.wav"
            subprocess.run(["say", "-o", str(aiff), text], check=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff), "-ar", "16000",
                            "-ac", "1", str(wav)], check=True)
            cache[text] = load_audio(str(wav))
        return cache[text]

    return make
