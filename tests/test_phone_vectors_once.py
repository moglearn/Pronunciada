"""Scorer.score computes the per-sound encoder vectors once per call, however many readers."""
import numpy as np
import pytest


@pytest.mark.model
def test_phone_vectors_computed_once(monkeypatch):
    from pronunciada import sound_emb as SE
    from pronunciada.scorer import Scorer

    calls = []
    orig = SE.phone_vectors

    def counting(emb, ev, n):
        calls.append(n)
        return orig(emb, ev, n)

    monkeypatch.setattr(SE, "phone_vectors", counting)
    sc = Scorer("ru", precision="int8", threads=1)
    wav = (0.05 * np.random.default_rng(0).standard_normal(16000 * 2)).astype(np.float32)
    res = sc.score(wav, "Привет, как дела?")
    assert len(calls) == 1                      # contrast heads and identity evidence share them
    assert res.words and res._pronunciada["identity_s"] is not None
