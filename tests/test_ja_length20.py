"""Round 20 (Japanese): the padded-window vowel-length head may only lower length probabilities."""
from types import SimpleNamespace

import numpy as np
import pytest


def test_lower_never_raises():
    from pronunciada.ja_length20 import lower

    rng = np.random.default_rng(0)
    for _ in range(500):
        p = float(rng.uniform(0.001, 0.999))
        zh, a, k = (float(x) for x in rng.normal(0, 4, 3))
        assert lower(p, zh, abs(a), k) <= p + 1e-12
    assert lower(0.9, 5.0, 1.0, 0.0) == pytest.approx(0.9)          # evidence above the knee: unchanged
    assert lower(0.9, -6.0, 1.0, -2.0) < 0.2                         # clear mismatch: pushed down
    assert lower(0.9, -6.0, 1.0, None) == 0.9                        # class switched off
    assert lower(0.9, -60.0, 1.0, -2.0, cap=3.0) > 0.25              # drop capped at 3 log-odds


def test_switch(monkeypatch):
    from pronunciada import ja_length20 as JL

    monkeypatch.delenv("PRONUNCIADA_JA_LENGTH20", raising=False)
    assert not JL.on()                                               # off by default
    monkeypatch.setenv("PRONUNCIADA_JA_LENGTH20", "0")
    assert not JL.on()
    monkeypatch.setenv("PRONUNCIADA_JA_LENGTH20", "1")
    monkeypatch.setenv("PRONUNCIADA_JA_ROUND19", "0")                # needs the round-19 length model
    assert not JL.on()
    monkeypatch.delenv("PRONUNCIADA_JA_ROUND19")
    assert JL.on()


def _toy():
    """Two words, a long and a short vowel, on a 40-frame clip."""
    ph = [SimpleNamespace(ipa="k", word=0), SimpleNamespace(ipa="oː", word=0),
          SimpleNamespace(ipa="t", word=1), SimpleNamespace(ipa="a", word=1)]
    words = [SimpleNamespace(phones=(0, 2), syllables=(0, 1)), SimpleNamespace(phones=(2, 4), syllables=(1, 2))]
    ref = SimpleNamespace(phones=ph, words=words)
    slots = [SimpleNamespace(start_frame=s, end_frame=e) for s, e in ((4, 6), (6, 14), (20, 22), (22, 26))]
    ev = SimpleNamespace(slots=slots)
    return ref, ev


def test_adjust_lowers_only():
    from pronunciada import ja_length20 as JL

    P = JL.params()
    if P is None:
        pytest.skip("no packaged round-20 length head")
    ref, ev = _toy()
    rng = np.random.default_rng(1)
    T = 40
    emb = {k: rng.normal(0, 1, (T, d)).astype(np.float32) for k, d in (("enc1", 256), ("enc5", 256))}
    lpa = np.log(np.full((T, 127), 1 / 127, np.float32))
    lres = {"expected": [None, "long", None, "short"], "prob": [None, 0.8, None, 0.7]}
    out = JL.adjust(ref, ev, lpa, emb, lres)
    assert out["prob"][0] is None and out["prob"][2] is None
    for i in (1, 3):
        assert out["prob"][i] <= lres["prob"][i] + 1e-12
        assert np.isfinite(out["head20"][i])
    z = JL.head_logits(ref, ev, lpa, emb, [1, 3])
    assert z.shape == (2,) and np.all(np.isfinite(z))
