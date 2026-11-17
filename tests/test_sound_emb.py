"""Round 12b: per-sound activation inputs (English detail sound model) and the Mandarin
consonant / vowel coaching score."""
import numpy as np
import pytest

from pronunciada import calibrate as C
from pronunciada import coach as Co
from pronunciada import coach_zh as CZ
from pronunciada import sound_emb as SE


class _Slot:
    def __init__(self, a, b):
        self.start_frame, self.end_frame = a, b


class _Ev:
    def __init__(self, spans):
        self.slots = [_Slot(a, b) for a, b in spans]


def test_phone_vectors_pool_with_one_frame_margin():
    T = 10
    emb = {"enc1": np.arange(T, dtype=np.float32)[:, None] * np.ones((1, 256), np.float32),
           "enc3": np.zeros((T, 512), np.float32), "enc5": np.ones((T, 256), np.float32)}
    V = SE.phone_vectors(emb, _Ev([(2, 4), (5, 5), (9, 10)]), 3)
    assert V.shape == (3, 1024)
    assert V[0, 0] == pytest.approx(np.mean([1, 2, 3, 4]))     # frames 1..4 (span 2..3 widened)
    assert V[1, 0] == pytest.approx(np.mean([3, 4, 5, 6]))     # empty span: 4..5, widened
    assert V[2, 0] == pytest.approx(np.mean([8, 9]))           # clipped at the end
    assert np.all(V[:, 300] == 0) and np.all(V[:, 900] == 1)


def test_english_detail_model_shipped_and_wired():
    cfg = C.config()
    assert cfg["sound_detail_model"]["en"] == "sound_E"
    assert C.sound_model_for("en")[0] == "sound_B"             # shown probabilities unchanged
    aux = SE.load("sound_E")
    assert aux is not None and aux.pW.shape == (1024,)
    m = C.load_level("sound_E")
    assert set(SE.SoundEmb.COLS) <= set(m.num)
    assert SE.load("sound_B") is None


def test_augment_without_activations_gives_nan_columns():
    aux = SE.load("sound_E")

    class R:
        phones = [object(), object()]
    tab = {}
    aux.augment(tab, R(), None, None)
    assert all(np.isnan(tab[c]).all() and len(tab[c]) == 2 for c in SE.SoundEmb.COLS)


def test_coach_language_specific_logistic():
    mdl = Co.model()
    assert "sound_model" in mdl["by_lang"]["en"]
    n = 5
    X = {k: np.zeros(n) for k in Co.SOUND_FEATURES}
    X["zl"] = X["zs"] = np.linspace(-6, 6, n)
    a, b = Co.error_prob(X, mdl, "en"), Co.error_prob(X, mdl, "es")
    assert a.shape == b.shape == (n,)
    assert not np.allclose(a, b)                               # en uses its own (sound_E) logistic
    assert np.all(np.diff(b) <= 0)                             # monotone: higher log-odds, lower score


def test_zh_score_falls_back_without_vectors():
    assert CZ.params() is not None
    ep = np.array([0.1, 0.5, 0.9])
    X = {"phone_vec": None, "expected": ["p", "a", "n"], "alts": [[], [], []], "initial": np.array([1, 0, 0], bool)}
    assert np.array_equal(CZ.error_prob(X, ep), ep)
    X["phone_vec"] = np.zeros((3, 1024), np.float32)
    out = CZ.error_prob(X, ep)
    assert out.shape == (3,) and np.all((out > 0) & (out < 1))
