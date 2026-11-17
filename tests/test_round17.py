"""Round 17: head-based contrast evidence (pronunciada.contrast_heads) and the Japanese length
channel's head input (pronunciada.length.head_combine)."""
import numpy as np
import pytest

from pronunciada import contrast_heads as CH
from pronunciada import length as L


def test_config_languages_and_switch(monkeypatch):
    langs = {e["contrast"] for lang in ("ko", "ru", "zh") for e in CH.entries(lang)}
    assert langs == {"aspiration", "hard_soft", "tense"}          # tense: round 18 (Korean)
    for lang in ("en", "es", "fr", "de", "it", "pt", "ja"):
        assert CH.entries(lang) == []                     # other languages untouched
    monkeypatch.setenv("PRONUNCIADA_HEAD_CAPS", "0")
    assert CH.entries("ko") == []


def test_evidence_only_on_members():
    rng = np.random.default_rng(0)
    ipa = ["kʰ", "a", "k", "n", "tʰ"]
    V = rng.standard_normal((len(ipa), 1024)).astype(np.float32)
    ev = CH.evidence("ko", ipa, V)
    z = ev["aspiration"]
    assert np.isfinite(z[[0, 2, 4]]).all() and np.isnan(z[[1, 3]]).all()
    cap = CH.cap_column("ko", ev)
    assert np.allclose(cap[[0, 2, 4]], z[[0, 2, 4]]) and np.isnan(cap[1])


def test_length_head_combine_monotone_and_native_quantiles():
    par = L.params("ja")
    assert par is not None and "head" in par
    H = par["head"]["v"]
    ps = [L.head_combine(par, "v", 0.9, s) for s in (-6.0, -2.0, 0.0, 4.0)]
    assert ps == sorted(ps)                                # more "long as expected" evidence -> higher
    assert L.head_combine(par, "v", 0.9, float("nan")) == 0.9
    assert len(H["q_new"]) == len(H["q_orig"]) and H["q_new"] == sorted(H["q_new"])


@pytest.mark.model
def test_runtime_caps_only_lower_shown_probabilities(tts, monkeypatch):
    import pronunciada

    wav = tts("hello")          # any audio; scored as Korean text
    sc = pronunciada.Scorer("ko", threads=2)
    r1 = sc.score((wav, 16000), "탈 달 팔")
    monkeypatch.setenv("PRONUNCIADA_HEAD_CAPS", "0")
    sc0 = pronunciada.Scorer("ko", threads=2)
    r0 = sc0.score((wav, 16000), "탈 달 팔")
    p1 = [p.prob for w in r1.words for p in w.phones]
    p0 = [p.prob for w in r0.words for p in w.phones]
    assert all(a <= b + 1e-12 for a, b in zip(p1, p0))
    assert [w.prob for w in r1.words] == [w.prob for w in r0.words]     # words untouched
