"""Round 19: Korean sound-identity evidence and word factor."""
import numpy as np

from pronunciada import ko_identity as KI
from pronunciada.text import get_g2p


def test_classes_cover_korean_sounds():
    ref = get_g2p("ko")("학교에 갔다. 저기 곳이 있다")
    cls = KI.classes(ref)
    assert len(cls) == len(ref.phones)
    assert "ʌ" in cls and "o" in cls                   # 저 / 곳
    assert any(c == "k*" for c in cls)                 # 학교 -> 학꾜 (tensification)
    assert all(c is None or c in KI._head()["index"] for c in cls)


def test_evidence_and_factor_shapes():
    ref = get_g2p("ko")("그래서 어제는 걷기에도")
    V = np.random.default_rng(0).standard_normal((len(ref.phones), 1024)).astype(np.float32)
    s = KI.evidence(ref, V)
    assert s.shape == (len(ref.phones),)
    cal = KI._word_params()
    assert KI.word_factor(cal, [np.nan], 0.5) is None
    hi = KI.word_factor(cal, [50.0], 0.5)
    lo = KI.word_factor(cal, [-50.0], 0.5)
    assert hi == 1.0 and abs(lo - cal["floor"]) < 1e-9


def test_switch(monkeypatch):
    monkeypatch.setenv("PRONUNCIADA_KO_IDENTITY", "0")
    assert not KI.enabled()
    monkeypatch.setenv("PRONUNCIADA_KO_IDENTITY", "1")
    assert KI.enabled()
