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


def test_class_follows_edited_reference_sound():
    import copy

    ref = get_g2p("ko")("저기 곳이 있다")
    cls = KI.classes(ref)
    i = next(k for k, p in enumerate(ref.phones) if p.is_nucleus and p.ipa == "ɔ")    # 저
    assert cls[i] == "ʌ"
    ed = copy.deepcopy(ref)
    ed.phones[i].ipa = "o"                         # a caller's reference now demands o here
    assert KI.classes(ed)[i] == "o"
    assert KI.classes(ed)[:i] + KI.classes(ed)[i + 1:] == cls[:i] + cls[i + 1:]


def test_evidence_and_factor_shapes():
    ref = get_g2p("ko")("그래서 어제는 걷기에도")
    V = np.random.default_rng(0).standard_normal((len(ref.phones), 1024)).astype(np.float32)
    s = KI.evidence(ref, V)
    assert s.shape == (len(ref.phones),)
    cal = {"form": "mult", "knee": [-1.0, 0.0, 1.0], "width": 4.0, "floor": 0.5}
    assert KI.word_factor(cal, [np.nan], 0.5) is None
    hi = KI.word_factor(cal, [50.0], 0.5)
    lo = KI.word_factor(cal, [-50.0], 0.5)
    assert hi == 1.0 and abs(lo - cal["floor"]) < 1e-9


def test_word_deltas_shipped_form():
    from pronunciada.align import score_reference

    cal = KI._word_params()
    assert cal["form"] == "delta" and len(cal["coef"]) == 8
    ref = get_g2p("ko")("그래서 어제는 걷기에도")
    rng = np.random.default_rng(1)
    lp = rng.standard_normal((90, 127))
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    ev = score_reference(lp, ref, insertion=True)
    assert ev.llr_ins is not None and len(ev.llr_ins) == len(ref.phones) and np.isnan(ev.llr_ins[-1])
    assert np.all(ev.llr_ins[:-1] <= 0)
    V = rng.standard_normal((len(ref.phones), 1024)).astype(np.float32)
    d = KI.word_deltas(ref, ev, KI.evidence(ref, V), cal)
    assert d.shape == (len(ref.words),) and np.all(d <= cal["cap_up"] + 1e-9)


def test_switch(monkeypatch):
    monkeypatch.setenv("PRONUNCIADA_KO_IDENTITY", "0")
    assert not KI.enabled()
    monkeypatch.setenv("PRONUNCIADA_KO_IDENTITY", "1")
    assert KI.enabled()
