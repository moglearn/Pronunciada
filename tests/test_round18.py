"""Round 18: Korean lax / tense marker and evidence, word-level contrast factor."""
import numpy as np
import pytest

from pronunciada import contrast_heads as CH
from pronunciada import contrast_word as CW
from pronunciada.text import get_g2p


def test_korean_tense_marker_from_spelling_and_rules():
    g = get_g2p("ko")
    for text, tense in (("달", True), ("딸", True), ("자다", True), ("짜다", True), ("사다", True), ("싸다", True)):
        ref = g(text)
        assert ref.phones[0].tense is (text in ("딸", "짜다", "싸다")), text
    a, b = g("달"), g("딸")
    assert [p.ipa for p in a.phones] == [p.ipa for p in b.phones]       # same symbols: scoring unchanged
    assert [p.tense for p in g("학교").phones][3] is True                 # post-obstruent tensification
    assert all(p.tense is None for p in get_g2p("en")("the cat").phones)


def test_tense_evidence_follows_the_flag():
    ipa = ["t", "a", "t", "a"]
    V = np.random.default_rng(1).standard_normal((4, 1024)).astype(np.float32)
    zt = CH.evidence("ko", ipa, V, tense=[True, None, False, None])["tense"]
    zf = CH.evidence("ko", ipa, V, tense=[False, None, True, None])["tense"]
    assert np.isfinite(zt[[0, 2]]).all() and np.isnan(zt[[1, 3]]).all()
    e = [x for x in CH.entries("ko") if x["contrast"] == "tense"][0]
    # flipping the expected class mirrors the evidence around s0
    assert np.allclose(zt[0] + zf[0], -2 * e["slope"] * e["s0"])


def test_word_factor_shape():
    cal = {"knee": [-3.0, -2.0, -1.0], "width": 4.0, "floor": 0.4}
    assert CW.word_factor(cal, [np.nan, np.nan], 0.5) is None
    assert CW.word_factor(cal, [0.0, 5.0], 0.5) == 1.0
    assert CW.word_factor(cal, [-6.0], 0.5) == pytest.approx(0.4)
    assert CW.word_factor(cal, [-4.0], 0.5) == pytest.approx(1 - 0.6 * 0.5)
    assert CW.word_factor(cal, [-2.5], 0.0) == 1.0 and CW.word_factor(cal, [-2.5], 1.0) < 1.0
