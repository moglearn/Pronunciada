"""Round 9 prompt-match channel (pronunciada/match.py) and its use in the Scorer."""
import numpy as np
import pytest

from pronunciada import match as M
from pronunciada.text.ipa import to_ids


class _Ph:
    def __init__(self, ipa):
        self.ipa = ipa


class _Ref:
    def __init__(self, ipas):
        self.phones = [_Ph(x) for x in ipas]


def _lp_for(ipas, frames_per=3, gap=2, V=127):
    """Peaky log-posteriors whose greedy decode is exactly `ipas` (blank = id 0)."""
    seq = []
    for x in ipas:
        for t in to_ids(x):
            seq += [t] * frames_per + [0] * gap
    lp = np.full((len(seq) + 4, V), -20.0)
    lp[:, 0] = 0.0
    for i, t in enumerate(seq):
        lp[i, :] = -20.0
        lp[i, t] = 0.0
    return lp


def test_evidence_perfect_and_empty():
    ipas = ["k", "a", "t", "a"]
    ref = _Ref(ipas)
    e = M.evidence(_lp_for(ipas), ref, None, {"llr_sub": np.zeros(4)})
    assert e["ins_rate"] == e["sub_rate"] == e["del_rate"] == 0.0
    assert e["tok_ratio"] == pytest.approx(1.0)
    silent = np.full((50, 127), -20.0)
    silent[:, 0] = 0.0
    e = M.evidence(silent, ref, None, {"llr_sub": np.full(4, -5.0)})
    assert e["del_rate"] == 1.0 and e["against_share"] == 1.0 and e["tok_ratio"] == pytest.approx(0.2)
    x = M.inputs(e, [0.1, 0.2, 0.9, None])
    assert x[M.INPUTS.index("absent_share")] == pytest.approx(2 / 3)
    assert x[M.INPUTS.index("short_tokens")] == pytest.approx(-np.log(0.2))


def test_inputs_clipped_and_prob_monotone():
    e = {"ins_rate": 9.0, "sub_rate": -1.0, "del_rate": 0.5, "against_share": 0.2, "tok_ratio": 1e-9}
    x = M.inputs(e, [])
    for k, v in zip(M.INPUTS, x):
        assert M.CLIPS[k][0] <= v <= M.CLIPS[k][1]
    cal = {"coef": [4.0] + [-1.0] * len(M.INPUTS)}
    base = np.zeros(len(M.INPUTS))
    p0 = M.prob(cal, base)
    for i in range(len(M.INPUTS)):
        xi = base.copy()
        xi[i] = 0.5
        assert M.prob(cal, xi) < p0


def test_shipped_coefficients_are_monotone():
    import json

    from pronunciada.calibrate import data_dir

    p = data_dir() / "match.json"
    if not p.exists():
        pytest.skip("match channel not shipped")
    cal = json.loads(p.read_text())
    assert cal["inputs"] == list(M.INPUTS)
    assert all(c <= 0 for c in cal["coef"][1:])


@pytest.mark.model
def test_scorer_multiplies_match_and_leaves_words_alone(tts):
    import pronunciada

    text = "The quick brown fox jumps over the lazy dog"
    wav = tts(text)
    on = pronunciada.Scorer("en", threads=2)
    off = pronunciada.Scorer("en", threads=2, match=False)
    a, b = on.score((wav, 16000), text), off.score((wav, 16000), text)
    assert b.match_prob is None
    assert [(w.prob, [p.prob for p in w.phones]) for w in a.words] == \
        [(w.prob, [p.prob for p in w.phones]) for w in b.words]
    if on.match_cal is None:
        assert a.prob == b.prob and a.match_prob is None
    else:
        assert 0 < a.match_prob <= 1
        assert a.prob == pytest.approx(b.prob * a.match_prob)
        assert on.rescore(a, 0.5).prob == pytest.approx(a.prob)
