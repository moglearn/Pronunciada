"""Round 20: tone-aware raise (Mandarin) and the extra-sound no-raise gate of pronunciada/identity.py."""
import math

import numpy as np

from pronunciada import identity as ID

STEP = {"form": "step", "coef": [0.0, 0.0, 0.5, 0.0, 0.5], "cap": 3.0, "no_raise_below": -2.0}


def _lo(p):
    return math.log(p / (1 - p))


def test_round19_behaviour_without_round20_keys():
    v = [3.0, 4.0]
    p = ID.apply_word(STEP, v, 0.5, 0.5)
    assert p > 0.5
    # the new inputs change nothing unless the rule asks for them
    assert ID.apply_word(STEP, v, 0.5, 0.5, tone=0.1, contrast=0.5, cls=["a", "b"], ins=-9.0) == p


def test_tone_ramp_scales_only_a_raise():
    rule = dict(STEP, tone_ramp=[0.2, 0.98])
    up = ID.apply_word(STEP, [3.0, 4.0], 0.5, 0.5)
    assert ID.apply_word(rule, [3.0, 4.0], 0.5, 0.5, tone=0.1) == 0.5          # weak tone: no raise
    assert ID.apply_word(rule, [3.0, 4.0], 0.5, 0.5, tone=0.99) == up          # confident tone: full raise
    mid = ID.apply_word(rule, [3.0, 4.0], 0.5, 0.5, tone=0.59)
    assert abs(_lo(mid) - 0.5 * _lo(up)) < 1e-9                               # half way: half the step
    low = dict(STEP, coef=[1.0, 0.0, 0.0, 0.0, 0.0])
    assert ID.apply_word(dict(low, tone_ramp=[0.2, 0.98]), [-3.0], 0.5, 0.5, tone=0.1) == \
        ID.apply_word(low, [-3.0], 0.5, 0.5)                                    # a lowering is untouched


def test_tone_gate_and_segmental():
    rule = dict(STEP, tone_gate=0.5)
    assert ID.apply_word(rule, [3.0, 4.0], 0.5, 0.5, tone=0.4) == 0.5
    seg = dict(STEP, on="segmental")
    p = ID.apply_word(seg, [3.0, 4.0], 0.3, 0.5, sounds=0.8, tone=0.5, contrast=0.75)
    d = _lo(ID.apply_word(STEP, [3.0, 4.0], 0.5, 0.5))
    assert abs(p - 0.3 * ID._step(0.6, d) / 0.6) < 1e-12


def test_extra_sound_gate():
    rule = dict(STEP, no_raise_ins_below=-3.0)
    up = ID.apply_word(STEP, [3.0, 4.0], 0.5, 0.5)
    assert ID.apply_word(rule, [3.0, 4.0], 0.5, 0.5, ins=-5.0) == 0.5
    assert ID.apply_word(rule, [3.0, 4.0], 0.5, 0.5, ins=-1.0) == up
    assert ID.apply_word(rule, [3.0, 4.0], 0.5, 0.5, ins=None) == up
    assert ID.word_ins(np.array([0.0, -4.0, np.nan, -1.0]), 1, 3) == -4.0
    assert ID.word_ins(None, 0, 2) is None


def test_class_no_raise():
    rule = dict(STEP, no_raise_cls={"a": 5.0})
    assert ID.apply_word(rule, [3.0, 6.0], 0.5, 0.5, cls=["a", "b"]) == 0.5
    assert ID.apply_word(rule, [6.0, 3.0], 0.5, 0.5, cls=["a", "b"]) > 0.5


def test_zh_rule_and_switch(monkeypatch):
    monkeypatch.delenv("PRONUNCIADA_IDENTITY_ZH_R20", raising=False)
    r = ID.rule("zh")
    assert r["tone_ramp"] == [0.2, 0.98] and "round20" not in r
    monkeypatch.setenv("PRONUNCIADA_IDENTITY_ZH_R20", "0")
    r0 = ID.rule("zh")
    assert "tone_ramp" not in r0 and r0 is ID._cal("zh")
    assert not ID.needs_insertion("zh")
