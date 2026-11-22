"""Round 19 rollout: the generic sound-identity module (pronunciada/identity.py)."""
import numpy as np

from pronunciada import identity as ID


class _P:
    def __init__(self, ipa, alts=()):
        self.ipa, self.alts = ipa, list(alts)


class _R:
    def __init__(self, phones):
        self.phones = phones


def test_classes_merge_skip_and_allowed_variants():
    spec = {"merge": {"l̴": "l"}, "skip": ["ə"], "skip_deletable": -2.0}
    ref = _R([_P("l̴"), _P("ə"), _P("t", [("ɾ", -1.0), ("d", -1.5)]), _P("d", [("", -0.7)]), _P("s", [("", -3.5)])])
    cls, allow = ID.classes(ref, spec)
    assert cls == ["l", None, "t", None, "s"]          # merged; schwa skipped; often-dropped d not judged
    assert allow[2] == {"t", "ɾ", "d"}


def test_word_rules():
    knee = {"form": "knee", "knee": [-6.0, -5.0, -4.0], "floor": 0.3, "width": 4.0}
    assert ID.apply_word(knee, [10.0, 2.0], 0.8, 0.5) == 0.8
    assert abs(ID.apply_word(knee, [-50.0], 0.8, 0.5) - 0.8 * 0.3) < 1e-9
    assert ID.apply_word(knee, [np.nan], 0.8, 0.5) == 0.8          # nothing judged: unchanged
    step = {"form": "step", "coef": [1.0, 0.0, 0.0, 0.0, 0.5], "cap": 3.0, "no_raise_below": -2.0}
    assert ID.apply_word(step, [5.0], 0.5, 0.5) > 0.5               # raised (intercept)
    assert ID.apply_word(step, [-3.0, 5.0], 0.5, 0.5) < 0.5          # weakest sound clearly wrong: not raised
    never = dict(step, cap=0.0)
    assert ID.apply_word(never, [5.0], 0.5, 0.5) == 0.5


def test_shipped_languages_and_switch(monkeypatch):
    for lang in ("es", "ru", "pt", "it"):
        assert ID.enabled(lang), lang
        H, cal = ID._head(lang), ID._cal(lang)
        assert cal["form"] == "step" and len(cal["coef"]) == 5
        assert all(c in H["index"] for c in cal["offset"])
        monkeypatch.setenv(f"PRONUNCIADA_IDENTITY_{lang.upper()}", "0")
        assert not ID.enabled(lang)
    for lang in ("fr", "en", "de"):                   # not shipped here (de keeps its own module)
        assert ID._cal(lang) is None


def test_no_data_means_off(monkeypatch):
    assert not ID.enabled("xx")
    monkeypatch.setenv("PRONUNCIADA_IDENTITY", "0")
    assert not ID.enabled("fr")
