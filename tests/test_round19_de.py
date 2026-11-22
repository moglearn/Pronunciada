"""Round 19 (German): front-end text handling and the word recalibration."""
import numpy as np

from pronunciada import word_lang as WL


def test_hyphen_pieces_and_english_reading():
    from pronunciada.text import get_g2p

    g = get_g2p("de")
    r = g("Kivu-See")
    assert [p.ipa for p in r.phones][-2:] == ["z", "eː"]          # "see" from the lexicon
    r = g("Hampton")                                               # unknown to the German lexicon
    alts = {a for p in r.phones for a, _ in p.alts}
    assert "æ" in alts                                             # English reading allowed
    r = g("Haus")                                                  # a lexicon word: unchanged
    assert not any(a == "æ" for p in r.phones for a, _ in p.alts)


def test_word_lang_params_and_order():
    cal = WL.params("de")
    assert cal is not None and cal["coef"][0] > 0
    WL._check(cal)
    W = {"wl_min": np.array([3.0, -1.0]), "ws_min": np.array([1.0, -2.0]), "wl_mean": np.array([5.0, 2.0]),
         "ws_mean": np.array([3.0, 0.5]), "n_phones": np.array([4.0, 9.0])}
    zl, zs = WL.transform(cal, W, np.array([4.0, -1.0]), np.array([2.0, -3.0]), np.array([False, True]))
    assert np.all(zs <= zl)
    assert WL.params("en") is None


def test_identity_classes_and_factor():
    from pronunciada import de_identity as DI
    from pronunciada.text import get_g2p

    ref = get_g2p("de")("Der Wagen fährt")
    cls = DI.classes(ref)
    assert all(c is None for p, c in zip(ref.phones, cls) if p.ipa in ("ʀ", "ə"))
    cal = DI._word_params()
    assert cal is not None and DI._head() is not None
    assert DI.word_factor(cal, [50.0], 0.5) == 1.0
    assert abs(DI.word_factor(cal, [-50.0], 0.5) - cal["floor"]) < 1e-9


def test_switch_off(monkeypatch):
    monkeypatch.setenv("PRONUNCIADA_DE_R19", "0")
    assert WL.params("de") is None
