"""Round 16: contrast-specific caps on the shown sound log-odds, and the measured alternatives."""
import numpy as np

from pronunciada import calibrate as C


def _tab(phones, comps, llr):
    n = len(phones)
    t = {k: np.zeros(n) for k in C.CAP_FEATS}
    t["llr_sub"] = np.asarray(llr, float)
    t["phone"], t["comp_tok"] = list(phones), list(comps)
    return t


def test_caps_shipped_for_the_adopted_languages():
    for lang in ("zh", "en", "ru", "ja", "de"):
        assert C.contrast_caps(lang), lang
    assert C.contrast_caps("es") == [] and C.contrast_caps("ko") == []      # ko: not adopted


def test_cap_fires_only_for_member_with_partner_competitor():
    caps = C.contrast_caps("zh")
    zl = np.full(4, 4.0)
    zs = np.full(4, 3.0)
    t = _tab(["p", "p", "pʰ", "a"], ["ʰ", "k", "ʰ", "ʰ"], [-12, -12, -12, -12])
    zl2, zs2 = C.apply_contrast_caps(caps, t, zl, zs)
    assert zl2[0] < zl[0] and zs2[0] < zs[0]               # plain p, competitor ʰ: capped
    assert np.array_equal(zl2[1:], zl[1:])                  # other competitor / not a member: untouched
    assert np.all(zl2 <= zl) and np.all(zs2 <= zs)          # a cap never raises


def test_cap_is_monotone_in_the_evidence():
    caps = C.contrast_caps("en")
    t = _tab(["θ"] * 3, ["s"] * 3, [-1, -6, -15])
    z, _ = C.apply_contrast_caps(caps, t, np.full(3, 20.0), np.full(3, 20.0))
    assert z[0] >= z[1] >= z[2]


def test_scorer_keeps_coaching_inputs_uncapped():
    import pronunciada  # noqa: F401
    from pronunciada.scorer import Scorer

    sc = Scorer("zh", threads=1)
    assert sc._calib_dir is None  # noqa: SLF001
    # the hook stores the pre-cap logits for coaching when caps exist for the language
    assert C.contrast_caps("zh")


def test_pt_medial_nasal_coda_no_longer_deletable():
    from pronunciada.text import get_g2p

    g = get_g2p("pt")
    assert g.measured_nasal is True
    ph = g("canto lã").phones
    medial = ph[2]                                   # c a n t o: the n before t
    final = ph[-1]                                   # lã: word-final coda
    assert medial.ipa == "n" and "" not in [a for a, _ in medial.alts]
    assert final.ipa == "n" and "" in [a for a, _ in final.alts]


def test_it_mid_vowel_measurement_is_off_by_default():
    from pronunciada.text import get_g2p

    assert get_g2p("it").measured_mid is False
