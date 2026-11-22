"""Round 20 (insdel): the greedy-decode vs expected-sounds alignment (offline ablation code)."""
import numpy as np

from bench.round20_insdel import greedy_match as G


def _feats(ipa, alts, word, heard, **kw):
    n = len(ipa)
    start = np.arange(n) * 10
    end = start + 8
    free = [(t, 10 * k + 1, 10 * k + 3) for k, t in enumerate(heard)]
    return G.features(ipa, alts, np.asarray(word), start, end, max(word) + 1, 10 * n, free, **kw)


def test_units_drop_length_and_merge_repeats():
    assert G.units("aː") == ("a",)
    assert G.units("tt") == ("t",)
    assert G.units("tʰa") == ("tʰ", "a")


def test_exact_match_has_no_edits():
    f = _feats(["k", "a", "t"], [[], [], []], [0, 0, 0], ["k", "a", "t"])
    assert f["ins"][0] + f["del"][0] == 0
    assert not f["dropped"].any()


def test_dropped_and_inserted_sounds():
    f = _feats(["k", "a", "t"], [[], [], []], [0, 0, 0], ["k", "t"])
    assert f["n_del"][0] == 1 and f["dropped"][1]
    g = _feats(["k", "a", "t"], [[], [], []], [0, 0, 0], ["k", "a", "s", "t"])
    assert g["n_ins"][0] == 1 and g["ins"][0] == 1.0 and not g["dropped"].any()


def test_allowed_empty_alternative_makes_a_drop_free():
    f = _feats(["k", "a", "t"], [[], [("", -1.0)], []], [0, 0, 0], ["k", "t"])
    assert f["del"][0] == 0.0 and not f["dropped"].any()
    g = _feats(["k", "a", "t"], [[], [("", -1.0)], []], [0, 0, 0], ["k", "t"], allow_alts=False)
    assert g["del"][0] == 1.0


def test_weighted_costs():
    C = G.Costs(ins={"s": 0.3}, dele={"a": 0.5}, weighted=True)
    f = _feats(["k", "a", "t"], [[], [], []], [0, 0, 0], ["k", "s", "t"], costs=C)
    # a -> s substitution (vowel vs consonant = 1.0) costs more than drop a + insert s = 0.8
    assert abs(f["ins"][0] + f["del"][0] - 0.8) < 1e-9
    assert G.sub_cost("t", "d", True) < 1.0 and G.sub_cost("t", "a", True) == 1.0
