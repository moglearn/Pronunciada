"""The two ZIPA-alone baseline scorers (bench/eval_zipa_baseline.py) on tiny synthetic posteriors."""
import numpy as np
import pytest

from bench.eval_zipa_baseline import GOP_FLOOR, gop_scores, levenshtein_align, match_scores

V = 127
A, B, C = 10, 20, 30          # arbitrary speech token ids (>= 4)


def posterior(frames, p=0.9):
    """[T, V] log-posteriors: frame t puts mass p on token frames[t] (0 = blank), the rest uniform."""
    lp = np.full((len(frames), V), np.log((1 - p) / (V - 1)))
    for t, k in enumerate(frames):
        lp[t, k] = np.log(p)
    return lp


def test_gop_alignment_and_means():
    # audio: A A blank B C ; reference: [A] [B C] in two words
    lp = posterior([A, A, 0, B, C])
    lp[1, A] = np.log(0.5)                       # second A frame less confident
    g = gop_scores(lp, [(A,), (B, C)], [(0, 1), (1, 2)])
    assert g["aligned"]
    assert g["phone"][0] == pytest.approx((np.log(0.9) + np.log(0.5)) / 2)   # both A frames, blank excluded
    assert g["phone"][1] == pytest.approx(np.log(0.9))                      # B and C frames
    assert g["word"] == pytest.approx(g["phone"])
    assert g["sentence"] == pytest.approx(np.mean(g["phone"]))


def test_gop_wrong_sound_scores_lower_and_too_short_is_floored():
    lp = posterior([A, 0, C])
    good = gop_scores(lp, [(A,), (C,)], [(0, 2)])
    bad = gop_scores(lp, [(A,), (B,)], [(0, 2)])
    assert bad["phone"][1] < good["phone"][1] and bad["phone"][0] == pytest.approx(good["phone"][0])
    short = gop_scores(posterior([A]), [(A,), (A,)], [(0, 2)])   # A A needs >= 3 frames (blank between)
    assert not short["aligned"] and short["phone"] == [GOP_FLOOR, GOP_FLOOR]


def test_levenshtein():
    m, d = levenshtein_align([A, B, C], [A, C, C])
    assert d == 1 and m == [True, False, True]
    m, d = levenshtein_align([A, B], [])
    assert d == 2 and m == [False, False]


def test_match_scores():
    # greedy: A A blank B blank B -> A B B ; reference [A] [B C] -> B matched, C substituted by B
    lp = posterior([A, A, 0, B, 0, B])
    m = match_scores(lp, [(A,), (B, C)], [(0, 1), (1, 2)])
    assert m["phone"] == [1.0, 0.0]                  # a multi-token sound needs every token matched
    assert m["word"] == [1.0, 0.0]
    assert m["edit_distance"] == 1 and m["sentence"] == pytest.approx(1 - 1 / 3)
    # insertions count, sentence floored at 0
    lp = posterior([B, 0, C, 0, B, 0, C, 0, B])
    m = match_scores(lp, [(A,)], [(0, 1)])
    assert m["sentence"] == 0.0 and m["phone"] == [0.0]
