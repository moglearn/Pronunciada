import itertools

import numpy as np

from bench.learners.evaluate import cindex
from bench.metrics import roc_auc


def brute(h, s, strata=None):
    conc = pairs = 0.0
    for i, j in itertools.combinations(range(len(h)), 2):
        if strata is not None and strata[i] != strata[j]:
            continue
        if h[i] == h[j]:
            continue
        hi, lo = (i, j) if h[i] > h[j] else (j, i)
        pairs += 1
        conc += 1.0 if s[hi] > s[lo] else 0.5 if s[hi] == s[lo] else 0.0
    return conc / pairs


def test_binary_equals_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = np.round(rng.normal(size=60) + y, 1)
    assert abs(cindex(y, s) - roc_auc(y, s)) < 1e-12


def test_ordinal_and_strata_match_brute_force():
    rng = np.random.default_rng(1)
    h = rng.integers(0, 5, 40).astype(float)
    s = np.round(rng.normal(size=40) + 0.3 * h, 1)
    st = rng.integers(0, 4, 40)
    assert abs(cindex(h, s) - brute(h, s)) < 1e-12
    assert abs(cindex(h, s, st) - brute(h, s, st)) < 1e-12
