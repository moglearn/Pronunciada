def test_threshold_keeps_ties_and_rejects_infeasible_rule():
    import numpy as np

    from bench.coach.en_precision_fit import select_threshold, tune

    scores = np.array([0.9, 0.9, 0.7, 0.5, 0.3])
    y = np.array([True, False, True, True, False])
    assert select_threshold(scores, y, base_tp=1, base_fp=1) is None
    assert select_threshold(scores, y, base_tp=1, base_fp=2) == 0.5
    assert select_threshold(scores, y, base_tp=2, base_fp=0) is None
    assert tune(scores, np.repeat("vowel", 5), y, np.ones(5, bool), y) is None
    assert select_threshold(np.array([5, 4, 3, 2]), np.array([True, True, True, False]),
                            base_tp=1, base_fp=1) == 3
    assert select_threshold(np.array([5, 5, 4, 4]), np.array([True, False, False, True]),
                            base_tp=1, base_fp=3) == 4


def test_exported_threshold_parity_and_fallback():
    import numpy as np
    from scipy.special import expit

    from bench.coach.en_precision_fit import masks, safe_logit, shifted_threshold

    p = np.array([0.0, 1e-8, 0.1, 0.31, 0.31, 0.8, 1 - 1e-8, 1.0])
    for delta in (-2.5, 0.0, 1.4):
        corrected = safe_logit(p) + delta
        for threshold in [*corrected, corrected[-1] + 1.0]:
            exported = shifted_threshold(p, delta, threshold)
            np.testing.assert_array_equal(p >= exported, corrected >= threshold)
    rule = {"thresholds": {"vowel": 0.8}, "fallback": float(expit(0.0))}
    assert masks(np.array([0.7, 0.5, 0.9]), ["vowel", "unknown", "unknown"],
                 np.array([True, True, False]), rule).tolist() == [False, True, False]


def test_tune_preserves_tp_and_excludes_omitted_class():
    import numpy as np

    from bench.coach.en_precision_fit import masks, tune

    p = np.array([0.65] * 5 + [0.8] * 5 + [0.99])
    y = np.array([True] * 5 + [False] * 5 + [True])
    classes = np.array(["vowel"] * 5 + ["stop"] * 5 + ["omitted"])
    eligible = classes != "omitted"
    rule = tune(p, classes, y, eligible, eligible & (p >= 0.6))
    assert set(rule["offsets"]) == {"vowel", "stop"}
    assert masks(p, classes, eligible, rule).tolist() == [True] * 5 + [False] * 6


def test_coaching_fits_and_tuning_exclude_held_speakers(monkeypatch):
    import numpy as np

    from bench.coach import en_precision_fit as module

    groups = np.repeat(np.arange(25), 2)
    F = groups[:, None]
    y = np.tile([False, True], 25)
    eligible = np.ones(len(y), bool)
    classes = groups.astype(str)
    calls = []
    tuning = []

    def predict(F, y, tr, te):
        assert not set(F[tr, 0]) & set(F[te, 0])
        calls.append((set(F[tr, 0]), set(F[te, 0])))
        return np.full(len(te), 0.8)

    def reject(p, classes, y, eligible, base):
        tuning.append(set(map(int, classes)))
        return None

    monkeypatch.setattr(module, "probabilities", predict)
    monkeypatch.setattr(module, "tune", reject)
    matrices = {(outer, inner): F for outer in range(5) for inner in [None, *range(5)]}
    _, base, candidate, folds = module.evaluate(matrices, y, classes, groups, groups % 5, eligible, 0.5)
    assert len(calls) == 25
    assert len(folds) == len(tuning) == 5
    for f, train_groups in zip(folds, tuning):
        assert train_groups == set(groups) - set(f["held_speakers"])
        assert not f["has_rule"]
    np.testing.assert_array_equal(base, candidate)
