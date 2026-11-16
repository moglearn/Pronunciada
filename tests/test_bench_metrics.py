import numpy as np
import pytest
from scipy import stats

from bench import metrics as M

sk = pytest.importorskip("sklearn.metrics")


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = y * 0.8 + rng.normal(0, 1, 500)
    s[:50] = np.round(s[:50], 1)  # some ties
    return y, s


def test_correlations(data):
    y, s = data
    x = s + np.random.default_rng(2).normal(0, 1, len(s))
    assert M.pearson(x, s) == pytest.approx(stats.pearsonr(x, s)[0], abs=1e-12)
    assert M.spearman(x, s) == pytest.approx(stats.spearmanr(x, s)[0], abs=1e-12)
    assert np.isnan(M.pearson([1, 1, 1], [1, 2, 3]))


def test_auc_vs_sklearn(data):
    y, s = data
    assert M.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-12)
    assert M.pr_auc(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-12)
    fpr, tpr, _ = M.roc_curve(y, s)
    f2, t2, _ = sk.roc_curve(y, s, drop_intermediate=False)
    assert np.allclose(fpr, f2) and np.allclose(tpr, t2)


def test_auc_hand():
    assert M.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert M.roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    # AP: ranks 0.8(+) 0.4(-) 0.35(+) 0.1(-): 0.5*1 + 0.5*(2/3)
    assert M.pr_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.5 + 1 / 3)


def test_prf(data):
    y, s = data
    r = M.prf_at(y, s, 0.3)
    pred = (s >= 0.3).astype(int)
    assert r.precision == pytest.approx(sk.precision_score(y, pred))
    assert r.recall == pytest.approx(sk.recall_score(y, pred))
    assert r.f1 == pytest.approx(sk.f1_score(y, pred))
    b = M.best_f1(y, s)
    brute = max(sk.f1_score(y, (s >= t).astype(int)) for t in np.unique(s))
    assert b.f1 == pytest.approx(brute)


def test_eer():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9]
    e, _ = M.eer(y, s)
    assert e == pytest.approx(0.25)
    e, _ = M.eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert e == pytest.approx(0.0)


def test_calibration():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 1])
    p = np.array([0.05, 0.15, 0.15, 0.55, 0.55, 0.55, 0.95, 0.95, 0.95, 0.95])
    # equal width, 10 bins: bins 0 (.05,y0), 1 (.15,.15 y0,1), 5 (.55 x3, 2/3), 9 (.95 x4, 1)
    exp = (1 * 0.05 + 2 * abs(0.15 - 0.5) + 3 * abs(0.55 - 2 / 3) + 4 * 0.05) / 10
    assert M.ece(y, p, 10, "width") == pytest.approx(exp)
    assert M.brier(y, p) == pytest.approx(np.mean((p - y) ** 2))
    rows = M.reliability_table(y, p, 5, "mass")
    assert [r["n"] for r in rows] == [2, 2, 2, 2, 2]
    assert M.ece(y, p, 1, "mass") == pytest.approx(abs(p.mean() - y.mean()))
    # sklearn calibration_curve agrees on equal-width frac_pos / mean_p
    from sklearn.calibration import calibration_curve
    fp, mp = calibration_curve(y, p, n_bins=10, strategy="uniform")
    rows = M.reliability_table(y, p, 10, "width")
    assert np.allclose([r["frac_pos"] for r in rows], fp)
    assert np.allclose([r["mean_p"] for r in rows], mp)


def test_bootstrap(data):
    y, s = data
    r = M.bootstrap_ci(M.roc_auc, y, s, n_boot=300)
    assert r["lo"] < r["value"] < r["hi"]
    clusters = np.arange(len(y)) // 10
    rc = M.bootstrap_ci(M.roc_auc, y, s, clusters=clusters, n_boot=300)
    assert rc["lo"] < rc["value"] < rc["hi"]
    # degenerate cluster: one cluster -> resample is the full set every time
    r1 = M.bootstrap_ci(M.pearson, s, s * 2, clusters=np.zeros(len(s)), n_boot=20)
    assert r1["lo"] == pytest.approx(1.0) and r1["hi"] == pytest.approx(1.0)


def test_paired_bootstrap(data):
    y, s = data
    noisy = s + np.random.default_rng(3).normal(0, 3, len(s))
    r = M.paired_bootstrap(M.roc_auc, y, s, noisy, n_boot=500)
    assert r["delta"] > 0 and r["p_value"] < 0.01 and r["lo"] > 0
    r2 = M.paired_bootstrap(M.roc_auc, y, noisy, s, n_boot=500)
    assert r2["p_value"] > 0.9
    # lower-is-better metric (ECE on probabilities)
    p_good = np.clip(0.5 + 0.3 * (y - 0.5) * 2, 0, 1)
    p_bad = np.full(len(y), 0.9)
    r3 = M.paired_bootstrap(M.brier, y, p_good, p_bad, n_boot=200, higher_is_better=False)
    assert r3["delta"] > 0 and r3["p_value"] < 0.05
