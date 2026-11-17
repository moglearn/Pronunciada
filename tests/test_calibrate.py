"""pronunciada.calibrate: numpy tree walker, calibration maps, strictness rule, packaged models."""
import json
import time

import numpy as np
import pytest

from pronunciada import calibrate as C


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[:, 2] = rng.integers(0, 60, n)          # categorical (phone-like)
    X[:, 4] = rng.integers(0, 5, n)           # categorical (class-like)
    X[rng.random(n) < 0.1, 0] = np.nan
    X[rng.random(n) < 0.05, 2] = np.nan
    y = X[:, 1] + np.nan_to_num(X[:, 0]) + (X[:, 2] % 7 == 1) - (X[:, 4] == 3) + rng.normal(size=n)
    return X, y


@pytest.mark.parametrize("kind", ["reg", "prob"])
def test_walker_reproduces_sklearn(kind):
    pytest.importorskip("sklearn")
    from bench.calib.common import export_trees, fit, raw

    X, y = _data()
    cm = [False, False, True, False, True, False]
    params = dict(learning_rate=0.1, max_iter=120, max_leaf_nodes=31, min_samples_leaf=20)
    m = fit(kind, params, X, y if kind == "reg" else (y > 0.3), cm)
    te = C.TreeEnsemble(export_trees(m, X.shape[1]))
    Xt, _ = _data(2000, 1)
    Xt[:50, 2] = 250                            # code never seen in training -> missing
    Xt[50:100, 4] = np.nan
    ref = raw(m, Xt)
    assert np.max(np.abs(te.raw(Xt) - ref)) < 1e-5
    assert np.max(np.abs(te.raw_reference(Xt) - ref)) < 1e-5
    assert te.depth <= 6


def test_blend_monotone_and_consistent():
    pl = np.array([0.99, 0.9, 0.5, 0.2, 0.01])
    ps = np.array([0.95, 0.95, 0.1, 0.05, 0.001])        # 2nd violates strict <= lenient
    prev = None
    for s in np.linspace(-1, 2, 13):
        p = C.blend(pl, ps, s)
        if prev is not None:
            assert np.all(p <= prev + 1e-12)
        prev = p
    assert np.allclose(C.blend(pl, ps, 0.0), np.clip(pl, C.P_CLIP, 1 - C.P_CLIP))
    assert C.blend(pl, ps, 1.0)[1] == pytest.approx(0.9)        # strict capped at lenient
    assert np.allclose(C.blend(pl, ps, 5.0), C.blend(pl, ps, 1.5))  # clipped to [-0.5, 1.5]
    assert np.allclose(C.blend(pl, ps, -3.0), C.blend(pl, ps, -0.5))


def test_calib_maps():
    z = np.linspace(-5, 5, 11)
    p = C.CalibMap("platt", np.array([1.0, 0.0]))(z)
    assert np.allclose(p, np.clip(1 / (1 + np.exp(-z)), C.P_CLIP, 1 - C.P_CLIP))
    iso = C.CalibMap("isotonic", np.array([0.0, 0.5, 1.0]), np.array([-1.0, 0.0, 1.0]))(z)
    assert np.all(np.diff(iso) >= 0) and iso[0] == C.P_CLIP and iso[-1] == 1 - C.P_CLIP


def test_save_load_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    from bench.calib.common import export_trees, fit

    X, y = _data(1500)
    X = X[:, [0, 1, 3, 5]]
    m = fit("prob", dict(learning_rate=0.1, max_iter=30, max_leaf_nodes=15, min_samples_leaf=20), X,
            (y > 0), [False] * 4)
    heads = {"len": {"kind": "prob", "trees": export_trees(m, 4), "calib": {"kind": "platt", "params": [1.0, 0.0]}},
             "str": {"kind": "prob", "trees": export_trees(m, 4), "calib": {"kind": "platt", "params": [1.0, -1.0]}}}
    C.save_level(tmp_path / "x.npz", ["a", "b", "c", "d"], [], {}, heads)
    lm = C.LevelModel(tmp_path / "x.npz")
    t = {"a": X[:, 0], "b": X[:, 1], "c": X[:, 2], "d": X[:, 3]}
    out = lm.predict(t)
    assert np.all(out["str"] <= out["len"])
    assert np.allclose(out["len"], np.clip(m.predict_proba(X)[:, 1], C.P_CLIP, 1 - C.P_CLIP), atol=1e-6)


def test_packaged_models_and_budget():
    d = C.data_dir()
    cfg = json.loads((d / "config.json").read_text())
    for lang in ("en", "zh", "es", "it", "ja", "ko", "fr", "de", "pt", "ru"):
        name, bias = C.sound_model_for(lang)
        assert (d / f"{name}.npz").exists()
        assert lang == "en" or name == cfg["universal_sound_model"]
        assert np.isfinite(bias)
    assert C.sound_model_for("en")[1] == 0.0
    total = sum(p.stat().st_size for p in d.glob("*.npz"))
    assert total <= 3 * 2 ** 20, total
    for name in ("sound_U", "word", "sentence", cfg["sound_model"]["en"]):
        lm = C.load_level(name)
        for h in lm.heads.values():
            assert h.trees.n_trees <= 400 and h.trees.depth <= 6


def _fake_table(n, seed=0):
    from pronunciada import features as F

    rng = np.random.default_rng(seed)
    t = {c: rng.normal(size=n) for c in F.EVIDENCE_NUM + F.RATE_NUM}
    t["cls"] = list(rng.choice(["vowel", "stop", "fricative", "nasal"], n))
    t["position"] = list(rng.choice(["initial", "medial", "final"], n))
    t["comp_cls"] = list(rng.choice(["vowel", "stop", "modifier"], n))
    t["phone"] = list(rng.choice(["t", "ə", "s", "æ", "ɹ"], n))
    t["comp_tok"] = list(rng.choice(["d", "ɪ", "z"], n))
    return t


def test_calibrator_latency_100_phones():
    """All calibrators the scorer runs for an English utterance (shipped English model's two
    probability heads, the universal model's three heads for the word inputs, word, sentence)
    on 100 phones / 20 words: <= 5 ms (best of repeats, to be robust to machine load)."""
    cfgd = C.config()
    rng = np.random.default_rng(0)
    U = C.load_level("sound_U")
    B = C.load_level(cfgd["sound_model"]["en"])
    W, S = C.load_level("word"), C.load_level("sentence")
    t = _fake_table(100)
    XU, XB = U.matrix(t), B.matrix(t)
    XW = rng.normal(size=(20, len(W.num)))
    XS = rng.normal(size=(1, len(S.num)))
    best = np.inf
    for _ in range(20):
        t0 = time.perf_counter()
        U.logits(X=XU)
        B.logits(X=XB, heads=("len", "str"))
        W.logits(X=XW)
        S.logits(X=XS)
        best = min(best, (time.perf_counter() - t0) * 1e3)
    from conftest import TIME_SCALE
    assert best <= 5.0 * TIME_SCALE, best
