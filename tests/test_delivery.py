"""Round 5: sentence delivery features, encoder-activation outputs, projection, per-language
sentence model choice."""
import itertools
import json
import shutil

import numpy as np
import pytest

from pronunciada import delivery as D
from pronunciada.acoustic import EMBED_OUTPUTS, add_graph_outputs


def _lev(a, b):
    """Reference Levenshtein distance."""
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def test_edit_ops_matches_levenshtein():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = rng.integers(0, 4, rng.integers(0, 8)).tolist()
        b = rng.integers(0, 4, rng.integers(0, 8)).tolist()
        i, s, d = D.edit_ops(a, b)
        assert i + s + d == _lev(a, b)
        assert len(a) - d + i == len(b)                     # path consistency
    assert D.edit_ops([1, 2, 3], [1, 2, 3]) == (0, 0, 0)
    assert D.edit_ops([1, 2, 3], [1, 3]) == (0, 0, 1)
    assert D.edit_ops([1, 3], [1, 2, 3]) == (1, 0, 0)
    assert D.edit_ops([1, 2], [1, 4]) == (0, 1, 0)


def test_speech_segments_and_pauses():
    # spans in 20 ms frames: phones 0-1 contiguous, then a 0.3 s gap, phone 2, a 0.1 s gap, phone 3
    s = np.array([10, 12, 30, 36])
    e = np.array([12, 15, 31, 40])
    segs, pauses = D.speech_segments(s, e)
    assert segs == [(10, 15), (30, 40)]
    assert pauses == [(15, 30, 1, 2)]
    m = D.speech_mask(50, segs)
    assert m.sum() == 15 and m[10] and not m[20]
    assert D.speech_segments([5], [5]) == ([], [])          # no non-empty span


def test_project_and_pool():
    rng = np.random.default_rng(1)
    pooled = {"final": rng.standard_normal(6), "enc3": rng.standard_normal(4)}
    proj = {"names": ["a", "b"], "layers": ["enc3", "final"], "mu": np.zeros(10), "sd": np.ones(10) * 2,
            "W": rng.standard_normal((10, 2)), "b": np.array([1.0, -1.0])}
    out = D.project(pooled, proj)
    x = np.concatenate([pooled["enc3"], pooled["final"]])
    assert np.allclose([out["a"], out["b"]], (x / 2) @ proj["W"] + proj["b"])
    assert all(np.isnan(v) for v in D.project(None, proj).values())
    assert D.project(pooled, None) == {}

    class Ev:
        slots = [type("S", (), {"start_frame": a, "end_frame": b})() for a, b in ((2, 4), (4, 6))]

    emb = {"final": np.arange(20, dtype=np.float32).reshape(10, 2)}
    p = D.pool_embeddings(emb, Ev())
    assert np.allclose(p["final"], np.r_[emb["final"][2:6].mean(0), emb["final"][2:6].std(0)])
    assert D.pool_embeddings(None, Ev()) is None


def test_add_graph_outputs_tiny_model():
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])
    g = helper.make_graph([helper.make_node("Relu", ["x"], ["h"]), helper.make_node("Neg", ["h"], ["y"])],
                          "g", [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    b = add_graph_outputs(m.SerializeToString(), ["h"])
    m2 = onnx.load_from_string(b)
    onnx.checker.check_model(m2)
    assert [o.name for o in m2.graph.output] == ["y", "h"]
    s = ort.InferenceSession(b, providers=["CPUExecutionProvider"])
    v = np.array([[-1.0, 0.5, 2.0]], dtype=np.float32)
    yy, hh = s.run(["y", "h"], {"x": v})
    assert np.array_equal(hh, np.maximum(v, 0)) and np.array_equal(yy, -hh)


@pytest.mark.model
def test_embedding_outputs_leave_logprobs_bit_identical():
    from pronunciada.acoustic import AcousticModel

    x = np.random.default_rng(0).standard_normal((300, 80)).astype(np.float32) - 5
    for prec in ("fp32", "int8"):
        a = AcousticModel(precision=prec, threads=2)
        b = AcousticModel(precision=prec, threads=2, embeddings=True)
        assert b.embeddings
        lp, emb = b.posteriors_from_features(x, return_embeddings=True)
        assert np.array_equal(lp, a.posteriors_from_features(x))
        assert set(emb) == set(EMBED_OUTPUTS) and all(len(v) == len(lp) for v in emb.values())
        # without embeddings the same session returns (lp, None) when asked
        assert a.posteriors_from_features(x, return_embeddings=True)[1] is None


@pytest.mark.model
def test_features_on_speech(tts):
    import pronunciada

    sc = pronunciada.Scorer("en", threads=2, delivery=True)
    wav = tts("The quick brown fox jumps over the lazy dog")
    res = sc.score(wav, "The quick brown fox jumps over the lazy dog")
    d = res._pronunciada["delivery"]
    assert set(D.ALL) <= set(d)
    assert 0 < d["voiced_share"] <= 1 and 0 < d["phonation_ratio"] <= 1
    assert d["v_int_mean"] > 0 and d["w_rate_mean"] > 0
    assert d["clipped_share"] == 0 and d["dec_ins_rate"] >= 0
    assert 0 < d["post_max_mean"] <= 1
    assert "delivery" in res.timing_ms
    pooled = res._pronunciada["pooled"]
    assert pooled is not None and pooled["final"].shape == (1024,)
    # the same features as calling the module directly on the cached evidence
    from pronunciada.audio import fbank
    from pronunciada.prosody.f0 import track

    c = res._pronunciada
    lpa = sc._posteriors(wav, {})[0]
    d2 = D.features(wav, fbank(wav), lpa, c["ref"], c["ev"], track(wav))
    for k in D.ALL:
        assert (np.isnan(d[k]) and np.isnan(d2[k])) or d[k] == d2[k], k


@pytest.mark.model
def test_sentence_model_per_language_config(tmp_path, tts):
    """config "sentence_model" picks the round-4 heads for a language: no delivery inputs, the
    plain acoustic model, and the round-4 outputs."""
    import pronunciada
    from pronunciada import calibrate as C

    src = C.data_dir()
    d = tmp_path / "calib"
    shutil.copytree(src, d)
    cfg = json.loads((d / "config.json").read_text())
    cfg["sentence_model"] = {"en": "sentence_r4"}
    cfg.pop("word_score_model", None)
    (d / "config.json").write_text(json.dumps(cfg))
    r4 = pronunciada.Scorer("en", threads=2, calib_dir=str(d))
    from pronunciada.prosody import needs_embeddings

    # the acoustic model runs plain unless the stress model needs its activations (stress_v3, round 11)
    assert r4.sentence_name == "sentence_r4" and not r4.delivery
    assert r4.am.embeddings == needs_embeddings("en")
    new = pronunciada.Scorer("en", threads=2)
    wav = tts("The quick brown fox jumps over the lazy dog")
    a = r4.score(wav, "The quick brown fox jumps over the lazy dog")
    b = new.score(wav, "The quick brown fox jumps over the lazy dog")
    assert a.prob == pytest.approx(b.prob, abs=1e-12)        # acceptance heads unchanged
    assert set(a.scores) == set(b.scores)
    if new.delivery:
        assert new.sentence_name == "sentence"
        assert any(c in D.ALL or c.startswith("emb_") for c in new.sentence.num)


def test_groups_cover_all_names():
    names = list(itertools.chain.from_iterable(D.GROUPS.values()))
    assert names == D.ALL and len(set(names)) == len(names)


@pytest.mark.model
def test_word_score_model(tts):
    """Round 6: the reported 0-10 word score reads word-pooled activations; the sentence level
    keeps the round-2 word regression; word probabilities are unchanged."""
    import pronunciada
    from pronunciada import calibrate as C

    sc = pronunciada.Scorer("en", threads=2)
    if sc.word_score is None:
        pytest.skip("no word_score_model configured for en")
    text = "The quick brown fox jumps over the lazy dog"
    wav = tts(text)
    res = sc.score(wav, text)
    c = res._pronunciada
    assert set(c["word_pooled"]) >= set(sc.word_score_proj["layers"])
    assert len(c["w_reg"]) == len(c["w_reg_r2"]) == len(res.words)
    assert np.all((c["w_reg"] >= 0) & (c["w_reg"] <= 10))
    assert np.array_equal(sc.details(res)["word_accuracy"], c["w_reg"])
    r2 = C.load_level("word_reg").predict(X=np.column_stack(
        [np.asarray(c["W"][k], dtype=np.float64) for k in C.load_level("word_reg").num]))["reg"]
    assert np.allclose(r2, c["w_reg_r2"])
    # spans: one per word, inside the utterance
    assert len(c["word_spans"]) == len(res.words) and all(a < b for a, b in c["word_spans"])


def test_word_frame_spans_and_pool():
    class P:
        def __init__(self, w):
            self.word = w

    class W:
        def __init__(self, a, b):
            self.phones = (a, b)

    class R:
        phones = [P(0), P(0), P(1)]
        words = [W(0, 2), W(2, 3)]

    class Ev:
        slots = [type("S", (), {"start_frame": a, "end_frame": b})() for a, b in ((2, 4), (5, 7), (9, 9))]

    sp = D.word_frame_spans(R(), Ev(), 20)
    assert sp == [(2, 7), (8, 10)]
    emb = np.arange(40, dtype=np.float32).reshape(20, 2)
    p = D.pool_words(emb, sp)
    assert p.shape == (2, 4) and np.allclose(p[0, :2], emb[2:7].mean(0)) and np.allclose(p[1, 2:], emb[8:10].std(0))
