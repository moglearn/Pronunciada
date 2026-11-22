"""Round 19: every-stack encoder outputs, std-pooled projections, voice cues and the English
sentence head (trees + sequence head blend, PRONUNCIADA_SENTENCE_HEAD switch)."""
import numpy as np
import pytest

from pronunciada import delivery as D
from pronunciada import sentence_head as SH
from pronunciada.acoustic import ALL_LAYER_OUTPUTS, EMBED_OUTPUTS, embed_outputs_for


def test_embed_output_selection():
    assert embed_outputs_for(True) == EMBED_OUTPUTS
    assert list(embed_outputs_for("all")) == list(ALL_LAYER_OUTPUTS)
    sel = embed_outputs_for(["enc0", "enc2"])
    assert set(sel) == {"enc0", "enc1", "enc2", "enc3", "enc5", "final"}
    assert list(sel) == [k for k in ALL_LAYER_OUTPUTS if k in sel]      # graph order kept


def test_pooled_part_and_projection():
    v = np.arange(8, dtype=np.float32)
    assert np.array_equal(D.pooled_part(v, "enc1"), v)
    assert np.array_equal(D.pooled_part(v, "enc1.mean"), v[:4])
    assert np.array_equal(D.pooled_part(v, "enc1.std"), v[4:])
    proj = {"names": ["a"], "layers": ["enc1.std", "enc0.std"], "mu": np.zeros(4), "sd": np.ones(4),
            "W": np.ones((4, 1)), "b": np.zeros(1)}
    pooled = {"enc1": np.array([9, 9, 1, 2.0]), "enc0": np.array([9, 9, 3, 4.0])}
    assert D.project(pooled, proj)["a"] == 10.0
    assert np.isnan(D.project({"enc1": pooled["enc1"]}, proj)["a"])


def test_voice_cues_follow_spectrum():
    sr = 16000
    t = np.arange(sr) / sr
    lo = SH.voice_cues(0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float32))
    hi = SH.voice_cues(0.3 * np.sin(2 * np.pi * 3000 * t).astype(np.float32))
    assert 250 < lo["centroid_med_hz"] < 400 and 2800 < hi["centroid_med_hz"] < 3200
    assert all(np.isnan(v) for v in SH.voice_cues(np.zeros(100, np.float32)).values())


def test_word_pos():
    assert SH.word_pos([0, 0, 0, 1, 2, 2]).tolist() == [1, 2, 3, 4, 1, 3]


@pytest.mark.model
def test_all_layer_outputs_leave_logprobs_bit_identical():
    from pronunciada.acoustic import AcousticModel

    x = np.random.default_rng(0).standard_normal((300, 80)).astype(np.float32) - 5
    for prec in ("fp32", "int8"):
        a = AcousticModel(precision=prec, threads=2)
        b = AcousticModel(precision=prec, threads=2, embeddings="all")
        lp, emb = b.posteriors_from_features(x, return_embeddings=True)
        assert np.array_equal(lp, a.posteriors_from_features(x))
        assert list(emb) == list(ALL_LAYER_OUTPUTS) and all(len(v) == len(lp) for v in emb.values())


@pytest.mark.model
def test_english_sentence_head_and_switch(tts, monkeypatch):
    import pronunciada

    text = "The quick brown fox jumps over the lazy dog"
    wav = tts(text)
    new = pronunciada.Scorer("en", threads=2)
    assert new.sentence_name == "sentence_r19" and new.seq_head is not None
    r = new.score(wav, text)
    c = r._pronunciada
    s = new.details(r)["sentence"]
    assert all(0 <= v <= 10 for v in s.values())
    assert c["sent_seq"].shape == (4,) and "sentence_head" in r.timing_ms
    lv = c["seq_level"]                       # sound / word outputs blended into the reported scores
    assert lv["phone"].shape == (len(c["ref"].phones),) and lv["word"].shape == (len(c["ref"].words),)
    d = new.details(r)
    assert np.all((d["phone_score"] >= 0) & (d["phone_score"] <= 2))
    assert np.all((d["word_accuracy"] >= 0) & (d["word_accuracy"] <= 10))
    assert all(np.isfinite(float(np.asarray(c["S"][k]).reshape(-1)[0])) for k in ("emb_total", "f0_med_hz"))
    monkeypatch.setenv("PRONUNCIADA_SENTENCE_HEAD", "0")
    old = pronunciada.Scorer("en", threads=2)
    assert old.sentence_name == "sentence_r4" and old.seq_head is None
    r0 = old.score(wav, text)
    assert "sent_seq" not in r0._pronunciada
    # sounds, words and the acceptance probability do not depend on the sentence head
    assert r0.prob == r.prob
    assert [w.prob for w in r0.words] == [w.prob for w in r.words]
    other = pronunciada.Scorer("es", threads=2, prosody=False)
    assert other.seq_head is None and other.sentence_name == "sentence_r4"
