"""pronunciada.features: the one feature implementation shared by training and inference."""
import math
import os

import numpy as np
import pytest

from pronunciada import features as F
from pronunciada.align import score_reference
from pronunciada.text import get_g2p


def _lp(T, seed=0, V=127):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, V)) * 2
    x[:, 0] += 3                                   # blank-heavy, like ZIPA
    x -= np.log(np.exp(x).sum(1, keepdims=True))
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def ref():
    return get_g2p("en")("the cat sat on a mat")


def test_phone_rows_columns_and_lengths(ref):
    lp = _lp(120)
    ev = score_reference(lp, ref)
    rows = F.phone_rows(lp, ref, ev, None, 2.4)
    assert len(rows) == len(ref.phones)
    for k in ("llr_sub", "llr_del", "start_frame", "end_frame", "can_mean", "comp_tok", "comp_dist",
              "word_mean_llr_sub", "utt_mean_llr_sub", "phones_per_s", "z_llr_sub"):
        assert k in rows[0]
    assert all(math.isnan(r["z_llr_sub"]) for r in rows)            # no native stats given
    assert rows[0]["utt_dur"] == 2.4
    assert rows[0]["phones_per_s"] == pytest.approx(len(ref.phones) / 2.4)
    for i, r in enumerate(rows):
        assert r["llr_sub"] <= 0 and r["phone_index"] == i
        assert r["n_frames"] == r["end_frame"] - r["start_frame"]


def test_phone_table_derived_and_matrix(ref):
    from pronunciada.variants import NativeStats

    lp = _lp(150, 1)
    t = F.phone_table(lp, ref, score_reference(lp, ref), NativeStats.load("en"), 3.0)
    n = len(ref.phones)
    for c in F.EVIDENCE_NUM:
        assert len(t[c]) == n, c
    assert np.isfinite(t["z_llr_sub"]).all()
    assert np.isnan(t["prev_llr_del"][0]) and np.isnan(t["next_llr_del"][-1])
    w0 = [i for i in range(n) if ref.phones[i].word == 0]
    assert np.allclose(t["word_min_llr_del"][w0], min(t["llr_del"][w0]))
    vocab = {"cls": sorted(set(t["cls"])),
             "position": sorted(set(t["position"])), "comp_cls": sorted({"none" if v is None else v for v in t["comp_cls"]}),
             "phone": sorted(set(t["phone"])), "comp_tok": ["k"]}
    X = F.matrix(t, F.EVIDENCE_NUM, F.EVIDENCE_CAT + F.IDENTITY_CAT, vocab, "mixed")
    cols = F.matrix_columns(F.EVIDENCE_NUM, F.EVIDENCE_CAT + F.IDENTITY_CAT, vocab, "mixed")
    assert X.shape == (n, len(cols))
    oh = [j for j, c in enumerate(cols) if c.startswith("cls=")]
    assert np.allclose(X[:, oh].sum(1), 1)                 # one-hot: exactly one class
    j = cols.index("comp_tok")
    assert np.all(np.isnan(X[:, j]) | (X[:, j] == 0))      # unknown codes -> NaN
    mask = F.categorical_mask(F.EVIDENCE_NUM, F.EVIDENCE_CAT + F.IDENTITY_CAT, vocab, "mixed")
    assert [cols[k] for k, m in enumerate(mask) if m] == ["phone", "comp_tok"]


def test_derive_groups_equal_separate(ref):
    """derive() on two stacked references == derive() on each alone (training stacks many)."""
    ref2 = get_g2p("en")("dogs bark loudly")
    tabs, rows = [], []
    for k, r in enumerate((ref, ref2)):
        lp = _lp(100 + 20 * k, 3 + k)
        ev = score_reference(lp, r)
        tabs.append(F.phone_table(lp, r, ev, None, 2.0))
        rows.append(F.phone_rows(lp, r, ev, None, 2.0))
    both = F.to_columns(rows[0] + rows[1])
    F.derive(both, np.array(["a"] * len(rows[0]) + ["b"] * len(rows[1])))
    for c in ("word_mean_llr_del", "utt_mean_llr_del", "utt_p10_llr_sub", "prev_llr_del", "next_llr_del",
              "llr_sub_rel_word", "utt_share_del"):
        sep = np.concatenate([tabs[0][c], tabs[1][c]])
        assert np.allclose(both[c], sep, equal_nan=True), c


def test_word_and_sentence_tables(ref):
    lp = _lp(150, 5)
    ev = score_reference(lp, ref)
    t = F.phone_table(lp, ref, ev, None, 3.0)
    n = len(ref.phones)
    sl = np.linspace(-2, 3, n)
    ss = sl - 1
    sr = np.full(n, 1.5)
    W = F.word_table(t, sl, ss, sr)
    assert len(W["wl_mean"]) == len(ref.words)
    p0, p1 = ref.words[0].phones
    assert W["wl_min"][0] == pytest.approx(sl[p0:p1].min())
    assert W["n_phones"][0] == p1 - p0
    assert np.isnan(W["stress_prob"]).all()
    nw = len(ref.words)
    S = F.sentence_table(t, sl, ss, sr, W, np.zeros(nw), np.zeros(nw), np.full(nw, 8.0), 3.0)
    for c in F.SENT_NUM:
        assert len(S[c]) == 1, c
    assert 0 <= S["completeness"][0] <= 1


def test_pacing():
    # two runs of speech separated by a 0.5 s gap (25 frames) and a 0.1 s gap (not a pause)
    st = np.array([0, 10, 20, 55, 62])
    en = np.array([10, 20, 30, 60, 70])
    p = F.pacing(st, en, dur=2.0)
    assert p["n_pauses"] == 1
    assert p["pause_total_s"] == pytest.approx(0.5)
    assert p["speech_share"] == pytest.approx((70 * 0.02 - 0.5) / 2.0)
    assert F.pacing(np.zeros(3), np.zeros(3), 1.0)["articulation_rate"] == 0.0


def test_token_distance():
    from pronunciada.text.ipa import token_ids

    ids = token_ids()
    d = F.token_distance(ids["p"], ids["b"])
    assert 0 < d < F.token_distance(ids["p"], ids["a"]) or math.isnan(F.token_distance(ids["p"], ids["a"]))
    assert F.token_distance(ids["p"], ids["b"]) == F.token_distance(ids["b"], ids["p"])
    assert math.isnan(F.token_distance(ids["p"], ids["ʰ"]))
    assert F.token_distance(ids["t"], ids["t"]) == 0.0


def _feature_cache():
    root = os.environ.get("PRONUNCIADA_CACHE") or os.path.expanduser("~/.cache/pronunciada")
    return os.path.join(root, "features", "speechocean762.parquet")


@pytest.mark.skipif(not os.path.exists(_feature_cache()), reason="feature tables not built")
def test_parity_with_feature_table():
    """Runtime features recomputed from the cached posteriors equal the stored table rows."""
    pd = pytest.importorskip("pandas")
    from bench.calib.parity import runtime_rows
    from bench.pipeline.common import records

    df = pd.read_parquet(_feature_cache())
    recs = {r["id"]: r for r in records("speechocean762", split="test")}
    for u in list(dict.fromkeys(df[df.split == "test"].utt))[:5]:
        old = df[df.utt == u].sort_values("phone_index")
        new = runtime_rows("speechocean762", recs[u])
        assert len(new) == len(old)
        for c in ("llr_sub", "llr_del", "can_mean", "comp_mean", "comp_dist", "z_llr_sub", "word_min_llr_sub"):
            a = old[c].to_numpy(float)
            b = np.array([np.nan if r[c] is None else r[c] for r in new], float)
            assert np.allclose(a, b, equal_nan=True, atol=1e-9), c
        assert list(old.comp_tok.fillna("∅")) == [r["comp_tok"] or "∅" for r in new]
