"""Cheaper forms of two per-clip computations give the same values as the forms they replaced."""
import numpy as np

from pronunciada import sentence_head as SH
from pronunciada.text.ipa import load_tokens, to_ids


def _voice_cues_stacked(wav, sr=16000):
    x = np.asarray(wav, np.float32).reshape(-1)
    n, hop = int(0.025 * sr), int(0.01 * sr)
    if len(x) < n:
        return {k: np.nan for k in SH.VOICE_KEYS}
    idx = np.arange(0, len(x) - n + 1, hop)
    fr = np.stack([x[i:i + n] for i in idx]) * np.hanning(n).astype(np.float32)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / sr)
    e = P.sum(1)
    edb = 10 * np.log10(e + 1e-12)
    keep = edb > edb.max() - 30.0
    P, e = P[keep], e[keep]
    cen = (P @ f) / (e + 1e-12)
    cs = np.cumsum(P, axis=1)
    roll = f[np.argmax(cs >= 0.85 * cs[:, -1:], axis=1)]
    return {"centroid_med_hz": float(np.median(cen)), "centroid_mean_hz": float(np.mean(cen)),
            "rolloff85_med_hz": float(np.median(roll))}


def test_voice_cues_identical_to_stacked_frames():
    rng = np.random.default_rng(5)
    for n in (399, 400, 401, 560, 16000, 16000 * 3 + 77):
        w = (0.1 * rng.standard_normal(n)).astype(np.float32)
        w[: n // 3] *= 0.001
        a, b = SH.voice_cues(w), _voice_cues_stacked(w)
        assert a.keys() == b.keys()
        assert all(a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])) for k in a)


def test_competitor_mask_equals_isin():
    nb = np.arange(4, len(load_tokens()))
    for ids in [[k] for k in range(127)] + [to_ids(p) for p in ("tʃ", "aː", "ts", "kʰ", "ŋ", "ɑː")] + [[0, 3, 5, 5]]:
        old = np.isin(nb, ids)
        new = np.zeros(len(nb), bool)
        new[[k - 4 for k in ids if 4 <= k < 4 + len(nb)]] = True
        assert np.array_equal(old, new), ids
