"""Round 19: sequence sentence head (English), numpy + onnxruntime.

A small transformer over one token per expected sound reads, per sound, the mean of every
selected ZIPA encoder stack over the sound's frames (learned weighted sum of per-stack linear
maps), the aligner evidence (the phone-table inputs of the sound models), the expected sound and
its position in the word; one utterance token reads the shipped sentence inputs (SENT_NUM), the
delivery features (pronunciada.delivery) and an F0 summary. It outputs the four 0-10 sentence
scores (accuracy, fluency, prosodic, total), which replace the regression heads of the sentence
calibrator. Trained on speechocean762 train (bench/round19/train_seq_final.py).

Files: pronunciada/data/calib/<name>.onnx (the network) and <name>.npz (normalisation,
vocabulary, column names, layer list). Off for a language unless config "sentence_head" names a
model; PRONUNCIADA_SENTENCE_HEAD=0 turns it off everywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

POS = {"B": 1, "I": 2, "E": 3, "S": 4}
F0_KEYS = ("f0_med_hz", "f0_p10_hz", "f0_p90_hz")


def enabled() -> bool:
    return os.environ.get("PRONUNCIADA_SENTENCE_HEAD", "1") != "0"


def word_pos(phone_word) -> np.ndarray:
    pw = np.asarray(phone_word)
    out = np.zeros(len(pw), np.int64)
    for i in range(len(pw)):
        first = i == 0 or pw[i - 1] != pw[i]
        last = i == len(pw) - 1 or pw[i + 1] != pw[i]
        out[i] = POS["S"] if first and last else POS["B"] if first else POS["E"] if last else POS["I"]
    return out


def phone_windows(ev, n: int, T: int) -> list[tuple[int, int]]:
    """Frame window per sound: its aligned span widened by one frame on each side (empty span:
    the 2 frames around its position), as pronunciada.sound_emb.phone_vectors."""
    out = []
    for sl in ev.slots[:n]:
        a, b = int(sl.start_frame), int(sl.end_frame)
        if b <= a:
            a, b = a - 1, a + 1
        a, b = max(a - 1, 0), min(b + 1, T)
        if b <= a:
            a = max(min(a, T - 1), 0)
            b = a + 1
        out.append((a, b))
    return out


def f0_summary(trk) -> dict:
    if trk is None or not len(trk.f0):
        return {k: np.nan for k in F0_KEYS}
    v = trk.f0[trk.voiced & (trk.f0 > 0)]
    if not len(v):
        return {k: np.nan for k in F0_KEYS}
    return {"f0_med_hz": float(np.median(v)), "f0_p10_hz": float(np.percentile(v, 10)),
            "f0_p90_hz": float(np.percentile(v, 90))}


VOICE_KEYS = ("centroid_med_hz", "centroid_mean_hz", "rolloff85_med_hz")


def voice_cues(wav, sr: int = 16000) -> dict:
    """Speaker cues of the voice (child / adult): spectral centroid (median, mean) and 85 %
    roll-off (median) over 25 ms frames (10 ms hop) within 30 dB of the loudest frame."""
    x = np.asarray(wav, np.float32).reshape(-1)
    n, hop = int(0.025 * sr), int(0.01 * sr)
    if len(x) < n:
        return {k: np.nan for k in VOICE_KEYS}
    # frames starting every hop samples (a strided view: the same values as stacking slices)
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n).astype(np.float32)
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


def _norm(x, mu, sd, clip):
    z = (np.asarray(x, np.float64) - mu) / sd
    return np.clip(np.nan_to_num(z, nan=0.0), -clip, clip).astype(np.float32)


class SentenceHead:
    def __init__(self, name: str, calib_dir: str | None = None, threads: int | None = None):
        import onnxruntime as ort

        from .calibrate import data_dir

        d = Path(calib_dir) if calib_dir else data_dir()
        z = np.load(d / f"{name}.npz", allow_pickle=False)
        self.name = name
        self.layers = [str(x) for x in z["layers"]]
        self.tab_cols = [str(x) for x in z["tab_cols"]]
        self.utt_cols = [str(x) for x in z["utt_cols"]]
        self.vocab = {str(p): i + 1 for i, p in enumerate(z["vocab"])}
        self.tab = (z["tab_mu"], z["tab_sd"], float(z["tab_clip"][0]))
        self.utt = (z["utt_mu"], z["utt_sd"], float(z["utt_clip"][0]))
        self.lnorm = {k: (z[f"lay_{k}_mu"], z[f"lay_{k}_sd"]) for k in self.layers}
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = int(threads or 1)
        so.inter_op_num_threads = 1
        from ._overrides import apply_ort_options

        apply_ort_options(so, "sentence_head")
        self.sess = ort.InferenceSession(str(d / f"{name}.onnx"), so, providers=["CPUExecutionProvider"])

    def inputs(self, tab: dict, ref, ev, emb: dict, S: dict, dv: dict | None, trk) -> dict | None:
        n = len(ref.phones)
        if n == 0 or not emb or any(k not in emb for k in self.layers):
            return None
        T = len(emb[self.layers[0]])
        win = phone_windows(ev, n, T)
        feeds = {}
        for k in self.layers:
            v = np.asarray(emb[k], np.float32)
            m = np.stack([v[a:b].mean(0) for a, b in win])
            mu, sd = self.lnorm[k]
            feeds[f"x_{k}"] = ((m - mu) / sd).astype(np.float32)[None]
        Xt = np.column_stack([np.asarray(tab[c], np.float64) for c in self.tab_cols])
        feeds["xt"] = _norm(Xt, *self.tab)[None]
        feeds["pid"] = np.array([[self.vocab.get(p.ipa, 0) for p in ref.phones]], np.int64)
        feeds["ppos"] = word_pos([p.word for p in ref.phones])[None]
        f0 = f0_summary(trk)
        u = []
        for c in self.utt_cols:
            if c in S:
                u.append(float(np.asarray(S[c]).reshape(-1)[0]))
            elif c in F0_KEYS:
                u.append(f0[c])
            else:
                u.append(float((dv or {}).get(c, np.nan)))
        feeds["xu"] = _norm(np.array(u)[None], *self.utt)
        return feeds

    def __call__(self, feeds: dict) -> np.ndarray:
        """-> [4] 0-10 scores (accuracy, fluency, prosodic, total)."""
        return self.run(feeds)["sentence"]

    def run(self, feeds: dict, phone_word=None) -> dict:
        """sentence: [4] 0-10 scores; and, when the network has them, phone: [n] 0-2 sound scores
        and word: [n_words] 0-10 word scores (mean of the word's per-sound contributions)."""
        names = [o.name for o in self.sess.get_outputs()]
        res = dict(zip(names, self.sess.run(names, feeds)))
        out = {"sentence": np.clip(np.asarray(res["sentence"][0], np.float64), 0.0, 1.0) * 10.0}
        if "phone" in res:
            out["phone"] = np.clip(np.asarray(res["phone"][0], np.float64), 0.0, 1.0) * 2.0
        if "word_part" in res and phone_word is not None:
            pw = np.asarray(phone_word)
            wp = np.asarray(res["word_part"][0], np.float64)
            nw = int(pw.max()) + 1 if len(pw) else 0
            w = np.array([wp[pw == k].mean() if np.any(pw == k) else np.nan for k in range(nw)])
            out["word"] = np.clip(w, 0.0, 1.0) * 10.0
        return out
