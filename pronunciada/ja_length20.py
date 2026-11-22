"""Round 20 (Japanese): vowel length read from a padded window of encoder activations.

The CTC output of the acoustic model collapses a held vowel to one spike, so start / end times from
the alignment measure length poorly. The encoder's hidden states before the CTC layer still carry
the vowel across frames. For every vowel the length channel judges, a small head reads a fixed
window of one encoder stack's activations centred on the vowel's CTC peak (+-w output frames of
20 ms, whatever the alignment says about its boundaries) and the local mora duration (speaking
rate), and gives the log-odds that the vowel as SAID is long. It was trained on native Common Voice
ja / FLEURS ja train vowels as they are, and on the same clips with single vowels time-stretched or
compressed by WSOLA (the rest of the clip untouched), with real background noise mixed into part
of them (bench/round20_ja_length/, results/round20_japanese_length.md).

Use: lower-only. With ``zh`` the head's log-odds that the vowel matches its text (long expected:
the long log-odds; short expected: minus it), the length channel's log-odds of that vowel become

    z' = z + min(0, a * (zh - k))

(k = k_long for an expected-long vowel, k_short for an expected-short one) so the head can only lower a vowel's length probability (and through it the word step and the
error flag), never raise it; ``k`` sits at a low native quantile of zh on dev natives, so native
vowels are rarely touched. Parameters: data/length/ja_len20.json and the head
data/length/ja_len20.onnx (exported from torch; onnxruntime only at runtime).

Switch: OFF by default. On the round-19 dev sets every setting tried raised the share of native
words below 0.6 (by 0.03 to 1 point), which fails the Japanese native guard, so the head does not
change any output unless PRONUNCIADA_JA_LENGTH20=1 (results/round20_japanese_length.md). Unset or 0
gives the previous outputs exactly. Only active with the round-19 length model
(pronunciada.ja_round19 part "lenmodel").
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from importlib import resources

import numpy as np


def on() -> bool:
    from .ja_round19 import on as _r19

    return os.environ.get("PRONUNCIADA_JA_LENGTH20", "0") == "1" and _r19("lenmodel")


@lru_cache(maxsize=None)
def params() -> dict | None:
    p = resources.files("pronunciada").joinpath("data/length/ja_len20.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None


@lru_cache(maxsize=None)
def _session():
    import onnxruntime as ort

    p = resources.files("pronunciada").joinpath("data/length/ja_len20.onnx")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def layers_needed() -> tuple[str, ...]:
    P = params()
    return tuple(P["layers"]) if P else ()


def mora_durations(ref, ev) -> tuple[np.ndarray, float]:
    """Per word: the local mora duration outside the word (20 ms frames), and the utterance's."""
    from .length_r19 import _morae

    on_ = [s.start_frame for s in ev.slots]
    nW = len(ref.words)
    wdur, wmor = np.zeros(nW), np.zeros(nW)
    for k, w in enumerate(ref.words):
        a, b = w.phones
        s1 = int(on_[b]) if b < len(on_) else int(ev.slots[b - 1].end_frame)
        wdur[k] = max(s1 - int(on_[a]), 1)
        wmor[k] = _morae(ref, w)
    tot_d, tot_m = wdur.sum(), wmor.sum()
    md_utt = max(tot_d / max(tot_m, 1.0), 1.5)
    loc = np.where(tot_m - wmor >= 3, (tot_d - wdur) / np.maximum(tot_m - wmor, 1.0), md_utt)
    return np.maximum(loc, 1.5), md_utt


def peak_frame(p_ipa: str, sl, lpa: np.ndarray, T: int) -> int:
    from .text.ipa import token_ids

    s, e = int(sl.start_frame), int(sl.end_frame)
    tk = token_ids().get(p_ipa[0])
    c = s + int(np.argmax(lpa[s:e, tk])) if e > s and tk is not None else s
    return min(max(c, 0), T - 1)


def window(X: np.ndarray, c: int, W: int) -> tuple[np.ndarray, int, int]:
    """[2W+1, D] activations centred on frame c (zeros outside the clip) and the valid range."""
    T = len(X)
    win = np.zeros((2 * W + 1, X.shape[1]), np.float32)
    a, b = c - W, c + W + 1
    aa, bb = max(a, 0), min(b, T)
    win[aa - a: aa - a + (bb - aa)] = X[aa:bb]
    return win, aa - a, aa - a + (bb - aa)


def head_logits(ref, ev, lpa, emb, idx: list[int]) -> np.ndarray:
    """log-odds that each vowel in idx (reference sound indices) is said long."""
    P = params()
    w = int(P["w"])
    X = np.concatenate([np.asarray(emb[k], np.float32) for k in P["layers"]], axis=1)
    T = len(X)
    loc, _ = mora_durations(ref, ev)
    xs, ms, rs = [], [], []
    for i in idx:
        c = peak_frame(ref.phones[i].ipa, ev.slots[i], lpa, T)
        win, v0, v1 = window(X, c, w)
        m = np.zeros(2 * w + 1, np.float32)
        m[v0:v1] = 1.0
        xs.append(win)
        ms.append(m)
        rs.append(math.log(float(loc[ref.phones[i].word])) - math.log(P["med_md"]))
    out = _session().run(["logit"], {"x": np.stack(xs).astype(np.float32), "mask": np.stack(ms),
                                     "lrate": np.asarray(rs, np.float32)})[0]
    return np.asarray(out, np.float64).reshape(-1)


def judged_vowels(ref, lres) -> list[int]:
    from .text.ipa import is_vowel

    return [i for i, p in enumerate(ref.phones)
            if lres["prob"][i] is not None and lres["expected"][i] in ("long", "short") and is_vowel(p.ipa)]


def gated(ref, i: int, gate: int) -> bool:
    """Positions left to the round-19 channel alone (bit 1: the word's last sound, where natives
    lengthen vowels before particles and phrase edges; bit 2: a vowel next to another vowel of the
    same word; bit 4, applied by the callers: a sound the round-19 channel already finds more likely
    wrong than right, p < 0.5, so the head only adds detections the channel missed)."""
    from .text.ipa import is_vowel

    if not gate:
        return False
    P = ref.phones
    w = P[i].word
    if gate & 1 and i == ref.words[w].phones[1] - 1:
        return True
    if gate & 2 and ((i > 0 and P[i - 1].word == w and is_vowel(P[i - 1].ipa))
                     or (i + 1 < len(P) and P[i + 1].word == w and is_vowel(P[i + 1].ipa))):
        return True
    return False


def lower(p: float, zh: float, a: float, k: float | None, cap: float = 0.0, mode: str = "add") -> float:
    """p lowered in log-odds z: mode "add": z + min(0, a (zh - k)) (by at most `cap` when cap > 0);
    mode "min": min(z, a (zh - k)), the head's own verdict replaces a higher one. k None: unchanged."""
    if k is None:
        return p
    if mode == "min":
        q = min(max(float(p), 1e-6), 1 - 1e-6)
        z = math.log(q / (1 - q))
        zn = a * (zh - k)
        if zn >= z:
            return p
        return 1.0 / (1.0 + math.exp(-max(zn, -30.0)))
    d = min(0.0, a * (zh - k))
    if cap > 0:
        d = max(d, -cap)
    if d == 0.0:
        return p
    q = min(max(float(p), 1e-6), 1 - 1e-6)
    z = math.log(q / (1 - q)) + d
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def adjust(ref, ev, lpa, emb, lres: dict) -> dict:
    """The length channel's result with every judged vowel's probability lowered where the head
    says the vowel as said does not have the expected length (see the module docstring)."""
    P = params()
    if P is None or lres is None or not emb or any(k not in emb for k in P["layers"]):
        return lres
    idx = judged_vowels(ref, lres)
    if not idx:
        return lres
    z = head_logits(ref, ev, lpa, emb, idx)
    prob = list(lres["prob"])
    a = float(P["a"])
    zh_all = [None] * len(prob)
    gate = int(P.get("gate", 0))
    for i, zi in zip(idx, z):
        long_ = lres["expected"][i] == "long"
        if gated(ref, i, gate) or (gate & 4 and prob[i] < 0.5):
            continue
        zh = float(zi) if long_ else -float(zi)
        zh_all[i] = zh
        k = P["k_long"] if long_ else P["k_short"]
        prob[i] = lower(prob[i], zh, a, None if k is None else float(k), float(P.get("cap", 0.0)),
                        P.get("mode", "add"))
    return {**lres, "prob": prob, "head20": zh_all}
