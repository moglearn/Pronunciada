"""Japanese pitch accent, round 13: word-level judgement on dependable words only.

A word is judged when it has >= 2 morae, every mora has a dictionary high/low label, and its
accent class (RefSyllable.pitch_class -> class_key) is in the model's dependable list (chosen on
FLEURS-ja validation natives, bench/prosody/pitch_ja_eval.py). Its score is the mean over its
morae of -log P(dictionary high / low) from the pitch model; it is flagged when the score exceeds
the per-sensitivity threshold in the model JSON ("judge": {"classes": [...], "thresholds":
{"careful": .., "balanced": .., "eager": ..}}). Everything else is not judged (score None).
"""
from __future__ import annotations

import numpy as np


def class_key(pitch_class: str | None) -> str | None:
    """"<pos>/<detail>/<head|chain>/c<n>/<t>" -> "<pos>/<head|chain>/<single|compound>/<t>"."""
    if not pitch_class:
        return None
    pos, _grp, head, comp, t = pitch_class.split("/")
    return f"{pos}/{head}/{'single' if comp == 'c1' else 'compound'}/{t}"


def word_morae(ref, wi: int) -> list[int] | None:
    idx = list(range(*ref.words[wi].syllables))
    if len(idx) < 2 or any(ref.syllables[i].pitch_high is None for i in idx):
        return None
    return idx


def word_score(p_high, ref, idx) -> float:
    """Mean over the word's morae of -log P(dictionary label); p_high indexed like ref.syllables."""
    lab = np.array([ref.syllables[i].pitch_high for i in idx], bool)
    p = np.clip(np.asarray([p_high[i] for i in idx], np.float64), 1e-4, 1 - 1e-4)
    return float(-np.mean(np.log(np.where(lab, p, 1 - p))))


def drop_after(pattern: list[bool], next_high: bool | None) -> int | None:
    """Mora index (within the word) after which the pitch falls, the fall onto the next word's
    first mora included; None = no fall in or right after the word."""
    for k in range(len(pattern) - 1):
        if pattern[k] and not pattern[k + 1]:
            return k
    if pattern and pattern[-1] and next_high is False:
        return len(pattern) - 1
    return None


def judge(p_high, ref, judge_cfg: dict) -> dict[int, dict]:
    """{word index: {"score", "flag": {sensitivity: bool}, "expected_drop", "heard_drop"}} for the
    judged words."""
    classes = set(judge_cfg.get("classes", []))
    thr = judge_cfg.get("thresholds", {})
    out = {}
    for wi in range(len(ref.words)):
        idx = word_morae(ref, wi)
        if idx is None or class_key(ref.syllables[idx[0]].pitch_class) not in classes:
            continue
        s = word_score(p_high, ref, idx)
        nxt = idx[-1] + 1
        exp = [bool(ref.syllables[i].pitch_high) for i in idx]
        heard = [bool(p_high[i] > 0.5) for i in idx]
        nh = ref.syllables[nxt].pitch_high if nxt < len(ref.syllables) else None
        nh_heard = bool(p_high[nxt] > 0.5) if nxt < len(ref.syllables) and p_high[nxt] is not None else None
        out[wi] = {"score": s, "flag": {k: s > v for k, v in thr.items()},
                   "expected_drop": drop_after(exp, nh), "heard_drop": drop_after(heard, nh_heard)}
    return out
