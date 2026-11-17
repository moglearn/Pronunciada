"""Round 9: the prompt-match channel — P(the recording is a reading of this prompt at all).

NOT ADOPTED (results/round9_match.md): no coefficients were shipped, so this module was moved out
of the runtime package (formerly pronunciada/match.py) and lives here for bench/calib/fit_match.py
only. The text below describes the design as it was tried.

The sentence acceptance heads were trained on speechocean762 only, where every speaker reads
the right sentence, so they barely react when the audio is not the prompt (another sentence,
silence, a truncated take). P_match is a small monotone logistic model on recogniser-side
evidence, pooled over the ten languages and fitted on FLEURS validation
(bench/calib/fit_match.py); SentenceResult.prob = sentence head x P_match, and
SentenceResult.match_prob exposes it. Mild errors (one changed word) are the word level's job:
the fit keeps P_match ~ 1 there.

    e = evidence(lpa, ref, ev, tab)          # sentence-level recogniser evidence (no present_prob)
    x = inputs(e, present_probs)             # the model's clipped inputs, in INPUTS order
    p = prob(cal, x)
"""
from __future__ import annotations

import math

import numpy as np

INPUTS = ("ins_rate", "sub_rate", "del_rate", "absent_share", "against_share", "short_tokens")
CLIPS = {"ins_rate": (0.0, 1.5), "sub_rate": (0.0, 1.0), "del_rate": (0.0, 1.0), "absent_share": (0.0, 1.0),
         "against_share": (0.0, 1.0), "short_tokens": (0.0, 3.0)}
AGAINST_LLR = -3.0       # a sound with llr_sub below this: strong evidence it was not said as written


def evidence(lpa, ref, ev, tab) -> dict:
    """Greedy decode vs reference edit rates (the round-5 disfluency definition of
    pronunciada.delivery), share of sounds with strong evidence against them, and how many
    recognised tokens there are per reference token."""
    from pronunciada.delivery import edit_ops
    from pronunciada.text.ipa import to_ids

    lpa = np.asarray(lpa)
    T = len(lpa)
    ids = lpa.argmax(-1) if T else np.zeros(0, np.int64)
    change = np.r_[True, ids[1:] != ids[:-1]] if T else np.zeros(0, bool)
    tok = ids[np.flatnonzero(change)]
    hyp = tok[tok >= 4]
    refids = [t for p in ref.phones for t in to_ids(p.ipa)]
    i_, s_, d_ = edit_ops(refids, hyp)
    nr = max(len(refids), 1)
    sub = np.nan_to_num(np.asarray(tab["llr_sub"], dtype=np.float64), nan=0.0)
    return {"ins_rate": i_ / nr, "sub_rate": s_ / nr, "del_rate": d_ / nr,
            "against_share": float(np.mean(sub < AGAINST_LLR)) if len(sub) else 0.0,
            "tok_ratio": (len(hyp) + 1.0) / (nr + 1.0)}


def inputs(e: dict, present_probs) -> np.ndarray:
    pr = [p for p in present_probs if p is not None]
    raw = {**e, "absent_share": float(np.mean([p < 0.5 for p in pr])) if pr else 0.0,
           "short_tokens": max(0.0, -math.log(max(e["tok_ratio"], 1e-6)))}
    return np.array([float(np.clip(raw[k], *CLIPS[k])) for k in INPUTS])


def prob(cal: dict, x) -> float:
    b = np.asarray(cal["coef"], dtype=np.float64)
    return float(1.0 / (1.0 + np.exp(-(b[0] + np.asarray(x, np.float64) @ b[1:]))))
