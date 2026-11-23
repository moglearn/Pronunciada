"""From posteriors to the universal sound model's and the word model's log-odds (the first part
of pronunciada/scorer.py Scorer._calibrate).

    ev = align.score_phones(lp, target["phones"], tokens, insertion=...)
    out = universal_scores(lp, target, ev, "es", tokens, dur)

out["sound"]: per sound z_len, z_str (log-odds of P(lenient ok) / P(strict ok), plus the
language's logit bias) and reg (0-2 regression); out["word"]: per word z_len, z_str of the word
model (data/calib/word.npz) fed with those sound values. These are the inputs of the word and
sentence levels for every language. What the scorer finally reports adds more: the language's own
sound model where config "sound_model" names one (English: sound_B), contrast caps, the round-2
word regression, and the per-language word factors (prosody, length, contrast, identity,
omission) of Scorer._assemble; those are not ported (see the package docstring).
"""
from __future__ import annotations

import json
import os

from . import features as F
from .calibrate import LevelModel

_MODELS = {}


def _data(*parts) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", *parts)


def _model(name: str) -> LevelModel:
    if name not in _MODELS:
        _MODELS[name] = LevelModel(_data("calib", f"{name}.npz"))
    return _MODELS[name]


def config() -> dict:
    with open(_data("calib", "config.json"), encoding="utf-8") as f:
        return json.load(f)


def universal_scores(lp, target, ev, lang: str, tokens, dur=None) -> dict:
    cfg = config()
    uni = _model(cfg["universal_sound_model"])
    word = _model("word")
    bias = float(cfg.get("bias", {}).get(lang, 0.0))
    ns = F.NativeStats.load(lang)
    tab = F.phone_table(lp, target, ev, tokens, ns, dur)
    X = F.matrix(tab, uni.num, uni.cat, uni.vocab, uni.encoding)
    zl, zs, reg = [], [], []
    for row in X:
        o = uni.logits(row)
        zl.append(o["z_len"] + bias)
        zs.append(o["z_str"] + bias)
        reg.append(o["reg"])
    W = F.word_table(tab, zl, zs, reg)
    wzl, wzs = [], []
    for k in range(len(W["wl_mean"])):
        o = word.logits([float(W[c][k]) for c in word.num])
        wzl.append(o["z_len"])
        wzs.append(o["z_str"])
    return {"table": tab, "sound": {"z_len": zl, "z_str": zs, "reg": reg},
            "word": {"z_len": wzl, "z_str": wzs}, "word_table": W}
