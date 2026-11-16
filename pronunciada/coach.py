"""Coaching feedback: structured, versioned diagnoses of what went wrong in one utterance.

    items = result.coaching()            # == pronunciada.coach.coaching(result)
    tips = pronunciada.tips.tips(items)  # optional default English tip sentences

Every item is a plain JSON-serialisable dict with the same keys (SCHEMA_VERSION 1):

  schema_version  1
  kind            "substitution" | "missing" | "extra" | "tone" | "stress" | "missing_word"
  word, word_index        the prompt word and its index in result.words
  syllable_index          index into that word's syllables (None for missing_word)
  phone_index             index into that word's phones (sounds only, else None)
  position                "initial" | "medial" | "final" | "only" (sound in its word)
  expected, heard         sounds: IPA strings; tones: 1-5; stress: syllable index in the word;
                          heard None = flagged, but what was heard is not reliable enough to say
  expected_ipa, heard_ipa sounds only (same as expected / heard), else None
  diff_keys               articulatory features that differ (pronunciada.articulatory.diff);
                          ["tone"] / ["stress_position"]; [] when heard is None
  prob                    the calibrated probability the item was acceptable (PhoneResult.prob,
                          SyllableResult.tone_prob / stress_prob, WordResult.present_prob)
  confidence              estimated precision of the diagnosis: P(heard is right within one
                          articulatory feature) from the calibration table (None without heard)
  start, end              seconds
  gate                    the gate inputs, for logging (heard_post, log_ratio, n_frames)

The reliability gate (when `heard` is filled in) was set on FLEURS validation with planted
substitutions (bench/coach/fit_gate.py, results/coaching.md); thresholds live in
pronunciada/data/coach/gate.json and can be overridden per call.
Nothing here changes any probability: it reads the evidence cached on the result.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources

import numpy as np

from . import articulatory as A
from .text.ipa import MODIFIERS, token_ids

SCHEMA_VERSION = 1
KINDS = ("substitution", "missing", "extra", "tone", "stress", "missing_word")
DEFAULT_GATE = {
    "flag_prob": 0.5,          # a sound / tone / stress item is reported when prob < this
    "min_log_ratio": 0.0,      # log(P heard / P expected) over the sound's span, at least
    "min_heard_post": 0.05,    # geometric-mean span posterior of the heard sound, at least
    "min_frames": 0,           # aligned frames of the sound, at least
    "missing_max_frames": 0,   # "missing" when the sound got at most this many frames ...
    "missing_min_llr_del": 0.0,  # ... and the "skipped" evidence llr_del is at most this
    "tone_min_post": 0.5,      # heard tone: its posterior at least this ...
    "tone_min_log_ratio": 0.0,  # ... and log(P heard tone / P expected tone) at least this
    "stress_min_post": 0.5,    # heard stressed syllable: its share at least this
    "missing_word_prob": 0.5,  # missing_word when present_prob < this
    "confidence": {"edges": [], "values": [0.85]},   # log_ratio bins -> precision
    "tone_confidence": 0.8, "stress_confidence": 0.8, "missing_confidence": 0.8,
    "by_lang": {},             # per-language overrides of the above (fitted where needed)
}


@lru_cache(maxsize=1)
def _shipped_gate() -> dict:
    p = resources.files("pronunciada").joinpath("data/coach/gate.json")
    try:
        g = json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        g = {}
    return {**DEFAULT_GATE, **g.get("thresholds", {})}


def gate_params(**overrides) -> dict:
    """Shipped gate thresholds with keyword overrides."""
    return {**_shipped_gate(), **overrides}


@lru_cache(maxsize=None)
def inventory(lang: str) -> tuple[str, ...]:
    p = resources.files("pronunciada").joinpath("data/coach/inventory.json")
    return tuple(json.loads(p.read_text(encoding="utf-8"))["phones"].get(lang, ()))


@lru_cache(maxsize=None)
def _cand_ids(lang: str, vowel: bool):
    """Candidate sounds of the same type (vowel / consonant) as the expected one: planted and
    annotated learner substitutions are within a type, and a cross-type "heard" is nearly
    always a boundary effect of the alignment."""
    tid = token_ids()
    cands = [c for c in inventory(lang) if A.covered(c) and (A.features(c)["type"] == "vowel") == vowel]
    return cands, [np.array([tid[ch] for ch in c]) for c in cands]


def span_mean(lpa: np.ndarray, s: int, e: int) -> np.ndarray:
    """Mean posterior over the sound's span padded by one frame (pronunciada.features rule)."""
    T = len(lpa)
    a, b = (max(s - 1, 0), min(s + 1, T)) if e <= s else (max(s - 1, 0), min(e + 1, T))
    if b <= a:
        a, b = max(T - 1, 0), T
    return np.exp(lpa[a:b].astype(np.float64)).mean(0)


def heard_candidate(m: np.ndarray, expected: str, alts, lang: str):
    """Best other sound of the language by the geometric-mean span posterior of its tokens.
    Returns (heard, log P heard, log P expected)."""
    tid = token_ids()
    lm = np.log(m + 1e-6)
    cands, ids = _cand_ids(lang, A.features(expected)["type"] == "vowel")
    skip = {expected, *alts}
    best, bs = None, -math.inf
    for c, ix in zip(cands, ids):
        if c in skip:
            continue
        s = float(lm[ix].mean())
        if s > bs:
            best, bs = c, s
    le = float(lm[[tid[ch] for ch in expected]].mean())
    return best, bs, le


def _confidence(x: float, table: dict) -> float:
    edges, vals = table["edges"], table["values"]
    k = int(np.searchsorted(np.asarray(edges, dtype=np.float64), x, side="right")) if edges else 0
    return float(vals[min(k, len(vals) - 1)])


def _position(k: int, n: int) -> str:
    return "only" if n == 1 else "initial" if k == 0 else "final" if k == n - 1 else "medial"


def _item(kind, w, wi, si, pi, pos, expected, heard, diff, prob, conf, start, end, gate=None, ipa=True):
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "word": w.text, "word_index": wi,
            "syllable_index": si, "phone_index": pi, "position": pos, "expected": expected,
            "heard": heard, "expected_ipa": expected if ipa else None,
            "heard_ipa": heard if ipa else None, "diff_keys": list(diff),
            "prob": round(float(prob), 4), "confidence": None if conf is None else round(float(conf), 3),
            "start": start, "end": end, "gate": gate}


def phone_diagnosis(m, expected, alts, lang, prob, n_frames, slot_heard, llr_del, g):
    """(kind, heard, diff_keys, confidence, gate inputs) for one flagged sound."""
    if n_frames <= g["missing_max_frames"] and not slot_heard and llr_del <= g["missing_min_llr_del"]:
        return "missing", "", [], g["missing_confidence"], {"n_frames": int(n_frames), "llr_del": round(float(llr_del), 3)}
    h, lh, le = heard_candidate(m, expected, alts, lang)
    lr = lh - le
    gin = {"heard_post": round(math.exp(lh), 4), "log_ratio": round(lr, 3), "n_frames": int(n_frames),
           "candidate": h}
    if (h is None or lr < g["min_log_ratio"] or math.exp(lh) < g["min_heard_post"]
            or n_frames < g["min_frames"]):
        return "substitution", None, [], None, gin
    try:
        d = A.diff(expected, h)
    except KeyError:
        return "substitution", None, [], None, gin
    return "substitution", h, d, _confidence(lr, g["confidence"]), gin


def _rank(it) -> float:
    c = it["confidence"] if it["confidence"] is not None else 0.4
    return (1.0 - it["prob"]) * c


def coaching(result, max_items: int | None = None, **gate) -> list[dict]:
    """Coaching items for a SentenceResult from Scorer.score / rescore, most important first
    (by (1 - prob) x confidence; flag-only items count with confidence 0.4). Gate thresholds:
    pronunciada/data/coach/gate.json, override with keyword arguments (see DEFAULT_GATE)."""
    c = getattr(result, "_pronunciada", None)
    if c is None:
        raise ValueError("result has no cached evidence (was it produced by Scorer.score?)")
    ref, ev, tab = c["ref"], c["ev"], c["tab"]
    lang = ref.lang
    g = gate_params(**gate)
    g = {**g, **{k: v for k, v in g.get("by_lang", {}).get(lang, {}).items() if k not in gate}}
    lpa = c.get("lpa")
    pr = c.get("prosody") or []
    items: list[dict] = []
    for wi, w in enumerate(result.words):
        rw = ref.words[wi]
        p0, p1 = rw.phones
        s0, _s1 = rw.syllables
        if w.present_prob is not None and w.present_prob < g["missing_word_prob"]:
            items.append(_item("missing_word", w, wi, None, None, None, w.text, None, [],
                               w.present_prob, g["missing_confidence"], w.start, w.end, ipa=False))
            continue
        for k, p in enumerate(w.phones):
            i = p0 + k
            if p.prob >= g["flag_prob"]:
                continue
            sl = ev.slots[i]
            rp = ref.phones[i]
            m = span_mean(lpa, sl.start_frame, sl.end_frame) if lpa is not None and len(lpa) else None
            if m is None:
                kind, h, d, conf, gin = "substitution", None, [], None, None
            else:
                kind, h, d, conf, gin = phone_diagnosis(
                    m, rp.ipa, [a for a, _ in rp.alts], lang, p.prob, sl.n_frames, sl.heard,
                    float(tab["llr_del"][i]), g)
            items.append(_item(kind, w, wi, rp.syllable - s0, k, _position(k, len(w.phones)),
                               rp.ipa, h, d, p.prob, conf, p.start, p.end, gin))
        for k, sy in enumerate(w.syllables):
            si = s0 + k
            d = pr[si] if si < len(pr) else {}
            # tones / stress: the syllable AND the word's calibrated channel (WordResult.tone_prob
            # / stress_prob, what the word score uses) must both be below the flag bar
            wt = w.tone_prob if w.tone_prob is not None else 0.0
            ws = w.stress_prob if w.stress_prob is not None else 0.0
            if (sy.tone_prob is not None and sy.tone_prob < g["flag_prob"] and wt < g["flag_prob"]
                    and d.get("tone_probs")):
                P = np.asarray(d["tone_probs"], dtype=np.float64)
                acc = {sy.tone_expected, *ref.syllables[si].tone_alts}
                ht = int(P.argmax()) + 1
                lr = math.log(P[ht - 1] + 1e-6) - math.log(P[sy.tone_expected - 1] + 1e-6)
                ok = ht not in acc and P[ht - 1] >= g["tone_min_post"] and lr >= g["tone_min_log_ratio"]
                items.append(_item("tone", w, wi, k, None, None, sy.tone_expected, ht if ok else None,
                                   ["tone"] if ok else [], sy.tone_prob,
                                   g["tone_confidence"] if ok else None, sy.start, sy.end,
                                   {"heard_post": round(float(P[ht - 1]), 4), "log_ratio": round(lr, 3)},
                                   ipa=False))
            if (sy.stress_prob is not None and sy.stress_prob < g["flag_prob"] and ws < g["flag_prob"]
                    and d.get("stress_dist")):
                P = np.asarray(d["stress_dist"], dtype=np.float64)
                hs = int(P.argmax())
                ok = hs != k and P[hs] >= g["stress_min_post"]
                items.append(_item("stress", w, wi, k, None, None, k, hs if ok else None,
                                   ["stress_position"] if ok else [], sy.stress_prob,
                                   g["stress_confidence"] if ok else None, w.start, w.end,
                                   {"heard_post": round(float(P[hs]), 4), "n_syllables": len(P)},
                                   ipa=False))
    items.sort(key=_rank, reverse=True)
    return items[:max_items] if max_items else items


def to_json(items: list[dict], **kw) -> str:
    return json.dumps({"schema_version": SCHEMA_VERSION, "items": items}, ensure_ascii=False, **kw)


__all__ = ["SCHEMA_VERSION", "KINDS", "DEFAULT_GATE", "coaching", "gate_params", "heard_candidate",
           "phone_diagnosis", "to_json", "MODIFIERS"]
