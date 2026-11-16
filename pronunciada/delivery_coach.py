"""Word-level reading miscues and delivery flags for coaching (round 3). numpy only; reads the
evidence cached on a SentenceResult, changes no score.

Inputs (one implementation shared by pronunciada.coach and bench/coach/dump_r3.py):
  extra_segments(c)    decoded speech that belongs to no expected word: greedy (free) tokens of
                       the recogniser lying outside every word's aligned span, grouped into
                       segments; position = (word before, word after), -1 / n at the ends
  delivery_inputs(c)   pauses with their positions and the prompt's punctuation there, the
                       silence at every sentence-internal punctuation mark, articulation rate,
                       pitch range (semitones, p95 - p5 of voiced F0 inside the speech span)
Thresholds come from pronunciada/data/coach/model.json ("extra_speech", "delivery"), fitted on
FLEURS validation natives (bench/coach/fit_r3.py).
"""
from __future__ import annotations

import re

import numpy as np

FRAME = 0.02
T0 = 0.03
STOP = re.compile(r"[.!?;:。！？；：…]")
COMMA = re.compile(r"[,，、—–\-]")


def _t(frame) -> float:
    return round(float(frame) * FRAME + T0, 3)


def word_spans(c) -> list[tuple[int, int] | None]:
    ref, ev = c["ref"], c["ev"]
    out = []
    for w in ref.words:
        sp = [(s.start_frame, s.end_frame) for s in ev.slots[w.phones[0]:w.phones[1]] if s.end_frame > s.start_frame]
        out.append((min(a for a, _ in sp), max(b for _, b in sp)) if sp else None)
    return out


def punctuation_between(ref, k: int) -> str | None:
    """"stop" / "comma" / None: the prompt's punctuation between word k and word k + 1."""
    if k < 0 or k + 1 >= len(ref.words):
        return None
    a = ref.words[k].char_span[1]
    b = ref.words[k + 1].char_span[0]
    gap = ref.text[a:b] if b >= a else ""
    if STOP.search(gap):
        return "stop"
    if COMMA.search(gap):
        return "comma"
    return None


def extra_segments(c, join_frames: int = 10) -> list[dict]:
    from .text.ipa import MODIFIERS

    ev, ref = c["ev"], c["ref"]
    spans = word_spans(c)
    have = [(k, s) for k, s in enumerate(spans) if s is not None]
    toks = list(getattr(ev, "free_tokens", []) or [])
    segs: list[dict] = []
    cur = None
    for tok, a, b in toks:
        mid = 0.5 * (a + b)
        inside = any(s[0] - 1 <= mid < s[1] + 1 for _, s in have)
        if inside:
            if cur is not None:
                segs.append(cur)
                cur = None
            continue
        if cur is not None and a - cur["end_frame"] <= join_frames:
            cur["end_frame"] = int(b)
            cur["tokens"].append(tok)
        else:
            if cur is not None:
                segs.append(cur)
            cur = {"start_frame": int(a), "end_frame": int(b), "tokens": [tok]}
    if cur is not None:
        segs.append(cur)
    out = []
    for g in segs:
        before = [k for k, s in have if s[1] <= g["start_frame"] + 1]
        after = [k for k, s in have if s[0] >= g["end_frame"] - 1]
        prev = before[-1] if before else -1
        nxt = after[0] if after else len(ref.words)
        heard = "".join(g["tokens"])
        n_base = sum(1 for t in g["tokens"] if t not in MODIFIERS)
        restart = False
        if 0 <= nxt < len(ref.words):          # a restarted word: the next word's first sounds
            exp = "".join(p.ipa for p in ref.phones[ref.words[nxt].phones[0]:ref.words[nxt].phones[1]])
            restart = n_base >= 1 and len(heard) >= 1 and exp.startswith(heard[:2])
        out.append({"start": _t(g["start_frame"]), "end": _t(g["end_frame"]), "heard_ipa": heard,
                    "n_sounds": n_base, "after_word": prev, "before_word": nxt, "restart": bool(restart)})
    return out


def _f0_range_st(c) -> float | None:
    trk = c.get("f0")
    if trk is None and c.get("wav") is not None:
        from .prosody.f0 import track

        trk = track(np.asarray(c["wav"], dtype=np.float32))
        c["f0"] = trk
    if trk is None:
        return None
    spans = [s for s in word_spans(c) if s is not None]
    if not spans:
        return None
    t0, t1 = _t(spans[0][0]), _t(max(b for _, b in spans))
    tm = trk.times
    m = (tm >= t0) & (tm <= t1) & np.asarray(trk.voiced, bool) & (np.asarray(trk.f0) > 0)
    if m.sum() < 10:
        return None
    st = 12.0 * np.log2(np.asarray(trk.f0)[m] / 100.0)
    return float(np.percentile(st, 95) - np.percentile(st, 5))


def delivery_inputs(c, pitch: bool = True) -> dict:
    from .delivery import speech_segments
    from .features import pacing

    ref, ev = c["ref"], c["ev"]
    s = np.array([x.start_frame for x in ev.slots])
    e = np.array([x.end_frame for x in ev.slots])
    word_of = np.array([p.word for p in ref.phones])
    _, pauses = speech_segments(s, e)
    P = []
    for a, b, i, j in pauses:
        wi, wj = int(word_of[i]), int(word_of[j])
        P.append({"start": _t(a), "end": _t(b), "duration": round((b - a) * FRAME, 3), "after_word": wi,
                  "before_word": wj, "in_word": wi == wj,
                  "punctuation": None if wi == wj else punctuation_between(ref, wj - 1)})
    spans = word_spans(c)
    marks = []
    for k in range(len(ref.words) - 1):
        pk = punctuation_between(ref, k)
        if pk and spans[k] is not None and spans[k + 1] is not None:
            marks.append({"after_word": k, "kind": pk, "gap": round((spans[k + 1][0] - spans[k][1]) * FRAME, 3),
                          "time": _t(spans[k][1])})
    dur = float(c["dur"]) if c.get("dur") else len(ev.slots and c["lpa"]) * FRAME
    pc = pacing(s, e, dur)
    return {"pauses": P, "marks": marks, "articulation_rate": pc["articulation_rate"],
            "n_sounds": int((e > s).sum()), "f0_range_st": _f0_range_st(c) if pitch else None}


# ----------------------------------------------------------------------------- items

def _it(kind, words, wi, start, end, eprob=None, extra=None, heard=None, expected=None):
    from .coach import SCHEMA_MINOR, SCHEMA_VERSION

    w = words[wi].text if wi is not None and 0 <= wi < len(words) else None
    d = {"schema_version": SCHEMA_VERSION, "schema_minor": SCHEMA_MINOR, "kind": kind, "word": w, "word_index": wi, "syllable_index": None,
         "phone_index": None, "position": None, "expected": expected, "heard": heard, "expected_ipa": None,
         "heard_ipa": heard if kind == "extra_speech" else None, "diff_keys": [], "error_prob": eprob,
         "prob": None, "confidence": None, "start": start, "end": end, "gate": None}
    d.update(extra or {})
    return d


def miscue_items(result, sound_items: list[dict], cfg: dict) -> list[dict]:
    """mispronounced_word (a word with flagged sounds) and extra_speech items."""
    c = result._pronunciada  # noqa: SLF001
    out = []
    by_word: dict[int, list] = {}
    for it in sound_items:
        if it["kind"] in ("substitution", "missing"):
            by_word.setdefault(it["word_index"], []).append(it)
    for wi, its in sorted(by_word.items()):
        w = result.words[wi]
        ep = max(i["error_prob"] for i in its)
        out.append(_it("mispronounced_word", result.words, wi, w.start, w.end, round(ep, 4),
                       {"n_sounds_flagged": len(its), "prob": round(float(w.prob), 4)}, expected=w.text))
    X = cfg.get("extra_speech")
    if X:
        for g in extra_segments(c):
            if g["n_sounds"] >= X["min_sounds"].get(c["ref"].lang, X["min_sounds"]["default"]):
                pos = ("start" if g["after_word"] < 0 else "end" if g["before_word"] >= len(result.words)
                       else "between")
                out.append(_it("extra_speech", result.words, None, g["start"], g["end"], None,
                               {"after_word": g["after_word"], "before_word": g["before_word"], "where": pos,
                                "restart": g["restart"], "n_sounds": g["n_sounds"]}, heard=g["heard_ipa"]))
    return out


def delivery_flags(D: dict, T: dict, n_words: int) -> list[tuple[str, dict]]:
    """(kind, payload) for every delivery flag of one utterance's delivery_inputs D under the
    language thresholds T (shared by coaching and the benchmarks)."""
    out = []
    for p in D["pauses"]:
        if p["duration"] >= T["long_pause_s"]:
            out.append(("long_pause", {"duration": p["duration"], "after_word": p["after_word"],
                                       "before_word": p["before_word"], "at_punctuation": p["punctuation"] is not None,
                                       "start": p["start"], "end": p["end"]}))
        elif p["punctuation"] is None and p["duration"] >= T["mid_pause_s"]:
            out.append(("pause_mid_phrase", {"duration": p["duration"], "after_word": p["after_word"],
                                             "before_word": p["before_word"], "in_word": p["in_word"],
                                             "start": p["start"], "end": p["end"]}))
    for m in D["marks"]:
        g = (T.get("rushed") or {}).get(m["kind"])
        if g is not None and m["gap"] < g:
            out.append(("rushed_punctuation", {"punctuation": m["kind"], "gap": m["gap"], "after_word": m["after_word"],
                                               "start": m["time"], "end": m["time"]}))
    r = D.get("f0_range_st")
    if r is not None and T.get("flat_range_st") is not None and r < T["flat_range_st"] and n_words >= 3:
        out.append(("flat_pitch", {"pitch_range_st": round(r, 2)}))
    rate = D["articulation_rate"]
    if D["n_sounds"] >= 8 and rate > 0:
        if rate < T["slow_rate"]:
            out.append(("slow", {"sounds_per_s": round(rate, 2)}))
        elif rate > T["fast_rate"]:
            out.append(("fast", {"sounds_per_s": round(rate, 2)}))
    return out


# Kinds reported by default. slow and flat_pitch are computed but off by default: on
# speechocean762 train they fire on 2/3 and 1/2 of the (short, read) learner utterances and do
# not go with lower expert fluency / prosody scores (results/coaching.md); coaching(delivery="all").
DEFAULT_DELIVERY = ("long_pause", "pause_mid_phrase", "rushed_punctuation", "fast")


def delivery_items(result, cfg: dict, kinds=DEFAULT_DELIVERY) -> list[dict]:
    c = result._pronunciada  # noqa: SLF001
    lang = c["ref"].lang
    T = (cfg.get("delivery") or {}).get(lang)
    if not T:
        return []
    if "flat_pitch" not in kinds:
        T = dict(T, flat_range_st=None)       # no pitch track needed
    W = result.words
    out = []
    for kind, p in delivery_flags(delivery_inputs(c, pitch=T.get("flat_range_st") is not None), T, len(W)):
        if kind not in kinds:
            continue
        wi = p.get("after_word")
        st, en = p.pop("start", W[0].start), p.pop("end", W[-1].end)
        out.append(_it(kind, W, wi, st, en, None, p))
    return out
