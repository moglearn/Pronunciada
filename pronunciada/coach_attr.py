"""Feature-level claims for flagged sounds (coaching round 4), from the attribute heads
(pronunciada.attr_heads). Shared by coaching() and bench/coach/fit_r4.py.

A head claims "this feature differs" for a flagged sound when 1 - P(the expected sound's class)
>= its bar (model.json "attr"), naming the most probable other class. Heads whose class the
sound's accepted alternatives already change are skipped. When a heard sound is named, a claim
must agree with it (same class as the heard sound, and different from the expected one);
otherwise it is dropped.
"""
from __future__ import annotations

import numpy as np

from . import attr_heads as AH

ORDER = {"height": ("close", "mid", "open"), "backness": ("front", "central", "back")}


def claims_for(P: dict, expected: str, alts, heard: str | None, bars: dict, cand: str | None = None,
               eprob: float | None = None, min_ep: dict | None = None, agree: bool = False) -> list[dict]:
    """P = {head: {class: prob}} for one sound. Returns [{feature, key, expected, heard, p}]
    (articulatory diff key in `key`), strongest first. With agree=True a claim also needs the
    best competing sound of the language (coach.heard_candidate, `cand`) to have the claimed
    class; min_ep[head] is a floor on the sound's coaching score `eprob`."""
    le = AH.labels(expected)
    alt_cls = [AH.labels(a) for a in alts if a]
    lh = AH.labels(heard) if heard else None
    out = []
    for h, probs in P.items():
        if h not in le or h not in bars or bars[h] is None:
            continue
        e = le[h]
        if any(h in la and la[h] != e for la in alt_cls):
            continue
        pd = 1.0 - probs.get(e, 0.0)
        if pd < bars[h]:
            continue
        other = max((c for c in probs if c != e), key=lambda c: probs[c])
        if lh is not None and (h not in lh or lh[h] != other):
            continue                                   # would contradict the named sound
        if agree:
            lc = AH.labels(cand) if cand else {}
            if lc.get(h) != other:
                continue
        if min_ep and eprob is not None and eprob < (min_ep.get(h) or 0.0):
            continue
        out.append({"feature": h, "key": AH.HEADS[h][2], "expected": e, "heard": other, "p": round(float(pd), 4)})
    out.sort(key=lambda d: -d["p"])
    return out


def sound_vectors(result) -> tuple[np.ndarray, np.ndarray] | None:
    from .sound_emb import phone_vectors

    c = result._pronunciada  # noqa: SLF001
    emb = c.get("emb")
    if not emb:
        return None
    V = phone_vectors(emb, c["ev"], len(c["ref"].phones))
    return V, AH.context_vectors(V)
