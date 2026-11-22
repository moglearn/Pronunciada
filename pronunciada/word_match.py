"""Round 19: does what the free (greedy) decode heard over a word match the word's text?

The sound scores judge each expected sound on the canonical alignment; a sound said in addition
(the text lost one: おばさん written, おばあさん / はおばさん said) or a different sequence the
alignment squeezes into the expected sounds leaves little trace there. This module compares,
per word, the greedy tokens heard over the word's region (its sounds' frames plus half of the gaps
to the neighbouring words) with the word's sounds, each sound allowed to match its canonical form
or any of its alternatives (length marks ignored; a doubled consonant matches one token).

    mismatch(ref, ev) -> [{"ins": k, "del": k, "sub": k, "n": n_sounds, "heard": "..."}] per word

Counts come from a minimum-cost alignment (insertion / deletion / substitution cost 1 each).
"""
from __future__ import annotations

from functools import lru_cache

_DROP = {"ː"}


def _toks(s: str) -> tuple[str, ...]:
    out = [c for c in s if c not in _DROP]
    # a doubled consonant (geminate) or a repeated vowel is one token here: length is the length
    # channel's business
    k = []
    for c in out:
        if k and c == k[-1]:
            continue
        k.append(c)
    return tuple(k)


def _variants(p) -> list[tuple[str, ...]]:
    vs = {_toks(p.ipa)}
    for a, _ in p.alts:
        vs.add(_toks(a))
    return sorted(vs, key=len)


@lru_cache(maxsize=1 << 16)
def _edit(a: tuple, b: tuple) -> tuple[int, int, int]:
    """(ins, del, sub) of the cheapest alignment of heard b against variant a (b - a insertions)."""
    n, m = len(a), len(b)
    D = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = None
            if i and j:
                c = D[i - 1][j - 1]
                s = int(a[i - 1] != b[j - 1])
                best = (c[0] + s, c[1], c[2], c[3] + s)
            if i:
                c = D[i - 1][j]
                x = (c[0] + 1, c[1], c[2] + 1, c[3])
                best = x if best is None or x < best else best
            if j:
                c = D[i][j - 1]
                x = (c[0] + 1, c[1] + 1, c[2], c[3])
                best = x if best is None or x < best else best
            D[i][j] = best
    return D[n][m][1:]


def align_word(phones, heard: tuple[str, ...]) -> tuple[int, int, int]:
    """Minimum-cost alignment of a word's sounds (each with its variants) to heard tokens."""
    n, m = len(phones), len(heard)
    INF = (1 << 30, 0, 0, 0)
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = (0, 0, 0, 0)
    for j in range(1, m + 1):
        D[0][j] = (j, j, 0, 0)
    vars_ = [_variants(p) for p in phones]
    for i in range(1, n + 1):
        V = vars_[i - 1]
        for j in range(m + 1):
            best = INF
            # the sound consumes heard[k:j] (0..3 tokens) as one of its variants
            for k in range(max(0, j - 3), j + 1):
                c = D[i - 1][k]
                if c[0] >= INF[0]:
                    continue
                seg = heard[k:j]
                for v in V:
                    ins, dl, sb = _edit(v, seg)
                    x = (c[0] + ins + dl + sb, c[1] + ins, c[2] + dl, c[3] + sb)
                    if x < best:
                        best = x
            # an extra heard token before this sound's segment
            if j:
                c = D[i][j - 1]
                if c[0] < INF[0]:
                    x = (c[0] + 1, c[1] + 1, c[2], c[3])
                    if x < best:
                        best = x
            D[i][j] = best
    return D[n][m][1:]


def mismatch(ref, ev) -> list[dict]:
    slots = ev.slots
    free = ev.free_tokens
    mids = [(a + b) / 2 for _, a, b in free]
    spans = []
    for w in ref.words:
        ss = [(slots[i].start_frame, slots[i].end_frame) for i in range(*w.phones)
              if slots[i].end_frame > slots[i].start_frame]
        spans.append((min(a for a, _ in ss), max(b for _, b in ss)) if ss else None)
    out = []
    for k, w in enumerate(ref.words):
        sp = spans[k]
        phones = [ref.phones[i] for i in range(*w.phones)]
        if sp is None:
            out.append({"ins": 0, "del": len(phones), "sub": 0, "n": len(phones), "heard": ""})
            continue
        lo, hi = sp
        prv = next((spans[j] for j in range(k - 1, -1, -1) if spans[j] is not None), None)
        nxt = next((spans[j] for j in range(k + 1, len(spans)) if spans[j] is not None), None)
        a = (prv[1] + lo) / 2 if prv is not None else -1e9
        b = (hi + nxt[0]) / 2 if nxt is not None else 1e9
        heard = tuple(t for (t, *_), mm in zip(free, mids) if a <= mm < b and t not in _DROP)
        heard = _toks("".join(heard))
        ins, dl, sb = align_word(phones, heard)
        out.append({"ins": ins, "del": dl, "sub": sb, "n": len(phones), "heard": "".join(heard)})
    return out


def extra_counts(ref, ev) -> list[int]:
    """Per word: sounds heard in addition to its text (the "ins" of mismatch)."""
    return [m["ins"] for m in mismatch(ref, ev)]
