"""Round 20: extra and missing sounds from the unconstrained greedy decode.

The aligner judges each expected sound at its own position; a sound said in addition, or an
expected sound that was not said at all, leaves only indirect traces there. This module compares,
per word, the collapsed greedy decode of the (adapter-corrected) frame scores with the word's
expected sounds:

* decoded units (a base token plus its modifiers; length marks ignored, a repeated unit counted
  once) are assigned to a word when their middle frame falls inside the word's aligned span
  widened by `tol` frames, or by half the gap to the neighbouring words (`tol` = "half");
* a weighted edit distance aligns them with the word's sounds; each sound may match its
  canonical form or (with `alts`) any allowed alternative, an allowed empty alternative makes
  dropping it free; substitution costs come from the articulatory feature distance
  (`pronunciada.articulatory`), insertion and deletion costs per unit from a small table of
  how often natives add or drop that unit (cheaper when natives often do);
* per word: weighted insertions, weighted deletions, their sum per sound, the mean top-1
  probability of the inserted units ("mass"), whether an edit touches the word's edge; per
  sound: dropped in the greedy decode.

Pure python + numpy; no model. Offline ablation only (bench/round20_insdel); it was not adopted, so it
lives outside the package.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

LEN = {"ː", "ˑ"}


@lru_cache(maxsize=None)
def _mods() -> frozenset:
    from pronunciada.articulatory import MODIFIERS

    return frozenset(MODIFIERS)


def units(s: str) -> tuple[str, ...]:
    """Base + modifier units of a token string; length marks dropped; repeats collapsed."""
    M = _mods()
    out: list[str] = []
    for ch in s:
        if ch in LEN:
            continue
        if ch in M and out:
            out[-1] += ch
        else:
            out.append(ch)
    k: list[str] = []
    for u in out:
        if k and u == k[-1]:
            continue
        k.append(u)
    return tuple(k)


@lru_cache(maxsize=1 << 16)
def sub_cost(a: str, b: str, weighted: bool) -> float:
    if a == b:
        return 0.0
    if not weighted:
        return 1.0
    from pronunciada.articulatory import diff

    try:
        d = diff(a, b)
    except KeyError:
        return 1.0
    if d == ["type"]:
        return 1.0
    return min(1.0, 0.2 + 0.2 * len(d))


def base(u: str) -> str:
    return u[:1]


class Costs:
    """Insertion / deletion costs per unit base symbol (default 1)."""

    def __init__(self, ins: dict | None = None, dele: dict | None = None, weighted: bool = True):
        self.ins = ins or {}
        self.dele = dele or {}
        self.weighted = weighted

    def i(self, u: str) -> float:
        return self.ins.get(base(u), 1.0) if self.weighted else 1.0

    def d(self, u: str) -> float:
        return self.dele.get(base(u), 1.0) if self.weighted else 1.0


def _edit(v: tuple, seg: tuple, C: Costs):
    """Cheapest alignment of a variant's units v with heard units seg:
    (cost, ins_cost, del_cost, n_ins, n_del, inserted positions in seg)."""
    n, m = len(v), len(seg)
    D = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = (0.0, 0.0, 0.0, 0, 0, ())
    for j in range(1, m + 1):
        c = D[0][j - 1]
        w = C.i(seg[j - 1])
        D[0][j] = (c[0] + w, c[1] + w, c[2], c[3] + 1, c[4], c[5] + (j - 1,))
    for i in range(1, n + 1):
        c = D[i - 1][0]
        w = C.d(v[i - 1])
        D[i][0] = (c[0] + w, c[1], c[2] + w, c[3], c[4] + 1, c[5])
        for j in range(1, m + 1):
            c = D[i - 1][j - 1]
            best = (c[0] + sub_cost(v[i - 1], seg[j - 1], C.weighted),) + c[1:]
            c = D[i - 1][j]
            w = C.d(v[i - 1])
            x = (c[0] + w, c[1], c[2] + w, c[3], c[4] + 1, c[5])
            if x[0] < best[0] - 1e-12:
                best = x
            c = D[i][j - 1]
            w = C.i(seg[j - 1])
            x = (c[0] + w, c[1] + w, c[2], c[3] + 1, c[4], c[5] + (j - 1,))
            if x[0] < best[0] - 1e-12:
                best = x
            D[i][j] = best
    return D[n][m]


def align_word(variants: list[list[tuple]], free_drop: list[bool], heard: tuple, C: Costs):
    """variants[i]: the unit tuples sound i may be said as; free_drop[i]: dropping it costs 0.
    Returns (cost, ins_cost, del_cost, n_ins, n_del, inserted heard positions, dropped sounds)."""
    n, m = len(variants), len(heard)
    INF = (math.inf,)
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = (0.0, 0.0, 0.0, 0, 0, (), ())
    for j in range(1, m + 1):
        c = D[0][j - 1]
        w = C.i(heard[j - 1])
        D[0][j] = (c[0] + w, c[1] + w, c[2], c[3] + 1, c[4], c[5] + (j - 1,), c[6])
    for i in range(1, n + 1):
        V = variants[i - 1]
        for j in range(m + 1):
            best = INF
            for k in range(max(0, j - 3), j + 1):
                c = D[i - 1][k]
                if c[0] == math.inf:
                    continue
                seg = heard[k:j]
                if not seg:
                    w = 0.0 if free_drop[i - 1] else min(sum(C.d(u) for u in v) for v in V)
                    x = (c[0] + w, c[1], c[2] + w, c[3], c[4] + (0 if free_drop[i - 1] else 1), c[5],
                         c[6] + ((i - 1,) if not free_drop[i - 1] else ()))
                else:
                    e = min((_edit(v, seg, C) for v in V), key=lambda t: t[0])
                    x = (c[0] + e[0], c[1] + e[1], c[2] + e[2], c[3] + e[3], c[4] + e[4],
                         c[5] + tuple(k + q for q in e[5]), c[6])
                if x[0] < best[0] - 1e-12:
                    best = x
            if j:
                c = D[i][j - 1]
                if c[0] < math.inf:
                    w = C.i(heard[j - 1])
                    x = (c[0] + w, c[1] + w, c[2], c[3] + 1, c[4], c[5] + (j - 1,), c[6])
                    if x[0] < best[0] - 1e-12:
                        best = x
            D[i][j] = best
    return D[n][m]


def heard_units(free: list[tuple[str, int, int]], top1_lp: np.ndarray | None = None):
    """Group greedy tokens into units: [(unit, start, end, mean top-1 prob)] (length marks dropped,
    modifiers attached to the preceding token, repeated units merged)."""
    M = _mods()
    out: list[list] = []
    for t, a, b in free:
        if t in LEN:
            if out:
                out[-1][2] = b
            continue
        if t in M and out:
            out[-1][0] += t
            out[-1][2] = b
            continue
        if out and out[-1][0] == t:
            out[-1][2] = b
            continue
        out.append([t, a, b])
    res = []
    for u, a, b in out:
        p = float(np.exp(np.asarray(top1_lp[a:b], np.float64)).mean()) if top1_lp is not None and b > a else 1.0
        res.append((u, a, b, p))
    return res


def word_regions(word_of_phone, start, end, n_words: int, T: int, tol):
    """[lo, hi) frame region of each word (None if none of its sounds has frames)."""
    word_of_phone = np.asarray(word_of_phone)
    spans = []
    for k in range(n_words):
        idx = np.flatnonzero((word_of_phone == k) & (np.asarray(end) > np.asarray(start)))
        spans.append((int(np.min(np.asarray(start)[idx])), int(np.max(np.asarray(end)[idx]))) if len(idx) else None)
    out = []
    for k, sp in enumerate(spans):
        if sp is None:
            out.append(None)
            continue
        lo, hi = sp
        if tol == "half":
            prv = next((spans[j] for j in range(k - 1, -1, -1) if spans[j] is not None), None)
            nxt = next((spans[j] for j in range(k + 1, n_words) if spans[j] is not None), None)
            a = (prv[1] + lo) / 2 if prv is not None else -1e9
            b = (hi + nxt[0]) / 2 if nxt is not None else 1e9
        else:
            a, b = lo - float(tol), hi + float(tol)
        out.append((a, b))
    return out


def features(ipa, alts, word_of_phone, start, end, n_words: int, T: int, free, top1_lp=None,
             top_ids=None, *, costs: Costs | None = None, allow_alts: bool = True, tol="half",
             soft_top2: bool = False, tok_id=None):
    """Per-word and per-sound greedy-vs-expected features (see module docstring).

    Returns {"ins", "del", "rate", "mass", "edge", "n_ins", "n_del"} arrays [n_words] and
    "dropped" [n_sounds] (bool)."""
    C = costs or Costs(weighted=False)
    hu = heard_units(free, top1_lp)
    mids = np.array([(a + b) / 2 for _, a, b, _ in hu]) if hu else np.zeros(0)
    regs = word_regions(word_of_phone, start, end, n_words, T, tol)
    wop = np.asarray(word_of_phone)
    out = {k: np.zeros(n_words) for k in ("ins", "del", "rate", "mass", "edge", "n_ins", "n_del")}
    dropped = np.zeros(len(ipa), bool)
    for k in range(n_words):
        idx = np.flatnonzero(wop == k)
        if not len(idx):
            continue
        variants, free_drop = [], []
        for i in idx:
            vs = {units(ipa[i])}
            fd = False
            if allow_alts:
                for a, _w in alts[i]:
                    if a == "":
                        fd = True
                    else:
                        vs.add(units(a))
            vs.discard(())
            variants.append(sorted(vs, key=lambda t: (len(t), t)) or [units(ipa[i])])
            free_drop.append(fd)
        reg = regs[k]
        if reg is None:
            sel = []
        else:
            sel = [q for q in range(len(hu)) if reg[0] <= mids[q] < reg[1]]
        heard = tuple(hu[q][0] for q in sel)
        cost, ic, dc, ni, nd, ins_pos, drops = align_word(variants, free_drop, heard, C)
        if soft_top2 and drops and top_ids is not None and reg is not None and tok_id is not None:
            lo, hi = max(int(math.floor(reg[0])), 0), min(int(math.ceil(reg[1])), T)
            seen = set(np.unique(top_ids[lo:hi, :2]).tolist()) if hi > lo else set()
            for d in drops:
                tid = tok_id.get(base(variants[d][0][0])) if variants[d] and variants[d][0] else None
                if tid is not None and tid in seen:
                    dc -= 0.5 * C.d(variants[d][0][0])
        n = len(idx)
        out["ins"][k] = ic
        out["del"][k] = dc
        out["n_ins"][k] = ni
        out["n_del"][k] = nd
        out["rate"][k] = (ic + dc) / n
        out["mass"][k] = sum(hu[sel[q]][3] for q in ins_pos)
        edge = any(d in (0, n - 1) for d in drops) or any(q in (0, len(heard) - 1) for q in ins_pos)
        out["edge"][k] = float(edge)
        for d in drops:
            dropped[idx[d]] = True
    out["dropped"] = dropped
    return out
