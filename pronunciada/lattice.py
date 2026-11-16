"""Reference -> CTC token lattice ("sausage"), DESIGN section 2.

State graph (every state has an implicit self-loop):

* state 0 is the leading blank b0 (slot -1);
* each alternative of slot i is a chain of token states u1..uK, each followed by its own blank
  state b(uk). Arcs: uk -> b(uk), uk -> uk+1 (only if the labels differ), b(uk) -> uk+1;
* a wildcard slot (used for L_wild_i) is slot i with every token state relabelled WILD (emits
  log(1 - p_blank), self-loops as usual). Alternatives of equal length give identical wildcard
  chains, so one chain is kept per distinct length, weighted with the best log-prior among
  those alternatives; the slot may also be skipped (weight 0). A WILD state counts as different
  from every neighbour (no mandatory blank). With log-priors <= 0 and distinct alternatives this
  guarantees L_wild_i >= L_can;
* exits(slot) = {last token, its blank} of every alternative; entries(slot) = first token of
  every alternative (arc weight = the alternative's log-prior). exits(j) -> entries(k) for j < k
  whenever every slot strictly between is skippable (weight = sum of their skip log-priors);
  a direct token -> token arc needs different labels, the blank route is always allowed;
* START -> b0 (weight 0) and START -> entries(k) if slots 0..k-1 are skippable;
  exits(j) -> END if slots j+1.. are skippable; b0 -> END if all slots are skippable.

Every path is therefore a CTC alignment of one lattice hypothesis (choice of alternative per
slot) with the standard rules (optional blanks, mandatory blank between identical tokens, also
across slot boundaries and skipped slots).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .text.ipa import to_ids

BLANK = -1
WILD = -2
NEG_INF = -np.inf


@dataclass
class Slot:
    alts: list[tuple[tuple[int, ...], float, int]]   # (token ids, log weight, alt index) non-empty
    skip: float = NEG_INF                              # log weight of skipping the slot
    wild: list[tuple[int, float]] = field(default_factory=list)   # wildcard chains (length, log w)

    @property
    def skippable(self) -> bool:
        return self.skip > NEG_INF


def slots_from_reference(ref) -> list[Slot]:
    """One Slot per RefPhone: canonical (weight 0, alt index 0) + RefPhone.alts (index k>=1);
    empty alts make the slot skippable (several empty alts: log-sum-exp of their priors)."""
    slots = []
    for p in ref.phones:
        alts = [(tuple(to_ids(p.ipa)), 0.0, 0)]
        skip = NEG_INF
        for k, (a, w) in enumerate(p.alts, start=1):
            if a:
                alts.append((tuple(to_ids(a)), float(w), k))
            else:
                skip = float(np.logaddexp(skip, w))
        slots.append(Slot(alts, skip))
    return slots


def wild_chains(slot: Slot) -> list[tuple[int, float]]:
    best: dict[int, float] = {}
    for ids, w, _ in slot.alts:
        best[len(ids)] = max(best.get(len(ids), NEG_INF), w)
    return sorted(best.items())


def wild_slot(slot: Slot) -> Slot:
    """Wildcard version of `slot`: anything (one chain of WILD states per alt length) or nothing."""
    return Slot([], 0.0, wild=wild_chains(slot))


@dataclass
class Lattice:
    lab: np.ndarray            # [S] token id, BLANK or WILD
    slot: np.ndarray           # [S] slot index (-1 for b0)
    alt: np.ndarray            # [S] alt index (0 canonical, k alts, -2 wildcard, -1 b0)
    is_tok: np.ndarray         # [S] bool: emitting a token / wildcard (not a blank)
    init: np.ndarray           # [S] log START weight
    fin: np.ndarray            # [S] log END weight
    src: np.ndarray            # [A] arcs (self-loops excluded)
    dst: np.ndarray
    w: np.ndarray
    entries: list[list[tuple[int, float]]] = field(default_factory=list)
    exits: list[list[int]] = field(default_factory=list)
    first_tok: list[dict] = field(default_factory=list)   # per slot: alt index -> first token id

    @property
    def n_states(self) -> int:
        return len(self.lab)


def _direct_ok(lab_u: int, lab_v: int) -> bool:
    """May token state u be followed by token state v without a blank in between?"""
    return lab_u == BLANK or lab_u == WILD or lab_v == WILD or lab_u != lab_v


def build_lattice(slots: list[Slot]) -> Lattice:
    lab, slot_of, alt_of, is_tok = [BLANK], [-1], [-1], [False]
    src, dst, w = [], [], []
    entries: list[list[tuple[int, float]]] = []
    exits: list[list[int]] = []
    first_tok: list[dict] = []

    def new(l, s, a, tok):
        lab.append(l); slot_of.append(s); alt_of.append(a); is_tok.append(tok)
        return len(lab) - 1

    for i, sl in enumerate(slots):
        ent, ex, ft = [], [], {}
        chains = [((WILD,) * K, wt, -2) for K, wt in sl.wild] + list(sl.alts)
        for ids, wt, k in chains:
            prev_u = prev_b = None
            for j, tid in enumerate(ids):
                u = new(tid, i, k, True)
                b = new(BLANK, i, k, False)
                src.append(u); dst.append(b); w.append(0.0)
                if prev_u is None:
                    ent.append((u, wt))
                    if k >= 0:
                        ft[k] = tid
                else:
                    if _direct_ok(lab[prev_u], tid):
                        src.append(prev_u); dst.append(u); w.append(0.0)
                    src.append(prev_b); dst.append(u); w.append(0.0)
                prev_u, prev_b = u, b
            ex += [prev_u, prev_b]
        entries.append(ent); exits.append(ex); first_tok.append(ft)

    n = len(slots)
    S = len(lab)
    lab_a = np.array(lab, dtype=np.int64)
    init = np.full(S, NEG_INF)
    fin = np.full(S, NEG_INF)
    init[0] = 0.0
    exits_ext = lambda j: [0] if j < 0 else exits[j]  # noqa: E731

    for k in range(n):
        gap = 0.0
        for j in range(k - 1, -2, -1):
            for p in exits_ext(j):
                for v, wt in entries[k]:
                    if _direct_ok(lab[p], lab[v]):
                        src.append(p); dst.append(v); w.append(gap + wt)
            if j < 0 or not slots[j].skippable:
                break
            gap += slots[j].skip
        # START -> entries(k) if slots 0..k-1 skippable
        g0 = sum(s.skip for s in slots[:k]) if all(s.skippable for s in slots[:k]) else NEG_INF
        if g0 > NEG_INF:
            for v, wt in entries[k]:
                init[v] = np.logaddexp(init[v], g0 + wt)
    for j in range(-1, n):
        rest = slots[j + 1:]
        if all(s.skippable for s in rest):
            g = sum(s.skip for s in rest)
            for p in exits_ext(j):
                fin[p] = np.logaddexp(fin[p], g)

    return Lattice(lab_a, np.array(slot_of), np.array(alt_of), np.array(is_tok, dtype=bool),
                   init, fin, np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64),
                   np.array(w, dtype=np.float64), entries, exits, first_tok)


def concat_lattices(lats: list[Lattice]) -> tuple[Lattice, np.ndarray]:
    """Block-diagonal union of independent lattices (for batched DP). Returns (lattice, offsets)."""
    offs = np.cumsum([0] + [l.n_states for l in lats])
    cat = lambda name: np.concatenate([getattr(l, name) for l in lats])  # noqa: E731
    src = np.concatenate([l.src + o for l, o in zip(lats, offs)])
    dst = np.concatenate([l.dst + o for l, o in zip(lats, offs)])
    big = Lattice(cat("lab"), cat("slot"), cat("alt"), cat("is_tok"), cat("init"), cat("fin"),
                  src, dst, cat("w"))
    return big, offs


@dataclass
class SlotArcs:
    """Arcs of the canonical lattice's surroundings of each slot, used by the fast evidence pass.
    Indices >= S are virtual: S = START (src only), S+1 = END (dst only)."""
    in_slot: np.ndarray; in_src: np.ndarray; in_w: np.ndarray          # p -> wildcard of slot i
    out_slot: np.ndarray; out_dst: np.ndarray; out_w: np.ndarray       # wildcard of slot i -> q
    cr_slot: np.ndarray; cr_src: np.ndarray; cr_dst: np.ndarray; cr_w: np.ndarray  # bypass slot i


def slot_arcs(slots: list[Slot], lat: Lattice) -> SlotArcs:
    S = lat.n_states
    START, END = S, S + 1
    n = len(slots)
    skippable = [s.skippable for s in slots]
    lab = lat.lab
    ex = lambda j: [0] if j < 0 else lat.exits[j]  # noqa: E731
    I, O, C = [], [], []
    for i in range(n):
        # predecessors of slot i's entries (any label: the wildcard differs from everything)
        gap = 0.0
        for j in range(i - 1, -2, -1):
            I += [(i, p, gap) for p in ex(j)]
            if j < 0 or not skippable[j]:
                break
            gap += slots[j].skip
        if all(skippable[:i]):
            I.append((i, START, sum(s.skip for s in slots[:i])))
        # successors of slot i's exits
        gap = 0.0
        for k in range(i + 1, n + 1):
            if k == n:
                O.append((i, END, gap))
                break
            O += [(i, v, gap + wt) for v, wt in lat.entries[k]]
            if not skippable[k]:
                break
            gap += slots[k].skip
        # bypass arcs: exits(j) -> entries(k), j < i < k, all of (j, k) \ {i} skippable
        lefts = []          # (j, gap_left) reachable going left from i
        gap = 0.0
        for j in range(i - 1, -2, -1):
            lefts.append((j, gap))
            if j < 0 or not skippable[j]:
                break
            gap += slots[j].skip
        start_ok = all(skippable[:i])
        g_start = sum(s.skip for s in slots[:i]) if start_ok else NEG_INF
        gap_r = 0.0
        for k in range(i + 1, n + 1):
            if k == n:
                for j, gl in lefts:
                    C += [(i, p, END, gl + gap_r) for p in ex(j)]
            else:
                for v, wt in lat.entries[k]:
                    for j, gl in lefts:
                        for p in ex(j):
                            if _direct_ok(lab[p], lab[v]):
                                C.append((i, p, v, gl + gap_r + wt))
                    if start_ok:
                        C.append((i, START, v, g_start + gap_r + wt))
                if not skippable[k]:
                    break
                gap_r += slots[k].skip

    def arr(rows, ncol):
        if not rows:
            return [np.zeros(0, dtype=np.int64 if c < ncol - 1 else np.float64) for c in range(ncol)]
        cols = list(zip(*rows))
        return [np.array(c, dtype=np.int64) for c in cols[:-1]] + [np.array(cols[-1], dtype=np.float64)]

    a_in, a_out, a_cr = arr(I, 3), arr(O, 3), arr(C, 4)
    return SlotArcs(*a_in, *a_out, *a_cr)
