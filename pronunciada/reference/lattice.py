"""Target sounds -> CTC token lattice (port of pronunciada/lattice.py).

Terms
-----
token id   index into the 127-line token file (zipa_tokens.txt); 0 <blk>, 1 <sos/eos>, 2 <unk>,
           3 the word marker, 4..126 one IPA code point each.
slot       one target sound. It has one or more alternatives (token sequences with a log weight
           <= 0: the canonical reading has weight 0, alternative k has its log prior) and may be
           skippable (log weight `skip` > -inf) when the target lists an empty alternative.
state      a node of the lattice. Every state has an implicit self-loop of weight 0.
           Labels: a token id, BLANK (-1) or WILD (-2, "any speech").

Structure (states are numbered in creation order, which is topological: every arc goes from a
lower to a higher number; the reachability pass relies on it):
  state 0              the leading blank (slot -1);
  per slot, per chain  (wildcard chains first, then the alternatives in order) a token state u_j
                       followed by its own blank b_j for every token j; arcs u_j -> b_j,
                       b_j -> u_{j+1}, and u_j -> u_{j+1} if the two labels may follow each other
                       directly (different labels, or either is WILD);
  entries(slot)        the first token state of every chain, with the chain's log weight;
  exits(slot)          the last token state and its blank, for every chain;
  exits(j) -> entries(k), j < k, if every slot strictly between is skippable (weight = sum of
                       their skip weights + the entry weight); a direct token -> token arc only
                       if the labels may follow each other;
  START                weight 0 into state 0, and into entries(k) when slots 0..k-1 are skippable;
  END                  from exits(j) when slots j+1.. are skippable; from state 0 if all are.
"""
from __future__ import annotations

from .numeric import NEG_INF, logaddexp

BLANK = -1
WILD = -2


class Slot:
    """alts: list of (tuple of token ids, log weight, alternative index) - may be empty for a
    wildcard slot; skip: log weight of skipping the slot (-inf = not skippable); wild: list of
    (chain length, log weight) of the wildcard chains (empty for an ordinary slot)."""

    def __init__(self, alts, skip=NEG_INF, wild=None):
        self.alts = list(alts)
        self.skip = skip
        self.wild = list(wild) if wild else []

    def skippable(self) -> bool:
        return self.skip > NEG_INF


def slots_from_phones(phones, token_id) -> list:
    """phones: list of {"ipa": str, "alts": [[ipa, log_prior], ...]} (or objects with .ipa / .alts).
    token_id: dict symbol -> id. Alternative index 0 = canonical (weight 0), k >= 1 = k-th entry
    of alts; an empty alternative makes the slot skippable (several: their log-sum-exp)."""
    slots = []
    for p in phones:
        ipa = p["ipa"] if isinstance(p, dict) else p.ipa
        alts_in = p.get("alts", []) if isinstance(p, dict) else p.alts
        alts = [(tuple(token_id[c] for c in ipa), 0.0, 0)]
        skip = NEG_INF
        k = 0
        for a, w in alts_in:
            k += 1
            if a:
                alts.append((tuple(token_id[c] for c in a), float(w), k))
            else:
                skip = logaddexp(skip, float(w))
        slots.append(Slot(alts, skip))
    return slots


def wild_chains(slot: Slot) -> list:
    """One wildcard chain per distinct alternative length, weight = best weight of that length;
    sorted by length."""
    best = {}
    for ids, w, _ in slot.alts:
        n = len(ids)
        if n not in best or w > best[n]:
            best[n] = w
    return sorted(best.items())


def wild_slot(slot: Slot) -> Slot:
    """Wildcard version of a slot: any speech (one WILD chain per alternative length) or nothing
    (skip weight 0)."""
    return Slot([], 0.0, wild=wild_chains(slot))


def direct_ok(lab_u: int, lab_v: int) -> bool:
    """May state u be followed by token state v without a blank in between?"""
    return lab_u == BLANK or lab_u == WILD or lab_v == WILD or lab_u != lab_v


class Lattice:
    """Plain lists, index = state: lab, slot, alt (0 canonical, k alternative, -2 wildcard,
    -1 leading blank), is_tok, init (log START weight), fin (log END weight); arcs src / dst / w
    in creation order (self-loops not listed); entries[i] = [(state, log w)], exits[i] = [state],
    first_tok[i] = {alt index: first token id}."""

    def __init__(self):
        self.lab, self.slot, self.alt, self.is_tok = [], [], [], []
        self.init, self.fin = [], []
        self.src, self.dst, self.w = [], [], []
        self.entries, self.exits, self.first_tok = [], [], []

    @property
    def n_states(self) -> int:
        return len(self.lab)


def build_lattice(slots: list) -> Lattice:
    L = Lattice()
    L.lab.append(BLANK); L.slot.append(-1); L.alt.append(-1); L.is_tok.append(False)

    def new(label, s, a, tok):
        L.lab.append(label); L.slot.append(s); L.alt.append(a); L.is_tok.append(tok)
        return len(L.lab) - 1

    for i, sl in enumerate(slots):
        ent, ex, ft = [], [], {}
        chains = [((WILD,) * K, wt, -2) for K, wt in sl.wild] + list(sl.alts)
        for ids, wt, k in chains:
            prev_u = prev_b = -1
            for tid in ids:
                u = new(tid, i, k, True)
                b = new(BLANK, i, k, False)
                L.src.append(u); L.dst.append(b); L.w.append(0.0)
                if prev_u < 0:
                    ent.append((u, wt))
                    if k >= 0:
                        ft[k] = tid
                else:
                    if direct_ok(L.lab[prev_u], tid):
                        L.src.append(prev_u); L.dst.append(u); L.w.append(0.0)
                    L.src.append(prev_b); L.dst.append(u); L.w.append(0.0)
                prev_u, prev_b = u, b
            ex.append(prev_u)
            ex.append(prev_b)
        L.entries.append(ent); L.exits.append(ex); L.first_tok.append(ft)

    n = len(slots)
    S = len(L.lab)
    L.init = [NEG_INF] * S
    L.fin = [NEG_INF] * S
    L.init[0] = 0.0

    def exits_ext(j):
        return [0] if j < 0 else L.exits[j]

    for k in range(n):
        gap = 0.0
        j = k - 1
        while j >= -1:
            for p in exits_ext(j):
                for v, wt in L.entries[k]:
                    if direct_ok(L.lab[p], L.lab[v]):
                        L.src.append(p); L.dst.append(v); L.w.append(gap + wt)
            if j < 0 or not slots[j].skippable():
                break
            gap += slots[j].skip
            j -= 1
        # START -> entries(k) when slots 0..k-1 are all skippable (sum left to right from 0)
        all_skip = True
        g0 = 0
        for s in slots[:k]:
            if not s.skippable():
                all_skip = False
            g0 += s.skip
        if all_skip:
            g0 = float(g0)
            if g0 > NEG_INF:
                for v, wt in L.entries[k]:
                    L.init[v] = logaddexp(L.init[v], g0 + wt)
    for j in range(-1, n):
        rest = slots[j + 1:]
        if all(s.skippable() for s in rest):
            g = 0
            for s in rest:
                g += s.skip
            for p in exits_ext(j):
                L.fin[p] = logaddexp(L.fin[p], float(g))
    return L


def concat_lattices(lats: list):
    """Block-diagonal union of independent lattices (one batched pass). Returns (lattice,
    offsets) with offsets[b] = first state of block b, offsets[-1] = total states."""
    big = Lattice()
    offs = [0]
    for l in lats:
        o = offs[-1]
        big.lab += l.lab; big.slot += l.slot; big.alt += l.alt; big.is_tok += l.is_tok
        big.init += l.init; big.fin += l.fin
        big.src += [x + o for x in l.src]
        big.dst += [x + o for x in l.dst]
        big.w += l.w
        offs.append(o + l.n_states)
    return big, offs


class SlotArcs:
    """Per slot, the arcs of the canonical lattice around it (for the stitched evidence pass).
    Virtual indices: S = START (source only), S + 1 = END (destination only).
    inn = [(slot, source state, log w)]: arcs into slot i's chains;
    out = [(slot, destination state, log w)]: arcs out of slot i's chains;
    cross = [(slot, source, destination, log w)]: arcs that bypass slot i."""

    def __init__(self, inn, out, cross):
        self.inn, self.out, self.cross = inn, out, cross


def slot_arcs(slots: list, lat: Lattice) -> SlotArcs:
    S = lat.n_states
    START, END = S, S + 1
    n = len(slots)
    skippable = [s.skippable() for s in slots]
    lab = lat.lab

    def ex(j):
        return [0] if j < 0 else lat.exits[j]

    I, O, C = [], [], []
    for i in range(n):
        gap = 0.0
        j = i - 1
        while j >= -1:
            for p in ex(j):
                I.append((i, p, gap))
            if j < 0 or not skippable[j]:
                break
            gap += slots[j].skip
            j -= 1
        if all(skippable[:i]):
            g = 0
            for s in slots[:i]:
                g += s.skip
            I.append((i, START, float(g)))
        gap = 0.0
        for k in range(i + 1, n + 1):
            if k == n:
                O.append((i, END, gap))
                break
            for v, wt in lat.entries[k]:
                O.append((i, v, gap + wt))
            if not skippable[k]:
                break
            gap += slots[k].skip
        lefts = []
        gap = 0.0
        j = i - 1
        while j >= -1:
            lefts.append((j, gap))
            if j < 0 or not skippable[j]:
                break
            gap += slots[j].skip
            j -= 1
        start_ok = all(skippable[:i])
        if start_ok:
            g_start = 0
            for s in slots[:i]:
                g_start += s.skip
            g_start = float(g_start)
        else:
            g_start = NEG_INF
        gap_r = 0.0
        for k in range(i + 1, n + 1):
            if k == n:
                for j, gl in lefts:
                    for p in ex(j):
                        C.append((i, p, END, gl + gap_r))
            else:
                for v, wt in lat.entries[k]:
                    for j, gl in lefts:
                        for p in ex(j):
                            if direct_ok(lab[p], lab[v]):
                                C.append((i, p, v, gl + gap_r + wt))
                    if start_ok:
                        C.append((i, START, v, g_start + gap_r + wt))
                if not skippable[k]:
                    break
                gap_r += slots[k].skip
    return SlotArcs(I, O, C)
