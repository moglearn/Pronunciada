"""Lattice CTC forward / Viterbi and per-sound evidence (DESIGN section 2).

Two implementations of the same quantities:

* ``exact=True`` builds all 2N+1 lattices (canonical, wildcard-i, deleted-i) explicitly and runs
  one batched log-domain forward pass over their block-diagonal union (reference implementation).
* default (fast): one scaled forward pass alpha and one backward pass beta on the canonical
  lattice. A variant that differs from the canonical lattice only in slot i shares every state
  left of slot i (same alpha) and right of slot i (same beta), so its total is obtained by
  "stitching": del_i = sum over bypass arcs (p -> q) and frames t of alpha[t-1,p] w em[t,q]
  beta[t,q]; wild_i additionally runs slot i's small wildcard-chain recursion over all frames,
  fed by alpha and drained into beta. The local window is therefore just slot i's own states
  over the full utterance, which makes the stitching exact (no time-window approximation).
  The only difference to the exact pass is floating point (scaled probabilities vs log domain);
  a slot whose stitched value underflows is recomputed with the exact pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lattice import (BLANK, WILD, Lattice, Slot, build_lattice, concat_lattices, slot_arcs,
                      slots_from_reference, wild_chains, wild_slot)

NEG_INF = -np.inf
LOG_FLOOR = -1.0e6          # finite stand-in for log(0) in returned evidence
IMPOSSIBLE_LLR = -1.0e3     # llr reported for every slot when the canonical lattice cannot fit


@dataclass
class SlotEvidence:
    llr_sub: float        # L_can - L_wild_i  (<= 0)
    llr_del: float        # L_can - L_del_i
    start_frame: int      # Viterbi span on the canonical lattice (end exclusive); zero-length if dropped
    end_frame: int
    alt_index: int        # 0 canonical, k>0 = k-th entry of RefPhone.alts, -1 dropped via empty alt
    peak_post: float      # max over span frames of the posterior of the chosen alternative's first token
    heard: str            # collapsed greedy non-blank tokens overlapping the span ("" if none)
    n_frames: int


@dataclass
class Evidence:
    slots: list[SlotEvidence]
    L_can: float
    n_frames: int
    frame_shift: float
    free_tokens: list[tuple[str, int, int]] = field(default_factory=list)
    # round 19 (insertion=True): L_can - L(an extra sound between slot i and i + 1), <= 0; NaN for
    # the last slot or when not computed
    llr_ins: np.ndarray | None = None


# --------------------------------------------------------------------------- emissions

def emission_table(lp: np.ndarray) -> np.ndarray:
    """[T, V+2] float64 log emissions: columns 0..V-1 = tokens, V = blank, V+1 = wildcard."""
    lp = np.asarray(lp, dtype=np.float64)
    T, V = lp.shape
    tab = np.empty((T, V + 2))
    tab[:, :V] = lp
    tab[:, V] = np.logaddexp(lp[:, 0], lp[:, 3])
    # log(1 - p_blank) computed stably as log-sum-exp of every non-blank column; clipped.
    nb = lp[:, [1, 2] + list(range(4, V))]
    m = nb.max(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        wild = (m + np.log(np.exp(nb - m).sum(axis=1, keepdims=True)))[:, 0]
    tab[:, V + 1] = np.maximum(wild, -50.0)
    return tab


def _cols(lab: np.ndarray, V: int) -> np.ndarray:
    return np.where(lab == BLANK, V, np.where(lab == WILD, V + 1, lab))


# --------------------------------------------------------------------------- exact (log domain)

def _padded_preds(lat: Lattice) -> tuple[np.ndarray, np.ndarray]:
    """[S, K+1] predecessor indices (column 0 = self) and log weights; pad -> dummy state S."""
    S = lat.n_states
    order = np.argsort(lat.dst, kind="stable")
    d, s, w = lat.dst[order], lat.src[order], lat.w[order]
    counts = np.bincount(d, minlength=S)
    K = int(counts.max()) if len(d) else 0
    P = np.full((S, K + 1), S, dtype=np.int64)
    W = np.full((S, K + 1), NEG_INF)
    P[:, 0] = np.arange(S)
    W[:, 0] = 0.0
    if len(d):
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pos = np.arange(len(d)) - starts[d] + 1
        P[d, pos] = s
        W[d, pos] = w
    return P, W


def _lse_rows(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1)
    mm = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return mm + np.log(np.exp(x - mm[:, None]).sum(axis=1))


def forward_log(lat: Lattice, tab: np.ndarray, blocks: np.ndarray | None = None) -> np.ndarray:
    """Full-sum log score of each block (offsets `blocks`, default one block)."""
    T = len(tab)
    S = lat.n_states
    if blocks is None:
        blocks = np.array([0, S])
    if T == 0:
        return np.full(len(blocks) - 1, NEG_INF)
    E = tab[:, _cols(lat.lab, tab.shape[1] - 2)]
    P, W = _padded_preds(lat)
    a = np.append(lat.init + E[0], NEG_INF)
    for t in range(1, T):
        a[:S] = _lse_rows(a[P] + W) + E[t]
    x = a[:S] + lat.fin
    out = np.empty(len(blocks) - 1)
    for b in range(len(blocks) - 1):
        seg = x[blocks[b]:blocks[b + 1]]
        m = seg.max() if len(seg) else NEG_INF
        out[b] = NEG_INF if not np.isfinite(m) else m + np.log(np.exp(seg - m).sum())
    return out


def exact_scores(slots: list[Slot], tab: np.ndarray, which: list[int] | None = None):
    """(L_can, L_wild[N], L_del[N]) by explicit lattices; `which` restricts the slot variants."""
    n = len(slots)
    which = list(range(n)) if which is None else which
    lats = [build_lattice(slots)]
    for i in which:
        lats.append(build_lattice(slots[:i] + [wild_slot(slots[i])] + slots[i + 1:]))
        lats.append(build_lattice(slots[:i] + slots[i + 1:]))
    big, offs = concat_lattices(lats)
    L = forward_log(big, tab, offs)
    L_wild = np.full(n, np.nan)
    L_del = np.full(n, np.nan)
    L_wild[which] = L[1::2]
    L_del[which] = L[2::2]
    return L[0], L_wild, L_del


# --------------------------------------------------------------------------- fast (scaled)

def _reach_masks(lat: Lattice, T: int, slack: int):
    """Boolean [T, S] masks: (can still reach END from s at t, can have reached s from START by t),
    each loosened by `slack` frames so they also hold for every single-slot variant lattice.
    Zeroing hopeless states keeps the scaled forward/backward from normalising on them (which
    would underflow the useful mass when the reference is long for the audio)."""
    S = lat.n_states
    big = 1 << 30
    d0 = np.where(np.isfinite(lat.init), 1, big)
    d1 = np.where(np.isfinite(lat.fin), 0, big)
    o = np.argsort(lat.dst, kind="stable")
    for a, b in zip(lat.src[o].tolist(), lat.dst[o].tolist()):     # indices are topological
        if d0[a] + 1 < d0[b]:
            d0[b] = d0[a] + 1
    o = np.argsort(-lat.src, kind="stable")
    for a, b in zip(lat.src[o].tolist(), lat.dst[o].tolist()):
        if d1[b] + 1 < d1[a]:
            d1[a] = d1[b] + 1
    t = np.arange(T)[:, None]
    m_end = (T - 1 - t) >= (d1[None, :] - slack)
    m_start = t >= (d0[None, :] - 1 - slack)
    min_len = int(np.min(np.where(np.isfinite(lat.init), d1 + 1, big)))
    return m_end, m_start, min_len


def _forward_backward(lat: Lattice, em: np.ndarray, em_b: np.ndarray | None = None):
    """Scaled forward/backward in the probability domain.
    alpha[t] = ahat[t] * exp(A[t]),  beta[t] = bhat[t] * exp(B[t]).  Returns None if impossible."""
    T, S = em.shape
    em_b = em if em_b is None else em_b
    src, dst, w = lat.src, lat.dst, np.exp(lat.w)
    ahat = np.empty((T, S))
    A = np.empty(T)
    a = np.exp(lat.init) * em[0]
    c = a.sum()
    if not c > 0:
        return None
    a /= c
    ahat[0] = a
    acc = np.log(c)
    A[0] = acc
    for t in range(1, T):
        a = (a + np.bincount(dst, weights=a[src] * w, minlength=S)) * em[t]
        c = a.sum()
        if not c > 0:
            return None
        a /= c
        ahat[t] = a
        acc += np.log(c)
        A[t] = acc
    fin = np.exp(lat.fin)
    z = (a * fin).sum()
    if not z > 0:
        return None
    L_can = A[-1] + np.log(z)

    bhat = np.empty((T, S))
    B = np.empty(T)
    b = fin.copy()
    bhat[-1] = b
    B[-1] = 0.0
    acc = 0.0
    for t in range(T - 1, 0, -1):
        g = b * em_b[t]
        b = g + np.bincount(src, weights=w * g[dst], minlength=S)
        d = b.sum()
        b /= d
        bhat[t - 1] = b
        acc += np.log(d)
        B[t - 1] = acc
    return ahat, A, bhat, B, L_can


def _group_sum(vals: np.ndarray, groups: np.ndarray, n: int) -> np.ndarray:
    """Sum columns of vals [R, M] into n groups (groups sorted, every group non-empty)."""
    starts = np.searchsorted(groups, np.arange(n))
    return np.add.reduceat(vals, starts, axis=1)


def _fast_scores(slots: list[Slot], lat: Lattice, tab: np.ndarray, insertion: bool = False):
    n = len(slots)
    T = len(tab)
    V = tab.shape[1] - 2
    E = tab[:, _cols(lat.lab, V)]
    em = np.exp(E)
    slack = 2 * max([len(a[0]) for sl in slots for a in sl.alts] + [1]) + 2
    m_end, m_start, min_len = _reach_masks(lat, T, slack)
    if min_len > T:              # canonical lattice cannot fit: nothing to stitch
        return NEG_INF, np.full(n, np.nan), np.full(n, np.nan)
    fb = _forward_backward(lat, em * m_end, em * m_start)
    if fb is None:
        return None
    ahat, A, bhat, B, L_can = fb
    em = em * m_start
    if n == 0:
        return L_can, np.zeros(0), np.zeros(0)
    S = lat.n_states
    ar = slot_arcs(slots, lat)

    # augmented tables, row r: alpha row = time r-1 (row 0 = START), em*beta row = time r
    # (row T = END). Column S = START, S+1 = END, S+2 = always-zero dummy.
    alpha_aug = np.zeros((T + 1, S + 3))
    alpha_aug[1:, :S] = ahat
    alpha_aug[0, S] = 1.0
    A_aug = np.concatenate([[0.0], A])                # scale of alpha_aug rows
    eb_aug = np.zeros((T + 1, S + 3))
    eb_aug[:T, :S] = em * bhat
    eb_aug[T, S + 1] = 1.0
    B_aug = np.concatenate([B, [0.0]])                # scale of eb_aug rows
    DUMMY = S + 2
    sl = np.arange(n)

    def with_dummy(slot_idx, *cols):
        # one dummy arc per slot so every group is non-empty; keep slot-sorted order
        s2 = np.concatenate([slot_idx, sl])
        order = np.argsort(s2, kind="stable")
        out = [s2[order]]
        for c, fill in cols:
            out.append(np.concatenate([c, np.full(n, fill, dtype=c.dtype)])[order])
        return out

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # ---- bypass (deletion) sums: row r = arc taken into frame r (r = T: into END)
        cs, cp, cq, cw = with_dummy(ar.cr_slot, (ar.cr_src, DUMMY), (ar.cr_dst, DUMMY),
                                    (ar.cr_w, 0.0))
        R = _group_sum(alpha_aug[:, cp] * np.exp(cw) * eb_aug[:, cq], cs, n)       # [T+1, n]
        scale = A_aug + B_aug                                                       # [T+1]
        L_del = _lse_cols(np.log(R) + scale[:, None])

        # ---- wildcard paths
        is_, ip, iw = with_dummy(ar.in_slot, (ar.in_src, DUMMY), (ar.in_w, 0.0))
        inj = np.log(_group_sum(alpha_aug[:T][:, ip] * np.exp(iw), is_, n))           # [T, n], scale A_aug[t]
        os_, oq, ow = with_dummy(ar.out_slot, (ar.out_dst, DUMMY), (ar.out_w, 0.0))
        out = np.log(_group_sum(eb_aug[1:][:, oq] * np.exp(ow), os_, n))              # [T, n], scale B_aug[t+1]
        # wildcard chains: one row per (slot, chain length); left-aligned in [n_pairs, Kmax]
        pairs = [(i, K, w) for i in range(n) for K, w in wild_chains(slots[i])]
        if not pairs:                       # every slot pre-wildcarded
            return L_can, L_del.copy(), L_del
        p_slot = np.array([q[0] for q in pairs])
        p_last = np.array([q[1] for q in pairs]) - 1
        p_w = np.array([q[2] for q in pairs])
        P = len(pairs)
        inj_p = inj[:, p_slot] + p_w                                                # [T, P]
        # round 19 extra-sound evidence: an extra "any speech" chain (1 or 2 sounds) between slot
        # i and slot i + 1, fed like slot i + 1's wildcard (inj[:, i + 1]) and drained like slot
        # i's (out[:, i]). Its chain states follow the same recursion as the wildcard chains, and
        # states 0 and 1 of a chain do not depend on the chain's length, so any wildcard row of
        # slot i + 1 with chain weight 0 already holds them (state 1 needs Kmax >= 2). A gap
        # without such a row gets an extra row in the same recursion.
        n_ins = n - 1 if insertion and n > 1 else 0
        Kmax = max(int(p_last.max()) + 1, 2 if n_ins else 1)
        n_extra = 0
        if n_ins:
            zero_row: dict[int, int] = {}
            for r in np.flatnonzero(p_w == 0.0):
                zero_row.setdefault(int(p_slot[r]), int(r))
            ins_row = np.array([zero_row.get(i + 1, -1) for i in range(n - 1)], dtype=np.int64)
            miss = np.flatnonzero(ins_row < 0)
            n_extra = len(miss)
            if n_extra:
                ins_row[miss] = P + np.arange(n_extra)
                inj_p = np.concatenate([inj_p, inj[:, miss + 1]], axis=1)          # [T, P + n_extra]
        logc = np.diff(A_aug)                                                       # log c_t
        lew = tab[:, V + 1] - logc
        leb = tab[:, V] - logc
        R2 = P + n_extra
        Wh = np.empty((T, R2, Kmax))       # log W_k[t] - A[t]   (wildcard token states)
        Bh = np.empty((T, R2, Kmax))       # log b_k[t] - A[t]   (their blanks)
        # Oh[t, :, k] = log(W_k[t] + b_k[t]) - A[t] for k < Kmax - 1: the recursion forms it one
        # frame later to feed state k + 1, so it is kept rather than formed again for the chain ends
        Oh = np.empty((T, R2, Kmax - 1))
        Wc = np.full((R2, Kmax), NEG_INF)
        Bc = np.full((R2, Kmax), NEG_INF)
        lae = np.logaddexp
        for t in range(T):
            Wn, Bn = Wh[t], Bh[t]
            lae(Wc[:, 0], inj_p[t], out=Wn[:, 0])
            if Kmax > 1:
                o = lae(Wc[:, :-1], Bc[:, :-1], out=Oh[t - 1] if t else None)
                lae(Wc[:, 1:], o, out=Wn[:, 1:])
            Wn += lew[t]
            lae(Bc, Wc, out=Bn)
            Bn += leb[t]
            Wc, Bc = Wn, Bn
        if Kmax > 1:
            lae(Wc[:, :-1], Bc[:, :-1], out=Oh[T - 1])
        # occupancy of each chain's last state [T, P], laid out like a fancy-indexed result
        # (column-major), so that the frame sums below add in the same order as before
        occ = np.empty((T, P), order="F")
        kept = np.flatnonzero(p_last < Kmax - 1)
        occ[:, kept] = Oh[:, kept, p_last[kept]]
        top = np.flatnonzero(p_last == Kmax - 1)
        occ[:, top] = lae(Wh[:, top, Kmax - 1], Bh[:, top, Kmax - 1])
        L_pair = _lse_cols(occ + out[:, p_slot] + (A + B_aug[1:])[:, None])
        L_x = np.full(n, NEG_INF)
        np.logaddexp.at(L_x, p_slot, L_pair)
        L_wild = np.logaddexp(L_x, L_del)
        if n_ins:
            L_ins = np.full(n, np.nan)
            ii = np.repeat(np.arange(n - 1), 2)        # (gap i, chain length kk + 1), gap-major
            kk = np.tile([0, 1], n - 1)
            # occupancy of state kk of gap i's chain (row ins_row[i]), row-major as before
            r2 = ins_row[ii]
            occ2 = np.full((T, len(ii)), NEG_INF)                                   # [T, 2(n - 1)]
            low = np.flatnonzero(kk < Kmax - 1)
            occ2[:, low] = Oh[:, r2[low], kk[low]]
            last = np.flatnonzero(kk == Kmax - 1)          # only when Kmax == 2: the 2-sound chain
            if len(last):
                # a weight-0 row of slot i + 1 whose chain ends at this state has it in `occ`;
                # the rest is formed only where the drain out[:, i] is above -inf (elsewhere the
                # frame's term is -inf, or NaN, whatever the occupancy, so -inf changes no number)
                occ_of = {int(p_slot[r]): int(r) for r in np.flatnonzero((p_w == 0.0) & (p_last == Kmax - 1))[::-1]}
                src = np.array([occ_of.get(int(ii[q]) + 1, -1) for q in last], dtype=np.int64)
                reuse, calc = last[src >= 0], last[src < 0]
                occ2[:, reuse] = occ[:, src[src >= 0]]
                if len(calc):
                    part = np.full((T, len(calc)), NEG_INF)
                    lae(Wh[:, r2[calc], Kmax - 1], Bh[:, r2[calc], Kmax - 1], out=part,
                        where=out[:, ii[calc]] > NEG_INF)
                    occ2[:, calc] = part
            L2 = _lse_cols(occ2 + out[:, ii] + (A + B_aug[1:])[:, None])
            Lc = np.full(n - 1, NEG_INF)
            np.logaddexp.at(Lc, ii, L2)
            L_ins[:-1] = np.logaddexp(L_can, Lc)
            return L_can, L_wild, L_del, L_ins
    return L_can, L_wild, L_del


def _lse_cols(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=0)
    mm = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return mm + np.log(np.exp(x - mm).sum(axis=0))


# --------------------------------------------------------------------------- Viterbi

def viterbi(lat: Lattice, tab: np.ndarray) -> tuple[float, np.ndarray]:
    """Best state path [T] on the lattice (log domain). Returns (score, path) or (-inf, [])."""
    T = len(tab)
    S = lat.n_states
    if T == 0:
        return NEG_INF, np.zeros(0, dtype=np.int64)
    E = tab[:, _cols(lat.lab, tab.shape[1] - 2)]
    P, W = _padded_preds(lat)
    # predecessors laid out [K+1, S]: the max over them is then an element-wise maximum of
    # contiguous rows (about twice as fast as a reduction over a short last axis); a maximum
    # has no rounding, so the scores are the same numbers
    PT, WT = np.ascontiguousarray(P.T), np.ascontiguousarray(W.T)
    D = np.empty((T, S + 1))           # Viterbi scores; back-pointers recomputed on the path only
    D[:, S] = NEG_INF                  # column S: the padding state
    D[0, :S] = lat.init + E[0]
    for t in range(1, T):
        x = D[t - 1][PT]
        x += WT
        np.add(x.max(axis=0), E[t], out=D[t, :S])
    x = D[T - 1, :S] + lat.fin
    s = int(x.argmax())
    score = float(x[s])
    if not np.isfinite(score):
        return NEG_INF, np.zeros(0, dtype=np.int64)
    path = np.empty(T, dtype=np.int64)
    path[-1] = s
    for t in range(T - 1, 0, -1):
        s = int(P[s, int((D[t - 1, P[s]] + W[s]).argmax())])
        path[t - 1] = s
    return score, path


# --------------------------------------------------------------------------- public API

def _greedy(lp: np.ndarray):
    from .acoustic import greedy_tokens

    return greedy_tokens(lp)


def score_reference(lp: np.ndarray, ref, *, exact: bool = False,
                    frame_shift: float = 0.02, wild_slots=None, insertion: bool = False) -> Evidence:
    """Per-slot acoustic evidence for `ref` (a pronunciada.types.Reference) given frame
    log-posteriors lp [T, 127].

    wild_slots: indices of slots that are pre-wildcarded in the BASE lattice: they emit the
    "any speech" wildcard and may be skipped (exactly the slot replacement of the L_wild
    construction), so the audio there no longer has to match their reference. Every other
    slot's llr_sub / llr_del / span is computed relative to that base ("explain-away" pass).
    The wildcarded slots themselves report llr_sub = llr_del = 0, alt_index -2."""
    lp = np.asarray(lp)
    slots = slots_from_reference(ref)
    fixed = sorted(set(int(i) for i in (wild_slots or ())))
    for i in fixed:
        slots[i] = wild_slot(slots[i])
    return score_slots(lp, slots, exact=exact, frame_shift=frame_shift, fixed=fixed, insertion=insertion)


def score_slots(lp: np.ndarray, slots: list[Slot], *, exact: bool = False,
                frame_shift: float = 0.02, fixed=(), insertion: bool = False) -> Evidence:
    T = int(lp.shape[0]) if lp.ndim == 2 else 0
    n = len(slots)
    free = _greedy(lp) if T else []
    lat = build_lattice(slots)
    tab = emission_table(lp) if T else np.zeros((0, 129))

    res = None
    if T and not exact:
        res = _fast_scores(slots, lat, tab, insertion=insertion)
    if T and (exact or res is None):
        which = [i for i in range(n) if i not in set(fixed)]
        L_can, L_wild, L_del = exact_scores(slots, tab, which=which)
        res = (L_can, L_wild, L_del)
    if T == 0:
        res = (NEG_INF, np.full(n, NEG_INF), np.full(n, NEG_INF))
    L_ins = res[3] if len(res) > 3 else None
    L_can, L_wild, L_del = res[:3]

    if np.isfinite(L_can) and n:
        # stitched values that underflowed (or violate L_wild >= L_can) -> exact recompute
        bad = ~np.isfinite(L_wild) | ~np.isfinite(L_del) | (L_wild < L_can - 1e-6)
        if len(fixed):
            bad[list(fixed)] = False
        if bad.any() and not exact:
            _, Lw2, Ld2 = exact_scores(slots, tab, which=list(np.flatnonzero(bad)))
            L_wild = np.where(bad, Lw2, L_wild)
            L_del = np.where(bad, Ld2, L_del)

    possible = bool(np.isfinite(L_can))
    if possible:
        llr_sub = np.minimum(L_can - L_wild, 0.0)
        llr_del = np.where(np.isfinite(L_del), L_can - np.maximum(L_del, LOG_FLOOR), -IMPOSSIBLE_LLR)
        llr_del = np.clip(llr_del, IMPOSSIBLE_LLR, -IMPOSSIBLE_LLR)
    else:
        llr_sub = np.full(n, IMPOSSIBLE_LLR)
        llr_del = np.full(n, IMPOSSIBLE_LLR)
    if len(fixed):
        llr_sub = np.array(llr_sub, dtype=np.float64)
        llr_del = np.array(llr_del, dtype=np.float64)
        llr_sub[list(fixed)] = 0.0
        llr_del[list(fixed)] = 0.0

    spans = _spans(lat, slots, tab, T, possible)
    free_start = np.array([a for _, a, _ in free], dtype=np.int64)
    free_end = np.array([b for _, _, b in free], dtype=np.int64)
    out = []
    for i, (s, e, k) in enumerate(spans):
        if k >= 0 and e > s:
            ft = lat.first_tok[i][k]
            peak = float(np.exp(lp[s:e, ft].max()))
        else:
            peak = 0.0
        if e > s:
            lo, hi = np.searchsorted(free_end, s, side="right"), np.searchsorted(free_start, e)
            heard = "".join(free[j][0] for j in range(lo, hi))
        else:
            heard = ""
        out.append(SlotEvidence(float(llr_sub[i]), float(llr_del[i]), int(s), int(e), int(k),
                                peak, heard, int(e - s)))
    ev = Evidence(out, float(L_can) if possible else LOG_FLOOR, T, frame_shift, free)
    if insertion:
        ev.llr_ins = (np.minimum(L_can - L_ins, 0.0) if (possible and L_ins is not None)
                      else np.full(n, np.nan))
    return ev


def _spans(lat: Lattice, slots: list[Slot], tab, T: int, possible: bool):
    n = len(slots)
    if not possible or n == 0:
        return [(0, 0, -1)] * n
    _, path = viterbi(lat, tab)
    if len(path) == 0:
        return [(0, 0, -1)] * n
    ps = lat.slot[path]
    tok = lat.is_tok[path]
    out = []
    for i in range(n):
        fr = np.flatnonzero((ps == i) & tok)
        if len(fr):
            out.append((int(fr[0]), int(fr[-1]) + 1, int(lat.alt[path[fr[0]]])))
        else:
            after = np.flatnonzero(ps > i)
            t = int(after[0]) if len(after) else T
            out.append((t, t, -1))
    return out
