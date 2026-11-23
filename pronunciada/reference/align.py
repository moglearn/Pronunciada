"""CTC forced alignment and per-sound evidence (port of pronunciada/align.py).

Input: adapted frame log-posteriors lp[T][127] (float32 values held as doubles; frame t covers
0.02 * t + 0.03 s .. + 0.02 s) and the slots of the target (pronunciada.reference.lattice).

Quantities (all natural-log, base e):
  L_can        log P(audio | canonical lattice), full CTC sum over every path;
  L_wild[i]    the same with slot i replaced by its wildcard slot ("any speech, or nothing");
  L_del[i]     the same with slot i removed;
  L_ins[i]     the same with one or two extra "any speech" sounds allowed between slot i and
               slot i + 1 (computed only with insertion=True);
  llr_sub[i] = min(L_can - L_wild[i], 0);  llr_del[i] = L_can - max(L_del[i], -1e6) clipped to
  [-1000, 1000] (1000 when L_del[i] = -inf);  llr_ins[i] = min(L_can - L_ins[i], 0);
  span[i]      first and last+1 frame where the best (Viterbi) path emits a token of slot i,
               with the alternative used; a slot the best path skips gets an empty span at the
               first frame of a later slot.

Two ways to the same numbers, as in production:
  fast   one scaled forward pass and one backward pass on the canonical lattice; each slot's
         variants are "stitched" from them (bypass arcs for L_del, a small chain recursion for
         L_wild and L_ins);
  exact  every variant lattice built explicitly, one log-domain forward pass over their union.
Production uses fast, and exact for slots whose stitched values are not finite or violate
L_wild >= L_can - 1e-6, and for the whole clip when the scaled pass underflows.
"""
from __future__ import annotations

from .lattice import (BLANK, WILD, build_lattice, concat_lattices, slot_arcs, wild_chains,
                      wild_slot)
from .numeric import (NAN, NEG_INF, div, exp, float32, is_finite, log, logaddexp, lse,
                      max_list, maximum, minimum, sum_first_pw, sum_pairwise)

LOG_FLOOR = -1.0e6            # finite stand-in for log(0) in returned evidence
IMPOSSIBLE_LLR = -1.0e3       # llr of every slot when the canonical lattice cannot fit
WILD_FLOOR = -50.0            # lower bound of the wildcard emission
NON_SPEECH = 4                # token ids 0..3 are not sounds (greedy decode skips them)
V = 127                       # token columns
COL_BLANK = V                 # emission column of a BLANK state
COL_WILD = V + 1              # emission column of a WILD state


# --------------------------------------------------------------------------- emissions

def emission_table(lp) -> list:
    """[T][129] log emissions: 0..126 the token log-posteriors, 127 blank = logaddexp(lp[0],
    lp[3]) (<blk> or the word marker), 128 wildcard = max(log sum of exp(lp) over every column
    except 0 and 3, -50); the log-sum-exp subtracts the row maximum m first:
    m + log(sum(exp(x - m))), summed left to right in column order 1, 2, 4, 5, ..., 126."""
    tab = []
    for row in lp:
        r = [float(x) for x in row]
        nb = [r[1], r[2]] + r[4:V]
        m = max_list(nb)
        s = 0.0
        for x in nb:
            s += exp(x - m)
        wild = m + log(s)
        r.append(logaddexp(r[0], r[3]))
        r.append(maximum(wild, WILD_FLOOR))
        tab.append(r)
    return tab


def _col(label: int) -> int:
    if label == BLANK:
        return COL_BLANK
    if label == WILD:
        return COL_WILD
    return label


def _emissions(lat, tab) -> list:
    """E[t][s] = tab[t][column of state s]."""
    cols = [_col(l) for l in lat.lab]
    return [[row[c] for c in cols] for row in tab]


def _preds(lat):
    """Per state: predecessor list in arc creation order, the state itself first (self-loop,
    weight 0), padded to the same length K + 1 with (dummy state S, weight -inf)."""
    S = lat.n_states
    P = [[s] for s in range(S)]
    W = [[0.0] for _ in range(S)]
    for a in range(len(lat.src)):
        P[lat.dst[a]].append(lat.src[a])
        W[lat.dst[a]].append(lat.w[a])
    K1 = max(len(p) for p in P) if S else 1
    for s in range(S):
        while len(P[s]) < K1:
            P[s].append(S)
            W[s].append(NEG_INF)
    return P, W


# --------------------------------------------------------------------------- exact (log domain)

def forward_log(lat, tab, blocks=None) -> list:
    """Full-sum log score per block of states (blocks = state offsets; default one block).
    alpha_0[s] = init[s] + E[0][s];
    alpha_t[s] = lse over predecessors p (self first) of (alpha_{t-1}[p] + w(p -> s)) + E[t][s],
    each lse summed left to right in predecessor order (pairwise order for 8 or more terms);
    result per block = lse over its states of alpha_{T-1}[s] + fin[s] (pairwise order)."""
    T = len(tab)
    S = lat.n_states
    if blocks is None:
        blocks = [0, S]
    if T == 0:
        return [NEG_INF] * (len(blocks) - 1)
    E = _emissions(lat, tab)
    P, W = _preds(lat)
    a = [lat.init[s] + E[0][s] for s in range(S)] + [NEG_INF]
    for t in range(1, T):
        new = []
        for s in range(S):
            terms = [a[P[s][k]] + W[s][k] for k in range(len(P[s]))]
            new.append(lse(terms, "pairwise") + E[t][s])
        a = new + [NEG_INF]
    x = [a[s] + lat.fin[s] for s in range(S)]
    out = []
    for b in range(len(blocks) - 1):
        seg = x[blocks[b]:blocks[b + 1]]
        m = max_list(seg)
        if not is_finite(m):
            out.append(NEG_INF)
        else:
            out.append(m + log(sum_pairwise([exp(v - m) for v in seg])))
    return out


def exact_scores(slots, tab, which=None):
    """(L_can, L_wild[n], L_del[n]) from explicit lattices; slots not in `which` get NaN."""
    n = len(slots)
    which = list(range(n)) if which is None else list(which)
    lats = [build_lattice(slots)]
    for i in which:
        lats.append(build_lattice(slots[:i] + [wild_slot(slots[i])] + slots[i + 1:]))
        lats.append(build_lattice(slots[:i] + slots[i + 1:]))
    big, offs = concat_lattices(lats)
    L = forward_log(big, tab, offs)
    L_wild = [NAN] * n
    L_del = [NAN] * n
    for q, i in enumerate(which):
        L_wild[i] = L[1 + 2 * q]
        L_del[i] = L[2 + 2 * q]
    return L[0], L_wild, L_del


# --------------------------------------------------------------------------- fast (scaled)

def _reach(lat, T: int, slack: int):
    """Shortest number of token / blank steps from START to each state (d0, START -> state 0 or
    an entry counts 1) and from each state to END (d1), by relaxing arcs in topological order.
    m_end[t][s]: from s at frame t END is still reachable (T - 1 - t >= d1[s] - slack);
    m_start[t][s]: s may be reached by frame t (t >= d0[s] - 1 - slack); slack makes both hold
    for every one-slot variant lattice too. min_len = fewest frames any path needs."""
    S = lat.n_states
    big = 1 << 30
    d0 = [1 if is_finite(lat.init[s]) else big for s in range(S)]
    d1 = [0 if is_finite(lat.fin[s]) else big for s in range(S)]
    order = sorted(range(len(lat.dst)), key=lambda a: lat.dst[a])          # stable sort
    for a in order:
        u, v = lat.src[a], lat.dst[a]
        if d0[u] + 1 < d0[v]:
            d0[v] = d0[u] + 1
    order = sorted(range(len(lat.src)), key=lambda a: -lat.src[a])         # stable sort
    for a in order:
        u, v = lat.src[a], lat.dst[a]
        if d1[v] + 1 < d1[u]:
            d1[u] = d1[v] + 1
    m_end = [[(T - 1 - t) >= (d1[s] - slack) for s in range(S)] for t in range(T)]
    m_start = [[t >= (d0[s] - 1 - slack) for s in range(S)] for t in range(T)]
    min_len = big
    for s in range(S):
        if is_finite(lat.init[s]) and d1[s] + 1 < min_len:
            min_len = d1[s] + 1
    return m_end, m_start, min_len


def _forward_backward(lat, em, em_b):
    """Scaled forward / backward in the probability domain:
    alpha_t = ahat[t] * exp(A[t]), beta_t = bhat[t] * exp(B[t]).
    Forward: a = exp(init) * em[0]; per frame a = (a + sum over arcs p->q of a[p] * exp(w)) * em[t]
    (arc contributions accumulated into q in arc order, starting from 0), then c = sum(a)
    (pairwise order), a /= c, A[t] = A[t-1] + log c. L_can = A[T-1] + log(sum(a * exp(fin))).
    Backward: b = exp(fin); per frame t = T-1 .. 1: g = b * em_b[t];
    b = g + (sum over arcs p->q of exp(w) * g[q] into p), d = sum(b), b /= d, B[t-1] = B[t] + log d.
    Returns None when the forward mass vanishes (then the exact pass is used)."""
    T = len(em)
    S = lat.n_states
    src, dst = lat.src, lat.dst
    w = [exp(x) for x in lat.w]
    nA = len(src)
    ahat, A = [], []
    a = [exp(lat.init[s]) * em[0][s] for s in range(S)]
    c = sum_pairwise(a)
    if not c > 0:
        return None
    a = [x / c for x in a]
    ahat.append(a)
    acc = log(c)
    A.append(acc)
    for t in range(1, T):
        inc = [0.0] * S
        for k in range(nA):
            inc[dst[k]] += a[src[k]] * w[k]
        a = [(a[s] + inc[s]) * em[t][s] for s in range(S)]
        c = sum_pairwise(a)
        if not c > 0:
            return None
        a = [x / c for x in a]
        ahat.append(a)
        acc += log(c)
        A.append(acc)
    fin = [exp(x) for x in lat.fin]
    z = sum_pairwise([a[s] * fin[s] for s in range(S)])
    if not z > 0:
        return None
    L_can = A[-1] + log(z)
    bhat = [None] * T
    B = [0.0] * T
    b = list(fin)
    bhat[T - 1] = b
    acc = 0.0
    for t in range(T - 1, 0, -1):
        g = [b[s] * em_b[t][s] for s in range(S)]
        inc = [0.0] * S
        for k in range(nA):
            inc[src[k]] += w[k] * g[dst[k]]
        b = [g[s] + inc[s] for s in range(S)]
        d = sum_pairwise(b)
        b = [div(x, d) for x in b]
        bhat[t - 1] = b
        acc += log(d)
        B[t - 1] = acc
    return ahat, A, bhat, B, L_can


def _lse_frames(col, pairwise: bool) -> float:
    """log-sum-exp over frames of one column (see the summation order notes in numeric)."""
    return lse(col, "pairwise" if pairwise else "seq")


def _grouped(n: int, rows, fill):
    """Per slot, its rows in input order followed by one dummy row `fill` (production adds a
    zero-valued dummy term per slot so no group is empty; it changes no value but is part of
    the summation order)."""
    g = [[] for _ in range(n)]
    for r in rows:
        g[r[0]].append(r[1:])
    for i in range(n):
        g[i].append(fill)
    return g


def _fast_scores(slots, lat, tab, insertion: bool = False):
    n = len(slots)
    T = len(tab)
    S = lat.n_states
    E = _emissions(lat, tab)
    em = [[exp(x) for x in row] for row in E]
    longest = 1
    for sl in slots:
        for ids, _, _ in sl.alts:
            if len(ids) > longest:
                longest = len(ids)
    slack = 2 * longest + 2
    m_end, m_start, min_len = _reach(lat, T, slack)
    if min_len > T:
        return NEG_INF, [NAN] * n, [NAN] * n
    em_f = [[em[t][s] if m_end[t][s] else 0.0 for s in range(S)] for t in range(T)]
    em_s = [[em[t][s] if m_start[t][s] else 0.0 for s in range(S)] for t in range(T)]
    fb = _forward_backward(lat, em_f, em_s)
    if fb is None:
        return None
    ahat, A, bhat, B, L_can = fb
    if n == 0:
        return L_can, [], []
    ar = slot_arcs(slots, lat)
    START, END, DUMMY = S, S + 1, S + 2

    # alpha_aug row r = alpha at frame r - 1 (row 0: START = 1), scale A_aug[r];
    # eb_aug row r = em_s[r] * beta at frame r (row T: END = 1), scale B_aug[r]
    def alpha_aug(r, s):
        if s == DUMMY or s == END:
            return 0.0
        if r == 0:
            return 1.0 if s == START else 0.0
        return 0.0 if s == START else ahat[r - 1][s]

    def eb_aug(r, s):
        if s == DUMMY or s == START:
            return 0.0
        if r == T:
            return 1.0 if s == END else 0.0
        return 0.0 if s == END else em_s[r][s] * bhat[r][s]

    A_aug = [0.0] + A
    B_aug = B + [0.0]
    pw_frames = n == 1           # a one-column table is summed in pairwise order (see numeric)

    # ---- deletion: L_del[i] = log sum over frames r and bypass arcs (p -> q) of
    #      alpha_aug[r][p] * exp(w) * eb_aug[r][q], each frame's terms rescaled by A_aug + B_aug
    cross = _grouped(n, ar.cross, (DUMMY, DUMMY, 0.0))
    L_del = []
    for i in range(n):
        col = []
        for r in range(T + 1):
            terms = [alpha_aug(r, p) * exp(wt) * eb_aug(r, q) for p, q, wt in cross[i]]
            col.append(log(sum_first_pw(terms)) + (A_aug[r] + B_aug[r]))
        L_del.append(_lse_frames(col, pw_frames))

    # ---- wildcard: flow into slot i at frame t (inj, scale A_aug[t]) and out of it (out, scale
    #      B_aug[t + 1])
    inn = _grouped(n, ar.inn, (DUMMY, 0.0))
    outa = _grouped(n, ar.out, (DUMMY, 0.0))
    inj = [[log(sum_first_pw([alpha_aug(t, p) * exp(wt) for p, wt in inn[i]])) for i in range(n)]
           for t in range(T)]
    out = [[log(sum_first_pw([eb_aug(t + 1, q) * exp(wt) for q, wt in outa[i]])) for i in range(n)]
           for t in range(T)]
    pairs = []                    # (slot, chain length, chain log weight), slot-major
    for i in range(n):
        for K, wt in wild_chains(slots[i]):
            pairs.append((i, K, wt))
    if not pairs:
        return L_can, list(L_del), L_del
    P = len(pairs)
    p_slot = [q[0] for q in pairs]
    p_last = [q[1] - 1 for q in pairs]
    p_w = [q[2] for q in pairs]
    n_ins = n - 1 if (insertion and n > 1) else 0
    Kmax = max(max(p_last) + 1, 2 if n_ins else 1)
    # chain rows: one per pair, fed by inj[:, slot] + chain weight; for the insertion evidence the
    # first weight-0 row of slot i + 1 is reused, or an extra row fed by inj[:, i + 1] is added
    feed = [[inj[t][p_slot[r]] + p_w[r] for r in range(P)] for t in range(T)]
    ins_row = []
    if n_ins:
        zero_row = {}
        for r in range(P):
            if p_w[r] == 0.0 and p_slot[r] not in zero_row:
                zero_row[p_slot[r]] = r
        ins_row = [zero_row.get(i + 1, -1) for i in range(n - 1)]
        miss = [i for i in range(n - 1) if ins_row[i] < 0]
        for q, i in enumerate(miss):
            ins_row[i] = P + q
            for t in range(T):
                feed[t].append(inj[t][i + 1])
    R2 = len(feed[0]) if T else P
    # chain recursion in the log domain, rescaled by the forward scale (A[t]):
    #   W_0[t] = logaddexp(W_0[t-1], feed[t]) + lew[t]
    #   W_k[t] = logaddexp(W_k[t-1], logaddexp(W_{k-1}[t-1], B_{k-1}[t-1])) + lew[t]
    #   B_k[t] = logaddexp(B_k[t-1], W_k[t-1]) + leb[t]
    # with lew = wildcard emission - log c_t, leb = blank emission - log c_t,
    # log c_t = A_aug[t + 1] - A_aug[t]; O_k[t] = logaddexp(W_k[t], B_k[t]) (k < Kmax - 1).
    Wh = [None] * T
    Bh = [None] * T
    Oh = [None] * T
    Wc = [[NEG_INF] * Kmax for _ in range(R2)]
    Bc = [[NEG_INF] * Kmax for _ in range(R2)]
    for t in range(T):
        logc = A_aug[t + 1] - A_aug[t]
        lew = tab[t][COL_WILD] - logc
        leb = tab[t][COL_BLANK] - logc
        Wn = [[0.0] * Kmax for _ in range(R2)]
        Bn = [[0.0] * Kmax for _ in range(R2)]
        Oprev = [[logaddexp(Wc[r][k], Bc[r][k]) for k in range(Kmax - 1)] for r in range(R2)]
        if t:
            Oh[t - 1] = Oprev
        for r in range(R2):
            Wn[r][0] = logaddexp(Wc[r][0], feed[t][r]) + lew
            for k in range(1, Kmax):
                Wn[r][k] = logaddexp(Wc[r][k], Oprev[r][k - 1]) + lew
            for k in range(Kmax):
                Bn[r][k] = logaddexp(Bc[r][k], Wc[r][k]) + leb
        Wh[t], Bh[t] = Wn, Bn
        Wc, Bc = Wn, Bn
    if T:
        Oh[T - 1] = [[logaddexp(Wc[r][k], Bc[r][k]) for k in range(Kmax - 1)] for r in range(R2)]

    def occupancy(t, r, k):
        """log(W_k + B_k) at frame t of row r (scale A[t])."""
        if k < Kmax - 1:
            return Oh[t][r][k]
        return logaddexp(Wh[t][r][k], Bh[t][r][k])

    # L_pair: chain of row r completed at frame t, then drained out of slot p_slot[r]
    L_x = [NEG_INF] * n
    for r in range(P):
        col = [occupancy(t, r, p_last[r]) + out[t][p_slot[r]] + (A[t] + B_aug[t + 1]) for t in range(T)]
        L_x[p_slot[r]] = logaddexp(L_x[p_slot[r]], _lse_frames(col, True))
    L_wild = [logaddexp(L_x[i], L_del[i]) for i in range(n)]
    if not n_ins:
        return L_can, L_wild, L_del
    # L_ins: one or two extra sounds in gap i (chain states 0 and 1 of row ins_row[i]), fed like
    # slot i + 1's wildcard and drained like slot i's
    L_ins = [NAN] * n
    for i in range(n - 1):
        Lc = NEG_INF
        for kk in (0, 1):
            col = []
            for t in range(T):
                o = out[t][i]
                if kk < Kmax - 1:
                    occ = Oh[t][ins_row[i]][kk]
                elif o > NEG_INF:
                    occ = logaddexp(Wh[t][ins_row[i]][kk], Bh[t][ins_row[i]][kk])
                else:
                    occ = NEG_INF
                col.append(occ + o + (A[t] + B_aug[t + 1]))
            Lc = logaddexp(Lc, _lse_frames(col, False))
        L_ins[i] = logaddexp(L_can, Lc)
    return L_can, L_wild, L_del, L_ins


# --------------------------------------------------------------------------- Viterbi

def viterbi(lat, tab):
    """Best path (log domain, max instead of sum). D[0][s] = init[s] + E[0][s];
    D[t][s] = max over predecessors (self first) of D[t-1][p] + w + E[t][s]. The final state is
    the first maximiser of D[T-1][s] + fin[s]; back-pointers are recomputed on the way back (the
    first predecessor reaching the maximum). Returns (score, [state per frame]) or (-inf, [])."""
    T = len(tab)
    S = lat.n_states
    if T == 0:
        return NEG_INF, []
    E = _emissions(lat, tab)
    P, W = _preds(lat)
    K1 = len(P[0]) if S else 1
    D = [[lat.init[s] + E[0][s] for s in range(S)] + [NEG_INF]]
    for t in range(1, T):
        prev = D[t - 1]
        row = []
        for s in range(S):
            best = max_list([prev[P[s][k]] + W[s][k] for k in range(K1)])
            row.append(best + E[t][s])
        row.append(NEG_INF)
        D.append(row)
    x = [D[T - 1][s] + lat.fin[s] for s in range(S)]
    s = 0
    for j in range(S):
        if x[j] != x[j]:
            s = j
            break
        if x[j] > x[s]:
            s = j
    score = x[s]
    if not is_finite(score):
        return NEG_INF, []
    path = [0] * T
    path[T - 1] = s
    for t in range(T - 1, 0, -1):
        cand = [D[t - 1][P[s][k]] + W[s][k] for k in range(K1)]
        kbest = 0
        for k in range(K1):
            if cand[k] != cand[k]:
                kbest = k
                break
            if cand[k] > cand[kbest]:
                kbest = k
        s = P[s][kbest]
        path[t - 1] = s
    return score, path


# --------------------------------------------------------------------------- evidence

def greedy_tokens(lp, tokens) -> list:
    """Collapsed greedy decode: per frame the first arg-max column; runs of equal ids merge;
    ids 0..3 dropped. Returns [(token, start frame, end frame exclusive)]."""
    ids = []
    for row in lp:
        best = 0
        for k in range(len(row)):
            if row[k] != row[k]:
                best = k
                break
            if row[k] > row[best]:
                best = k
        ids.append(best)
    out = []
    t = 0
    while t < len(ids):
        e = t + 1
        while e < len(ids) and ids[e] == ids[t]:
            e += 1
        if ids[t] >= NON_SPEECH:
            out.append((tokens[ids[t]], t, e))
        t = e
    return out


def _spans(lat, slots, tab, T, possible):
    n = len(slots)
    if not possible or n == 0:
        return [(0, 0, -1)] * n
    _, path = viterbi(lat, tab)
    if not path:
        return [(0, 0, -1)] * n
    out = []
    for i in range(n):
        fr = [t for t in range(T) if lat.slot[path[t]] == i and lat.is_tok[path[t]]]
        if fr:
            out.append((fr[0], fr[-1] + 1, lat.alt[path[fr[0]]]))
        else:
            after = [t for t in range(T) if lat.slot[path[t]] > i]
            t = after[0] if after else T
            out.append((t, t, -1))
    return out


def score_slots(lp, slots, tokens, *, fixed=(), insertion: bool = False, exact: bool = False,
                frame_shift: float = 0.02) -> dict:
    """Evidence for every slot. lp: [T][127] log-posteriors; slots from slots_from_phones (fixed
    slots already replaced by wild_slot). Returns {"L_can", "n_frames", "frame_shift",
    "free_tokens", "slots": [{llr_sub, llr_del, start_frame, end_frame, alt_index, peak_post,
    heard, n_frames}], "llr_ins" (list or None)}. peak_post = exp(max of lp over the span of the
    chosen alternative's first token), computed in float32 (production evaluates exp on a float32
    value)."""
    T = len(lp)
    n = len(slots)
    fixed = sorted(set(int(i) for i in fixed))
    free = greedy_tokens(lp, tokens) if T else []
    lat = build_lattice(slots)
    tab = emission_table(lp) if T else []
    res = None
    if T and not exact:
        res = _fast_scores(slots, lat, tab, insertion=insertion)
    if T and (exact or res is None):
        which = [i for i in range(n) if i not in fixed]
        res = exact_scores(slots, tab, which=which)
    if T == 0:
        res = (NEG_INF, [NEG_INF] * n, [NEG_INF] * n)
    L_ins = res[3] if len(res) > 3 else None
    L_can, L_wild, L_del = res[0], list(res[1]), list(res[2])
    if is_finite(L_can) and n:
        bad = [not is_finite(L_wild[i]) or not is_finite(L_del[i]) or (L_wild[i] < L_can - 1e-6)
               for i in range(n)]
        for i in fixed:
            bad[i] = False
        if any(bad) and not exact:
            _, Lw2, Ld2 = exact_scores(slots, tab, which=[i for i in range(n) if bad[i]])
            for i in range(n):
                if bad[i]:
                    L_wild[i], L_del[i] = Lw2[i], Ld2[i]
    possible = is_finite(L_can)
    if possible:
        llr_sub = [minimum(L_can - L_wild[i], 0.0) for i in range(n)]
        llr_del = []
        for i in range(n):
            v = (L_can - maximum(L_del[i], LOG_FLOOR)) if is_finite(L_del[i]) else -IMPOSSIBLE_LLR
            if v == v:
                v = min(max(v, IMPOSSIBLE_LLR), -IMPOSSIBLE_LLR)
            llr_del.append(v)
    else:
        llr_sub = [IMPOSSIBLE_LLR] * n
        llr_del = [IMPOSSIBLE_LLR] * n
    for i in fixed:
        llr_sub[i] = 0.0
        llr_del[i] = 0.0
    spans = _spans(lat, slots, tab, T, possible)
    out = []
    for i, (s, e, k) in enumerate(spans):
        if k >= 0 and e > s:
            ft = lat.first_tok[i][k]
            peak = float32(exp(max_list([float(lp[t][ft]) for t in range(s, e)])))
        else:
            peak = 0.0
        if e > s:
            heard = "".join(tok for tok, a, b in free if b > s and a < e)
        else:
            heard = ""
        out.append({"llr_sub": float(llr_sub[i]), "llr_del": float(llr_del[i]), "start_frame": s,
                    "end_frame": e, "alt_index": k, "peak_post": peak, "heard": heard, "n_frames": e - s})
    ev = {"slots": out, "L_can": L_can if possible else LOG_FLOOR, "n_frames": T,
          "frame_shift": frame_shift, "free_tokens": free, "llr_ins": None}
    if insertion:
        if possible and L_ins is not None:
            ev["llr_ins"] = [minimum(L_can - x, 0.0) for x in L_ins]
        else:
            ev["llr_ins"] = [NAN] * n
    return ev


def score_phones(lp, phones, tokens, *, wild_slots=(), insertion: bool = False,
                 exact: bool = False) -> dict:
    """Evidence for target phones ([{"ipa", "alts"}]); wild_slots = slots replaced by their
    wildcard in the base lattice (explain-away pass; they report llr 0 and alt_index -2...)."""
    from .lattice import slots_from_phones

    tid = {t: i for i, t in enumerate(tokens)}
    slots = slots_from_phones(phones, tid)
    fixed = sorted(set(int(i) for i in wild_slots))
    for i in fixed:
        slots[i] = wild_slot(slots[i])
    return score_slots(lp, slots, tokens, fixed=fixed, insertion=insertion, exact=exact)
