"""Extra-sound evidence (Evidence.llr_ins) of the fast pass against explicit lattices: an "any
speech" slot of one or two sounds, or nothing, inserted between slot i and slot i + 1."""
import numpy as np
import pytest

from pronunciada.align import emission_table, forward_log, score_slots
from pronunciada.lattice import Slot, build_lattice, wild_slot

TOKS = [4, 5, 6]


def rand_lp(rng, T, peaky=2.0):
    x = rng.standard_normal((T, 127)) * peaky
    x[:, 0] += 1.5
    return x - np.logaddexp.reduce(x, axis=1, keepdims=True)


def rand_slots(rng, n, max_len, shift_p):
    slots = []
    for _ in range(n):
        alts = [(tuple(int(t) for t in rng.choice(TOKS, size=rng.integers(1, max_len + 1))), 0.0, 0)]
        for _ in range(int(rng.integers(0, 3))):
            ids = tuple(int(t) for t in rng.choice(TOKS, size=rng.integers(1, max_len + 1)))
            if all(ids != a[0] for a in alts):
                alts.append((ids, float(-rng.random() * 2), len(alts)))
        if rng.random() < shift_p:          # no chain of weight 0 for this slot
            alts = [(ids, w - 0.3, k) for ids, w, k in alts]
        skip = float(-rng.random() * 2) if rng.random() < 0.3 else -np.inf
        slots.append(Slot(alts, skip))
    return slots


def exact_llr_ins(lp, slots):
    tab = emission_table(lp)
    L_can = forward_log(build_lattice(slots), tab)[0]
    ins = Slot([], 0.0, wild=[(1, 0.0), (2, 0.0)])          # nothing, one or two extra sounds
    out = np.full(len(slots), np.nan)
    for i in range(len(slots) - 1):
        L = forward_log(build_lattice(slots[:i + 1] + [ins] + slots[i + 1:]), tab)[0]
        out[i] = min(L_can - L, 0.0)
    return out


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("max_len,shift_p,wild_p", [(3, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.5, 0.0), (2, 0.2, 0.3)])
def test_fast_llr_ins_matches_explicit_lattices(seed, max_len, shift_p, wild_p):
    # (1, ...): single-sound slots only (one chain state); shift_p: slots without a weight-0 chain;
    # wild_p: pre-wildcarded slots (no chains of their own)
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(2, 10))
    slots = rand_slots(rng, n, max_len, shift_p)
    fixed = [i for i in range(n) if rng.random() < wild_p]
    for i in fixed:
        slots[i] = wild_slot(slots[i])
    lp = rand_lp(rng, int(rng.integers(3 * n, 8 * n)), peaky=3.0)
    ev = score_slots(lp, slots, fixed=fixed, insertion=True)
    assert ev.llr_ins is not None and len(ev.llr_ins) == n and np.isnan(ev.llr_ins[-1])
    if len(fixed) == n:
        return
    ref = exact_llr_ins(lp, slots)
    assert np.allclose(ev.llr_ins[:-1], ref[:-1], atol=1e-6, rtol=0), (ev.llr_ins, ref)
