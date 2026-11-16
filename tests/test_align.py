"""Lattice aligner: brute-force path enumeration on tiny inputs, textbook CTC, fast vs exact,
degenerate inputs, and model-based behaviour on synthetic speech."""
import itertools

import numpy as np
import pytest

from pronunciada.align import (IMPOSSIBLE_LLR, emission_table, exact_scores, score_reference,
                           score_slots, viterbi, _fast_scores)
from pronunciada.lattice import Slot, build_lattice, slots_from_reference
from pronunciada.text.ipa import segment, to_ids
from pronunciada.types import RefPhone, Reference

TOKS = [4, 5, 6]          # three arbitrary speech token ids


def rand_lp(rng, T, peaky=2.0):
    x = rng.standard_normal((T, 127)) * peaky
    x[:, 0] += 1.5
    return x - np.logaddexp.reduce(x, axis=1, keepdims=True)


def rand_slots(rng, n, max_len=2, p_skip=0.4, max_alts=2):
    slots = []
    for _ in range(n):
        alts = [(tuple(int(t) for t in rng.choice(TOKS, size=rng.integers(1, max_len + 1))), 0.0, 0)]
        seen = {alts[0][0]}
        for k in range(int(rng.integers(0, max_alts + 1))):
            ids = tuple(int(t) for t in rng.choice(TOKS, size=rng.integers(1, max_len + 1)))
            if ids in seen:
                continue
            seen.add(ids)
            alts.append((ids, float(-rng.random() * 2), len(alts)))
        skip = float(-rng.random() * 2) if rng.random() < p_skip else -np.inf
        slots.append(Slot(alts, skip))
    return slots


# ----------------------------------------------------------------------------- brute force

def collapse(labels):
    out, prev = [], None
    for l in labels:
        if l != prev and l != "B":
            out.append(l)
        prev = l
    return tuple(out)


def brute_by_sequence(lp):
    """dict: collapsed token sequence -> (logsumexp, max) over all frame labelings."""
    T = len(lp)
    blank = np.logaddexp(lp[:, 0], lp[:, 3])
    acc_sum, acc_max = {}, {}
    for lab in itertools.product(["B"] + TOKS, repeat=T):
        s = sum(blank[t] if l == "B" else lp[t, l] for t, l in enumerate(lab))
        y = collapse(lab)
        acc_sum[y] = np.logaddexp(acc_sum.get(y, -np.inf), s)
        acc_max[y] = max(acc_max.get(y, -np.inf), s)
    return acc_sum, acc_max


def brute_lattice(lp, slots, table=None):
    """(log full-sum, log max) over every lattice hypothesis (choice of alt/skip per slot)."""
    sums, maxs = table or brute_by_sequence(lp)
    options = []
    for s in slots:
        o = [(ids, w) for ids, w, _ in s.alts]
        if s.skippable:
            o.append(((), s.skip))
        options.append(o)
    tot, best = -np.inf, -np.inf
    for choice in itertools.product(*options):
        y = tuple(t for ids, _ in choice for t in ids)
        w = sum(w for _, w in choice)
        if y in sums:
            tot = np.logaddexp(tot, w + sums[y])
            best = max(best, w + maxs[y])
    return tot, best


def ctc_textbook(lp, y):
    """Standard CTC forward over the extended label sequence (blank = <blk> + ▁)."""
    blank = np.logaddexp(lp[:, 0], lp[:, 3])
    ext = [-1]
    for t in y:
        ext += [t, -1]
    S, T = len(ext), len(lp)
    em = lambda t, s: blank[t] if ext[s] < 0 else lp[t, ext[s]]  # noqa: E731
    a = np.full(S, -np.inf)
    a[0] = em(0, 0)
    if S > 1:
        a[1] = em(0, 1)
    for t in range(1, T):
        b = np.full(S, -np.inf)
        for s in range(S):
            v = a[s]
            if s >= 1:
                v = np.logaddexp(v, a[s - 1])
            if s >= 2 and ext[s] >= 0 and ext[s] != ext[s - 2]:
                v = np.logaddexp(v, a[s - 2])
            b[s] = v + em(t, s)
        a = b
    return np.logaddexp(a[-1], a[-2]) if S > 1 else a[-1]


def _close(a, b, tol=1e-8):
    if not np.isfinite(a) or not np.isfinite(b):
        return a == b
    return abs(a - b) < tol


# ----------------------------------------------------------------------------- tests

@pytest.mark.parametrize("seed", range(40))
def test_canonical_and_deletion_match_bruteforce(seed):
    rng = np.random.default_rng(seed)
    T = int(rng.integers(1, 7))
    n = int(rng.integers(1, 4))
    slots = rand_slots(rng, n)
    lp = rand_lp(rng, T)
    table = brute_by_sequence(lp)
    tab = emission_table(lp)
    ref_can, ref_max = brute_lattice(lp, slots, table)
    L_can, L_wild, L_del = exact_scores(slots, tab)
    assert _close(L_can, ref_can)
    for i in range(n):
        ref_del, _ = brute_lattice(lp, slots[:i] + slots[i + 1:], table)
        assert _close(L_del[i], ref_del)
        if np.isfinite(L_can):
            assert L_wild[i] >= L_can - 1e-9
    fast = _fast_scores(slots, build_lattice(slots), tab)
    assert fast is not None
    if not np.isfinite(fast[0]):     # infeasible canonical lattice: variants are not stitched
        assert not np.isfinite(ref_can)
    else:
        assert _close(fast[0], ref_can)
        for i in range(n):
            assert _close(fast[2][i], L_del[i], 1e-7)
            assert _close(fast[1][i], L_wild[i], 1e-7)
    score, path = viterbi(build_lattice(slots), tab)
    assert _close(score, ref_max)


def test_repeated_tokens_need_blank_across_slots_and_skips():
    rng = np.random.default_rng(0)
    lp = rand_lp(rng, 2)
    a, b = TOKS[0], TOKS[1]
    tab = emission_table(lp)
    # "a" "a" in two frames is impossible (needs a blank in between)
    L, _, _ = exact_scores([Slot([((a,), 0.0, 0)]), Slot([((a,), 0.0, 0)])], tab)
    assert L == -np.inf
    # "a" [skippable b] "a": the b-less path still needs a blank -> only "a b a"/... possible
    lp3 = rand_lp(rng, 3)
    slots = [Slot([((a,), 0.0, 0)]), Slot([((b,), 0.0, 0)], skip=-0.5), Slot([((a,), 0.0, 0)])]
    ref, _ = brute_lattice(lp3, slots)
    L, _, _ = exact_scores(slots, emission_table(lp3))
    assert _close(L, ref)
    # within one alternative: "aa" needs >= 3 frames
    L2, _, _ = exact_scores([Slot([((a, a), 0.0, 0)])], tab)
    assert L2 == -np.inf


@pytest.mark.parametrize("seed", range(10))
def test_no_alts_equals_textbook_ctc(seed):
    rng = np.random.default_rng(100 + seed)
    T = int(rng.integers(5, 40))
    y = [int(t) for t in rng.choice(TOKS + [7, 8], size=rng.integers(1, max(2, T // 3)))]
    lp = rand_lp(rng, T)
    slots = [Slot([((t,), 0.0, 0)]) for t in y]
    want = ctc_textbook(lp, y)
    tab = emission_table(lp)
    assert _close(exact_scores(slots, tab)[0], want, 1e-8)
    assert _close(_fast_scores(slots, build_lattice(slots), tab)[0], want, 1e-8)
    # grouping tokens into multi-token phones does not change the canonical score
    grouped = [Slot([(tuple(y[k:k + 2]), 0.0, 0)]) for k in range(0, len(y), 2)]
    assert _close(exact_scores(grouped, tab)[0], want, 1e-8)


@pytest.mark.parametrize("seed", range(15))
def test_fast_matches_exact_and_wild_dominates(seed):
    rng = np.random.default_rng(1000 + seed)
    T = int(rng.integers(10, 60))
    n = int(rng.integers(1, 12))
    slots = rand_slots(rng, n, max_len=3, p_skip=0.3, max_alts=3)
    lp = rand_lp(rng, T, peaky=3.0)
    fast = score_slots(lp, slots)
    ex = score_slots(lp, slots, exact=True)
    assert abs(fast.L_can - ex.L_can) < 1e-6
    for f, e in zip(fast.slots, ex.slots):
        assert abs(f.llr_sub - e.llr_sub) < 1e-6
        assert abs(f.llr_del - e.llr_del) < 1e-6
        assert f.llr_sub <= 0.0
        assert (f.start_frame, f.end_frame, f.alt_index) == (e.start_frame, e.end_frame, e.alt_index)


def test_fast_survives_reference_much_longer_than_audio(monkeypatch):
    """Learner stopped early: the reference is 2.5x what the audio holds. The scaled passes must
    not underflow (and must not need the exact fallback)."""
    import pronunciada.align as A

    rng = np.random.default_rng(7)
    said = [int(t) for t in rng.choice(TOKS + [7, 8, 9], size=40)]
    frames = []
    for t in said:
        frames += [t, t, 0]
    lp = np.full((len(frames), 127), -15.0)
    lp[np.arange(len(frames)), frames] = 0.0
    lp += rng.standard_normal(lp.shape)
    lp -= np.logaddexp.reduce(lp, axis=1, keepdims=True)
    slots = [Slot([((t,), 0.0, 0)]) for t in said * 2 + said[:20]]
    calls = []
    orig = A.exact_scores
    monkeypatch.setattr(A, "exact_scores", lambda *a, **k: calls.append(1) or orig(*a, **k))
    fast = score_slots(lp, slots)
    assert not calls
    monkeypatch.setattr(A, "exact_scores", orig)
    ex = score_slots(lp, slots, exact=True)
    assert abs(fast.L_can - ex.L_can) < 1e-6 and fast.L_can < -100
    for f, e in zip(fast.slots, ex.slots):
        assert abs(f.llr_sub - e.llr_sub) < 1e-6 and abs(f.llr_del - e.llr_del) < 1e-6
    # 4x: cannot fit at all -> detected up front, no slow path
    monkeypatch.setattr(A, "exact_scores", lambda *a, **k: calls.append(1) or orig(*a, **k))
    ev = score_slots(lp, [Slot([((t,), 0.0, 0)]) for t in said * 4])
    assert not calls and all(s.llr_sub == IMPOSSIBLE_LLR for s in ev.slots)


def _ref(phones, alts=None):
    alts = alts or {}
    return Reference("en", "", phones=[RefPhone(p, 0, 0, alts=alts.get(i, [])) for i, p in enumerate(phones)])


def _peaky_lp(frames, T):
    """Posteriors that strongly prefer the given token string per frame ('-' = blank)."""
    lp = np.full((T, 127), -12.0)
    for t, c in enumerate(frames):
        lp[t, 0 if c == "-" else to_ids(c)[0]] = 0.0
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)


def test_alternative_and_drop_are_reported():
    # audio says "s - ɑ - t"; reference "s ɪ t" with alt "ɑ" for the vowel, and a droppable "k"
    lp = _peaky_lp("-ss-ɑɑ-tt-", 10)
    ref = _ref(["s", "ɪ", "k", "t"], alts={1: [("ɑ", -1.0)], 2: [("", -0.5)]})
    for exact in (False, True):
        ev = score_reference(lp, ref, exact=exact)
        s, v, k, t = ev.slots
        assert v.alt_index == 1 and v.heard == "ɑ" and v.n_frames == 2
        assert k.alt_index == -1 and k.n_frames == 0 and k.heard == ""
        assert s.alt_index == 0 and (s.start_frame, s.end_frame) == (1, 3)
        assert t.peak_post > 0.99
        assert k.llr_del < 0 < s.llr_del
        assert max(x.llr_sub for x in ev.slots) <= 0
    assert [tok for tok, _, _ in ev.free_tokens] == ["s", "ɑ", "t"]


def test_degenerate_inputs():
    rng = np.random.default_rng(3)
    # empty reference
    ev = score_reference(rand_lp(rng, 5), _ref([]))
    assert ev.slots == [] and np.isfinite(ev.L_can)
    # more tokens than frames
    ev = score_reference(rand_lp(rng, 2), _ref(["a", "b", "k", "s"]))
    assert np.isfinite(ev.L_can) and ev.L_can < -1e5
    assert all(np.isfinite(s.llr_sub) and s.llr_sub == IMPOSSIBLE_LLR for s in ev.slots)
    ev2 = score_reference(rand_lp(rng, 2), _ref(["a", "b", "k", "s"]), exact=True)
    assert [s.llr_sub for s in ev2.slots] == [s.llr_sub for s in ev.slots]
    # zero and one frame
    ev = score_reference(np.zeros((0, 127)), _ref(["a"]))
    assert len(ev.slots) == 1 and ev.slots[0].llr_sub == IMPOSSIBLE_LLR and ev.n_frames == 0
    ev = score_reference(rand_lp(rng, 1), _ref(["a"]))
    assert np.isfinite(ev.slots[0].llr_sub) and ev.slots[0].n_frames == 1
    # everything droppable, silent audio
    lp = _peaky_lp("-----", 5)
    ev = score_reference(lp, _ref(["a", "b"], alts={0: [("", -1.0)], 1: [("", -1.0)]}))
    assert all(s.alt_index == -1 for s in ev.slots) and np.isfinite(ev.L_can)


# ----------------------------------------------------------------------------- with the model

def _greedy_reference(model, lp):
    phones = segment("".join(t for t, _, _ in model.greedy(lp)))
    return phones, _ref(phones)


SENT = "Yesterday the committee announced a remarkable new plan for public transport."


@pytest.mark.model
def test_correct_reference_scores_well(acoustic_model, tts):
    lp = acoustic_model.posteriors(tts(SENT))
    phones, ref = _greedy_reference(acoustic_model, lp)
    assert len(phones) > 30
    ev = score_reference(lp, ref)
    sub = np.array([s.llr_sub for s in ev.slots])
    dl = np.array([s.llr_del for s in ev.slots])
    # the wildcard can always also absorb neighbouring frames, so a correct sound sits around
    # -0.8 (segmentation freedom), never near the values of a wrong sound (see next test)
    assert np.median(sub) > -1.5
    assert (sub > -3.0).mean() > 0.95
    assert (dl > 0).mean() > 0.95
    assert all(s.heard for s in ev.slots if s.n_frames)


@pytest.mark.model
def test_substitution_is_local_and_strong(acoustic_model, tts):
    lp = acoustic_model.posteriors(tts(SENT))
    phones, ref = _greedy_reference(acoustic_model, lp)
    base = score_reference(lp, ref)
    vowels = [i for i, p in enumerate(phones) if p[0] in "aeiouɑɛɪʊəʌæɔ" and 8 <= i < len(phones) - 8]
    for i in vowels[:4]:
        bad = list(phones)
        bad[i] = "s" if phones[i - 1] != "s" and phones[i + 1] != "s" else "f"
        ev = score_reference(lp, _ref(bad))
        assert ev.slots[i].llr_sub < -5.0, (i, phones[i], ev.slots[i])
        assert ev.slots[i].llr_sub < base.slots[i].llr_sub - 4.0
        far = [j for j in range(len(phones)) if abs(j - i) > 4]
        diff = max(abs(ev.slots[j].llr_sub - base.slots[j].llr_sub) for j in far)
        assert diff < 0.05


@pytest.mark.model
def test_deletion_evidence_direction(acoustic_model, tts):
    lp = acoustic_model.posteriors(tts(SENT))
    phones, _ = _greedy_reference(acoustic_model, lp)
    i = len(phones) // 2
    # (1) spurious phone inserted into the reference (not in the audio): llr_del < 0
    spurious = "k" if "k" not in phones[i - 1:i + 1] else "m"
    ins = phones[:i] + [spurious] + phones[i:]
    ev_ins = score_reference(lp, _ref(ins))
    assert ev_ins.slots[i].llr_del < 0
    # (2) a phone that is really there: removing it from the reference hurts -> llr_del > 0
    ev = score_reference(lp, _ref(phones))
    assert ev.slots[i].llr_del > 2.0
    assert ev.slots[i].llr_del > ev_ins.slots[i].llr_del + 2.0
    # (3) the audio's phone removed from the reference: the neighbours absorb it, the
    # neighbour that has to explain the missing sound looks worse
    rem = phones[:i] + phones[i + 1:]
    ev_rem = score_reference(lp, _ref(rem))
    assert min(ev_rem.slots[i - 1].llr_sub, ev_rem.slots[i].llr_sub) < \
        min(ev.slots[i - 1].llr_sub, ev.slots[i + 1].llr_sub)


@pytest.mark.model
def test_fast_vs_exact_on_speech(acoustic_model, tts):
    lp = acoustic_model.posteriors(tts("I would like a cup of coffee with milk, please."))
    phones, ref = _greedy_reference(acoustic_model, lp)
    ref.phones[3].alts = [("", -2.0), ("ə", -1.0)]
    f = score_reference(lp, ref)
    e = score_reference(lp, ref, exact=True)
    err = max(max(abs(a.llr_sub - b.llr_sub), abs(a.llr_del - b.llr_del)) for a, b in zip(f.slots, e.slots))
    assert err < 1e-6
