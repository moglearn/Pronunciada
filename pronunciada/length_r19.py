"""Round 19 (Japanese): duration measured on the 10 ms filterbank frames, normalised by the local
mora rate, as extra inputs of the length model (pronunciada.length).

The round-15 features read the canonical alignment's onsets (20 ms output frames, soft CTC
boundaries). These read the signal around the aligned position instead:

* ``v_stable``  vowel: the run of consecutive 10 ms frames around the vowel's centre whose
                (utterance-mean-removed) log-mel spectrum stays close (cosine >= COS) to the
                centre frames and is not silent; log of its length over the local mora duration;
* ``c_dip``     consonant: the longest run of frames between the neighbouring vowels' centres whose
                energy lies DIP_DB below the lower of the two vowels' centre energies (the closure
                of っ, the hold of a long fricative); log of its length over the local mora duration;
* ``rate_loc``  log local mora duration: speech time of the utterance outside the word divided by
                the morae outside the word (long vowel and geminate = 2 morae);
* ``word_ratio`` log of the word's aligned duration over (its morae, this sound counted short, x the
                local mora duration): a long sound adds about +0.4 to a two-mora word.
* ``cf_llr``    counterfactual: best-path log-likelihood of the sound's long form against its short
                form (aː / a, tt / t) on the frames of the sound and its neighbours, from the frame
                posteriors (positive = the long form fits better, whatever the text expects).

Every feature reads "how long", never "how well the expected form fits": the trees are fitted on
natives, whose sounds always match their text, so a text-relative feature never varies with the
label there and the trees cannot learn it.
"""
from __future__ import annotations

import math

import numpy as np

from .text.ipa import is_vowel

COS = 0.85
DIP_DB = 8.0
EXTRA = ("v_stable", "c_dip", "rate_loc", "word_ratio", "cf_llr")
_LN10 = math.log(10.0)


def _fbi(t: float) -> int:
    return int(round((t * 0.02 + 0.03) / 0.01))


def _morae(ref, w) -> int:
    n = w.syllables[1] - w.syllables[0]
    for i in range(*w.phones):
        ip = ref.phones[i].ipa
        if ip.endswith("ː") or (len(ip) >= 2 and ip[0] == ip[1] and not is_vowel(ip)):
            n += 1
    return max(n, 1)


def _twin(ip: str) -> str | None:
    if is_vowel(ip[0]):
        return ip[:-1] if ip.endswith("ː") else ip + "ː"
    if len(ip) >= 2 and ip[0] == ip[1]:
        return ip[1:]
    if len(ip) >= 2 and ip.startswith("t") and ip[1:] in ("ttʃ", "tts"):
        return ip[1:]
    if ip in ("tʃ", "ts"):
        return "t" + ip
    return ip[0] + ip


def _best_path(lp: np.ndarray, toks: list[int], blank: int = 0) -> float:
    """Viterbi log-score of a CTC token sequence over frames lp [T, V] (blank optional between
    distinct tokens, required between repeats; the sequence may start / end with blanks)."""
    T = len(lp)
    S = 2 * len(toks) + 1
    lab = [blank if s % 2 == 0 else toks[s // 2] for s in range(S)]
    NEG = -1e30
    a = np.full(S, NEG)
    a[0] = lp[0, lab[0]]
    if S > 1:
        a[1] = lp[0, lab[1]]
    em = lp[:, lab]
    skip = np.array([s >= 2 and lab[s] != blank and lab[s] != lab[s - 2] for s in range(S)])
    for t in range(1, T):
        b = a.copy()
        b[1:] = np.maximum(b[1:], a[:-1])
        b[2:] = np.where(skip[2:], np.maximum(b[2:], a[:-2]), b[2:])
        a = b + em[t]
    return float(max(a[-1], a[-2] if S > 1 else NEG))


def _best_paths(lp: np.ndarray, jobs: list[tuple[int, int, list[int]]], blank: int = 0) -> list[float]:
    """_best_path(lp[t0:t1], toks) for every (t0, t1, toks) job (t1 - t0 >= 1, toks non-empty), all
    jobs in one recursion over frames. Each job is padded to the longest window and token sequence;
    a padded state only feeds states to its right and a job's score is read at its own last frame,
    so every number is the one _best_path computes (same operations, element by element)."""
    N = len(jobs)
    if not N:
        return []
    NEG = -1e30
    S = np.array([2 * len(tk) + 1 for _, _, tk in jobs])
    Tn = np.array([t1 - t0 for t0, t1, _ in jobs])
    Smax, Tmax = int(S.max()), int(Tn.max())
    lab = np.full((N, Smax), blank, dtype=np.int64)
    skip = np.zeros((N, Smax), dtype=bool)
    for k, (_, _, tk) in enumerate(jobs):
        lk = [blank if s % 2 == 0 else tk[s // 2] for s in range(S[k])]
        lab[k, :S[k]] = lk
        skip[k, :S[k]] = [s >= 2 and lk[s] != blank and lk[s] != lk[s - 2] for s in range(S[k])]
    t0 = np.array([j[0] for j in jobs])
    tt = np.minimum(t0[:, None] + np.arange(Tmax)[None, :], (t0 + Tn - 1)[:, None])   # [N, Tmax]
    em = lp[tt[:, :, None], lab[:, None, :]]                                          # [N, Tmax, Smax]
    a = np.full((N, Smax), NEG)
    a[:, 0] = em[:, 0, 0]
    a[:, 1] = em[:, 0, 1]
    rows = np.arange(N)
    out = np.empty(N)

    def read(sel):
        out[sel] = np.maximum(a[sel, S[sel] - 1], a[sel, S[sel] - 2])

    read(Tn == 1)
    sk = skip[:, 2:]
    for t in range(1, Tmax):
        b = a.copy()
        b[:, 1:] = np.maximum(b[:, 1:], a[:, :-1])
        b[:, 2:] = np.where(sk, np.maximum(b[:, 2:], a[:, :-2]), b[:, 2:])
        a = b + em[:, t]
        read(rows[Tn == t + 1])
    return [float(x) for x in out]


def extra(ref, inp: dict, idx: list[int], rows: list[dict], lpa=None, slot_se=None, fb=None) -> None:
    """Add EXTRA keys to the feature rows (in place). inp: pronunciada.length.frame_inputs; fb: the
    raw log-mel filterbank (10 ms); lpa / slot_se: the adapted posteriors and slot spans for cf_llr."""
    P = ref.phones
    on = inp["onset"]
    E = inp["energy"]
    nE = len(E)
    if fb is not None and len(fb):
        F = np.asarray(fb, np.float32)
        F = F - F.mean(0, keepdims=True)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-6)
    else:
        Fn = None
    loud = float(np.percentile(E, 95)) if nE else 0.0
    floor = float(np.percentile(E, 5)) if nE else 0.0
    sil = max(loud - 3.0 * _LN10, floor + 0.35 * (loud - floor))
    # speech time per word (aligned span, 10 ms frames) and morae
    nW = len(ref.words)
    wdur = np.zeros(nW)
    wmor = np.zeros(nW)
    for k, w in enumerate(ref.words):
        a, b = w.phones
        s0, s1 = int(on[a]), int(on[b]) if b < len(on) else (int(slot_se[b - 1][1]) if slot_se is not None else int(on[a]) + 4 * (b - a))
        wdur[k] = max(2 * (s1 - s0), 1)            # 20 ms output frames -> 10 ms
        wmor[k] = _morae(ref, w)
    tot_d, tot_m = wdur.sum(), wmor.sum()
    tid = None
    if lpa is not None and slot_se is not None:
        from .text.ipa import token_ids

        tid = token_ids()
        lp = np.array(lpa, np.float32)
        lp[:, 0] = np.logaddexp(lp[:, 0], lp[:, 3])        # blank = <blk> or the word separator
    # counterfactual best paths (long form vs short form) of every sound, in one batch
    cf_job: dict[int, int] = {}
    jobs = []
    if tid is not None:
        for i in idx:
            tw = _twin(P[i].ipa)
            if tw is not None and all(ch in tid for ch in tw) and all(ch in tid for ch in P[i].ipa):
                j0, j1 = max(i - 1, 0), min(i + 2, len(P))
                t0 = int(slot_se[j0][0])
                t1 = int(max(slot_se[j1 - 1][1], slot_se[i][1]))
                if t1 - t0 >= 2:
                    seq_a, seq_b = [], []
                    for j in range(j0, j1):
                        ip = P[j].ipa
                        seq_a += [tid[ch] for ch in ip]
                        seq_b += [tid[ch] for ch in (tw if j == i else ip)]
                    n_fr = len(lp[t0:t1])
                    if n_fr >= len(seq_a) and n_fr >= len(seq_b):
                        cf_job[i] = len(jobs)
                        jobs += [(t0, t0 + n_fr, seq_a), (t0, t0 + n_fr, seq_b)]
    bp = _best_paths(lp, jobs) if jobs else []
    for i, r in zip(idx, rows):
        k = P[i].word
        md = (tot_d - wdur[k]) / max(tot_m - wmor[k], 1.0) if tot_m - wmor[k] >= 3 else tot_d / max(tot_m, 1.0)
        md = max(md, 3.0)
        r["rate_loc"] = math.log(md)
        # morae with THIS sound counted short, so the ratio reads "how long" whatever the text expects
        ip = P[i].ipa
        own_long = ip.endswith("ː") or (len(ip) >= 2 and ip[0] == ip[1] and not is_vowel(ip)) or ip in ("ttʃ", "tts")
        r["word_ratio"] = math.log(wdur[k] / (max(wmor[k] - own_long, 1) * md))
        # centre of the sound (10 ms) and its neighbours' centres
        a = _fbi(on[i])
        b = _fbi(on[i + 1]) if i + 1 < len(on) else a + 8
        c = min(max((a + b) // 2, 0), max(nE - 1, 0))
        if is_vowel(P[i].ipa[0]):
            v = 0.0
            if Fn is not None and nE:
                ref_v = Fn[max(c - 1, 0):c + 2].mean(0)
                ref_v /= np.linalg.norm(ref_v) + 1e-6
                L = c
                while L - 1 >= 0 and float(Fn[L - 1] @ ref_v) >= COS and E[L - 1] >= sil:
                    L -= 1
                R = c
                while R + 1 < nE and float(Fn[R + 1] @ ref_v) >= COS and E[R + 1] >= sil:
                    R += 1
                v = math.log((R - L + 1) / md)
            r["v_stable"], r["c_dip"] = v, 0.0
        else:
            d = 0.0
            if nE and i > 0 and i + 1 < len(on):
                pa = _fbi(on[i - 1]) + (a - _fbi(on[i - 1])) // 2
                nb = b + max((_fbi(on[i + 2]) if i + 2 < len(on) else b + 8) - b, 0) // 2
                pa, nb = max(pa, 0), min(nb, nE - 1)
                if nb > pa:
                    thr = min(E[pa], E[nb]) - DIP_DB / 10.0 * _LN10
                    seg = E[pa:nb + 1] < thr
                    best = cur = 0
                    for x in seg:
                        cur = cur + 1 if x else 0
                        best = max(best, cur)
                    d = math.log((best + 1) / md)
            r["v_stable"], r["c_dip"] = 0.0, d
        cf = 0.0
        if i in cf_job:
            q = cf_job[i]
            cf = float(np.clip(bp[q] - bp[q + 1], -30.0, 30.0))
        # evidence for the LONG form (log P(long form) - log P(short form)), whatever the text expects
        r["cf_llr"] = cf if own_long else -cf
