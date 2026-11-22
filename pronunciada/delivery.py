"""Sentence "delivery" features (round 5): rhythm, pauses, pitch, loudness, recogniser
disfluency evidence and pooled encoder activations of one utterance. numpy only.

ONE implementation shared by training (bench/calib/delivery_extract.py runs the public
pronunciada.Scorer and reads these values back) and inference (pronunciada.scorer).

    D = features(wav, fb, lpa, ref, ev, track)     # {name: float}, NaN where undefined
    E = pool_embeddings(emb, ev)                   # {layer: [2C] mean ++ std over speech frames}
    D.update(project(E, proj))                     # supervised linear projection (shipped matrices)

Time grids: ZIPA output frame j covers [0.02 j + 0.03, 0.02 j + 0.05) s; fbank and the pitch
tracker use 10 ms frames centred at 0.01 i + 0.005 s, so ZIPA frame j = 10 ms frames 2j+3, 2j+4.

"Speech" = from the first sound's start to the last sound's end of the Viterbi alignment, minus
the silent stretches longer than PAUSE_MIN_S between consecutive sounds (the same definition as
pronunciada.features.pacing). CTC alignments are peaky, so timing uses onset-to-onset intervals
rather than span lengths wherever possible.
"""
from __future__ import annotations

import math

import numpy as np

from .features import FRAME_SHIFT, PAUSE_MIN_S

NAN = math.nan
T0 = 0.03                      # start time of ZIPA frame 0
HOP = 0.01                     # fbank / pitch frame hop
CLIP_LEVEL = 0.99
OCTAVE_JUMP_ST = 7.0           # |F0 change| between consecutive 10 ms voiced frames counted as a jump
FINAL_PITCH_S = 0.3

RHYTHM = ["v_int_mean", "v_int_std", "v_int_cv", "v_npvi", "w_rate_mean", "w_rate_std",
          "w_rate_min", "w_rate_max", "final_lengthening", "vowel_share"]
PAUSES = ["pauses_per_word", "pause_mean_s", "pause_max_s", "pauses_in_word", "lead_sil_s",
          "trail_sil_s", "run_length", "phonation_ratio"]
PITCH = ["voiced_share", "f0_std_st", "f0_range_st", "f0_abs_slope", "f0_declination",
         "f0_final_slope", "f0_turns_per_s", "f0_word_mean_std", "f0_word_range_mean",
         "f0_jump_rate"]
LOUDNESS = ["en_std_db", "en_range_db", "en_word_std_db", "en_speech_minus_pause_db",
            "clipped_share"]
DISFLUENCY = ["dec_ins_rate", "dec_sub_rate", "dec_del_rate", "dec_extra_in_pauses",
              "post_max_mean", "post_entropy_mean", "blank_share_speech"]
GROUPS = {"rhythm": RHYTHM, "pauses": PAUSES, "pitch": PITCH, "loudness": LOUDNESS,
          "disfluency": DISFLUENCY}
ALL = [c for g in GROUPS.values() for c in g]


# ----------------------------------------------------------------------------- helpers

def _slot_spans(ev):
    S = ev.slots if hasattr(ev, "slots") else None
    if S is not None:
        s = np.array([x.start_frame for x in S], dtype=np.int64)
        e = np.array([x.end_frame for x in S], dtype=np.int64)
    else:
        s = np.asarray(ev["start_frame"], dtype=np.int64)
        e = np.asarray(ev["end_frame"], dtype=np.int64)
    return s, e


def speech_segments(start, end) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    """ZIPA-frame speech segments [(a, b)] and pauses [(a, b, phone before, phone after)] from
    the phone spans (pauses = gaps > PAUSE_MIN_S between consecutive sounds)."""
    s = np.asarray(start, dtype=np.int64)
    e = np.asarray(end, dtype=np.int64)
    ok = np.flatnonzero(e > s)
    if not len(ok):
        return [], []
    o = ok[np.argsort(s[ok], kind="stable")]
    segs, pauses = [], []
    a, reach, last = int(s[o[0]]), int(e[o[0]]), int(o[0])
    for i in o[1:]:
        if (s[i] - reach) * FRAME_SHIFT > PAUSE_MIN_S:
            segs.append((a, reach))
            pauses.append((reach, int(s[i]), last, int(i)))
            a = int(s[i])
        if e[i] >= reach:
            reach, last = int(e[i]), int(i)
    segs.append((a, reach))
    return segs, pauses


def speech_mask(n_frames: int, segs) -> np.ndarray:
    m = np.zeros(int(n_frames), dtype=bool)
    for a, b in segs:
        m[max(a, 0):min(b, n_frames)] = True
    return m


def _to10(n10: int, segs) -> np.ndarray:
    """ZIPA-frame segments -> boolean mask over 10 ms frames."""
    m = np.zeros(int(n10), dtype=bool)
    for a, b in segs:
        m[max(2 * a + 3, 0):max(min(2 * b + 3, n10), 0)] = True
    return m


def _std(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.std(x)) if len(x) >= 2 else NAN


def _slope(t, y):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(t) < 3 or np.ptp(t) <= 0:
        return NAN
    tc = t - t.mean()
    return float((tc * (y - y.mean())).sum() / (tc * tc).sum())


def edit_ops(ref, hyp) -> tuple[int, int, int]:
    """Levenshtein alignment of two id sequences -> (insertions, substitutions, deletions)
    (one optimal path; ties prefer substitution, then deletion)."""
    ref = np.asarray(ref, dtype=np.int64)
    hyp = np.asarray(hyp, dtype=np.int64)
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return m, 0, n
    D = np.zeros((n + 1, m + 1), dtype=np.int64)
    D[0] = np.arange(m + 1)
    idx = np.arange(m + 1)
    for i in range(1, n + 1):
        sub = D[i - 1, :-1] + (hyp != ref[i - 1])
        row = np.empty(m + 1, dtype=np.int64)
        row[0] = i
        row[1:] = np.minimum(D[i - 1, 1:] + 1, sub)
        D[i] = np.minimum.accumulate(row - idx) + idx           # insertions along the row
    i, j = n, m
    ins = subs = dels = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (hyp[j - 1] != ref[i - 1]):
            subs += int(hyp[j - 1] != ref[i - 1])
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return ins, subs, dels


# ----------------------------------------------------------------------------- features

def features(wav, fb, lpa, ref, ev, track=None, pitch: bool = True) -> dict:
    """Flat {name: float} of the RHYTHM, PAUSES, PITCH, LOUDNESS and DISFLUENCY groups.
    wav: float32 16 kHz; fb: [T10, 80] log-mel fbank; lpa: adapted log-posteriors [T, 127];
    ref: pronunciada.types.Reference; ev: pronunciada.align.Evidence of ref on lpa;
    track: pronunciada.prosody.f0.F0Track of wav (computed when None and pitch is True; with
    pitch=False and no track the PITCH group stays NaN)."""
    out = {k: NAN for k in ALL}
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    dur = len(wav) / 16000.0
    lpa = np.asarray(lpa)
    T = len(lpa)
    s, e = _slot_spans(ev)
    n = len(ref.phones)
    ok = e > s
    out["clipped_share"] = float(np.mean(np.abs(wav) >= CLIP_LEVEL)) if len(wav) else NAN
    segs, pauses = speech_segments(s, e)
    if not segs or n == 0 or dur <= 0:
        return out
    word_of = np.array([p.word for p in ref.phones], dtype=np.int64)
    nucleus = np.array([bool(p.is_nucleus) for p in ref.phones])
    nw = len(ref.words)
    first, last = segs[0][0], segs[-1][1]
    speech_s = sum(b - a for a, b in segs) * FRAME_SHIFT
    span_s = (last - first) * FRAME_SHIFT

    # ---------------------------------------------------------------- rhythm
    on = np.sort(s[ok & nucleus]) * FRAME_SHIFT
    if len(on) >= 3:
        d = np.diff(on)
        d = d[d > 0]
        if len(d) >= 2:
            out["v_int_mean"] = float(d.mean())
            out["v_int_std"] = float(d.std())
            out["v_int_cv"] = float(d.std() / max(d.mean(), 1e-6))
            out["v_npvi"] = float(100.0 * np.mean(np.abs(np.diff(d)) / ((d[1:] + d[:-1]) / 2)))
    w_on = np.full(nw, np.nan)
    w_end = np.full(nw, np.nan)
    w_np = np.zeros(nw)
    for k, w in enumerate(ref.words):
        a, b = w.phones
        m = ok[a:b]
        w_np[k] = b - a
        if m.any():
            w_on[k] = s[a:b][m].min()
            w_end[k] = e[a:b][m].max()
    have = np.flatnonzero(np.isfinite(w_on))
    if len(have):
        # word duration = onset-to-next-onset, but never across a pause (then: own span end)
        wd = np.empty(len(have))
        for q, k in enumerate(have):
            nxt = w_on[have[q + 1]] if q + 1 < len(have) else np.nan
            own = w_end[k]
            gap_pause = np.isfinite(nxt) and (nxt - own) * FRAME_SHIFT > PAUSE_MIN_S
            wd[q] = ((own if (not np.isfinite(nxt) or gap_pause) else nxt) - w_on[k]) * FRAME_SHIFT
        wd = np.maximum(wd, FRAME_SHIFT)
        rate = w_np[have] / wd
        out["w_rate_mean"] = float(rate.mean())
        out["w_rate_std"] = _std(rate)
        out["w_rate_min"], out["w_rate_max"] = float(rate.min()), float(rate.max())
        if len(have) >= 2:
            per = wd / np.maximum(w_np[have], 1)
            out["final_lengthening"] = float(per[-1] / max(np.median(per[:-1]), 1e-6))
    tot = (e - s)[ok].sum()
    out["vowel_share"] = float((e - s)[ok & nucleus].sum() / tot) if tot > 0 else NAN

    # ---------------------------------------------------------------- pauses
    pl = np.array([(b - a) * FRAME_SHIFT for a, b, _, _ in pauses])
    out["pauses_per_word"] = len(pauses) / max(nw, 1)
    out["pause_mean_s"] = float(pl.mean()) if len(pl) else 0.0
    out["pause_max_s"] = float(pl.max()) if len(pl) else 0.0
    out["pauses_in_word"] = float(sum(word_of[i] == word_of[j] for _, _, i, j in pauses))
    out["lead_sil_s"] = float(T0 + first * FRAME_SHIFT)
    out["trail_sil_s"] = float(max(dur - (T0 + last * FRAME_SHIFT), 0.0))
    between = len(pauses) - out["pauses_in_word"]
    out["run_length"] = float(nw / (between + 1))
    out["phonation_ratio"] = float(speech_s / span_s) if span_s > 0 else NAN

    # ---------------------------------------------------------------- pitch
    if track is None and pitch:
        from .prosody.f0 import track as _track

        track = _track(wav)
    n10 = len(track.f0) if track is not None else 0
    sp10 = _to10(n10, segs)
    vo = (sp10 & track.voiced & (track.f0 > 0)) if track is not None else sp10
    if track is not None:
        out["voiced_share"] = float(vo.sum() / max(sp10.sum(), 1))
    if track is not None and vo.sum() >= 5:
        f0 = np.asarray(track.f0, dtype=np.float64)
        st = np.full(n10, np.nan)
        st[vo] = 12.0 * np.log2(f0[vo] / np.median(f0[vo]))
        tt = np.arange(n10) * HOP + HOP / 2
        sv = st[vo]
        out["f0_std_st"] = float(sv.std())
        p5, p95 = np.percentile(sv, [5, 95])
        out["f0_range_st"] = float(p95 - p5)
        both = vo[1:] & vo[:-1]
        dst = np.diff(st)[both]
        jump = np.abs(dst) > OCTAVE_JUMP_ST
        smooth = dst[~jump]
        out["f0_abs_slope"] = float(np.mean(np.abs(smooth)) / HOP) if len(smooth) else NAN
        out["f0_jump_rate"] = float(jump.sum() / (vo.sum() * HOP))
        out["f0_declination"] = _slope(tt[vo], sv)
        tv = tt[vo]
        fin = tv >= tv[-1] - FINAL_PITCH_S
        out["f0_final_slope"] = _slope(tv[fin], sv[fin])
        # turning points: sign changes of the slope of the 5-frame smoothed contour inside
        # voiced runs of >= 5 frames
        turns = 0
        idx = np.flatnonzero(vo)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for r in runs:
            if len(r) < 7:
                continue
            c = np.convolve(st[r], np.ones(5) / 5, mode="valid")
            dd = np.diff(c)
            dd = dd[np.abs(dd) > 0.05]
            turns += int((np.sign(dd[1:]) != np.sign(dd[:-1])).sum())
        out["f0_turns_per_s"] = float(turns / (vo.sum() * HOP))
        wm, wr = [], []
        for k in have:
            a10, b10 = int(2 * w_on[k] + 3), int(2 * w_end[k] + 3)
            v = st[a10:b10][vo[a10:b10]]
            if len(v) >= 3:
                wm.append(v.mean())
                wr.append(v.max() - v.min())
        out["f0_word_mean_std"] = _std(wm)
        out["f0_word_range_mean"] = float(np.mean(wr)) if wr else NAN

    # ---------------------------------------------------------------- loudness
    fb = np.asarray(fb, dtype=np.float64)
    if len(fb):
        mx = fb.max(1)
        en = (10.0 / np.log(10.0)) * (mx + np.log(np.exp(fb - mx[:, None]).sum(1)))   # dB
        sp = _to10(len(en), segs)
        if sp.sum() >= 5:
            es = en[sp]
            out["en_std_db"] = float(es.std())
            p5, p95 = np.percentile(es, [5, 95])
            out["en_range_db"] = float(p95 - p5)
            if (~sp).sum() >= 5:
                out["en_speech_minus_pause_db"] = float(np.median(es) - np.median(en[~sp]))
            wm = []
            for k in have:
                a10, b10 = int(2 * w_on[k] + 3), int(2 * w_end[k] + 3)
                if b10 > a10:
                    wm.append(en[a10:b10].mean())
            out["en_word_std_db"] = _std(wm)

    # ---------------------------------------------------------------- disfluency
    from .text.ipa import to_ids

    ids = lpa.argmax(-1)
    change = np.r_[True, ids[1:] != ids[:-1]] if T else np.zeros(0, bool)
    tok_start = np.flatnonzero(change)
    tok_id = ids[tok_start]
    keep = tok_id >= 4
    hyp, hyp_t = tok_id[keep], tok_start[keep]
    refids = [t for p in ref.phones for t in to_ids(p.ipa)]
    i_, s_, d_ = edit_ops(refids, hyp)
    nr = max(len(refids), 1)
    out["dec_ins_rate"], out["dec_sub_rate"], out["dec_del_rate"] = i_ / nr, s_ / nr, d_ / nr
    msk = speech_mask(T, segs)
    extra = int((~msk[np.clip(hyp_t, 0, max(T - 1, 0))]).sum()) if len(hyp_t) else 0
    out["dec_extra_in_pauses"] = extra / max(nw, 1)
    if msk.any():
        L = lpa[msk].astype(np.float64)
        P = np.exp(L)
        out["post_max_mean"] = float(P.max(1).mean())
        out["post_entropy_mean"] = float(-(P * np.where(P > 0, L, 0.0)).sum(1).mean())
        out["blank_share_speech"] = float(np.isin(ids[msk], (0, 3)).mean())
    return out


# ----------------------------------------------------------------------------- embeddings

def pool_embeddings(emb: dict | None, ev) -> dict | None:
    """{layer: [T, C]} -> {layer: float32 [2C] = mean ++ std over the speech frames} (all
    frames when the alignment has no speech)."""
    if not emb:
        return None
    s, e = _slot_spans(ev)
    segs, _ = speech_segments(s, e)
    out = {}
    for k, v in emb.items():
        v = np.asarray(v, dtype=np.float32)
        m = speech_mask(len(v), segs)
        x = v[m] if m.sum() >= 2 else v
        out[k] = np.concatenate([x.mean(0), x.std(0)]).astype(np.float32)
    return out


def pooled_part(v, key: str) -> np.ndarray:
    """Part of a pooled vector (mean ++ std) named by a projection layer key: "enc1" = all,
    "enc1.mean" = the mean half, "enc1.std" = the std half (round 19)."""
    v = np.asarray(v)
    if key.endswith(".mean"):
        return v[:len(v) // 2]
    if key.endswith(".std"):
        return v[len(v) // 2:]
    return v


def project(pooled: dict | None, proj: dict | None) -> dict:
    """Supervised linear projection of the pooled activations (fitted offline by
    bench/calib/train_delivery.py): z = ((concat(pooled[layers]) - mu) / sd) @ W + b, one output
    column per name in proj["names"]. NaN when the model gave no embeddings."""
    if not proj:
        return {}
    names = list(proj["names"])
    if pooled is None or any(k.split(".")[0] not in pooled for k in proj["layers"]):
        return {k: NAN for k in names}
    x = np.concatenate([pooled_part(pooled[k.split(".")[0]], k) for k in proj["layers"]]).astype(np.float64)
    z = ((x - proj["mu"]) / proj["sd"]) @ proj["W"] + proj["b"]
    return {k: float(v) for k, v in zip(names, z)}


def word_frame_spans(ref, ev, n_frames: int) -> list[tuple[int, int]]:
    """Per reference word, the ZIPA-frame span [a, b) from its first sound's start to its last
    sound's end; a word whose sounds all have empty spans gets the 2 frames around its first
    sound's position (round 6, word-level pooling)."""
    s, e = _slot_spans(ev)
    out = []
    T = max(int(n_frames), 1)
    for w in ref.words:
        a, b = w.phones
        ok = e[a:b] > s[a:b]
        if ok.any():
            lo, hi = int(s[a:b][ok].min()), int(e[a:b][ok].max())
        else:
            c = int(s[a]) if b > a else 0
            lo, hi = c - 1, c + 1
        lo = min(max(lo, 0), T - 1)
        hi = min(max(hi, lo + 1), T)
        out.append((lo, hi))
    return out


def pool_words(emb: np.ndarray, spans) -> np.ndarray:
    """[T, C] activations -> [n_words, 2C] float32 mean ++ std over each word's frame span."""
    emb = np.asarray(emb, dtype=np.float32)
    out = np.empty((len(spans), 2 * emb.shape[1]), dtype=np.float32)
    for k, (a, b) in enumerate(spans):
        x = emb[a:b]
        out[k, :emb.shape[1]] = x.mean(0)
        out[k, emb.shape[1]:] = x.std(0)
    return out
