"""Calibrator features: ONE implementation shared by training (bench/pipeline/features.py,
bench/calib/*) and inference (pronunciada.scorer). numpy only.

Three layers, all column-oriented (dict name -> 1-D array, one entry per reference phone):

1. ``phone_table(lp, ref, ev, ...)``: the BASE columns of the feature tables
   (~/.cache/pronunciada/features/*.parquet): evidence from pronunciada.align (llr_sub, llr_del, span,
   peak posterior ...), span-posterior statistics on the ADAPTED posteriors (canonical token
   mean/max, best competitor and its probs, log-ratio, entropy, blank prob, articulatory distance,
   the same for the phone's last token), native z-scores (pronunciada.variants.NativeStats),
   neighbour / word / utterance llr statistics, class / position / nucleus / stress.
2. ``derive(t, group)``: DERIVED columns computed from base columns only (so feature tables built
   before a derived column existed need no rebuild): flags, more neighbour / word / utterance
   statistics. `group` = one id per row identifying the scored reference (utterance x reference
   variant); rows of a group must be in phone order.
3. ``matrix(t, spec)``: numeric matrix for a calibrator: numeric columns as float (NaN kept),
   categorical columns as integer codes of the model's vocabulary (unknown / missing -> NaN).

Word and sentence inputs (``word_table``, ``sentence_table``) aggregate the calibrated sound
log-odds and add word evidence / omission / pacing statistics.
"""
from __future__ import annotations

import math
from functools import lru_cache
from importlib import resources

import numpy as np

from .text.ipa import MODIFIERS, load_tokens, to_ids
from .variants import broad_class, phone_position

FEATURE_VERSION = 3          # 3: start_frame / end_frame columns; features moved to pronunciada
FRAME_SHIFT = 0.02
PAUSE_MIN_S = 0.2            # silent stretch inside the utterance counted as a pause
OMIT_LLR_DEL = -2.0          # a word is "omitted" when EVERY sound has llr_del below this
EPS = 1e-8

UTT_COLS = ("utt_mean_llr_sub", "utt_median_llr_sub", "utt_dur", "phones_per_s", "utt_n_phones")
STR_COLS = ("phone", "cls", "position", "comp_tok", "comp_cls", "last_comp_tok", "heard",
            "word_text")


# ----------------------------------------------------------------------------- articulatory distance

@lru_cache(maxsize=1)
def _dist_table():
    p = resources.files("pronunciada").joinpath("data/calib/phone_dist.npz")
    with p.open("rb") as f:
        z = np.load(f, allow_pickle=False)
        toks = [str(t) for t in z["tokens"]]
        D = z["dist"].astype(np.float64)
    if tuple(toks) != load_tokens():
        raise RuntimeError("phone_dist.npz was built for a different token list")
    return D


def token_distance(a: int, b: int) -> float:
    """Articulatory (panphon weighted-feature) distance between two ZIPA token ids; NaN when
    panphon cannot parse one of them (modifier tokens)."""
    if a == b:
        return 0.0
    return float(_dist_table()[a, b])


# ----------------------------------------------------------------------------- layer 1: base table

def evidence_columns(ev) -> dict:
    """pronunciada.align.Evidence -> per-slot arrays."""
    S = ev.slots
    return {"llr_sub": np.array([s.llr_sub for s in S], dtype=np.float64),
            "llr_del": np.array([s.llr_del for s in S], dtype=np.float64),
            "n_frames": np.array([s.n_frames for s in S], dtype=np.int64),
            "peak_post": np.array([s.peak_post for s in S], dtype=np.float64),
            "alt_index": np.array([s.alt_index for s in S], dtype=np.int64),
            "start_frame": np.array([s.start_frame for s in S], dtype=np.int64),
            "end_frame": np.array([s.end_frame for s in S], dtype=np.int64),
            "heard": [s.heard for s in S]}


def span_posterior_features(lp: np.ndarray, ref, start, end) -> list[dict]:
    """Per slot, over the Viterbi span [start, end) padded by one frame each side (a zero-length
    span -> the 2 frames around its position), on the ADAPTED posteriors:
    can_* = canonical phone's first token (mean / max prob), comp_* = best competing non-blank
    token by mean prob (excluding the phone's own tokens; may be a modifier such as ʰ, then
    comp_cls = "modifier"), log_ratio_comp = log(can_mean/comp_mean), ent_nonblank = entropy
    (nats) of the mean non-blank distribution renormalised, blank_mean = mean P(<blk>)+P(▁),
    comp_dist = articulatory distance canonical token vs competitor, comp_cls; last_* the same
    for the phone's LAST token when it has >= 2 tokens (NaN otherwise)."""
    toks = load_tokens()
    T = len(lp)
    P = np.exp(np.asarray(lp).astype(np.float64))
    nb = np.arange(4, P.shape[1]) if P.ndim == 2 else np.arange(4, 127)
    out = []
    for i, p in enumerate(ref.phones):
        s, e = int(start[i]), int(end[i])
        if e <= s:
            a, b = max(s - 1, 0), min(s + 1, T)
        else:
            a, b = max(s - 1, 0), min(e + 1, T)
        if b <= a:
            a, b = max(T - 1, 0), T
        if T == 0:
            out.append(_empty_span())
            continue
        W = P[a:b]
        m = W.mean(0)
        mnb = m[nb]
        q = mnb / max(mnb.sum(), EPS)
        d = {"span_frames_padded": int(b - a), "blank_mean": float(m[0] + m[3]),
             "ent_nonblank": float(-(q * np.log(q + 1e-12)).sum())}
        ids = to_ids(p.ipa)
        for pref, tid in (("", ids[0]), ("last_", ids[-1] if len(ids) >= 2 else None)):
            if tid is None:
                d.update({f"{pref}can_mean": math.nan, f"{pref}can_max": math.nan,
                          f"{pref}comp_tok": None, f"{pref}comp_mean": math.nan,
                          f"{pref}comp_max": math.nan, f"{pref}log_ratio_comp": math.nan})
                if pref == "":
                    d.update({"comp_dist": math.nan, "comp_cls": None})
                continue
            mm = mnb.copy()
            # never the phone's own tokens (nb = 4, 5, ...: token k sits at position k - 4)
            mm[[k - 4 for k in ids if 4 <= k < 4 + len(nb)]] = -1.0
            c = int(nb[int(np.argmax(mm))])
            d.update({f"{pref}can_mean": float(m[tid]), f"{pref}can_max": float(W[:, tid].max()),
                      f"{pref}comp_tok": toks[c], f"{pref}comp_mean": float(m[c]),
                      f"{pref}comp_max": float(W[:, c].max()),
                      f"{pref}log_ratio_comp": float(np.log(m[tid] + EPS) - np.log(m[c] + EPS))})
            if pref == "":
                d.update({"comp_dist": token_distance(tid, c),
                          "comp_cls": "modifier" if toks[c] in MODIFIERS else broad_class(toks[c])})
        out.append(d)
    return out


def _empty_span() -> dict:
    d = {"span_frames_padded": 0, "blank_mean": math.nan, "ent_nonblank": math.nan,
         "comp_dist": math.nan, "comp_cls": None}
    for pref in ("", "last_"):
        d.update({f"{pref}can_mean": math.nan, f"{pref}can_max": math.nan, f"{pref}comp_tok": None,
                  f"{pref}comp_mean": math.nan, f"{pref}comp_max": math.nan,
                  f"{pref}log_ratio_comp": math.nan})
    return d


def phone_rows(lp: np.ndarray, ref, ev, ns=None, dur: float | None = None) -> list[dict]:
    """One dict per reference phone with every BASE column (the per-row content of the feature
    tables, without dataset / label columns). `lp` = ADAPTED log-posteriors, `ev` = the
    pronunciada.align.Evidence of `ref` on `lp` (or a dict from evidence_columns), `ns` =
    NativeStats of the language (None -> NaN z-scores), `dur` = audio duration in seconds
    (default: frames x 20 ms)."""
    E = ev if isinstance(ev, dict) else evidence_columns(ev)
    s = np.asarray(E["llr_sub"], dtype=np.float64)
    n = len(ref.phones)
    if dur is None:
        dur = len(lp) * FRAME_SHIFT
    dur = float(dur)
    utt = {"utt_mean_llr_sub": float(np.mean(s)) if n else math.nan,
           "utt_median_llr_sub": float(np.median(s)) if n else math.nan, "utt_dur": dur,
           "phones_per_s": n / dur if dur > 0 else math.nan, "utt_n_phones": n}
    post = span_posterior_features(lp, ref, E["start_frame"], E["end_frame"]) if n else []
    wmean = {w: float(np.mean(s[slice(*ref.words[w].phones)])) for w in range(len(ref.words))}
    wmin = {w: float(np.min(s[slice(*ref.words[w].phones)])) for w in range(len(ref.words))}
    rows = []
    for i, p in enumerate(ref.phones):
        pos = phone_position(ref, i)
        nf = int(E["n_frames"][i])
        r = dict(utt)
        r.update({"word_index": p.word, "syllable_index": p.syllable, "phone_index": i,
                  "phone": p.ipa, "cls": broad_class(p.ipa), "position": pos,
                  "is_nucleus": bool(p.is_nucleus), "stress": ref.syllables[p.syllable].stress,
                  "word_text": ref.words[p.word].text,
                  "llr_sub": float(s[i]), "llr_del": float(E["llr_del"][i]),
                  "n_frames": nf, "peak_post": float(E["peak_post"][i]),
                  "alt_index": int(E["alt_index"][i]), "heard": E["heard"][i],
                  "start_frame": int(E["start_frame"][i]), "end_frame": int(E["end_frame"][i]),
                  "llr_sub_per_frame": float(s[i]) / max(nf, 1)})
        for f in ("llr_sub", "llr_del", "n_frames"):
            r[f"z_{f}"] = ns.z(p.ipa, pos, f, r[f]) if ns is not None else math.nan
        r.update(post[i])
        r.update({"prev_llr_sub": float(s[i - 1]) if i > 0 else math.nan,
                  "next_llr_sub": float(s[i + 1]) if i + 1 < n else math.nan,
                  "word_mean_llr_sub": wmean[p.word], "word_min_llr_sub": wmin[p.word]})
        rows.append(r)
    return rows


def to_columns(rows: list[dict], keys=None) -> dict:
    keys = keys or (list(rows[0]) if rows else [])
    out = {}
    for k in keys:
        v = [r.get(k) for r in rows]
        out[k] = v if k in STR_COLS else np.asarray(
            [np.nan if x is None else x for x in v], dtype=np.float64)
    return out


def phone_table(lp: np.ndarray, ref, ev, ns=None, dur: float | None = None) -> dict:
    """Column form of phone_rows plus derive() for one scored reference."""
    t = to_columns(phone_rows(lp, ref, ev, ns, dur))
    if not ref.phones:
        return t
    return derive(t)


# ----------------------------------------------------------------------------- layer 2: derived

def _codes(group) -> tuple[np.ndarray, int]:
    """Run codes 0, 1, 2 ... in order of appearance; every group must be one contiguous run."""
    g = np.asarray(group)
    if len(g) == 0:
        return np.zeros(0, dtype=np.int64), 0
    ch = np.r_[False, g[1:] != g[:-1]]
    c = np.cumsum(ch).astype(np.int64)
    return c, int(c[-1]) + 1


def _gmean(x, g, n):
    ok = np.isfinite(x)
    s = np.bincount(g, weights=np.where(ok, x, 0.0), minlength=n)
    c = np.bincount(g, weights=ok.astype(float), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _gmin(x, g, n):
    out = np.full(n, np.inf)
    np.minimum.at(out, g, np.where(np.isfinite(x), x, np.inf))
    return np.where(np.isfinite(out), out, np.nan)


def _gmin2(x, g, n):
    """Per-group second-smallest value (the smallest for one-element groups)."""
    x = np.where(np.isfinite(x), x, np.inf)
    m1 = np.full(n, np.inf)
    np.minimum.at(m1, g, x)
    # drop one occurrence of the minimum per group
    order = np.lexsort((x, g))
    first = np.r_[True, g[order][1:] != g[order][:-1]]
    keep = order[~first]
    m2 = np.full(n, np.inf)
    np.minimum.at(m2, g[keep], x[keep])
    m2 = np.where(np.isfinite(m2), m2, m1)
    return np.where(np.isfinite(m2), m2, np.nan)


def _gsum(x, g, n):
    return np.bincount(g, weights=x, minlength=n)


def _gquantile(x, g, n, q):
    """Per-group quantile (linear interpolation, as np.quantile)."""
    order = np.lexsort((x, g))
    xs, gs = x[order], g[order]
    starts = np.searchsorted(gs, np.arange(n))
    ends = np.searchsorted(gs, np.arange(n), side="right")
    out = np.full(n, np.nan)
    for k in range(n):
        a, b = starts[k], ends[k]
        if b > a:
            out[k] = np.quantile(xs[a:b], q)
    return out


def derive(t: dict, group=None) -> dict:
    """Add derived columns in place (and return t). `group`: one id per row identifying the
    scored reference (default: all rows one reference); rows of a group are in phone order."""
    n = len(t["llr_sub"])
    g, ng = _codes(np.zeros(n, dtype=np.int64) if group is None else group)
    f = lambda k: np.asarray(t[k], dtype=np.float64)  # noqa: E731
    sub, dl, nf = f("llr_sub"), f("llr_del"), f("n_frames")
    t["alt_nonzero"] = (f("alt_index") != 0).astype(float)
    t["llr_sub_rel"] = sub - f("utt_median_llr_sub")
    t["is_nucleus_f"] = f("is_nucleus")
    t["stress_f"] = f("stress")                       # NaN where the language has no stress
    t["has_last"] = np.isfinite(f("last_can_mean")).astype(float)
    # neighbours (llr_del), inside the same reference only
    first = np.r_[True, g[1:] != g[:-1]] if n else np.zeros(0, bool)
    last = np.r_[g[1:] != g[:-1], True] if n else np.zeros(0, bool)
    t["prev_llr_del"] = np.where(first, np.nan, np.r_[np.nan, dl[:-1]]) if n else dl
    t["next_llr_del"] = np.where(last, np.nan, np.r_[dl[1:], np.nan]) if n else dl
    # word statistics
    wkey = g * 100000 + f("word_index").astype(np.int64)
    wg, nw = _codes(wkey)
    t["word_mean_llr_del"] = _gmean(dl, wg, nw)[wg]
    t["word_min_llr_del"] = _gmin(dl, wg, nw)[wg]
    t["word_n_phones"] = np.bincount(wg, minlength=nw)[wg].astype(float)
    t["word_share_del"] = _gmean((dl < 0).astype(float), wg, nw)[wg]
    t["word_frames"] = _gsum(nf, wg, nw)[wg]
    t["llr_sub_rel_word"] = sub - f("word_mean_llr_sub")
    # utterance (reference) statistics
    t["utt_mean_llr_del"] = _gmean(dl, g, ng)[g]
    t["utt_p10_llr_sub"] = _gquantile(sub, g, ng, 0.10)[g]
    t["utt_share_del"] = _gmean((dl < 0).astype(float), g, ng)[g]
    zs = f("z_llr_sub")
    t["utt_mean_z_llr_sub"] = _gmean(zs, g, ng)[g]
    t["z_llr_sub_rel"] = zs - t["utt_mean_z_llr_sub"]
    t["utt_mean_peak_post"] = _gmean(f("peak_post"), g, ng)[g]
    t["utt_mean_log_ratio_comp"] = _gmean(f("log_ratio_comp"), g, ng)[g]
    return t


# ----------------------------------------------------------------------------- feature sets

EVIDENCE_NUM = [
    "llr_sub", "llr_del", "n_frames", "peak_post", "alt_nonzero", "z_llr_sub", "z_llr_del",
    "z_n_frames", "llr_sub_per_frame", "llr_sub_rel", "is_nucleus_f", "stress_f",
    "can_mean", "can_max", "comp_mean", "comp_max", "log_ratio_comp", "ent_nonblank",
    "blank_mean", "comp_dist", "span_frames_padded", "has_last", "last_can_mean",
    "last_can_max", "last_comp_mean", "last_comp_max", "last_log_ratio_comp",
    "prev_llr_sub", "next_llr_sub", "prev_llr_del", "next_llr_del",
    "word_mean_llr_sub", "word_min_llr_sub", "word_mean_llr_del", "word_min_llr_del",
    "word_n_phones", "word_share_del", "word_frames", "llr_sub_rel_word",
    "utt_mean_llr_sub", "utt_median_llr_sub", "utt_mean_llr_del", "utt_p10_llr_sub",
    "utt_share_del", "utt_mean_z_llr_sub", "z_llr_sub_rel", "utt_mean_peak_post",
    "utt_mean_log_ratio_comp"]
EVIDENCE_CAT = ["cls", "position", "comp_cls"]
IDENTITY_CAT = ["phone", "comp_tok"]
RATE_NUM = ["utt_dur", "phones_per_s", "utt_n_phones"]

SOUND_SETS = {
    "A": {"num": EVIDENCE_NUM, "cat": EVIDENCE_CAT},
    "B": {"num": EVIDENCE_NUM, "cat": EVIDENCE_CAT + IDENTITY_CAT},
    "C": {"num": EVIDENCE_NUM + RATE_NUM, "cat": EVIDENCE_CAT + IDENTITY_CAT},
}


def _onehot(c: str, encoding: str) -> bool:
    return encoding == "onehot" or (encoding == "mixed" and c not in IDENTITY_CAT)


def matrix(t: dict, num: list[str], cat: list[str], vocab: dict[str, list[str]],
           encoding: str = "mixed") -> np.ndarray:
    """Calibrator input matrix: numeric columns (NaN kept), then the categorical columns.
    A one-hot column set has one 0/1 column per vocabulary entry (unknown value -> all zeros;
    missing -> the "none" entry); a code column holds integer codes into vocab[col] (unknown /
    missing -> NaN) for the trees' native categorical splits.
    encoding: "onehot" (all one-hot), "codes" (all codes), "mixed" (identity columns phone /
    comp_tok as codes, the small class / position columns one-hot). The shipped models use
    "mixed": one-hot keeps the walker free of categorical splits (2x faster) at equal CV
    accuracy for the small columns, while phone identity loses accuracy when one-hot encoded."""
    n = len(t[num[0]]) if num else len(t[cat[0]])
    cols = matrix_columns(num, cat, vocab, encoding)
    X = np.zeros((n, len(cols)))
    for j, c in enumerate(num):
        X[:, j] = np.asarray(t[c], dtype=np.float64) if c in t else np.nan
    j = len(num)
    for c in cat:
        idx = {v: i for i, v in enumerate(vocab[c])}
        col = t.get(c, [None] * n)
        codes = np.array([idx.get("none" if v is None else str(v), -1) for v in col], dtype=np.int64)
        if _onehot(c, encoding):
            ok = codes >= 0
            X[np.flatnonzero(ok), j + codes[ok]] = 1.0
            j += len(vocab[c])
        else:
            X[:, j] = np.where(codes >= 0, codes, np.nan)
            j += 1
    return X


def matrix_columns(num: list[str], cat: list[str], vocab: dict, encoding: str = "mixed") -> list[str]:
    out = list(num)
    for c in cat:
        out += [f"{c}={v}" for v in vocab[c]] if _onehot(c, encoding) else [c]
    return out


def categorical_mask(num: list[str], cat: list[str], vocab: dict, encoding: str = "mixed") -> list[bool]:
    """True for code columns (native categorical splits) of matrix()."""
    return [("=" not in c) and c in cat for c in matrix_columns(num, cat, vocab, encoding)]


# ----------------------------------------------------------------------------- word / sentence

WORD_NUM = [
    "wl_mean", "wl_min", "ws_mean", "ws_min", "wl_share_below", "ws_share_below", "wr_mean",
    "wr_min", "n_phones", "word_mean_llr_sub", "word_min_llr_sub", "word_mean_llr_del",
    "word_min_llr_del", "word_share_del", "word_frames", "omitted_f", "utt_ws_mean", "utt_wl_min"]
# stress_prob / tone_prob are computed by word_table (NaN when absent) but are not calibrator
# inputs: the human word ratings available for training (speechocean762) come without them; the
# scorer reports them next to the sounds-based word probability (pronunciada.scorer).

NEIGHBOUR_STATS = ["ws_min", "ws_min2", "wl_min", "ws_mean", "ws_share_below", "word_min_llr_sub",
                   "word_sum_llr_sub", "word_mean_llr_del"]
# round-2 word calibrator inputs ("would a rater accept every sound of this word?")
WORD2_NUM = (["wl_min", "wl_min2", "ws_min", "ws_min2", "wl_mean", "ws_mean", "wl_share_below",
              "ws_share_below", "n_phones", "word_sum_llr_sub", "word_min_llr_sub",
              "word_mean_llr_sub", "word_mean_llr_del", "word_min_llr_del", "word_share_del",
              "word_frames", "omitted_f", "wr_mean", "wr_min", "utt_ws_mean", "utt_wl_min",
              "word_pos_first", "word_pos_last"]
             + [f"{p}_{c}" for c in NEIGHBOUR_STATS for p in ("prev", "next", "dprev", "dnext")])

SENT_NUM = [
    "w_len_mean", "w_len_min", "w_str_mean", "w_str_min", "w_len_share_below",
    "w_str_share_below", "w_reg_mean", "w_reg_min", "s_len_mean", "s_len_min", "s_str_mean",
    "s_str_p10", "s_len_share_below", "s_str_share_below", "s_reg_mean", "completeness",
    "n_words", "articulation_rate", "n_pauses", "pause_total_s", "speech_share",
    "mean_llr_sub", "mean_z_llr_sub", "share_del"]
SENT2_NUM = SENT_NUM + ["w_len_min2", "w_str_min2"]      # round 2


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def word_table(t: dict, sl: np.ndarray, ss: np.ndarray, sr: np.ndarray, group=None,
               prosody: dict | None = None) -> dict:
    """Per (reference, word) inputs of the word calibrator.
    t: derived phone table; sl / ss / sr: per phone lenient logit, strict logit, regression
    (0-2); prosody: optional {"stress_prob": [per word], "tone_prob": [per word]} (NaN = absent).
    Returns columns incl. "group", "word_index" and "row" (index of the word's first phone row),
    one row per word, in order."""
    n = len(sl)
    g, ng = _codes(np.zeros(n, dtype=np.int64) if group is None else group)
    wi = np.asarray(t["word_index"], dtype=np.int64)
    wg, nw = _codes(g * 100000 + wi)
    first = np.r_[0, np.flatnonzero(np.diff(wg)) + 1] if n else np.zeros(0, dtype=np.int64)
    f = lambda k: np.asarray(t[k], dtype=np.float64)  # noqa: E731
    dl = f("llr_del")
    w = {"group": g[first], "word_index": wi[first], "row": first}
    w["wl_mean"], w["wl_min"] = _gmean(sl, wg, nw), _gmin(sl, wg, nw)
    w["ws_mean"], w["ws_min"] = _gmean(ss, wg, nw), _gmin(ss, wg, nw)
    w["wl_share_below"] = _gmean((sl < 0).astype(float), wg, nw)
    w["ws_share_below"] = _gmean((ss < 0).astype(float), wg, nw)
    w["wr_mean"], w["wr_min"] = _gmean(sr, wg, nw), _gmin(sr, wg, nw)
    w["n_phones"] = np.bincount(wg, minlength=nw).astype(float)
    for c in ("word_mean_llr_sub", "word_min_llr_sub", "word_mean_llr_del", "word_min_llr_del",
              "word_share_del", "word_frames"):
        w[c] = f(c)[first]
    # every sound looks dropped (by a margin: at llr_del < 0 the flag fired on 4.9 % of the
    # speechocean762-train words, whose mean human accuracy was 7.2 / 10; at -2: 1.4 %)
    w["omitted_f"] = (-_gmin(-dl, wg, nw) < OMIT_LLR_DEL).astype(float)
    gw = w["group"]
    w["utt_ws_mean"] = _gmean(ss, g, ng)[gw]
    w["utt_wl_min"] = _gmin(sl, g, ng)[gw]
    # round 2: second-lowest sound, llr sum, and spill-over context (a wrong word disturbs its
    # neighbours' alignment, so the model sees each statistic for the previous / next word of
    # the same reference and the difference to them)
    w["wl_min2"], w["ws_min2"] = _gmin2(sl, wg, nw), _gmin2(ss, wg, nw)
    w["word_sum_llr_sub"] = _gsum(f("llr_sub"), wg, nw)
    first_w = np.r_[True, gw[1:] != gw[:-1]] if nw else np.zeros(0, bool)
    last_w = np.r_[gw[1:] != gw[:-1], True] if nw else np.zeros(0, bool)
    for c in NEIGHBOUR_STATS:
        x = w[c]
        prv = np.where(first_w, np.nan, np.r_[np.nan, x[:-1]]) if nw else x
        nxt = np.where(last_w, np.nan, np.r_[x[1:], np.nan]) if nw else x
        w[f"prev_{c}"], w[f"next_{c}"] = prv, nxt
        w[f"dprev_{c}"], w[f"dnext_{c}"] = x - prv, x - nxt
    w["word_pos_first"] = first_w.astype(float)
    w["word_pos_last"] = last_w.astype(float)
    for c in ("stress_prob", "tone_prob"):
        v = None if prosody is None else prosody.get(c)
        w[c] = np.full(nw, np.nan) if v is None else np.asarray(
            [np.nan if x is None else x for x in v], dtype=np.float64)
    return w


def pacing(start, end, dur: float, frame_shift: float = FRAME_SHIFT) -> dict:
    """Pacing from the phone spans of one utterance: articulation rate (phones per second of
    speech time), number and total length of silent stretches > PAUSE_MIN_S between the first
    and the last sound, share of the audio that is speech."""
    s = np.asarray(start, dtype=np.int64)
    e = np.asarray(end, dtype=np.int64)
    ok = e > s
    if not ok.any() or dur <= 0:
        return {"articulation_rate": 0.0, "n_pauses": 0.0, "pause_total_s": 0.0,
                "speech_share": 0.0}
    s, e = s[ok], e[ok]
    o = np.argsort(s, kind="stable")
    s, e = s[o], e[o]
    reach = np.maximum.accumulate(e)
    gaps = (s[1:] - reach[:-1]) * frame_shift
    gaps = gaps[gaps > PAUSE_MIN_S]
    span = (e.max() - s.min()) * frame_shift
    speech = max(span - gaps.sum(), frame_shift)
    return {"articulation_rate": float(ok.sum() / speech), "n_pauses": float(len(gaps)),
            "pause_total_s": float(gaps.sum()), "speech_share": float(min(speech / dur, 1.0))}


def sentence_table(t: dict, sl, ss, sr, w: dict, wl, ws, wr, dur, group=None) -> dict:
    """Per reference inputs of the sentence calibrator. w: word_table output; wl/ws/wr: word
    lenient logit, strict logit, regression (0-10); dur: audio seconds per group (scalar or
    array indexed by group code)."""
    n = len(sl)
    g, ng = _codes(np.zeros(n, dtype=np.int64) if group is None else group)
    gw = np.asarray(w["group"], dtype=np.int64)
    f = lambda k: np.asarray(t[k], dtype=np.float64)  # noqa: E731
    out = {}
    out["w_len_mean"], out["w_len_min"] = _gmean(wl, gw, ng), _gmin(wl, gw, ng)
    out["w_len_min2"] = _gmin2(np.asarray(wl, dtype=np.float64), gw, ng)
    out["w_str_min2"] = _gmin2(np.asarray(ws, dtype=np.float64), gw, ng)
    out["w_str_mean"], out["w_str_min"] = _gmean(ws, gw, ng), _gmin(ws, gw, ng)
    out["w_len_share_below"] = _gmean((np.asarray(wl) < 0).astype(float), gw, ng)
    out["w_str_share_below"] = _gmean((np.asarray(ws) < 0).astype(float), gw, ng)
    out["w_reg_mean"], out["w_reg_min"] = _gmean(wr, gw, ng), _gmin(wr, gw, ng)
    out["s_len_mean"], out["s_len_min"] = _gmean(sl, g, ng), _gmin(sl, g, ng)
    out["s_str_mean"] = _gmean(ss, g, ng)
    out["s_str_p10"] = _gquantile(np.asarray(ss, dtype=np.float64), g, ng, 0.10)
    out["s_len_share_below"] = _gmean((np.asarray(sl) < 0).astype(float), g, ng)
    out["s_str_share_below"] = _gmean((np.asarray(ss) < 0).astype(float), g, ng)
    out["s_reg_mean"] = _gmean(sr, g, ng)
    out["completeness"] = 1.0 - _gmean(np.asarray(w["omitted_f"], dtype=np.float64), gw, ng)
    out["n_words"] = np.bincount(gw, minlength=ng).astype(float)
    out["mean_llr_sub"] = _gmean(f("llr_sub"), g, ng)
    out["mean_z_llr_sub"] = _gmean(f("z_llr_sub"), g, ng)
    out["share_del"] = _gmean((f("llr_del") < 0).astype(float), g, ng)
    durs = np.broadcast_to(np.asarray(dur, dtype=np.float64), (ng,))
    sf, ef = f("start_frame"), f("end_frame")
    pc = {k: np.zeros(ng) for k in ("articulation_rate", "n_pauses", "pause_total_s", "speech_share")}
    starts = np.searchsorted(g, np.arange(ng)) if np.all(np.diff(g) >= 0) else None
    for k in range(ng):
        if starts is not None:
            a = starts[k]
            b = starts[k + 1] if k + 1 < ng else n
            idx = slice(a, b)
        else:
            idx = g == k
        p = pacing(sf[idx], ef[idx], float(durs[k]))
        for c in pc:
            pc[c][k] = p[c]
    out.update(pc)
    return out


def sigmoid(x):
    return _sig(x)
