"""Calibrator input tables (port of pronunciada/features.py: phone_table, derive, matrix,
word_table; pronunciada/variants.py: broad_class, phone_position, NativeStats).

Target structure (the parts of pronunciada.types.Reference these tables read):
  {"words": [{"text": str, "phones": [start, end]}],
   "syllables": [{"stress": 0 | 1 | 2 | None}],
   "phones": [{"ipa": str, "alts": [...], "word": int, "syllable": int, "is_nucleus": bool}]}

A table is a dict column name -> list (one entry per sound, or per word for word_table).
Missing numbers are NaN; text columns hold str or None.

Data read: data/native/<lang>.json (native statistics: stats -> "phone_pos" / "phone" / "class" /
"global" -> feature -> {"count", "mean", "std", ...}) and data/calib/phone_dist.npz ("tokens":
127 symbols, "dist": 127 x 127 float32 articulatory distances, NaN for modifier tokens).
"""
from __future__ import annotations

import json
import math
import os

from .npz import load_npz
from .numeric import NAN, exp, is_finite, log, sum_pairwise
from .tokens import MODIFIERS

EPS = 1e-8
OMIT_LLR_DEL = -2.0
FRAME_SHIFT = 0.02
MIN_COUNT = 30                 # native statistics: smallest count a back-off level must have

_VOWELS = frozenset("aeiouyæøœɐɑɒɔɘəɛɜɞɤɨɪɯɵɶʉʊʌʏ")
_STOPS = frozenset("pbtdʈɖcɟkgqɢʔɓɗʄɠʛᶑǀǁǂǃʘ")
_FRICS = frozenset("ɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɕʑɧʍ")
_NASALS = frozenset("mɱnɳɲŋɴ")
_LIQ = frozenset("lɭʎʟɬɮrɾɹɻɽɺʀʙⱱjwɥɰʋ")
_AFFR = frozenset({("t", "s"), ("d", "z"), ("t", "ʃ"), ("d", "ʒ"), ("t", "ɕ"), ("d", "ʑ"),
                   ("ʈ", "ʂ"), ("ɖ", "ʐ"), ("p", "f"), ("t", "θ"), ("k", "x"), ("c", "ç")})

NEIGHBOUR_STATS = ["ws_min", "ws_min2", "wl_min", "ws_mean", "ws_share_below", "word_min_llr_sub",
                   "word_sum_llr_sub", "word_mean_llr_del"]
IDENTITY_CAT = ("phone", "comp_tok")      # categorical columns given as codes (not one-hot)


def _data(*parts) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", *parts)


# ----------------------------------------------------------------------------- phone classes

def broad_class(ipa: str) -> str:
    """vowel / stop / fricative / affricate / nasal / liquid-glide."""
    if ipa and ipa[0] in _VOWELS:
        return "vowel"
    base = [c for c in ipa if c not in MODIFIERS]
    if len(base) >= 2 and (base[0], base[1]) in _AFFR:
        return "affricate"
    c = base[0] if base else (ipa[:1] or "?")
    if c in _STOPS:
        return "stop"
    if c in _FRICS:
        return "fricative"
    if c in _NASALS:
        return "nasal"
    if c in _LIQ:
        return "liquid-glide"
    for x in base:
        if x in _VOWELS:
            return "vowel"
    return "fricative"


def phone_position(target, i: int) -> str:
    p0, p1 = target["words"][target["phones"][i]["word"]]["phones"]
    if p1 - p0 == 1:
        return "only"
    if i == p0:
        return "initial"
    if i == p1 - 1:
        return "final"
    return "medial"


class NativeStats:
    def __init__(self, table: dict):
        self.t = table

    @classmethod
    def load(cls, lang: str, path: str | None = None):
        p = path or _data("native", f"{lang}.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return cls(json.load(f))

    def stats(self, phone: str, position, feature: str) -> dict:
        """Back-off phone + position -> phone -> broad class -> global: first level with
        count >= 30."""
        s = self.t["stats"]
        if position is not None:
            d = s["phone_pos"].get(phone, {}).get(position, {}).get(feature)
            if d and d["count"] >= MIN_COUNT:
                return d
        d = s["phone"].get(phone, {}).get(feature)
        if d and d["count"] >= MIN_COUNT:
            return d
        d = s["class"].get(broad_class(phone), {}).get(feature)
        if d and d["count"] >= MIN_COUNT:
            return d
        return s["global"][feature]

    def z(self, phone: str, position, feature: str, value: float) -> float:
        d = self.stats(phone, position, feature)
        return (float(value) - d["mean"]) / max(d["std"], 1e-3)


_DIST = {}


def token_distance(a: int, b: int) -> float:
    if a == b:
        return 0.0
    if "D" not in _DIST:
        z = load_npz(_data("calib", "phone_dist.npz"))
        _DIST["D"] = z["dist"].rows()
    return float(_DIST["D"][a][b])


# ----------------------------------------------------------------------------- statistics

def mean(xs) -> float:
    """np.mean of a vector: pairwise sum / count."""
    return sum_pairwise(xs) / len(xs)


def median(xs) -> float:
    """np.median: NaN if any NaN; middle value, or (a + b) / 2 of the two middle values."""
    if any(x != x for x in xs):
        return NAN
    s = sorted(xs)
    n = len(s)
    if n % 2:
        return s[n // 2] / 1.0
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def quantile(xs, q: float) -> float:
    """np.quantile, method "linear": i = (n - 1) * q, a = sorted[floor(i)], b = the next value
    (the last value when i >= n - 1), g = i - floor(i);
    a + (b - a) * g, or b - (b - a) * (1 - g) when g >= 0.5. NaN if any value is NaN."""
    if any(x != x for x in xs):
        return NAN
    s = sorted(xs)
    n = len(s)
    vi = (n - 1) * q
    prev = math.floor(vi)
    if vi >= n - 1:
        a = b = s[n - 1]
        g = vi - (-1)
    else:
        a, b = s[prev], s[prev + 1]
        g = vi - prev
    diff = b - a
    if g >= 0.5:
        return b - diff * (1 - g)
    return a + diff * g


# ----------------------------------------------------------------------------- phone table

def span_posterior_features(lp, target, start, end, tokens) -> list:
    """Per sound, over its Viterbi span [s, e) padded by one frame on each side (empty span: the
    two frames around s; clamped to the clip): P = exp(lp) per entry; m = mean over the frames
    (left-to-right sum over frames, then / count); competitor c = the best non-blank token
    (ids 4..126, first maximum of m) that is not one of the sound's own tokens."""
    T = len(lp)
    V = len(lp[0]) if T else 127
    out = []
    for i, p in enumerate(target["phones"]):
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
        W = [[exp(float(x)) for x in lp[t]] for t in range(a, b)]
        nfr = b - a
        m = []
        for k in range(V):
            acc = 0.0
            for r in range(nfr):
                acc += W[r][k]
            m.append(acc / nfr)
        mnb = m[4:]
        tot = sum_pairwise(mnb)
        den = EPS if EPS > tot else tot          # Python max(tot, EPS)
        q = [x / den for x in mnb]
        ent = -sum_pairwise([x * log(x + 1e-12) for x in q])
        d = {"span_frames_padded": nfr, "blank_mean": m[0] + m[3], "ent_nonblank": ent}
        ids = [tokens.index(ch) for ch in p["ipa"]]
        for pref, tid in (("", ids[0]), ("last_", ids[-1] if len(ids) >= 2 else None)):
            if tid is None:
                d.update({f"{pref}can_mean": NAN, f"{pref}can_max": NAN, f"{pref}comp_tok": None,
                          f"{pref}comp_mean": NAN, f"{pref}comp_max": NAN, f"{pref}log_ratio_comp": NAN})
                continue
            mm = list(mnb)
            for k in ids:
                if 4 <= k < 4 + len(mnb):
                    mm[k - 4] = -1.0
            best = 0
            for k in range(len(mm)):
                if mm[k] > mm[best]:
                    best = k
            c = 4 + best
            d.update({f"{pref}can_mean": m[tid], f"{pref}can_max": max(W[r][tid] for r in range(nfr)),
                      f"{pref}comp_tok": tokens[c], f"{pref}comp_mean": m[c],
                      f"{pref}comp_max": max(W[r][c] for r in range(nfr)),
                      f"{pref}log_ratio_comp": log(m[tid] + EPS) - log(m[c] + EPS)})
            if pref == "":
                d.update({"comp_dist": token_distance(tid, c),
                          "comp_cls": "modifier" if tokens[c] in MODIFIERS else broad_class(tokens[c])})
        out.append(d)
    return out


def _empty_span() -> dict:
    d = {"span_frames_padded": 0, "blank_mean": NAN, "ent_nonblank": NAN, "comp_dist": NAN, "comp_cls": None}
    for pref in ("", "last_"):
        d.update({f"{pref}can_mean": NAN, f"{pref}can_max": NAN, f"{pref}comp_tok": None,
                  f"{pref}comp_mean": NAN, f"{pref}comp_max": NAN, f"{pref}log_ratio_comp": NAN})
    return d


TEXT_COLS = ("phone", "cls", "position", "comp_tok", "comp_cls", "last_comp_tok", "heard", "word_text")


def phone_table(lp, target, ev, tokens, ns=None, dur=None) -> dict:
    """Per-sound columns (base + derived), as pronunciada.features.phone_table. ev = output of
    pronunciada.reference.align.score_phones; ns = NativeStats or None; dur = audio seconds."""
    S = ev["slots"]
    s = [x["llr_sub"] for x in S]
    n = len(target["phones"])
    if dur is None:
        dur = len(lp) * FRAME_SHIFT
    dur = float(dur)
    utt = {"utt_mean_llr_sub": mean(s) if n else NAN, "utt_median_llr_sub": median(s) if n else NAN,
           "utt_dur": dur, "phones_per_s": n / dur if dur > 0 else NAN, "utt_n_phones": n}
    post = span_posterior_features(lp, target, [x["start_frame"] for x in S], [x["end_frame"] for x in S],
                                   tokens) if n else []
    words = target["words"]
    wmean = [mean(s[w["phones"][0]:w["phones"][1]]) for w in words]
    wmin = [min(s[w["phones"][0]:w["phones"][1]]) for w in words]
    rows = []
    for i, p in enumerate(target["phones"]):
        pos = phone_position(target, i)
        nf = int(S[i]["n_frames"])
        r = dict(utt)
        r.update({"word_index": p["word"], "syllable_index": p["syllable"], "phone_index": i,
                  "phone": p["ipa"], "cls": broad_class(p["ipa"]), "position": pos,
                  "is_nucleus": bool(p["is_nucleus"]), "stress": target["syllables"][p["syllable"]].get("stress"),
                  "word_text": words[p["word"]]["text"], "llr_sub": s[i], "llr_del": S[i]["llr_del"],
                  "n_frames": nf, "peak_post": S[i]["peak_post"], "alt_index": S[i]["alt_index"],
                  "heard": S[i]["heard"], "start_frame": S[i]["start_frame"], "end_frame": S[i]["end_frame"],
                  "llr_sub_per_frame": s[i] / max(nf, 1)})
        for f in ("llr_sub", "llr_del", "n_frames"):
            r[f"z_{f}"] = ns.z(p["ipa"], pos, f, r[f]) if ns is not None else NAN
        r.update(post[i])
        r.update({"prev_llr_sub": s[i - 1] if i > 0 else NAN, "next_llr_sub": s[i + 1] if i + 1 < n else NAN,
                  "word_mean_llr_sub": wmean[p["word"]], "word_min_llr_sub": wmin[p["word"]]})
        rows.append(r)
    t = {}
    for k in (list(rows[0]) if rows else []):
        v = [r.get(k) for r in rows]
        t[k] = v if k in TEXT_COLS else [NAN if x is None else float(x) for x in v]
    if n:
        derive(t)
    return t


def _groups(keys) -> list:
    """Run codes 0, 1, ... of consecutive equal keys."""
    out, c = [], -1
    for i, k in enumerate(keys):
        if i == 0 or k != keys[i - 1]:
            c += 1
        out.append(c)
    return out


def gmean(x, g, n) -> list:
    """Per group: sum of the finite values (left to right from 0) / their count; NaN if none."""
    s = [0.0] * n
    c = [0.0] * n
    for v, k in zip(x, g):
        ok = is_finite(v)
        s[k] += v if ok else 0.0
        c[k] += 1.0 if ok else 0.0
    return [s[k] / max(c[k], 1.0) if c[k] > 0 else NAN for k in range(n)]


def gmin(x, g, n) -> list:
    m = [float("inf")] * n
    for v, k in zip(x, g):
        if is_finite(v) and v < m[k]:
            m[k] = v
    return [v if is_finite(v) else NAN for v in m]


def gmin2(x, g, n) -> list:
    """Per group the second-smallest finite value (the smallest for one value); NaN if none."""
    vals = [[] for _ in range(n)]
    for v, k in zip(x, g):
        vals[k].append(v if is_finite(v) else float("inf"))
    out = []
    for vs in vals:
        vs = sorted(vs)
        m = vs[1] if len(vs) > 1 and is_finite(vs[1]) else (vs[0] if vs else float("inf"))
        out.append(m if is_finite(m) else NAN)
    return out


def gsum(x, g, n) -> list:
    s = [0.0] * n
    for v, k in zip(x, g):
        s[k] += v
    return s


def derive(t: dict) -> dict:
    """Derived columns of one scored target (production derive() with one group)."""
    n = len(t["llr_sub"])
    sub, dl, nf = t["llr_sub"], t["llr_del"], t["n_frames"]
    t["alt_nonzero"] = [1.0 if a != 0 else 0.0 for a in t["alt_index"]]
    t["llr_sub_rel"] = [a - b for a, b in zip(sub, t["utt_median_llr_sub"])]
    t["is_nucleus_f"] = list(t["is_nucleus"])
    t["stress_f"] = list(t["stress"])
    t["has_last"] = [1.0 if is_finite(v) else 0.0 for v in t["last_can_mean"]]
    t["prev_llr_del"] = [NAN] + dl[:-1]
    t["next_llr_del"] = dl[1:] + [NAN]
    wg = _groups(t["word_index"])
    nw = wg[-1] + 1
    g0 = [0] * n
    wm, wmin_, cnt = gmean(dl, wg, nw), gmin(dl, wg, nw), [0] * nw
    for k in wg:
        cnt[k] += 1
    share = gmean([1.0 if v < 0 else 0.0 for v in dl], wg, nw)
    frames = gsum(nf, wg, nw)
    t["word_mean_llr_del"] = [wm[k] for k in wg]
    t["word_min_llr_del"] = [wmin_[k] for k in wg]
    t["word_n_phones"] = [float(cnt[k]) for k in wg]
    t["word_share_del"] = [share[k] for k in wg]
    t["word_frames"] = [frames[k] for k in wg]
    t["llr_sub_rel_word"] = [a - b for a, b in zip(sub, t["word_mean_llr_sub"])]
    t["utt_mean_llr_del"] = gmean(dl, g0, 1) * n
    t["utt_p10_llr_sub"] = [quantile(sub, 0.10)] * n
    t["utt_share_del"] = gmean([1.0 if v < 0 else 0.0 for v in dl], g0, 1) * n
    zs = t["z_llr_sub"]
    t["utt_mean_z_llr_sub"] = gmean(zs, g0, 1) * n
    t["z_llr_sub_rel"] = [a - b for a, b in zip(zs, t["utt_mean_z_llr_sub"])]
    t["utt_mean_peak_post"] = gmean(t["peak_post"], g0, 1) * n
    t["utt_mean_log_ratio_comp"] = gmean(t["log_ratio_comp"], g0, 1) * n
    return t


def matrix(t: dict, num, cat, vocab, encoding: str = "mixed") -> list:
    """Calibrator input rows: numeric columns (NaN kept), then per categorical column either one
    0/1 column per vocabulary entry (one-hot; None -> the "none" entry, unknown -> all zeros) or one
    code column (index into the vocabulary; unknown / missing -> NaN). "mixed": phone / comp_tok as
    codes, the other categorical columns one-hot."""
    n = len(t[num[0]]) if num else len(t[cat[0]])
    rows = []
    for i in range(n):
        r = [float(t[c][i]) if c in t else NAN for c in num]
        for c in cat:
            idx = {v: j for j, v in enumerate(vocab[c])}
            v = t[c][i] if c in t else None
            code = idx.get("none" if v is None else str(v), -1)
            onehot = encoding == "onehot" or (encoding == "mixed" and c not in IDENTITY_CAT)
            if onehot:
                block = [0.0] * len(vocab[c])
                if code >= 0:
                    block[code] = 1.0
                r += block
            else:
                r.append(float(code) if code >= 0 else NAN)
        rows.append(r)
    return rows


def word_table(t: dict, sl, ss, sr) -> dict:
    """Per-word columns (production word_table, one scored target). sl / ss / sr: per sound
    lenient log-odds, strict log-odds, 0-2 regression of the universal sound model."""
    wi = t["word_index"]
    wg = _groups(wi)
    nw = wg[-1] + 1 if wg else 0
    n = len(sl)
    first = [i for i in range(n) if i == 0 or wg[i] != wg[i - 1]]
    dl = t["llr_del"]
    g0 = [0] * n
    w = {"word_index": [wi[i] for i in first], "row": [float(i) for i in first]}
    w["wl_mean"], w["wl_min"] = gmean(sl, wg, nw), gmin(sl, wg, nw)
    w["ws_mean"], w["ws_min"] = gmean(ss, wg, nw), gmin(ss, wg, nw)
    w["wl_share_below"] = gmean([1.0 if v < 0 else 0.0 for v in sl], wg, nw)
    w["ws_share_below"] = gmean([1.0 if v < 0 else 0.0 for v in ss], wg, nw)
    w["wr_mean"], w["wr_min"] = gmean(sr, wg, nw), gmin(sr, wg, nw)
    cnt = [0.0] * nw
    for k in wg:
        cnt[k] += 1.0
    w["n_phones"] = cnt
    for c in ("word_mean_llr_sub", "word_min_llr_sub", "word_mean_llr_del", "word_min_llr_del",
              "word_share_del", "word_frames"):
        w[c] = [t[c][i] for i in first]
    mx = gmin([-v for v in dl], wg, nw)
    w["omitted_f"] = [1.0 if (-v) < OMIT_LLR_DEL else 0.0 for v in mx]
    w["utt_ws_mean"] = gmean(ss, g0, 1) * nw
    w["utt_wl_min"] = gmin(sl, g0, 1) * nw
    w["wl_min2"], w["ws_min2"] = gmin2(sl, wg, nw), gmin2(ss, wg, nw)
    w["word_sum_llr_sub"] = gsum(t["llr_sub"], wg, nw)
    for c in NEIGHBOUR_STATS:
        x = w[c]
        prv = [NAN] + x[:-1]
        nxt = x[1:] + [NAN]
        w[f"prev_{c}"], w[f"next_{c}"] = prv, nxt
        w[f"dprev_{c}"] = [a - b for a, b in zip(x, prv)]
        w[f"dnext_{c}"] = [a - b for a, b in zip(x, nxt)]
    w["word_pos_first"] = [1.0 if k == 0 else 0.0 for k in range(nw)]
    w["word_pos_last"] = [1.0 if k == nw - 1 else 0.0 for k in range(nw)]
    w["stress_prob"] = [NAN] * nw
    w["tone_prob"] = [NAN] * nw
    return w

