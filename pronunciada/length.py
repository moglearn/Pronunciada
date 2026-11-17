"""Length channel (round 15): is each long / short sound held for the right time?

The acoustic model emits 20 ms frames and collapses repeated tokens, so it barely hears length
(results/contrast_audit.md: Japanese vowel length and っ are blind). This channel reads timing
instead, from evidence the scorer already has:

* the canonical alignment's onsets (pronunciada.align, 20 ms output frames): the onset-to-onset
  interval of a vowel, and the vowel-to-vowel span across a consonant, divided by the
  utterance's own median (speaking-rate normalisation);
* frame energy of the fbank (10 ms): where a vowel stops before a pause (so phrase-final
  lengthening and pauses do not count as length), and the closure silence of っ;
* the acoustic model's own length token: the peak posterior of "ː" over the vowel.

Labels. Every sound the channel judges gets length_expected "long" / "short" from the reference:
RefPhone.length when a front-end sets it, otherwise from the canonical symbol (Japanese "aː" /
"tt" = long; a plain vowel, or a single consonant between two vowels of the same word = short).
The canonical symbol is what the prompt asks for, so this is distinct even where the front-end
accepts the short form as an alternative pronunciation (おばあさん "aː{a}"): the alternative keeps
sound scores unchanged, the length channel judges the duration.

Model: boosted depth-3 trees per class (vowel / consonant) on the features below, P(long),
fitted on FLEURS train natives with the text's own labels (bench/length/fit_length.py; walked in
numpy here); length_prob = P(long) for an expected long sound, 1 - P(long) for a short one. The
word factor (word_factor) is 1 unless the word's lowest length logit falls below a knee placed at
a low native quantile. Parameters: pronunciada/data/length/<lang>.json.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources

import numpy as np

from .text.ipa import is_vowel, token_ids

LANGS = ("ja", "it", "de")
DE_PAIRS = {"iː": "ɪ", "yː": "ʏ", "uː": "ʊ", "eː": "ɛ", "øː": "œ", "oː": "ɔ", "ɑː": "a"}
DE_SHORT = set(DE_PAIRS.values())
JA_VOICELESS = set("ptksʃçhf")
FRIC_END = set("szʃʒɕʑçhfvxθð")      # fricatives and affricates (by their last symbol)
PAUSE_SKIP = 8          # 10 ms frames of silence after a vowel: before a pause, length is not judged
SIL_DB = 30.0           # a 10 ms fbank frame this far below the utterance's loud level is silence
_LN10 = math.log(10.0)

V_FEATURES = ("r_own", "r_voiced", "pause_after", "nb_ratio", "colon_post", "word_final", "utt_final",
              "close_vowel", "log_rate", "next_fric", "next_devoiced", "final_x_dur", "colon_x_dur")
C_FEATURES = ("r_span", "r_pre", "closure", "closure_share", "min_energy", "fricative", "log_rate",
              "colon_post")


# ----------------------------------------------------------------------------- labels

def _is_gem(ipa: str) -> bool:
    return len(ipa) >= 2 and ipa[0] == ipa[1] and not is_vowel(ipa)


def _word(ref, i):
    return ref.phones[i].word


def labels(ref) -> list[str | None]:
    """'long' / 'short' / None per reference sound (see the module docstring)."""
    lang = ref.lang
    P = ref.phones
    n = len(P)
    out: list[str | None] = [None] * n
    for i, p in enumerate(P):
        if getattr(p, "length", None) in ("long", "short"):
            out[i] = p.length
            continue
        prev_v = i > 0 and _word(ref, i - 1) == p.word and is_vowel(P[i - 1].ipa)
        next_v = i + 1 < n and _word(ref, i + 1) == p.word and is_vowel(P[i + 1].ipa)
        if lang == "ja":
            if p.is_nucleus and is_vowel(p.ipa) and len(p.ipa.replace("ː", "")) == 1:
                if p.ipa.endswith("ː"):
                    out[i] = "long"
                elif not (prev_v or next_v) and ref.words[p.word].syllables[1] - ref.words[p.word].syllables[0] > 1:
                    # a short vowel next to another vowel of the same word is not judged:
                    # sequences such as えい / おお / いう are often said as one long vowel; nor
                    # is the vowel of a one-mora word (particles を の で は: lengthened at
                    # phrase edges by natives, never a length contrast)
                    out[i] = "short"
            elif not is_vowel(p.ipa) and prev_v and next_v:
                if _is_gem(p.ipa):
                    out[i] = "long"
                elif not (p.ipa[0] in JA_VOICELESS and P[i + 1].ipa in ("i", "u", "ɯ")):
                    # a single voiceless consonant before i / u is not judged: the vowel is often
                    # devoiced and its silence reads as a held consonant (ます, です, した)
                    out[i] = "short"
        elif lang == "it":
            if is_vowel(p.ipa) or not next_v:
                continue
            if i > 1 and P[i - 1].ipa == p.ipa and _word(ref, i - 1) == p.word and is_vowel(P[i - 2].ipa) \
                    and _word(ref, i - 2) == p.word:
                out[i] = "long"                     # second copy of "C{∅} C"
            elif prev_v and not (i + 1 < n and P[i + 1].ipa == p.ipa):
                out[i] = "short"
        elif lang == "de":
            if p.is_nucleus and ref.syllables[p.syllable].stress == 1:
                if p.ipa in DE_PAIRS:
                    out[i] = "long"
                elif p.ipa in DE_SHORT:
                    out[i] = "short"
    return out


# ----------------------------------------------------------------------------- evidence

def frame_inputs(ref, ev, lpa: np.ndarray, fb: np.ndarray) -> dict:
    """Compact per-utterance inputs (cheap; also what the bench stores)."""
    tid = token_ids()
    e = (np.log(np.exp(np.asarray(fb, dtype=np.float32)).sum(1, dtype=np.float64) + 1e-10)
         if fb is not None and len(fb) else np.zeros(0))
    return {"onset": np.array([s.start_frame for s in ev.slots], dtype=np.int64),
            "ok": np.array([s.alt_index >= 0 for s in ev.slots], dtype=bool),
            "colon": np.exp(lpa[:, tid["ː"]].astype(np.float64)) if len(lpa) else np.zeros(0),
            "energy": e, "frame_shift": float(ev.frame_shift)}


def _fb_index(t: float) -> int:
    """fbank (10 ms) frame closest to the start of output frame t (20 ms, offset 30 ms)."""
    return int(round((t * 0.02 + 0.03) / 0.01))


def features(ref, lab: list, inp: dict):
    """(indices, classes 'v'/'c', X dict-of-arrays) for the labelled sounds."""
    P = ref.phones
    n = len(P)
    on, ok = inp["onset"], inp["ok"]
    E = inp["energy"]
    floor, loud = (float(x) for x in np.percentile(E, [5, 95])) if len(E) else (0.0, 0.0)
    # silence: 30 dB below the loud level, or 35 % of the way up from the floor in noisy / low
    # dynamic-range recordings
    sil = max(loud - SIL_DB / 10.0 * _LN10, floor + 0.35 * (loud - floor))
    colon = inp["colon"]
    okidx = np.flatnonzero(ok)
    pos = np.searchsorted(okidx, np.arange(n), side="right")
    nxt = (np.where(pos < len(okidx), okidx[np.minimum(pos, len(okidx) - 1)], -1) if len(okidx)
           else np.full(n, -1))
    silent = E < sil
    # k such that frames k-2, k-1, k are all silent: a vowel's voiced stretch ends at k-2
    r3 = np.flatnonzero(silent[2:] & silent[1:-1] & silent[:-2]) + 2 if len(E) >= 3 else np.zeros(0, int)
    own = np.full(n, np.nan)
    voi = np.full(n, np.nan)
    has = ok & (nxt >= 0)
    ii = np.flatnonzero(has)
    if len(ii):
        own[ii] = on[nxt[ii]] - on[ii]
        a = np.rint((on[ii] * 0.02 + 0.03) / 0.01).astype(np.int64)
        b = np.maximum(np.rint((on[nxt[ii]] * 0.02 + 0.03) / 0.01).astype(np.int64), a + 1)
        bb = np.minimum(b, len(E))
        j = np.searchsorted(r3, a + 2)
        k = r3[np.minimum(j, max(len(r3) - 1, 0))] if len(r3) else np.full(len(ii), 1 << 40)
        hit = (j < len(r3)) & (k < bb)
        voi[ii] = np.where(hit, np.maximum(k - 2 - a, 1), np.maximum(bb - a, 1))
    vow = [is_vowel(p.ipa) for p in P]
    isv = np.array([p.is_nucleus and v for p, v in zip(P, vow)], dtype=bool)
    med_v = np.nanmedian(own[isv]) if np.isfinite(own[isv]).any() else np.nan
    med_vv = np.nanmedian(voi[isv]) if np.isfinite(voi[isv]).any() else np.nan
    rate = math.log(max(med_v, 1.0)) if np.isfinite(med_v) else 0.0
    # consonant spans: previous vowel onset -> next vowel onset
    span = np.full(n, np.nan)
    pre = np.full(n, np.nan)
    prv = np.full(n, -1)
    for i in range(n):
        if lab[i] is None or vow[i]:
            continue
        j = i - 1
        while j >= 0 and not vow[j] and P[j].word == P[i].word:
            j -= 1
        k = i + 1
        if j >= 0 and k < n and vow[j] and vow[k] and ok[j] and ok[k] and ok[i]:
            span[i] = on[k] - on[j]
            pre[i] = on[i] - on[j]
            prv[i] = j
    short_c = [i for i in range(n) if lab[i] == "short" and np.isfinite(span[i])]
    med_span = np.median(span[short_c]) if short_c else (np.nanmedian(span) if np.isfinite(span).any() else np.nan)
    med_pre = np.median(pre[short_c]) if short_c else (np.nanmedian(pre) if np.isfinite(pre).any() else np.nan)
    idx, cls, rows = [], [], []
    vlist = np.flatnonzero(isv)
    ownl, voil, onl = own.tolist(), voi.tolist(), on.tolist()
    fin = math.isfinite
    for i in range(n):
        if lab[i] is None:
            continue
        if vow[i]:
            if not fin(ownl[i]) or not np.isfinite(med_v):
                continue
            if 2 * ownl[i] - voil[i] >= PAUSE_SKIP:
                continue                       # phrase-final lengthening / pause: not judged
            k = int(np.searchsorted(vlist, i))
            nb = [voil[j] for j in (vlist[k - 1] if k > 0 else -1, vlist[k + 1] if k + 1 < len(vlist) else -1)
                  if j >= 0 and fin(voil[j])]
            a, b = onl[i], onl[nxt[i]]
            cp = float(colon[a:max(b, a + 1)].max()) if len(colon) > a else 0.0
            wf = float(i + 1 == ref.words[P[i].word].phones[1])
            r = {"r_own": math.log(max(ownl[i], 1) / max(med_v, 1)),
                 "r_voiced": math.log(voil[i] / max(med_vv, 1)),
                 "pause_after": math.log1p(max(2 * ownl[i] - voil[i], 0.0)),
                 "nb_ratio": math.log(voil[i] / (sum(nb) / len(nb))) if nb else 0.0,
                 "colon_post": cp, "word_final": wf, "utt_final": float(nxt[i] < 0 or
                                                                         P[i].word == P[-1].word and wf),
                 "close_vowel": float(P[i].ipa[0] in "iuɯyɪʊʏ"), "log_rate": rate}
            nx = P[i + 1].ipa if i + 1 < n else ""
            r["next_fric"] = float(bool(nx) and nx[-1] in FRIC_END)
            r["next_devoiced"] = float(bool(nx) and nx[0] in JA_VOICELESS and i + 2 < n
                                       and P[i + 2].ipa in ("i", "u", "ɯ"))
            r["final_x_dur"] = wf * r["r_voiced"]
            r["colon_x_dur"] = cp * r["r_voiced"]
            idx.append(i), cls.append("v"), rows.append(r)
        else:
            if not np.isfinite(span[i]) or not np.isfinite(med_span):
                continue
            j = prv[i]
            a, b = _fb_index(on[j]), _fb_index(on[i + 1])
            seg = E[a:max(b, a + 1)]
            cl = float((seg < sil).sum()) if len(seg) else 0.0
            cb = on[i + 1]
            cp = float(colon[on[j]:max(cb, on[j] + 1)].max()) if len(colon) > on[j] else 0.0
            r = {"r_span": math.log(max(span[i], 1) / max(med_span, 1)),
                 "r_pre": math.log(max(pre[i], 0.5) / max(med_pre, 0.5)),
                 "closure": math.log1p(cl), "closure_share": cl / max(len(seg), 1),
                 "min_energy": float((seg.min() - loud) / _LN10) if len(seg) else 0.0,
                 "fricative": float(P[i].ipa[0] in "sʃzfhɕ"), "log_rate": rate, "colon_post": cp}
            idx.append(i), cls.append("c"), rows.append(r)
    return idx, cls, rows


# ----------------------------------------------------------------------------- model


@lru_cache(maxsize=None)
def params(lang: str) -> dict | None:
    p = resources.files("pronunciada").joinpath(f"data/length/{lang}.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None


def _z_trees(M: dict, x: np.ndarray) -> np.ndarray:
    """Boosted depth-3 trees stored complete (BFS: internal nodes 0-6, leaves 7-14; x <= thr goes
    left, as in sklearn). x: [n, n_features] -> raw log-odds [n]."""
    T = M.get("_np")
    if T is None:
        T = M["_np"] = (np.asarray(M["feat"], dtype=np.int64), np.asarray(M["thr"], dtype=np.float64),
                        np.asarray(M["leaf"], dtype=np.float64))
    feat, thr, leaf = T
    n, K = len(x), len(feat)
    node = np.zeros((n, K), dtype=np.int64)
    kk = np.arange(K)[None, :]
    for _ in range(3):
        f = feat[kk, node]
        right = np.take_along_axis(x, f, axis=1) > thr[kk, node]
        node = 2 * node + 1 + right
    return M["init"] + M["learning_rate"] * leaf[kk, node - 7].sum(1)


def p_long(model: dict, cls: str, rows) -> np.ndarray:
    """P(long) for feature rows (dicts) of one class."""
    M = model[cls]
    x = np.array([[r[k] for k in M["features"]] for r in rows], dtype=np.float64).reshape(-1, len(M["features"]))
    if "feat" in M:
        z = _z_trees(M, x)
    else:
        z = ((x - np.asarray(M["mean"])) / np.asarray(M["scale"])) @ np.asarray(M["coef"]) + M["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def word_factor(cal: dict, probs, s: float) -> float:
    """Word-level length factor: 1 while the lowest length logit of the word's judged sounds is
    above a knee (a low native quantile, by strictness), then falling linearly in log-odds to
    `floor` over `width`. Natives are almost never touched, so word probabilities of everything
    else keep their ranking."""
    W = cal["word"]
    q = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    zmin = float(np.min(np.log(q / (1 - q))))
    k = W["knee"]
    s = min(max(float(s), 0.0), 1.0)
    knee = k[0] + (k[1] - k[0]) * 2 * s if s <= 0.5 else k[1] + (k[2] - k[1]) * (2 * s - 1)
    t = min(max((knee - zmin) / W["width"], 0.0), 1.0)
    return 1.0 - (1.0 - W["floor"]) * t


def head_combine(par: dict, cls: str, p: float, s: float) -> float:
    """Round 17: combine the timing model's probability p with the attribute head's long-evidence s
    (logit P(head "long" = the expected value)): a logistic on [logit p, s] fitted on FLEURS
    validation natives vs planted length swaps, mapped back onto the timing model's own native
    distribution by quantiles, so native thresholds (word knee, coaching bars) keep their rates."""
    H = par.get("head", {}).get(cls)
    if H is None or not np.isfinite(s):
        return p
    q = min(max(p, 1e-6), 1 - 1e-6)
    z = H["coef"][0] * math.log(q / (1 - q)) + H["coef"][1] * s + H["b"]
    qn, qo = H["q_new"], H["q_orig"]
    if z <= qn[0]:
        zo = qo[0] + (z - qn[0])
    elif z >= qn[-1]:
        zo = qo[-1] + (z - qn[-1])
    else:
        zo = float(np.interp(z, qn, qo))
    return float(1.0 / (1.0 + math.exp(-min(max(zo, -30.0), 30.0))))


def score(ref, inp: dict, par: dict, head_s=None) -> dict:
    """{"expected": [...], "prob": [...]} per reference sound (None where not judged). head_s:
    optional per-sound attribute-head evidence (round 17, pronunciada.contrast_heads)."""
    lab = labels(ref)
    n = len(ref.phones)
    prob: list[float | None] = [None] * n
    expected: list[str | None] = [None] * n
    idx, cls, rows = features(ref, lab, inp)
    for c in ("v", "c"):
        sel = [k for k, x in enumerate(cls) if x == c]
        if not sel or c not in par["model"]:
            continue
        pl = p_long(par["model"], c, [rows[k] for k in sel])
        for k, p in zip(sel, pl):
            i = idx[k]
            expected[i] = lab[i]
            prob[i] = float(p if lab[i] == "long" else 1.0 - p)
            if head_s is not None and par.get("head"):
                prob[i] = head_combine(par, c, prob[i], float(head_s[i]))
    return {"expected": expected, "prob": prob}
