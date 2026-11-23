"""Per-word tests for the native false-alarm taxonomy (all automatic, written down here).

Every word row of core.word_rows gets the fields below; taxonomy.py turns them into buckets.

noise (corpus noise; not a scorer error), any of
  asr_word   the local Whisper large-v3 transcript disagrees with the prompt at this word: character-level
             alignment of prompt and transcript (letters and digits only, case-folded), the word's
             characters have edits on more than 25 % of them (any edit for words of <= 3 characters).
             Words containing digits are not tested (Whisper writes numbers in words or digits).
  low_snr    the clip's SNR (pronunciada.quality) is below 15 dB
  truncated  the word is the first (last) prompt word and the file starts (ends) inside speech:
             RMS of the first (last) 50 ms within 10 dB of the clip's speech level, and less than
             50 ms of leading (trailing) silence
neighbour    changed items: the word next to the changed word (for an inserted word, the words on either
             side). Changed-item words only.
variant      has_lex: the lexicon (or Britfone, en) has another reading; has_rule: a written optional rule
             of variants.py applies; rescued: the best of N alternatives (counterfactual re-scoring)
             lifts the word's benchmark score above the FA@80 threshold (a below-0.6 word: to >= 0.6)
boundary     first or last word of the clip; or a clipped onset / offset (as in truncated, without
             the ASR or position test); or a click (an isolated 5 ms energy spike >= 20 dB over its
             +-40 ms neighbourhood) within 100 ms of the word; or alignment jitter: the aligner's word
             span and Whisper's word times (cross-attention alignment) differ by more than 150 ms at
             either edge (words matched by the character alignment above)
function     the word is in the spaCy 3.7.5 stop-word list of the language (MIT). Korean: the eojeol
             is a spaCy ko stop stem, alone or followed by one particle of a fixed list, and has at most
             3 syllables. Also: short = aligned duration < 120 ms or at most 2 sounds.
tail         none of the above and the word's score is within 1 logit unit below the threshold score
residual     none of the above and further below the threshold
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache

import numpy as np

from bench.round20_native_fa.rebuild import ROOT

KO_PARTICLES = ("은", "는", "이", "가", "을", "를", "의", "에", "도", "만", "와", "과", "로", "고", "다", "서", "게", "며", "면")


@lru_cache(maxsize=None)
def stop_words(lang: str) -> frozenset:
    p = ROOT / "ext" / "spacy" / f"stop_words_{lang}.py"
    s = p.read_text(encoding="utf-8")
    m = re.search(r'"""(.*?)"""', s, re.S)
    words = set(m.group(1).split()) if m else set()
    if lang == "en":                    # contractions the file adds programmatically
        words |= {"n't", "'d", "'ll", "'m", "'re", "'s", "'ve"}
    return frozenset(w.lower() for w in words)


def key(t: str) -> str:
    return re.sub(r"[^\w]", "", t.lower().replace("’", "'"))


def is_function(lang: str, tok: str) -> bool:
    k = key(tok)
    if lang != "ko":
        return k in stop_words(lang) or tok.lower().strip(".,;:!?«»\"()") in stop_words(lang)
    sw = stop_words("ko")
    if not k or len(k) > 3:
        return False
    if k in sw:
        return True
    return any(k == s + p for s in sw for p in KO_PARTICLES)


# ----------------------------------------------------------------------------- ASR

@lru_cache(maxsize=None)
def asr(lang: str) -> dict:
    p = ROOT / "asr" / f"{lang}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["clip"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def _chars(s: str):
    """(letters/digits lower-cased, index of the source character of each)."""
    out, idx = [], []
    for i, ch in enumerate(s):
        if ch.isalnum():
            out.append(ch.lower())
            idx.append(i)
    return "".join(out), idx


def char_align(a: str, b: str):
    """Levenshtein alignment of strings a, b: for each char of a, (matched b index or None, edited flag)."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = D[i], D[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
    out = [None] * n
    ins_before = [0] * (n + 1)
    i, j = n, m
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            out[i - 1] = (j - 1, a[i - 1] != b[j - 1])
            i, j = i - 1, j - 1
        elif i and D[i, j] == D[i - 1, j] + 1:
            out[i - 1] = (None, True)
            i -= 1
        else:
            ins_before[i] += 1
            j -= 1
    return out, ins_before


def asr_words(prompt: str, rec: dict | None):
    """Per prompt token: {'mismatch': bool|None, 't': (start, end)|None} from the Whisper transcript."""
    toks = list(re.finditer(r"\S+", prompt))
    if not rec:
        return [{"mismatch": None, "t": None} for _ in toks]
    a, ai = _chars(prompt)
    hyp = rec["transcript"]
    b, bi = _chars(hyp)
    al, ins = char_align(a, b)
    # whisper word of each hypothesis character
    wspans, cur = [], 0
    for w, s, e in rec.get("words") or []:
        j = hyp.find(w.strip(), cur)
        if j >= 0:
            wspans.append((j, j + len(w.strip()), s, e))
            cur = j + len(w.strip())
    def word_at(ci):
        for a0, a1, s, e in wspans:
            if a0 <= ci < a1:
                return s, e
        return None
    pos_of = {src: k for k, src in enumerate(ai)}
    out = []
    for m in toks:
        ks = [pos_of[i] for i in range(m.start(), m.end()) if i in pos_of]
        if not ks or any(ch.isdigit() for ch in m.group()):
            out.append({"mismatch": None, "t": None})
            continue
        edits = sum(1 for k in ks if al[k][1]) + sum(ins[k] for k in ks[1:])
        n = len(ks)
        mm = edits >= 1 if n <= 3 else edits / n > 0.25
        ts = [word_at(bi[al[k][0]]) for k in ks if al[k][0] is not None]
        ts = [t for t in ts if t]
        out.append({"mismatch": bool(mm), "t": (min(t[0] for t in ts), max(t[1] for t in ts)) if ts else None})
    return out


# ----------------------------------------------------------------------------- audio

@lru_cache(maxsize=4096)
def clip_audio(path: str):
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def edge_db(path: str, ms: float = 50.0):
    x, sr = clip_audio(path)
    n = int(sr * ms / 1000)
    f = lambda y: 10 * math.log10(float(np.mean(y.astype(np.float64) ** 2)) + 1e-12)  # noqa: E731
    return f(x[:n]), f(x[-n:])


def frame_db(path: str, hop_ms: float = 2.5, win_ms: float = 5.0):
    x, sr = clip_audio(path)
    h, w = int(sr * hop_ms / 1000), int(sr * win_ms / 1000)
    n = max(1, (len(x) - w) // h + 1)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    e = np.mean(x[np.clip(idx, 0, len(x) - 1)].astype(np.float64) ** 2, axis=1)
    return 10 * np.log10(e + 1e-12), hop_ms / 1000


def click_near(path: str, t0: float, t1: float, pad: float = 0.1) -> bool:
    db, hop = frame_db(path)
    a, b = max(0, int((t0 - pad) / hop)), min(len(db), int((t1 + pad) / hop) + 1)
    k = int(0.04 / hop)
    for i in range(a, b):
        lo, hi = max(0, i - k), min(len(db), i + k + 1)
        nb = np.concatenate([db[lo:max(lo, i - 2)], db[min(hi, i + 3):hi]])
        if len(nb) < 4:
            continue
        if db[i] - np.median(nb) >= 20 and db[i] > -60:
            return True
    return False
