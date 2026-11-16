"""Align human phone labels (ARPAbet, on the corpora's own canonical phones) with OUR reference
phones (en front-end on the prompt text).

Words are aligned first (normalised text, Levenshtein over words); runs of unmatched words
between two matched words are treated as one segment (handles tokenisation differences such as
"it's" / "IT S"). Within a matched word / segment, phones are aligned by weighted edit distance
after mapping ARPAbet to our conventions (cost 0 identical or one of our alts, 0.4 near
(same vowel family, ɾ~t/d, l~l̴ ...), 0.7 same broad class, 1 otherwise; indel 1). A pair aligned
by a diagonal move is "matched"; everything else is unaligned (NaN labels).

EpaDB has no word grouping of its reference phones, so its phones are aligned globally over the
utterance (one segment).
"""
from __future__ import annotations

import re

import numpy as np

from pronunciada.variants import broad_class

ARPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɜ˞",
    "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "g", "HH": "h", "JH": "dʒ",
    "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
    # extensions (EpaDB / TIMIT style)
    "AX": "ə", "AXR": "ə˞", "IX": "ɪ", "UX": "u", "DX": "ɾ", "Th": "t", "Kh": "k", "Ph": "p",
    "BH": "b", "EL": "l", "EM": "m", "EN": "n", "NX": "n",
}
NEAR = [set("ɑɔʌəɒ"), {"ɜ˞", "ə˞"}, {"ʌ", "ə"}, {"ɪ", "i", "ə"}, {"ʊ", "u"}, {"ɛ", "æ", "eɪ"},
        {"oʊ", "ɔ", "o"}, {"t", "ɾ", "d", "ʔ"}, {"l", "l̴"}, {"ɹ", "ɻ"}, {"ð", "d", "θ"},
        {"n", "ŋ"}, {"z", "s"}]


def arpa_to_ours(p: str) -> str | None:
    """ARPAbet (+stress digit) -> our en convention, None if not a phone."""
    if p is None:
        return None
    p = p.strip()
    st = p[-1] if p and p[-1] in "012" else ""
    base = p.rstrip("012")
    if base in ("AH",) and st == "0":
        return "ə"
    if base == "ER" and st == "0":
        return "ə˞"
    if base in ARPA:
        return ARPA[base]
    if base.upper() in ARPA and base.upper() not in ("TH",):
        return ARPA[base.upper()]
    return None


def norm_word(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower()).replace("'", "")


def sub_cost(ours: str, alts: tuple, theirs: str) -> float:
    if ours == theirs or theirs in alts:
        return 0.0
    for s in NEAR:
        if ours in s and theirs in s:
            return 0.4
    if broad_class(ours) == broad_class(theirs):
        return 0.7
    return 1.0


def align_phones(ours: list[tuple[str, tuple]], theirs: list[str]) -> list[tuple[int, int]]:
    """Weighted edit-distance alignment; returns matched (i, j) pairs (diagonal moves)."""
    n, m = len(ours), len(theirs)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    C = np.array([[sub_cost(o, a, t) for t in theirs] for o, a in ours]).reshape(n, m)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + C[i - 1, j - 1])
    i, j, out = n, m, []
    while i > 0 and j > 0:
        if np.isclose(D[i, j], D[i - 1, j - 1] + C[i - 1, j - 1]):
            if C[i - 1, j - 1] < 1.0:
                out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif np.isclose(D[i, j], D[i - 1, j] + 1):
            i -= 1
        else:
            j -= 1
    return out[::-1]


def align_words(ours: list[str], theirs: list[str]) -> list[tuple[int, int]]:
    """Matched (our word, their word) pairs by normalised text."""
    a = [norm_word(w) for w in ours]
    b = [norm_word(w) for w in theirs]
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1.5))
    i, j, out = n, m, []
    while i > 0 and j > 0:
        c = 0 if a[i - 1] == b[j - 1] else 1.5
        if np.isclose(D[i, j], D[i - 1, j - 1] + c):
            if c == 0:
                out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif np.isclose(D[i, j], D[i - 1, j] + 1):
            i -= 1
        else:
            j -= 1
    return out[::-1]


def align_reference(ref, their_words: list[str] | None, their_phones: list[list[str | None]]):
    """their_phones: per their word, the list of mapped phones (None = unmappable symbol).
    their_words None -> one global segment (their_phones must have one entry).
    Returns (ours_to_theirs: dict our phone idx -> (word j, phone k), word_pairs list)."""
    our_ph = [(p.ipa, tuple(a for a, _ in p.alts)) for p in ref.phones]
    if their_words is None:
        segs = [(list(range(len(ref.words))), [0])]
        word_pairs = []
    else:
        word_pairs = align_words([w.text for w in ref.words], their_words)
        segs = []
        pi = pj = 0
        for (i, j) in word_pairs + [(len(ref.words), len(their_words))]:
            if i > pi or j > pj:
                segs.append((list(range(pi, i)), list(range(pj, j))))
            if i < len(ref.words):
                segs.append(([i], [j]))
            pi, pj = i + 1, j + 1
    o2t = {}
    for ow, tw in segs:
        if not ow or not tw:
            continue
        oi = [k for w in ow for k in range(*ref.words[w].phones)]
        tj = [(w, k) for w in tw for k, ph in enumerate(their_phones[w]) if ph is not None]
        if not oi or not tj:
            continue
        pairs = align_phones([our_ph[k] for k in oi], [their_phones[w][k] for w, k in tj])
        for a, b in pairs:
            o2t[oi[a]] = tj[b]
    return o2t, word_pairs
