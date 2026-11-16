"""Text-level planted errors for the Azure head-to-head ("fleurs-swap").

Azure Pronunciation Assessment only takes a reference TEXT, so an error is planted by changing
the text, not the audio: native audio is scored once against its own transcript (`orig`) and
again against a transcript in which ONE evaluation unit was replaced. A good scorer flags the
replaced unit and nothing else.

Evaluation unit ("word"):
  * en es fr de it pt ru ko: whitespace token with leading/trailing punctuation stripped
    (same rule as bench.azure_pa.tokenize); its phones are those of the front-end RefWords
    whose char spans fall inside the token;
  * zh ja: the front-end RefWord (char_span).

Variants (one replacement each, position never the first or last unit):
  swap_far   another real word (FLEURS train+test vocabulary), phone count within +-2,
             phone edit distance >= 60 % of the longer word;          zh: one hanzi replaced
  swap_near  a real word differing by exactly one phone substitution (same phone count);
             FLEURS vocabulary first, then the shipped lexicons; ko falls back to a one-jamo
             change; zh: one hanzi with the same tone and a one-phone-different syllable
  tone       zh only: one hanzi with the same syllable and a different tone (no sandhi or
             neutral tone at the position, before or after)
Every replacement is verified with the front-end on the full new sentence: all other units keep
their text, phones and (zh) tones; the replaced unit has the intended phones; no new warnings.

Everything here is deterministic (seeded per utterance id and variant).
"""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

SPACE_LANGS = ("en", "es", "fr", "de", "it", "pt", "ru", "ko")
CJK_UNIT_LANGS = ("zh", "ja")
LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
VARIANTS = ("swap_far", "swap_near", "tone")
LEXICON = {"en": "en.tsv.gz", "de": "de.tsv.gz", "fr": "fr.tsv.gz", "it": "it.tsv.gz",
           "pt": "pt.tsv.gz", "ru": "ru.tsv.gz", "es": "es_words.txt.gz"}
MIN_PHONES = 3
FAR_MIN_DIST = 0.6
FAR_LEN_TOL = 2

_STRIP = re.compile(r"^\W+|\W+$")
_HAN = re.compile(r"^[一-鿿]+$")
_JA_WORD = re.compile(r"^[ぁ-ゟ゠-ヿ一-鿿々ー]+$")
_HANGUL = re.compile(r"^[가-힣]+$")
_CJK_BAD = re.compile(r"[0-9A-Za-z０-９Ａ-Ｚａ-ｚ]")


# ----------------------------------------------------------------------------- units

@dataclass
class Unit:
    text: str
    span: tuple[int, int]
    phones: tuple[str, ...]
    tones: tuple = ()                 # zh surface tones per syllable
    oov: bool = False
    syl: list = field(default_factory=list)   # zh: [(char_offset, phones, tone, tone_lexical)]


def _phones(ref, w) -> tuple[str, ...]:
    return tuple(ref.phones[i].ipa for i in range(*w.phones))


def units(ref, lang: str) -> list[Unit]:
    """Evaluation units of a Reference (see module docstring)."""
    text = ref.text
    oov = {w.split(":", 1)[1] for w in ref.warnings if w.startswith("oov:")}
    if lang in CJK_UNIT_LANGS:
        out = []
        for w in ref.words:
            syl = []
            for j, si in enumerate(range(*w.syllables)):
                s = ref.syllables[si]
                syl.append((w.char_span[0] + j, tuple(ref.phones[i].ipa for i in range(*s.phones)),
                            s.tone, s.tone_lexical))
            out.append(Unit(w.text, tuple(w.char_span), _phones(ref, w),
                            tuple(s.tone for s in ref.syllables[w.syllables[0]:w.syllables[1]]),
                            False, syl))
        return out
    out = []
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        core = _STRIP.sub("", tok)
        if not core:
            continue
        a = m.start() + tok.index(core)
        b = a + len(core)
        ph: list[str] = []
        is_oov = False
        for w in ref.words:
            c0, c1 = w.char_span
            if c0 < b and c1 > a:
                ph.extend(_phones(ref, w))
                is_oov |= w.text.lower() in oov or _STRIP.sub("", w.text).lower() in oov
        out.append(Unit(core, (a, b), tuple(ph), (), is_oov))
    return out


def unit_list(us: list[Unit]) -> list[dict]:
    return [{"text": u.text, "span": list(u.span)} for u in us]


# ----------------------------------------------------------------------------- helpers

def edit_distance(a, b) -> int:
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]


def is_far(a, b) -> bool:
    return (abs(len(a) - len(b)) <= FAR_LEN_TOL
            and edit_distance(a, b) >= FAR_MIN_DIST * max(len(a), len(b)))


def is_near(a, b) -> bool:
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def rng_for(uid: str, variant: str) -> random.Random:
    return random.Random(int(hashlib.sha1(f"{uid}|{variant}".encode()).hexdigest()[:16], 16))


def source_ok(lang: str, text: str, ref) -> bool:
    """Selection rule: no front-end warnings other than OOV notes; zh/ja/ko: no digits or Latin."""
    if any(not w.startswith("oov:") for w in ref.warnings):
        return False
    if lang in ("zh", "ja", "ko") and _CJK_BAD.search(unicodedata.normalize("NFKC", text)):
        return False
    return len(units(ref, lang)) >= 3


def target_ok(lang: str, u: Unit) -> bool:
    """Can unit u be replaced (space languages and ja: >= 3 phones, a plain word)?"""
    if u.oov or len(u.phones) < MIN_PHONES:
        return False
    t = u.text
    if lang == "ja":
        return bool(_JA_WORD.match(t))
    if lang == "ko":
        return bool(_HANGUL.match(t))
    if not t.isalpha():
        return False
    if lang == "de":
        return not (len(t) > 1 and t.isupper())
    return t == t.lower()


# ----------------------------------------------------------------------------- vocabulary

@dataclass
class Entry:
    text: str
    phones: tuple[str, ...]
    freq: int
    source: str          # "fleurs" | "lexicon"


class Vocab:
    """Candidate replacement words for one language.

    Built from texts (FLEURS train+test transcripts) and, for near swaps, from the shipped
    lexicon word list. Phones come from the front-end: isolated-word G2P for space languages,
    in-context RefWords for ja; zh keeps a hanzi inventory {hanzi: (phones, tone, count)}.
    """

    def __init__(self, lang: str, texts: list[str], g2p, lexicon_words: list[str] | None = None):
        self.lang = lang
        self.g2p = g2p
        self.entries: list[Entry] = []
        self.hanzi: list[tuple[str, tuple[str, ...], int, int]] = []   # zh
        if lang == "zh":
            self._build_zh(texts)
        elif lang == "ja":
            self._build_ja(texts)
        else:
            self._build_space(texts, lexicon_words or [])
        self._index()

    # --- builders
    def _iso(self, word: str) -> tuple[str, ...] | None:
        ref = self.g2p(word)
        if ref.warnings or len(ref.words) != 1:
            return None
        us = units(ref, self.lang)
        if len(us) != 1 or us[0].text != word:
            return None
        return us[0].phones

    def _build_space(self, texts, lexicon_words) -> None:
        counts, noninit = Counter(), Counter()
        for t in texts:
            toks = [_STRIP.sub("", x) for x in t.split()]
            toks = [x for x in toks if x]
            for i, tok in enumerate(toks):
                if not tok.isalpha():
                    continue
                if self.lang == "ko" and not _HANGUL.match(tok):
                    continue
                counts[tok] += 1
                if i > 0:
                    noninit[tok] += 1
        forms: dict[str, str] = {}
        if self.lang == "de":
            for f in sorted(counts):
                if len(f) > 1 and f.isupper():
                    continue
                k = f.lower()
                if k not in forms or (noninit[f], counts[f]) > (noninit[forms[k]], counts[forms[k]]):
                    forms[k] = f
        else:
            for f in sorted(counts):
                if self.lang != "ko" and f != f.lower():
                    continue
                cap = f[:1].upper() + f[1:]
                if cap != f and noninit[cap] > noninit[f]:
                    continue                     # mostly capitalised mid-sentence: a name
                forms[f.lower()] = f
        seen = set()
        for k in sorted(forms):
            f = forms[k]
            if len(f) < 2:
                continue
            ph = self._iso(f)
            if ph:
                self.entries.append(Entry(f, ph, counts[f], "fleurs"))
                seen.add(k)
        for w in lexicon_words:
            if len(w) < 2 or not w.isalpha() or w.lower() in seen:
                continue
            if self.lang != "de" and w != w.lower():
                continue
            if self.lang == "de" and len(w) > 1 and w.isupper():
                continue
            ph = self._iso(w)
            if ph and len(ph) >= MIN_PHONES:
                self.entries.append(Entry(w, ph, 0, "lexicon"))
                seen.add(w.lower())

    def _build_ja(self, texts) -> None:
        cnt: dict[str, Counter] = defaultdict(Counter)
        for t in texts:
            ref = self.g2p(t)
            for u in units(ref, "ja"):
                if _JA_WORD.match(u.text) and u.phones:
                    cnt[u.text][u.phones] += 1
        for w in sorted(cnt):
            ph, n = cnt[w].most_common(1)[0]
            self.entries.append(Entry(w, ph, sum(cnt[w].values()), "fleurs"))

    def _build_zh(self, texts) -> None:
        cnt: dict[str, Counter] = defaultdict(Counter)
        for t in texts:
            ref = self.g2p(t)
            for u in units(ref, "zh"):
                if len(u.syl) != len(u.text) or not _HAN.match(u.text):
                    continue
                for (off, ph, tone, tlex), ch in zip(u.syl, u.text):
                    cnt[ch][(ph, tlex)] += 1
        for ch in sorted(cnt):
            if ch in "一不":
                continue
            (ph, tlex), n = cnt[ch].most_common(1)[0]
            tot = sum(cnt[ch].values())
            if tot >= 3 and n / tot >= 0.9 and tlex in (1, 2, 3, 4):
                self.hanzi.append((ch, ph, tlex, tot))

    def _index(self) -> None:
        self.near_idx: dict[tuple, list[int]] = defaultdict(list)
        for k, e in enumerate(self.entries):
            for i in range(len(e.phones)):
                self.near_idx[(len(e.phones), i, e.phones[:i] + e.phones[i + 1:])].append(k)

    # --- candidate lists (preference order)
    def far_candidates(self, word: str, ph: tuple[str, ...]) -> list[Entry]:
        c = [e for e in self.entries if e.source == "fleurs" and e.freq >= 2
             and e.text.lower() != word.lower() and is_far(ph, e.phones)]
        c.sort(key=lambda e: (-e.freq, e.text))
        return c

    def near_candidates(self, word: str, ph: tuple[str, ...]) -> list[Entry]:
        ks = set()
        for i in range(len(ph)):
            ks.update(self.near_idx.get((len(ph), i, ph[:i] + ph[i + 1:]), ()))
        c = [self.entries[k] for k in sorted(ks)]
        c = [e for e in c if e.text.lower() != word.lower() and is_near(ph, e.phones)]
        c.sort(key=lambda e: (e.source != "fleurs", -e.freq, len(e.text), e.text))
        return c


# ----------------------------------------------------------------------------- verification

def verify(lang: str, g2p, orig_units: list[Unit], k: int, new_text: str, new_word: str,
           orig_warnings: list[str], want=None):
    """Run the front-end on new_text; return (new_ref, new_units) if only unit k changed and
    want(new_unit) holds, else None."""
    ref = g2p(new_text)
    if not set(ref.warnings) <= set(orig_warnings):
        return None
    nu = units(ref, lang)
    if len(nu) != len(orig_units):
        return None
    for i, (a, b) in enumerate(zip(orig_units, nu)):
        if i == k:
            continue
        if a.text != b.text or a.phones != b.phones or a.tones != b.tones:
            return None
    if nu[k].text != new_word or nu[k].oov:
        return None
    if want is not None and not want(nu[k]):
        return None
    return ref, nu


# ----------------------------------------------------------------------------- korean jamo

_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_T = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
      "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_PAIRS = ["ㄱㅋ", "ㄷㅌ", "ㅂㅍ", "ㅈㅊ", "ㅅㅆ", "ㄴㅁ", "ㄱㄲ", "ㄷㄸ", "ㅂㅃ", "ㅈㅉ", "ㄹㄴ", "ㅅㅈ",
          "ㅗㅓ", "ㅜㅡ", "ㅏㅓ", "ㅗㅜ", "ㅐㅔ", "ㅣㅡ", "ㅕㅛ", "ㅑㅕ"]
JAMO_SWAP: dict[str, list[str]] = defaultdict(list)
for _a, _b in _PAIRS:
    JAMO_SWAP[_a].append(_b)
    JAMO_SWAP[_b].append(_a)


def jamo_variants(word: str) -> list[tuple[int, str]]:
    """All one-jamo changes of a Hangul word: [(syllable index, new word)] in a fixed order."""
    out = []
    for i, ch in enumerate(word):
        c = ord(ch) - 0xAC00
        if not 0 <= c < 11172:
            continue
        li, vi, ti = c // 588, (c % 588) // 28, c % 28
        L, V, T = _L[li], _V[vi], _T[ti]
        for pos, cur in (("L", L), ("V", V), ("T", T)):
            for alt in JAMO_SWAP.get(cur, []):
                nl, nv, nt = li, vi, ti
                if pos == "L" and alt in _L:
                    nl = _L.index(alt)
                elif pos == "V" and alt in _V:
                    nv = _V.index(alt)
                elif pos == "T" and alt in _T:
                    nt = _T.index(alt)
                else:
                    continue
                new = chr(0xAC00 + nl * 588 + nv * 28 + nt)
                out.append((i, word[:i] + new + word[i + 1:]))
    return out


# ----------------------------------------------------------------------------- generation

def _labels(variant, src_id, k, nu: list[Unit], ou: list[Unit], extra=None) -> dict:
    lab = {"variant": variant, "src_id": src_id, "word_index": k,
           "char_span_in_new_text": list(nu[k].span) if k is not None else None,
           "orig_word": ou[k].text if k is not None else None,
           "new_word": nu[k].text if k is not None else None,
           "phone_edit_distance": edit_distance(ou[k].phones, nu[k].phones) if k is not None else 0,
           "n_phones": len(nu[k].phones) if k is not None else None,
           "words": unit_list(nu)}
    if extra:
        lab.update(extra)
    return lab


def _try_order(rng: random.Random, cands: list, top: int = 5) -> list:
    """Seeded shuffle of the `top` preferred candidates, then the rest in preference order."""
    head = cands[:top]
    rng.shuffle(head)
    return head + cands[top:]


def make_variants(uid: str, lang: str, text: str, g2p, vocab: Vocab, max_tries: int = 40) -> dict:
    """{variant: (new_text, labels)} for one source utterance (variants that failed are absent),
    plus "orig". Raises nothing; an impossible variant is simply missing."""
    ref = g2p(text)
    ou = units(ref, lang)
    out = {"orig": (text, _labels("orig", uid, None, ou, ou))}
    inner = list(range(1, len(ou) - 1))
    if lang == "zh":
        for v in VARIANTS:
            r = _zh_variant(uid, v, text, ref, ou, inner, g2p, vocab, max_tries)
            if r:
                out[v] = r
        return out
    pos0 = [k for k in inner if target_ok(lang, ou[k])]
    for v in ("swap_far", "swap_near"):
        rng = rng_for(uid, v)
        pos = list(pos0)
        rng.shuffle(pos)
        # near: FLEURS-vocabulary minimal pairs at any position first, then the lexicon
        # (ko: one-jamo changes) -- so the less common fallback words are used only when needed
        passes = ["fleurs"] if v == "swap_far" else ["fleurs", "jamo" if lang == "ko" else "lexicon"]
        done = None
        for p in passes:
            for k in pos:
                u = ou[k]
                if v == "swap_far":
                    tries = [(e.text, e.source) for e in vocab.far_candidates(u.text, u.phones)]
                    rng.shuffle(tries)
                    want = lambda nu, u=u: is_far(u.phones, nu.phones)
                elif p == "jamo":
                    tries = [(w, "jamo") for _, w in jamo_variants(u.text)]
                    want = lambda nu, u=u: is_near(u.phones, nu.phones)
                else:
                    c = [e for e in vocab.near_candidates(u.text, u.phones) if e.source == p]
                    tries = [(e.text, e.source) for e in _try_order(rng, c)]
                    want = lambda nu, u=u: is_near(u.phones, nu.phones)
                for new_word, src in tries[:max_tries]:
                    nt = text[:u.span[0]] + new_word + text[u.span[1]:]
                    r = verify(lang, g2p, ou, k, nt, new_word, ref.warnings, want)
                    if r:
                        done = (nt, _labels(v, uid, k, r[1], ou, {"source": src}))
                        break
                if done:
                    break
            if done:
                out[v] = done
                break
    return out


def _zh_variant(uid, v, text, ref, ou, inner, g2p, vocab, max_tries):
    rng = rng_for(uid, v)
    slots = []                      # (unit k, syllable j)
    for k in inner:
        u = ou[k]
        if len(u.syl) != len(u.text) or not _HAN.match(u.text):
            continue
        for j in range(len(u.syl)):
            slots.append((k, j))
    rng.shuffle(slots)
    for k, j in slots:
        u = ou[k]
        off, ph, tone, tlex = u.syl[j]
        ch = u.text[j]
        if ch in "一不":
            continue
        if v == "swap_far":
            c = [h for h in vocab.hanzi if h[0] != ch and is_far(ph, h[1])]
            c.sort(key=lambda h: (-h[3], h[0]))
            c = c[:300]
            rng.shuffle(c)
        elif v == "swap_near":
            if tone not in (1, 2, 3, 4):
                continue
            c = [h for h in vocab.hanzi if h[0] != ch and h[2] == tone and is_near(ph, h[1])]
            c.sort(key=lambda h: (-h[3], h[0]))
            c = _try_order(rng, c)
        else:  # tone
            if tone not in (1, 2, 3, 4) or tone != tlex:
                continue
            c = [h for h in vocab.hanzi if h[0] != ch and h[1] == ph and h[2] != tone]
            c.sort(key=lambda h: (-h[3], h[0]))
            c = _try_order(rng, c)
        for h in c[:max_tries]:
            nch, nph, ntone, _ = h
            new_word = u.text[:j] + nch + u.text[j + 1:]
            nt = text[:off] + nch + text[off + 1:]

            def want(nu, j=j, nph=nph, ntone=ntone):
                if len(nu.syl) != len(u.syl) or nu.span != u.span:
                    return False
                for jj, (a, b) in enumerate(zip(u.syl, nu.syl)):
                    if jj != j and (a[1], a[2]) != (b[1], b[2]):
                        return False
                s = nu.syl[j]
                if s[1] != nph or s[2] != ntone:
                    return False
                if v == "tone":
                    return s[3] == s[2] and s[2] != tone and s[1] == ph
                if v == "swap_near":
                    return s[2] == tone and is_near(ph, s[1])
                return is_far(ph, s[1])

            # tones of the other units must stay too (verify compares unit tones except unit k)
            r = verify("zh", g2p, ou, k, nt, new_word, ref.warnings, want)
            if r:
                nu = r[1]
                lab = _labels(v, uid, k, nu, ou, {
                    "source": "fleurs", "orig_char": ch, "new_char": nch,
                    "char_span_in_new_text": [off, off + 1],
                    "unit_span_in_new_text": list(nu[k].span),
                    "orig_syllable": list(ph), "new_syllable": list(nph),
                    "orig_tone": tone, "new_tone": ntone,
                    "phone_edit_distance": edit_distance(ph, nph)})
                return nt, lab
    return None
