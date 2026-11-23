"""Round 20 word-level passes against native false alarms (all OFF by default; each behind its own switch).

Each pass works on a finished SentenceResult and only replaces WordResult objects of the words it concerns;
sentence-level outputs, coaching inputs and details() keep the first pass. Re-scoring passes call
Scorer.score(wav, Reference, _second=True), which reuses the cached posteriors (no acoustic model run) and
skips the sentence-only delivery features and sequence head.

Switches (environment, "1" = on):
  PRONUNCIADA_R20_UNSAID   B: unsaid words. A word whose presence probability is below TAU
                           (PRONUNCIADA_R20_UNSAID_TAU, default from data/calib/r20.json) is taken out of the
                           reference, the clip is aligned again, and the words up to 2 positions on either side
                           take their results from that second alignment. The unsaid word keeps its own result.
  PRONUNCIADA_R20_NUMALT   C: number tokens are also scored against their other standard readings (best kept).
  PRONUNCIADA_R20_GUESS    D: words the front-end had to guess. "d1": other letter-to-sound readings at the
                           ambiguity points of the existing rules (best kept); "d2": the error signal of such a
                           word is shrunk by a factor (gross mismatches, P < GROSS, untouched).
  PRONUNCIADA_R20_DE_LEXALT  E (de): every lexicon reading of a word is a candidate reading (best kept).
Front-end switches of round 20 (read by the text modules): PRONUNCIADA_R20_ES_GLIDE, PRONUNCIADA_R20_KO_NINS.
Candidate readings are tried only for a word whose probability is below ALT_GATE (0.9; a word already scored
as fine cannot gain), at most MAX_ALTS readings per word and MAX_WORDS words per clip.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
from functools import lru_cache
from importlib import resources

ALT_GATE = 0.9
MAX_ALTS = 3
MAX_WORDS = 4
GROSS = 0.05


def _on(name: str) -> bool:
    return os.environ.get(name, "0") not in ("0", "", "off")


@lru_cache(maxsize=1)
def params() -> dict:
    p = resources.files("pronunciada").joinpath("data/calib/r20.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return {}


def unsaid_tau() -> float:
    v = os.environ.get("PRONUNCIADA_R20_UNSAID_TAU")
    return float(v) if v else float(params().get("unsaid_tau", 0.1))


def guess_mode() -> str:
    return os.environ.get("PRONUNCIADA_R20_GUESS", "0")


def guess_shrink() -> float:
    v = os.environ.get("PRONUNCIADA_R20_GUESS_S")
    return float(v) if v else float(params().get("guess_shrink", 1.0))


def any_on(lang: str) -> bool:
    return (_on("PRONUNCIADA_R20_UNSAID") or _on("PRONUNCIADA_R20_NUMALT") or guess_mode() in ("d1", "d2")
            or (lang == "de" and _on("PRONUNCIADA_R20_DE_LEXALT")))


# ----------------------------------------------------------------------------- reference editing

def _rebuild(ref, words):
    """words: list of (RefWord-like with text / char_span, [syllables as (RefSyllable, [RefPhone])])."""
    from .types import Reference

    out = Reference(lang=ref.lang, text=ref.text, warnings=list(ref.warnings))
    for wi, (w, syls) in enumerate(words):
        s0, p0 = len(out.syllables), len(out.phones)
        for sy, phs in syls:
            si = len(out.syllables)
            a = len(out.phones)
            for p in phs:
                p.word, p.syllable = wi, si
                out.phones.append(p)
            sy.word, sy.phones = wi, (a, len(out.phones))
            out.syllables.append(sy)
        w.syllables, w.phones = (s0, len(out.syllables)), (p0, len(out.phones))
        out.words.append(w)
    out.validate()
    return out


def _nest(ref):
    out = []
    for w in ref.words:
        syls = []
        for si in range(*w.syllables):
            s = ref.syllables[si]
            syls.append((copy.deepcopy(s), [copy.deepcopy(ref.phones[i]) for i in range(*s.phones)]))
        out.append((copy.deepcopy(w), syls))
    return out


def remove_words(ref, idxs):
    keep = [x for i, x in enumerate(_nest(ref)) if i not in set(idxs)]
    return _rebuild(ref, keep)


def replace_word(ref, wi, frag):
    """Word wi gets the sounds and syllables of every word of the Reference `frag` (one alternative reading)."""
    nest = _nest(ref)
    syls = [x for _, s in _nest(frag) for x in s]
    nest[wi] = (nest[wi][0], syls)
    return _rebuild(ref, nest)


# ----------------------------------------------------------------------------- B: unsaid words

def _unsaid(sc, res, wav, s):
    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    n = len(ref.words)
    tau = unsaid_tau()
    gone = [i for i, w in enumerate(res.words) if w.present_prob is not None and w.present_prob < tau]
    if not gone or len(gone) >= n:
        return res, None
    ref2 = remove_words(ref, gone)
    res2 = sc.score(wav, ref2, s, _second=True)
    keep = [i for i in range(n) if i not in gone]
    new = {i: j for j, i in enumerate(keep)}
    near = [i for i in keep if min(abs(i - g) for g in gone) <= 2]
    for i in near:
        res.words[i] = res2.words[new[i]]
    return res, {"removed": gone, "replaced": near}


# ----------------------------------------------------------------------------- C / D1 / E: candidate readings

_DIGIT = re.compile(r"\d")


def number_alternatives(lang: str, text: str) -> list[str]:
    """Other standard readings of a number token, as text (spelled-out words), written from the language."""
    from .text._eu_common import _num2words

    n2w = _num2words()
    t = text.strip()
    m = re.fullmatch(r"(\d[\d    ]*)([^\d]*)", t)
    if not m or n2w is None:
        return []
    digits = re.sub(r"\D", "", m.group(1))
    suffix = m.group(2)
    if not digits or len(digits) > 7:
        return []
    v = int(digits)
    out = []
    try:
        if lang == "ko":
            return _ko_number_alternatives(v, suffix)
        if suffix and lang != "ko" and not re.fullmatch(r"[\W]*", suffix):
            return []                                    # ordinals, units glued to the number: leave as they are
        if lang == "en":
            card = n2w(v, lang="en")
            out += [card, card.replace(" and ", " ")]
            if 100 <= v < 200:
                out.append(re.sub(r"^one hundred", "a hundred", card))
                out.append(re.sub(r"^one hundred", "a hundred", card).replace(" and ", " "))
            if 1000 <= v < 2000:
                out.append(re.sub(r"^one thousand", "a thousand", card))
            if len(digits) == 4 and 1000 <= v <= 2099:
                out.append(n2w(v, lang="en", to="year"))
        elif lang == "de":
            card = n2w(v, lang="de")
            out += [card, re.sub(r"^ein(hundert|tausend)", r"\1", card)]
            if len(digits) == 4 and 1100 <= v <= 1999:
                out.append(n2w(v // 100, lang="de") + "hundert" + (n2w(v % 100, lang="de") if v % 100 else ""))
        elif lang == "fr":
            card = n2w(v, lang="fr")
            out.append(card)
            if len(digits) == 4 and 1100 <= v <= 1999:
                out.append(n2w(v // 100, lang="fr") + " cent " + (n2w(v % 100, lang="fr") if v % 100 else ""))
            out.append(n2w(v, lang="fr_BE") if v % 100 >= 70 or (v % 100) // 10 == 9 else card)
        elif lang == "es":
            card = n2w(v, lang="es")
            out.append(card)
            if card.endswith("uno"):
                out.append(card[:-1])                    # "veintiún", "un" before a noun
        else:
            return []
    except Exception:  # noqa: BLE001
        return []
    seen, res = set(), []
    for x in out:
        x = x.replace("-", " ").replace(",", " ").strip()
        if x and x not in seen:
            seen.add(x)
            res.append(x)
    return res


_KO_NATIVE = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열",
              20: "스무", 30: "서른", 40: "마흔", 50: "쉰", 60: "예순", 70: "일흔", 80: "여든", 90: "아흔"}
_KO_SINO = "영일이삼사오육칠팔구"


def _ko_number_alternatives(v: int, suffix: str) -> list[str]:
    """Korean: Sino-Korean and native readings of 1-99 before a Hangul counter / particle."""
    if not suffix or not re.match(r"[가-힣]", suffix):
        return []
    out = []
    if 1 <= v <= 99:
        tens, ones = (v // 10) * 10, v % 10
        if v in _KO_NATIVE:
            nat = _KO_NATIVE[v]
        else:
            nat = ({10: "열", 20: "스물", 30: "서른", 40: "마흔", 50: "쉰", 60: "예순", 70: "일흔", 80: "여든", 90: "아흔"}
                   .get(tens, "") + _KO_NATIVE.get(ones, ""))
        out.append(nat + suffix)
    sino = ""
    if v < 10000:
        for unit, name in ((1000, "천"), (100, "백"), (10, "십")):
            q = (v // unit) % 10
            if q:
                sino += ("" if q == 1 else _KO_SINO[q]) + name
        if v % 10:
            sino += _KO_SINO[v % 10]
        out.append(sino + suffix)
    return out


def _token_fragment(g2p, text: str, toks: list[str]):
    """A one-word Reference from phone tokens (lexicon notation after the language's rewrite)."""
    from .text import _eu_common as C

    w = C.word_from_prons(text, (0, len(text)), toks, getattr(g2p, "onsets", frozenset()),
                          default_stress=getattr(g2p, "default_stress", 0))
    return C.assemble(g2p.lang, text, [w], [])


def _norm(text: str) -> str:
    import unicodedata

    return re.sub(r"^\W+|\W+$", "", unicodedata.normalize("NFC", text).lower().replace("’", "'"))


def lts_alternatives(g2p, norm: str) -> list[list[str]]:
    """Other letter-to-sound readings of an unknown word at the ambiguity points of the existing rules: at a vowel
    letter whose most specific stored context and its back-off parent give different outputs, the parent's output
    (one point at a time, at most MAX_ALTS readings)."""
    from .text._eu_lts import LADDER, _key

    lts = getattr(g2p, "lts", None)
    if lts is None or not norm:
        return []
    t = lts.table
    padded = "^" + norm + "$"
    base, choices = [], []
    for i in range(1, len(padded) - 1):
        found = []
        for k, m in reversed(LADDER):
            v = t.get(_key(padded, i, k, m))
            if v is not None and (not found or v != found[-1]):
                found.append(v)
            if len(found) == 2:
                break
        base.append(found[0] if found else None)
        if len(found) == 2 and padded[i] in "aeiouyäöüéèêàâîôûáíóú":
            choices.append((i - 1, found[1]))
    out = []
    for pos, alt in choices[:MAX_ALTS]:
        seq = list(base)
        seq[pos] = alt
        ph = [p for x in seq if x for p in x.split("_") if p]
        out.append(ph)
    return out


def _fragments(sc, ref, wi, word_text, res_word, guessed: set):
    g2p = sc.g2p
    lang = sc.lang
    frags = []
    if _on("PRONUNCIADA_R20_NUMALT") and _DIGIT.search(word_text):
        for alt in number_alternatives(lang, word_text)[:MAX_ALTS + 1]:
            try:
                r = g2p(alt)
            except Exception:  # noqa: BLE001
                continue
            if r.words and not r.warnings:
                frags.append(r)
    if guess_mode() == "d1" and wi in guessed and hasattr(g2p, "rewrite"):
        norm = _norm(word_text)
        for ph in lts_alternatives(g2p, norm):
            try:
                toks = g2p.rewrite(g2p.fix_stress(ph, norm), norm)
                frags.append(_token_fragment(g2p, word_text, toks))
            except Exception:  # noqa: BLE001
                continue
    if lang == "de" and _on("PRONUNCIADA_R20_DE_LEXALT") and hasattr(g2p, "lex"):
        from .text import _eu_common as C

        raw = g2p.lex.get(_norm(word_text))
        if raw is not None:
            for p in C.split_prons(raw)[1:MAX_ALTS + 1]:
                try:
                    frags.append(_token_fragment(g2p, word_text, g2p.rewrite(list(p), _norm(word_text))))
                except Exception:  # noqa: BLE001
                    continue
    # drop readings identical to the canonical one
    canon = tuple(p.ipa for p in ref.phones[ref.words[wi].phones[0]:ref.words[wi].phones[1]])
    out, seen = [], {canon}
    for f in frags:
        key = tuple(p.ipa for p in f.phones)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out[:MAX_ALTS]


def guessed_words(res) -> set:
    """Words the front-end had to guess: no lexicon entry (letter rules: warning oov:<word>) and, for en / es / fr,
    names (capitalised, not the first word, not after . ! ? :)."""
    return guessed_in_ref(res._pronunciada["ref"])  # noqa: SLF001


def guessed_in_ref(ref) -> set:
    oov = {w.split(":", 1)[1] for w in ref.warnings if w.startswith("oov:")}
    out = set()
    for i, w in enumerate(ref.words):
        parts = [x for x in re.split(r"[-']", _norm(w.text)) if x]
        if _norm(w.text) in oov or any(x in oov for x in parts):
            out.add(i)
            continue
        if ref.lang in ("en", "es", "fr") and i > 0 and w.text[:1].isupper():
            before = ref.text[:w.char_span[0]].rstrip()
            if before and before[-1] not in ".!?:":
                out.add(i)
    return out


def _alternatives(sc, res, wav, s):
    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    guessed = guessed_words(res) if guess_mode() == "d1" else set()
    cand = [i for i, w in enumerate(res.words) if w.prob < ALT_GATE]
    done = []
    for wi in cand:
        if len(done) >= MAX_WORDS:
            break
        frags = _fragments(sc, ref, wi, ref.words[wi].text, res.words[wi], guessed)
        if not frags:
            continue
        best = res.words[wi]
        for f in frags:
            try:
                r2 = sc.score(wav, replace_word(ref, wi, f), s, _second=True)
            except Exception:  # noqa: BLE001
                continue
            if r2.words[wi].prob > best.prob:
                best = r2.words[wi]
        if best is not res.words[wi]:
            # word-level values only: the sounds, syllables and times keep the first pass (the canonical
            # reading's sound list, which every per-sound consumer, e.g. compat.azure, indexes by the reference)
            w = res.words[wi]
            for f in ("prob", "sounds_prob", "stress_prob", "tone_prob", "present_prob", "omitted", "length_prob",
                      "contrast_prob"):
                setattr(w, f, getattr(best, f))
        done.append(wi)
    return res, done


# ----------------------------------------------------------------------------- D2: uncertainty discount

def _discount(res):
    sh = guess_shrink()
    if sh >= 1.0:
        return res, []
    out = []
    for i in guessed_words(res):
        w = res.words[i]
        if w.prob >= GROSS and w.prob < 1.0:
            w.prob = 1.0 - sh * (1.0 - w.prob)
            out.append(i)
    return res, out


# ----------------------------------------------------------------------------- entry point

def apply(sc, res, wav, s):
    info = {}
    if _on("PRONUNCIADA_R20_UNSAID"):
        res, info["unsaid"] = _unsaid(sc, res, wav, s)
    if _on("PRONUNCIADA_R20_NUMALT") or guess_mode() == "d1" or (sc.lang == "de" and _on("PRONUNCIADA_R20_DE_LEXALT")):
        res, info["alts"] = _alternatives(sc, res, wav, s)
    if guess_mode() == "d2":
        res, info["discount"] = _discount(res)
    res._pronunciada["r20"] = info  # noqa: SLF001
    return res
