"""Round 20 fixes against native false alarms, on by default per language (DEFAULTS), each switchable.

Fixes (DEFAULTS: language -> fixes on by default; other languages have none):
  numalt     C: a number token is also scored against its other standard readings (best kept).
             Word-level pass, here. Environment: PRONUNCIADA_R20_NUMALT.
  de_lexalt  E (de): every lexicon reading of a word is a candidate reading (best kept). Word-level pass, here.
             Environment: PRONUNCIADA_R20_DE_LEXALT.
  es_glide   es front-end: the glides of rising diphthongs may be heard as vowels (pronunciada.text.es).
             Environment: PRONUNCIADA_R20_ES_GLIDE.
  ko_nins    ko front-end: optional n-insertion inside an eojeol (pronunciada.text.ko).
             Environment: PRONUNCIADA_R20_KO_NINS.
An environment variable set to "0" (or "off") switches its fix off for every language; set to "1" it switches it
on for every language where the fix applies. Unset, the language's DEFAULTS entry decides.

The word-level passes work on a finished SentenceResult and only change word-level values of the words they
concern; the sounds, syllables and times, sentence-level outputs, coaching inputs and details() keep the first
pass. Re-scoring calls Scorer.score(wav, Reference, _second=True), which reuses the cached posteriors (no
acoustic model run) and skips the sentence-only delivery features and sequence head. Candidate readings are
tried only for a word whose probability is below ALT_GATE (0.9; a word already scored as fine cannot gain), at
most MAX_ALTS readings per word and MAX_WORDS words per clip.

Measured in results/round20_native_fa.md; the held-out check is results/round20_fa_release_check.md.
"""
from __future__ import annotations

import copy
import os
import re

ALT_GATE = 0.9
MAX_ALTS = 3
MAX_WORDS = 4

# Decided by the held-out release check (results/round20_fa_release_check.md): en, de and fr failed the
# accuracy rule (planted-error audit rows significantly worse), ko the latency budget (mean +4 ms per clip on
# natural text); they are off, i.e. identical to the round-19 / 1.0.x outputs. Spanish passed both.
DEFAULTS: dict[str, tuple[str, ...]] = {
    "en": (),
    "de": (),
    "es": ("numalt", "es_glide"),
    "fr": (),
    "ko": (),
}
ENV = {"numalt": "PRONUNCIADA_R20_NUMALT", "de_lexalt": "PRONUNCIADA_R20_DE_LEXALT",
       "es_glide": "PRONUNCIADA_R20_ES_GLIDE", "ko_nins": "PRONUNCIADA_R20_KO_NINS"}


def enabled(fix: str, lang: str) -> bool:
    """Is round-20 fix `fix` on for language `lang`: its environment variable if set, else DEFAULTS."""
    v = os.environ.get(ENV[fix], "").strip().lower()
    if v:
        return v not in ("0", "off", "false", "no")
    return fix in DEFAULTS.get(lang, ())


def any_on(lang: str) -> bool:
    """Does the word-level pass run for this language."""
    return enabled("numalt", lang) or (lang == "de" and enabled("de_lexalt", lang))


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


def replace_word(ref, wi, frag):
    """Word wi gets the sounds and syllables of every word of the Reference `frag` (one alternative reading)."""
    nest = _nest(ref)
    syls = [x for _, s in _nest(frag) for x in s]
    nest[wi] = (nest[wi][0], syls)
    return _rebuild(ref, nest)


# ----------------------------------------------------------------------------- C / E: candidate readings

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



def _fragments(sc, ref, wi, word_text):
    # pronunciada.core.score_phones: the candidate readings were computed from text beforehand
    # (pronunciada.core.phones_for -> text_fragments) and arrive on the Reference, keyed by word index
    given = getattr(ref, "word_alternatives", None)
    if given is not None:
        return list(given.get(wi, ()))
    return text_fragments(sc.g2p, sc.lang, ref, wi, word_text)


def text_fragments(g2p, lang, ref, wi, word_text):
    """Candidate readings (one-word References) of word wi from its text with the front-end `g2p`."""
    frags = []
    if enabled("numalt", lang) and _DIGIT.search(word_text):
        for alt in number_alternatives(lang, word_text)[:MAX_ALTS + 1]:
            try:
                r = g2p(alt)
            except Exception:  # noqa: BLE001
                continue
            if r.words and not r.warnings:
                frags.append(r)
    if lang == "de" and enabled("de_lexalt", lang) and hasattr(g2p, "lex"):
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


def _alternatives(sc, res, wav, s):
    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    cand = [i for i, w in enumerate(res.words) if w.prob < ALT_GATE]
    done = []
    for wi in cand:
        if len(done) >= MAX_WORDS:
            break
        frags = _fragments(sc, ref, wi, ref.words[wi].text)
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


# ----------------------------------------------------------------------------- entry point

def apply(sc, res, wav, s):
    info = {}
    if any_on(sc.lang):
        res, info["alts"] = _alternatives(sc, res, wav, s)
    res._pronunciada["r20"] = info  # noqa: SLF001
    return res
