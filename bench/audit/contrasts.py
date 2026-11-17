"""Single-feature swap generators for the contrast audit (bench/audit/contrast_audit.py).

A swap edits the *reference* of a native recording at one eligible position so that exactly one
feature now disagrees with what the native audio contains (bench/corrupt.py protocol): the audio
has member A of a contrast, the edited reference demands member B. That simulates a learner who
said A where B was expected; the scorer should flag the edited sound and its word.

Contrast            members (A / B)                  how the swap is made
sub (default)       two reference symbols            the phone's symbol A -> B, alternatives cleared
final_dev           devoiced / voiced                word-final voiceless obstruent of a word spelt
                                                     with a final voiced letter (de, ru) -> voiced
gem_it              geminate / single (Italian)      "C{∅} C" -> "C" (delete the first copy), or an
                                                     intervocalic "C" -> "C C" (insert a copy)
nasal_pt            nasal / oral (Portuguese)        "V n{∅,ŋ}" (nasal vowel = vowel + coda nasal in
                                                     the front-end) -> "V" with nasal alternatives
                                                     removed, or open-syllable "V" -> "V n"
tone                tone / other tone (Mandarin)     the syllable's tone -> another tone
ru_unstressed       reduced / full (Russian)         unstressed vowel spelt о / е, reduced in the
                                                     reference (a, ɐ, ə / i, ɪ) -> full o / ɛ
                                                     (= the learner reads the spelling)

Every function here is pure (no audio, no models, no network) so that it can be unit-tested.
"""
from __future__ import annotations

import copy
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from pronunciada.text.ipa import is_valid, is_vowel
from pronunciada.types import RefPhone, Reference

# ----------------------------------------------------------------------------- definitions


@dataclass(frozen=True)
class Contrast:
    name: str
    label: str
    members: tuple[str, str]                       # names of member A and member B
    pairs: tuple[tuple[str, str], ...] = ()        # (A symbol, B symbol)
    kind: str = "sub"
    where: str = "any"                             # position filter, see _ok
    both: bool = True                              # also B -> A
    extra: tuple[tuple[str, str, str], ...] = ()   # one-way (src, dst, direction) additions
    phonemic: bool = True
    family: str = ""                               # "length" | "gemination" | "aspiration" | ...


VOICING = (("b", "p"), ("d", "t"), ("g", "k"), ("v", "f"), ("z", "s"), ("ʒ", "ʃ"), ("dʒ", "tʃ"))


def _c(name, label, members, pairs=(), **kw) -> Contrast:
    return Contrast(name, label, members, tuple(pairs), **kw)


NASAL_MN = _c("m_n", "nasal place m / n (before a vowel)", ("m", "n"), [("m", "n")], where="onset")

CONTRASTS: dict[str, list[Contrast]] = {
    "en": [
        _c("voicing", "voicing (b/p d/t g/k v/f z/s ʒ/ʃ dʒ/tʃ ð/θ)", ("voiced", "voiceless"),
           VOICING + (("ð", "θ"),)),
        _c("tense_lax", "tense / lax vowel (i/ɪ, u/ʊ)", ("tense", "lax"), [("i", "ɪ"), ("u", "ʊ")],
           family="length_en"),
        _c("th_s", "θ / s", ("θ", "s"), [("θ", "s")]),
        _c("dh_z", "ð / z", ("ð", "z"), [("ð", "z")]),
        _c("dh_d", "ð / d", ("ð", "d"), [("ð", "d")]),
        _c("v_w", "v / w", ("v", "w"), [("v", "w")]),
        _c("v_b", "v / b", ("v", "b"), [("v", "b")]),
        _c("ae_e", "æ / ɛ", ("æ", "ɛ"), [("æ", "ɛ")]),
        _c("r_l", "r / l (ɹ / l)", ("ɹ", "l"), [("ɹ", "l")], extra=(("l̴", "ɹ", "B>A"),)),
        _c("s_sh", "s / ʃ", ("s", "ʃ"), [("s", "ʃ")]),
        NASAL_MN,
        _c("n_ng", "final n / ŋ", ("n", "ŋ"), [("n", "ŋ")], where="coda"),
    ],
    "zh": [
        _c("aspiration", "aspiration (p/pʰ t/tʰ k/kʰ ts/tsʰ ʈʂ/ʈʂʰ tɕ/tɕʰ)", ("aspirated", "plain"),
           [("pʰ", "p"), ("tʰ", "t"), ("kʰ", "k"), ("tsʰ", "ts"), ("ʈʂʰ", "ʈʂ"), ("tɕʰ", "tɕ")],
           family="aspiration"),
        _c("retroflex", "retroflex / dental (zh ch sh / z c s)", ("retroflex", "dental"),
           [("ʈʂ", "ts"), ("ʈʂʰ", "tsʰ"), ("ʂ", "s")]),
        _c("n_ng", "final n / ng", ("n", "ŋ"), [("n", "ŋ")], where="coda"),
        _c("y_u", "ü / u", ("y", "u"), [("y", "u")]),
        _c("r_l", "r / l (ʐ / l)", ("ʐ", "l"), [("ʐ", "l")]),
        _c("tone", "tone (reference point)", ("dictionary tone", "other tone"), kind="tone",
           both=False),
        NASAL_MN,
    ],
    "ja": [
        _c("vowel_length", "vowel length (incl. ー, おう / えい)", ("long", "short"),
           [("aː", "a"), ("iː", "i"), ("uː", "u"), ("eː", "e"), ("oː", "o")], family="length"),
        _c("gemination", "consonant doubling (っ)", ("geminate", "single"),
           [("tt", "t"), ("kk", "k"), ("pp", "p"), ("ss", "s"), ("tts", "ts"), ("ttʃ", "tʃ"),
            ("ʃʃ", "ʃ"), ("dd", "d")], family="gemination"),
        _c("voicing", "voicing (g/k d/t z/s b/p dʒ/ʃ)", ("voiced", "voiceless"),
           [("g", "k"), ("d", "t"), ("z", "s"), ("b", "p"), ("dʒ", "ʃ")]),
        _c("s_sh", "s / ʃ", ("s", "ʃ"), [("s", "ʃ")]),
        NASAL_MN,
    ],
    "ko": [
        _c("aspiration", "aspirated / unaspirated (ㅋㅌㅍㅊ vs ㄱㄷㅂㅈ, ㄲㄸㅃㅉ)", ("aspirated", "plain"),
           [("kʰ", "k"), ("tʰ", "t"), ("pʰ", "p"), ("tʃʰ", "tʃ")], family="aspiration"),
        _c("eo_o", "ㅓ / ㅗ (ɔ / o)", ("ɔ", "o"), [("ɔ", "o")]),
        NASAL_MN,
        _c("n_ng", "final n / ŋ", ("n", "ŋ"), [("n", "ŋ")], where="coda"),
    ],
    "es": [
        _c("tap_trill", "tap / trill (pero / perro)", ("trill", "tap"), [("ɾ", "r")]),
        _c("voicing", "voicing (b/p d/t g/k)", ("voiced", "voiceless"), [("b", "p"), ("d", "t"), ("g", "k")]),
        NASAL_MN,
    ],
    "fr": [
        _c("nasal", "nasal / oral vowel (ɑ̃/a ɔ̃/ɔ ɛ̃/ɛ œ̃/œ)", ("nasal", "oral"),
           [("ɑ̃", "a"), ("ɔ̃", "ɔ"), ("ɛ̃", "ɛ"), ("œ̃", "œ")]),
        _c("round_front", "front rounded / unrounded (y/i ø/e œ/ɛ)", ("rounded", "unrounded"),
           [("y", "i"), ("ø", "e"), ("œ", "ɛ")]),
        _c("front_back", "front / back rounded (y/u ø/o œ/ɔ)", ("front", "back"),
           [("y", "u"), ("ø", "o"), ("œ", "ɔ")]),
        _c("mid_vowels", "close / open mid (e/ɛ o/ɔ ø/œ)", ("close", "open"),
           [("e", "ɛ"), ("o", "ɔ"), ("ø", "œ")]),
        _c("voicing", "voicing", ("voiced", "voiceless"), VOICING),
        _c("s_sh", "s / ʃ", ("s", "ʃ"), [("s", "ʃ")]),
        NASAL_MN,
    ],
    "de": [
        _c("vowel_length", "long / short vowel, stressed (iː/ɪ yː/ʏ uː/ʊ eː/ɛ øː/œ oː/ɔ ɑː/a)",
           ("long", "short"), [("iː", "ɪ"), ("yː", "ʏ"), ("uː", "ʊ"), ("eː", "ɛ"), ("øː", "œ"),
                               ("oː", "ɔ"), ("ɑː", "a")], where="stressed", family="length"),
        _c("round_front", "front rounded / unrounded (yː/iː ʏ/ɪ øː/eː œ/ɛ)", ("rounded", "unrounded"),
           [("yː", "iː"), ("ʏ", "ɪ"), ("øː", "eː"), ("œ", "ɛ")]),
        _c("front_back", "front / back rounded (yː/uː ʏ/ʊ øː/oː œ/ɔ)", ("front", "back"),
           [("yː", "uː"), ("ʏ", "ʊ"), ("øː", "oː"), ("œ", "ɔ")]),
        _c("voicing", "voicing, not word-final", ("voiced", "voiceless"), VOICING, where="nonfinal"),
        _c("final_devoicing", "final devoicing (Rad = [t])", ("devoiced", "voiced"),
           [(b, a) for a, b in VOICING], kind="final_dev", both=False),
        _c("s_sh", "s / ʃ", ("s", "ʃ"), [("s", "ʃ")]),
        _c("ich_isch", "ç / ʃ (ich / isch)", ("ç", "ʃ"), [("ç", "ʃ")]),
        NASAL_MN,
        _c("n_ng", "final n / ŋ", ("n", "ŋ"), [("n", "ŋ")], where="coda"),
    ],
    "it": [
        _c("gemination", "consonant doubling (fato / fatto)", ("geminate", "single"), kind="gem_it",
           family="gemination"),
        _c("mid_vowels", "close / open mid (e/ɛ o/ɔ)", ("close", "open"), [("e", "ɛ"), ("o", "ɔ")]),
        _c("voicing", "voicing (b/p d/t g/k v/f dʒ/tʃ dz/ts)", ("voiced", "voiceless"),
           [("b", "p"), ("d", "t"), ("g", "k"), ("v", "f"), ("dʒ", "tʃ"), ("dz", "ts")]),
        _c("s_sh", "s / ʃ", ("s", "ʃ"), [("s", "ʃ")]),
        NASAL_MN,
    ],
    "pt": [
        _c("nasal", "nasal / oral vowel (lã / lá)", ("nasal", "oral"), kind="nasal_pt"),
        _c("mid_vowels", "close / open mid (e/ɛ o/ɔ: avô / avó)", ("close", "open"),
           [("e", "ɛ"), ("o", "ɔ")]),
        _c("tap_trill", "tap / strong r between vowels (caro / carro)", ("strong r", "tap"),
           [("ʁ", "ɾ")], where="intervocalic"),
        _c("voicing", "voicing", ("voiced", "voiceless"), VOICING),
        _c("s_sh", "s / ʃ, before a vowel", ("s", "ʃ"), [("s", "ʃ")], where="onset"),
        NASAL_MN,
    ],
    "ru": [
        _c("hard_soft", "hard / soft consonant (Cʲ written in the reference)", ("soft", "hard"),
           [(c + "ʲ", c) for c in ("n", "s", "v", "k", "p", "b", "x", "f", "g", "z", "m")]
           + [("ɭʲ", "ɭ"), ("tsʲ", "ts"), ("dzʲ", "dz")]),
        _c("unstressed_vowel", "unstressed vowel reduction (о / е read as written)", ("reduced", "full"),
           kind="ru_unstressed", both=False),
        _c("voicing", "voicing, not word-final", ("voiced", "voiceless"),
           [("b", "p"), ("d", "t"), ("g", "k"), ("v", "f"), ("z", "s"), ("ʐ", "ʂ"), ("bʲ", "pʲ"),
            ("vʲ", "fʲ"), ("zʲ", "sʲ"), ("gʲ", "kʲ")], where="nonfinal"),
        _c("final_devoicing", "final devoicing (город = [t])", ("devoiced", "voiced"),
           [("p", "b"), ("t", "d"), ("k", "g"), ("f", "v"), ("s", "z"), ("ʂ", "ʐ"), ("ts", "dz")],
           kind="final_dev", both=False),
        _c("s_sh", "s / ʂ", ("s", "ʂ"), [("s", "ʂ")]),
        NASAL_MN,
    ],
}

# Contrasts the reference cannot express (no swap possible): minimal pairs probed in the report
NOT_REPRESENTABLE: dict[str, list[dict]] = {
    "en": [{"contrast": "aspiration (allophonic in English)", "pair": ["pin", "spin"], "phonemic": False}],
    "ko": [{"contrast": "lax / tense stops (ㄷ / ㄸ)", "pair": ["달", "딸"], "phonemic": True},
           {"contrast": "lax / tense stops (ㄱ / ㄲ)", "pair": ["굴", "꿀"], "phonemic": True},
           {"contrast": "lax / tense s (ㅅ / ㅆ)", "pair": ["살", "쌀"], "phonemic": True},
           {"contrast": "ㅐ / ㅔ (merged in Seoul speech)", "pair": ["개", "게"], "phonemic": False},
           {"contrast": "r / l (allophones of ㄹ)", "pair": ["라면", "말"], "phonemic": False}],
    "ru": [{"contrast": "hard / soft r", "pair": ["рад", "ряд"], "phonemic": True},
           {"contrast": "hard / soft t (word-final)", "pair": ["мат", "мать"], "phonemic": True},
           {"contrast": "hard / soft m", "pair": ["мол", "мёл"], "phonemic": True}],
    "ja": [{"contrast": "r / l (not phonemic)", "pair": ["ら", "ら"], "phonemic": False}],
    "es": [{"contrast": "b / v (not phonemic)", "pair": ["baca", "vaca"], "phonemic": False}],
}

VOICED_LETTERS = {"de": set("bdgw"), "ru": set("бвгджз")}
RU_VOWELS = set("аеёиоуыэюя")
GEM_IT = {"p", "b", "t", "d", "k", "g", "f", "v", "s", "m", "n", "l", "r", "tʃ", "dʒ", "ts", "dz"}
NASALS = {"n", "m", "ŋ"}


def contrasts_for(lang: str) -> list[Contrast]:
    return CONTRASTS.get(lang, [])


def get_contrast(lang: str, name: str) -> Contrast:
    for c in contrasts_for(lang):
        if c.name == name:
            return c
    raise KeyError(f"{lang}/{name}")


# ----------------------------------------------------------------------------- positions


def _syllable(ref: Reference, i: int):
    return ref.syllables[ref.phones[i].syllable]


def syl_position(ref: Reference, i: int) -> str:
    s = _syllable(ref, i)
    nuc = next(k for k in range(*s.phones) if ref.phones[k].is_nucleus)
    return "nucleus" if i == nuc else ("onset" if i < nuc else "coda")


def word_bounds(ref: Reference, i: int) -> tuple[int, int]:
    return ref.words[ref.phones[i].word].phones


def _prev(ref, i):
    a, _ = word_bounds(ref, i)
    return ref.phones[i - 1].ipa if i - 1 >= a else None


def _next(ref, i):
    _, b = word_bounds(ref, i)
    return ref.phones[i + 1].ipa if i + 1 < b else None


def _ok(ref: Reference, i: int, where: str) -> bool:
    if where == "any":
        return True
    if where == "onset":
        return syl_position(ref, i) == "onset"
    if where == "coda":
        return syl_position(ref, i) == "coda"
    if where == "nonfinal":
        return i != word_bounds(ref, i)[1] - 1
    if where == "final":
        return i == word_bounds(ref, i)[1] - 1
    if where == "stressed":
        return _syllable(ref, i).stress == 1
    if where == "intervocalic":
        p, n = _prev(ref, i), _next(ref, i)
        return p is not None and n is not None and is_vowel(p) and is_vowel(n)
    raise ValueError(where)


def letters(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFC", text.lower()) if ch.isalpha())


# ----------------------------------------------------------------------------- sites


@dataclass(frozen=True)
class Site:
    contrast: str
    direction: str          # "A>B": the audio has member A, the edited reference demands member B
    index: int              # phone index in the native reference (tone: syllable index)
    src: str
    dst: str
    kind: str = "sub"


def _directed(c: Contrast) -> list[tuple[str, str, str]]:
    out = [(a, b, "A>B") for a, b in c.pairs]
    if c.both:
        out += [(b, a, "B>A") for a, b in c.pairs]
    return out + list(c.extra)


def sites(ref: Reference, c: Contrast) -> list[Site]:
    """Every eligible position of contrast c in the (native) reference."""
    out: list[Site] = []
    if c.kind == "sub":
        m: dict[str, tuple[str, str]] = {}
        for src, dst, d in _directed(c):
            m.setdefault(src, (dst, d))
        for i, p in enumerate(ref.phones):
            if p.ipa in m and _ok(ref, i, c.where):
                dst, d = m[p.ipa]
                if is_valid(dst):
                    out.append(Site(c.name, d, i, p.ipa, dst, kind=c.kind))
    elif c.kind == "final_dev":
        m = {a: b for a, b in c.pairs}
        voiced = VOICED_LETTERS[ref.lang]
        for w in ref.words:
            lt = letters(w.text)
            i = w.phones[1] - 1
            if lt and lt[-1] in voiced and ref.phones[i].ipa in m:
                out.append(Site(c.name, "A>B", i, ref.phones[i].ipa, m[ref.phones[i].ipa], kind=c.kind))
    elif c.kind == "gem_it":
        for i, p in enumerate(ref.phones):
            a, b = word_bounds(ref, i)
            if p.is_nucleus:
                continue
            nxt = ref.phones[i + 1].ipa if i + 1 < b else None
            prv = ref.phones[i - 1].ipa if i - 1 >= a else None
            if nxt == p.ipa:                                        # first half of a geminate
                out.append(Site(c.name, "A>B", i, p.ipa + p.ipa, p.ipa, kind=c.kind))
            elif (p.ipa in GEM_IT and prv is not None and nxt is not None and is_vowel(prv)
                  and is_vowel(nxt) and prv != p.ipa):
                out.append(Site(c.name, "B>A", i, p.ipa, p.ipa + p.ipa, kind=c.kind))
    elif c.kind == "nasal_pt":
        for i, p in enumerate(ref.phones):
            if not p.is_nucleus:
                continue
            s = _syllable(ref, i)
            if i + 1 == s.phones[1] - 1 and ref.phones[i + 1].ipa in NASALS:
                out.append(Site(c.name, "A>B", i, p.ipa + "~", p.ipa, kind=c.kind))
            elif i == s.phones[1] - 1 and not any("̃" in a for a, _ in p.alts):
                nxt = _next(ref, i)
                if nxt is None or nxt not in NASALS:
                    out.append(Site(c.name, "B>A", i, p.ipa, p.ipa + "~", kind=c.kind))
    elif c.kind == "tone":
        for si, s in enumerate(ref.syllables):
            if s.tone is not None and s.tone in (1, 2, 3, 4):
                out.append(Site(c.name, "A>B", si, str(s.tone), "?", kind=c.kind))
    elif c.kind == "ru_unstressed":
        for w in ref.words:
            vl = [ch for ch in letters(w.text) if ch in RU_VOWELS]
            syls = list(range(*w.syllables))
            if len(vl) != len(syls) or len(syls) < 2:
                continue
            for k, si in enumerate(syls):
                s = ref.syllables[si]
                if s.stress != 0:
                    continue
                nuc = next(j for j in range(*s.phones) if ref.phones[j].is_nucleus)
                ipa = ref.phones[nuc].ipa
                if vl[k] == "о" and ipa in ("a", "ɐ", "ə", "ʌ", "ɑ"):
                    out.append(Site(c.name, "A>B", nuc, ipa, "o", kind=c.kind))
                elif vl[k] == "е" and ipa in ("i", "ɪ"):
                    out.append(Site(c.name, "A>B", nuc, ipa, "ɛ", kind=c.kind))
    else:
        raise ValueError(c.kind)
    return out


# ----------------------------------------------------------------------------- edits


@dataclass
class Edit:
    site: Site
    ref: Reference                # edited reference
    flag: list[int]               # phones of `ref` that are now wrong (tone: syllable indices)
    word: int
    partner_in_alts: bool         # the native reference already accepts member B here
    detail: dict = field(default_factory=dict)


def _rebuild(ref: Reference, phones: list[RefPhone], syl_of: list[int]) -> Reference:
    """Reference with the given phone list (each with its syllable index in `ref`)."""
    new = copy.deepcopy(ref)
    new.phones = []
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    for p, si in zip(phones, syl_of):
        q = copy.deepcopy(p)
        q.syllable = si
        q.word = ref.syllables[si].word
        starts.setdefault(si, len(new.phones))
        new.phones.append(q)
        ends[si] = len(new.phones)
    for si, s in enumerate(new.syllables):
        s.phones = (starts[si], ends[si])
    for w in new.words:
        s0, s1 = w.syllables
        w.phones = (new.syllables[s0].phones[0], new.syllables[s1 - 1].phones[1])
    new.validate()
    return new


def apply(ref: Reference, site: Site, rng: np.random.Generator | None = None) -> Edit:
    """Edited deep copy of ref for one site."""
    if site.kind == "tone":
        rng = rng or np.random.default_rng(0)
        new = copy.deepcopy(ref)
        s = new.syllables[site.index]
        choices = [t for t in (1, 2, 3, 4) if t != s.tone and t not in s.tone_alts]
        t = choices[int(rng.integers(len(choices)))]
        old = s.tone
        s.tone = s.tone_lexical = t
        s.tone_alts = []
        new.validate()
        return Edit(site, new, [site.index], s.word, False, {"from": old, "to": t})
    i = site.index
    p = ref.phones[i]
    alts = {a for a, _ in p.alts}
    phones = list(ref.phones)
    syl_of = [q.syllable for q in phones]
    if site.kind == "nasal_pt":
        if site.direction == "A>B":                           # delete coda nasal, oral vowel only
            v = copy.deepcopy(p)
            v.alts = [(a, w) for a, w in v.alts if "̃" not in a]
            phones[i] = v
            removed = phones[i + 1].ipa
            del phones[i + 1], syl_of[i + 1]
            return Edit(site, _rebuild(ref, phones, syl_of), [i], p.word, "" in {a for a, _ in ref.phones[i + 1].alts},
                        {"removed": removed})
        phones.insert(i + 1, RefPhone("n", p.word, p.syllable, False, []))
        syl_of.insert(i + 1, p.syllable)
        return Edit(site, _rebuild(ref, phones, syl_of), [i, i + 1], p.word,
                    any("̃" in a for a in alts), {})
    if site.kind == "gem_it":
        if site.direction == "A>B":                           # geminate -> single: drop first copy
            del phones[i], syl_of[i]
            return Edit(site, _rebuild(ref, phones, syl_of), [i], p.word, "" in alts, {})
        a, _ = word_bounds(ref, i)                            # single -> geminate: insert a copy
        phones.insert(i, RefPhone(p.ipa, p.word, p.syllable, False, []))
        syl_of.insert(i, ref.phones[i - 1].syllable if i - 1 >= a else p.syllable)
        return Edit(site, _rebuild(ref, phones, syl_of), [i, i + 1], p.word, site.dst in alts, {})
    if site.kind not in ("sub", "final_dev", "ru_unstressed"):
        raise ValueError(site.kind)
    new = copy.deepcopy(ref)                                  # plain substitution
    q = new.phones[i]
    new.phones[i] = RefPhone(site.dst, q.word, q.syllable, q.is_nucleus, [])
    new.validate()
    return Edit(site, new, [i], q.word, site.dst in alts, {})
