"""German text front-end.

Lexicon: ipa-dict de (MIT), stress from its ˈ/ˌ marks (monosyllabic content words: primary).
OOV: compound splitting into lexicon words (with linking -s-/-es-/-n-/-en-), else the learned
letter-context table.
"""
from __future__ import annotations

import os

from . import _eu_common as C
from ._eu_base import EuG2P

# Round 19 text handling (hyphen pieces, English readings of unknown words); 0 = round-18 front-end
TEXT_R19 = os.environ.get("PRONUNCIADA_DE_TEXT", "1") != "0" and os.environ.get("PRONUNCIADA_DE_R19", "1") != "0"

FUNC = {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "eines", "und",
        "oder", "aber", "in", "im", "an", "am", "auf", "aus", "bei", "mit", "nach", "von", "vom", "zu", "zum",
        "zur", "für", "um", "durch", "über", "unter", "vor", "ist", "sind", "war", "hat", "es", "er", "sie",
        "wir", "ihr", "ich", "du", "man", "sich", "nicht", "als", "wie", "dass", "ob", "wenn", "so", "auch",
        "noch", "schon", "nur", "doch", "ja", "mir", "mich", "dir", "dich", "ihm", "ihn", "uns", "euch"}
# weak forms (canonical stays the dictionary form)
WEAK = {
    "und": ["ʊ n", "n̩", "ʊ n t"], "ist": ["ɪ s"], "nicht": ["n ɪ ç", "n ɪ"], "ein": ["n̩", "aɪ n"],
    "eine": ["n ə", "aɪ n"], "einen": ["n̩", "aɪ n ə n"], "das": ["d a s", "s"], "es": ["ə s", "s"],
    "haben": ["h a m"], "sie": ["z ə"], "der": ["d ɐ", "d ɛ ɐ"], "den": ["d ə n", "n̩"], "dem": ["d ə m"],
}


# Function words: ZIPA's modal form on FLEURS de_de (canonical) + dictionary form (alt).
FUNC_PRON = {
    "mit": ("m iː t", ["m ɪ t"]), "um": ("uː m", ["ʊ m"]), "von": ("f oː n", ["f ɔ n"]),
    "vom": ("f oː m", ["f ɔ m"]), "dass": ("d ɑː s", ["d a s"]), "das": ("d a s", ["d ɑː s"]),
    "hat": ("h ɑː t", ["h a t"]), "es": ("eː s", ["ɛ s", "ə s"]), "des": ("d eː s", ["d ə s", "d ɛ s"]),
    "er": ("eːɐ", ["eː ʀ", "ə ʀ", "ɛ ʀ"]), "der": ("d eːɐ", ["d ə ʀ", "d eː ʀ", "d ɐ"]),
    "den": ("d ə n", ["d eː n"]), "und": ("ʊ n d", ["ʊ n t", "ʊ n"]), "in": ("ɪ n", ["iː n"]),
    "im": ("ɪ m", ["iː m"]), "an": ("a n", ["aː n"]), "für": ("f yːɐ", ["f yː ʀ", "f yː"]),
    "war": ("v aːɐ", ["v ɑː ʀ"]), "werden": ("v eː1 ʀ d ə n", ["v ə ʀ d ə n", "v eːɐ d n"]),
}


FINAL_E = True
LETTER_PRON = {"#aa": "ɑː1", "#ee": "eː1", "#ii": "iː1", "#oo": "oː1", "#uu": "uː1", "#be": "b eː1", "#ce": "ts eː1", "#de": "d eː1", "#ef": "ɛ1 f", "#ge": "g eː1", "#ha": "h ɑː1",
               "#jot": "j ɔ1 t", "#ka": "k ɑː1", "#el": "ɛ1 l", "#em": "ɛ1 m", "#en": "ɛ1 n", "#pe": "p eː1",
               "#ku": "k uː1", "#er": "ɛ1 ʀ", "#es": "ɛ1 s", "#te": "t eː1", "#vau": "f au1", "#we": "v eː1",
               "#ix": "ɪ1 k s", "#zet": "ts ɛ1 t", "#ä": "ɛː1", "#ö": "øː1", "#ü": "yː1"}


class German(EuG2P):
    lang = "de"
    lex_name = "de"
    decimal_comma = True
    thousand_space = True
    ordinals = ""
    num_lang = "de"
    split_min_len = 7
    split_min_piece = 3
    split_primary = "first"      # German compounds: primary stress on the first element
    split_links = ("s", "es", "n", "en", "e")
    link_phones = {"s": ["s"], "es": ["ə", "s"], "n": ["n"], "en": ["ə", "n"], "e": ["ə"]}
    letters = {"a": "#aa", "b": "#be", "c": "#ce", "d": "#de", "e": "#ee", "f": "#ef", "g": "#ge", "h": "#ha", "i": "#ii",
               "j": "#jot", "k": "#ka", "l": "#el", "m": "#em", "n": "#en", "o": "#oo", "p": "#pe", "q": "#ku", "r": "#er",
               "s": "#es", "t": "#te", "u": "#uu", "v": "#vau", "w": "#we", "x": "#ix", "y": "#ypsilon", "z": "#zet",
               "ä": "#ä", "ö": "#ö", "ü": "#ü"}
    symbols = {"%": "prozent", "&": "und", "$": "dollar", "€": "euro", "+": "plus", "=": "gleich",
               "@": "at", "°": "grad", "§": "paragraf"}
    abbrev = {"z.b": "zum beispiel", "d.h": "das heißt", "u.a": "unter anderem", "usw": "und so weiter",
              "bzw": "beziehungsweise", "ca": "circa", "dr": "doktor", "nr": "nummer", "km": "kilometer",
              "kg": "kilogramm", "cm": "zentimeter", "mio": "millionen", "mrd": "milliarden", "st": "sankt",
              "prof": "professor", "hr": "herr", "fr": "frau", "evtl": "eventuell", "ggf": "gegebenenfalls"}

    def rewrite(self, phones, word):
        """ZIPA conventions on FLEURS de_de: syllabic n̩/m̩/l̩ written ən/əm/əl (69 %), r as ʀ,
        vocalic ɐ after a vowel / in -er mostly as ʀ / əʀ, tense vowels written long even when
        unstressed (poːliːtsaɪ)."""
        out: list[str] = []
        for t in phones:
            ipa, st = C.parse_phone(t)
            sd = "" if st is None else str(st)
            if ipa in ("n̩", "m̩", "l̩"):
                out += ["ə" + sd, ipa[0]]
                continue
            if ipa in ("ʁ", "r"):
                out.append("ʀ")
                continue
            if ipa == "ɐ":
                out += ["ə" + sd, "ʀ"]
                continue
            if len(ipa) > 1 and ipa.endswith("ɐ") and ipa[0] in "aeiouyøɛɪʊɔʏœ":
                if "ː" in ipa:                       # long vowel + ɐ stays (deːɐ, vaːɐ)
                    out.append(ipa.replace("aː", "aː") + sd)
                else:
                    out += [ipa[:-1] + sd, "ʀ"]
                continue
            if ipa in ("i", "e", "o", "u", "y", "ø"):
                ipa = ipa + "ː"
            ipa = {"aɪ": "ai", "aʊ": "au", "aː": "ɑː"}.get(ipa, ipa)
            out.append(ipa + sd)
        # word-final schwa (-e): ZIPA writes eː (49 %) more often than ə (37 %)
        if FINAL_E and out and out[-1] == "ə" and len(out) > 2 and word[-1:] == "e":
            out[-1] = "eː"
        return out

    def lookup(self, w):
        if w in LETTER_PRON:
            return [LETTER_PRON[w].split()]
        if w in FUNC_PRON:
            return [FUNC_PRON[w][0].split()]
        r = super().lookup(w)
        if r is None and len(w) == 1 and w in self.letters:
            r = super().lookup(self.letters[w])
        return r

    def _pron_uncached(self, norm):
        # round 19: a hyphenated word whose pieces are not all in the lexicon is read piece by
        # piece (each piece: lexicon, compound split or letter rules) instead of sending the
        # whole string, hyphen included, through the letter rules ("kivu-see" -> kivu + see)
        if TEXT_R19 and "-" in norm and self.lookup_compound(norm) is None:
            parts = [p for p in norm.split("-") if p]
            if len(parts) > 1:
                out, oov = [], False
                for p in parts:
                    prons, o, _ = self._cache(p)
                    if not prons:
                        break
                    out.extend(prons[0])
                    oov |= o
                else:
                    return (tuple(out),), oov, "pieces"
        return super()._pron_uncached(norm)

    def _english(self):
        if self._en is None:
            from .en import build as _en_build

            self._en = _en_build()
        return self._en

    _en = None

    def english_variant(self, tok_text, items):
        """Round 19: English reading of a word the German lexicon does not know (letter rules,
        compound split of unknown pieces, hyphen pieces), when the English lexicon has it:
        names and English loans are common in German prompts and are said the English way."""
        if not TEXT_R19 or len(items) != 1:
            return None
        hows = [self._cache(p)[2] for p in [items[0]]]
        if not any(h in ("rules", "split", "pieces") for h in hows):
            return None
        en = self._english()
        norm = en.normalise_word(tok_text.strip(".,;:!?\"'„“”»«()"))
        pieces = [p for p in norm.split("-") if p]
        out = []
        for p in pieces:
            prons, oov, how = en._cache(p)
            if not prons or oov or how != "lex":
                return None
            out.extend(C.parse_phone(x)[0] for x in prons[0])
        return out or None

    def build_word(self, tok, items, warnings):
        w = super().build_word(tok, items, warnings)
        if not w.phones:
            return w
        ev = self.english_variant(tok.text, items)
        if ev:
            C.add_word_variant(w, ev, C.P_MED)
        single = len(items) == 1
        if single and items[0] in FUNC:
            if len(w.syls) == 1:
                w.syls = [(a, b, 0) for a, b, _ in w.syls]
        elif w.syls and not any(st == 1 for _, _, st in w.syls):
            # ipa-dict leaves monosyllables unmarked; compounds of unmarked pieces: first syllable
            a, b, _ = w.syls[0]
            w.syls[0] = (a, b, 1)
        if single and items[0] in WEAK:
            for alt in WEAK[items[0]]:
                C.add_word_variant(w, alt.split(), C.P_MED)
        if single and items[0] in FUNC_PRON:
            for alt in FUNC_PRON[items[0]][1]:
                C.add_word_variant(w, alt.split(), C.P_COMMON)
        ph = w.phones
        n = len(ph)
        for k, p in enumerate(ph):
            nxt = ph[k + 1] if k + 1 < n else None
            # -en / -em / -el: schwa + sonorant ~ syllabic sonorant
            if p.ipa == "ə" and nxt is not None and nxt.ipa in ("n", "m", "l") and (k + 2 == n or not ph[k + 2].nucleus):
                p.add_alt("", C.P_COMMON)
            # r: uvular trill / fricative / tap; vocalised (ɐ) or lost after a vowel
            if p.ipa == "ʀ":
                for a in ("ʁ", "ɾ", "r"):
                    p.add_alt(a, C.P_COMMON)
                if k > 0 and ph[k - 1].nucleus and (nxt is None or not nxt.nucleus):
                    p.add_alt("ɐ", C.P_COMMON)
                    p.add_alt("", C.P_COMMON)
                    if ph[k - 1].ipa == "ə":
                        ph[k - 1].add_alt("ɐ", C.P_COMMON)
            if p.nucleus and p.ipa.endswith("ː") and p.ipa[0] in "ieouyø" and not any(
                    a <= k < b and st == 1 for a, b, st in w.syls):
                p.add_alt(p.ipa[:-1], C.P_COMMON)      # unstressed tense vowels: short
            if p.ipa == "z":
                p.add_alt("s", C.P_COMMON)              # southern voiceless s
            alt = {"ai": "aɪ", "au": "aʊ", "ɑː": "aː", "ç": "x"}.get(p.ipa)
            if alt:
                p.add_alt(alt, C.P_COMMON)
            if p.ipa == "eː" and k == n - 1 and p.nucleus and w.syls[-1][2] == 0:
                p.add_alt("ə", C.P_COMMON)
            if p.nucleus and p.ipa.endswith("ɐ") and len(p.ipa) > 1:
                p.add_alt(p.ipa[:-1] + "ʀ", C.P_COMMON)
                p.add_alt(p.ipa[:-1], C.P_MED)
            if p.ipa in ("t", "d") and k == n - 1 and k > 0 and ph[k - 1].ipa in ("n", "l", "ʀ"):
                p.add_alt("d" if p.ipa == "t" else "t", C.P_COMMON)
            if p.ipa == "ʃ" and k + 1 < n and ph[k + 1].ipa in ("t", "p") and (k == 0 or not ph[k - 1].nucleus):
                p.add_alt("s", C.P_MED)
        # optional glottal stop before a word-initial vowel
        if ph[0].nucleus:
            ph[0].add_alt("ʔ" + ph[0].ipa, C.P_MED)
        return w

    def connected(self, words):
        for i in range(len(words) - 1):
            w, nx = words[i], words[i + 1]
            # word-final t after a consonant before a consonant ("ist das", "nicht so") may drop
            last = w.phones[-1]
            if last.ipa == "t" and len(w.phones) > 1 and not w.phones[-2].nucleus and not nx.phones[0].nucleus:
                last.add_alt("", C.P_MED)


def build() -> German:
    return German()
