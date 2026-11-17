"""US English text front-end.

Lexicon: CMUdict (BSD-2-Clause), converted to ZIPA tokens, stress from CMUdict digits.
OOV: learned letter-context table trained on the same lexicon (pronunciada/text/_eu_lts.py).
"""
from __future__ import annotations

from . import _eu_common as C
from ._eu_base import EuG2P

LETTERS = {
    "a": "a.", "b": "b.", "c": "c.", "d": "d.", "e": "e.", "f": "f.", "g": "g.", "h": "h.", "i": "i.",
    "j": "j.", "k": "k.", "l": "l.", "m": "m.", "n": "n.", "o": "o.", "p": "p.", "q": "q.", "r": "r.",
    "s": "s.", "t": "t.", "u": "u.", "v": "v.", "w": "w.", "x": "x.", "y": "y.", "z": "z.",
}

# function words: canonical (weak) form in ARPAbet-free ZIPA phones, strong/other forms as alts.
# (pron, prior). The first entry is canonical. Monosyllabic function words get stress 0.
WEAK: dict[str, list[tuple[str, float]]] = {
    # canonical = ZIPA's modal form on FLEURS en_us ("a" [ɪ] 61 %, "an" [æn] 84 %, "was" [wɑz] 88 %)
    "a": [("ɪ", 0), ("ə", C.P_COMMON), ("eɪ", C.P_LOW)],
    "an": [("æ n", 0), ("ə n", C.P_COMMON), ("ɪ n", C.P_MED)],
    "with": [("w ɪ θ", 0), ("w ɪ ð", C.P_COMMON)],
    "on": [("ɔ n", 0), ("ɑ n", C.P_COMMON)],
    "the": [("ð ə", 0), ("ð i", C.P_MED)],
    "to": [("t u", 0), ("t ə", C.P_COMMON)],
    "be": [("b i", 0), ("b ɪ", C.P_LOW)],
    "do": [("d u", 0), ("d ə", C.P_LOW)],
    "of": [("ə v", 0), ("ʌ v", C.P_MED), ("ə", C.P_MED)],
    "and": [("ə n d", 0), ("ə n", C.P_COMMON), ("æ n d", C.P_LOW), ("n", C.P_LOW)],
    "for": [("f ɔ ɹ", 0), ("f ɜ˞", C.P_COMMON), ("f ə˞", C.P_COMMON), ("f ɹ ɜ˞", C.P_COMMON)],
    "from": [("f ɹ ə m", 0), ("f ɹ ʌ m", C.P_MED)],
    "at": [("æ t", 0), ("ə t", C.P_MED)],
    "as": [("æ z", 0), ("ə z", C.P_MED)],
    "can": [("k æ n", 0), ("k ə n", C.P_COMMON)],
    "have": [("h æ v", 0), ("h ə v", C.P_MED), ("ə v", C.P_MED)],
    "has": [("h æ z", 0), ("h ə z", C.P_MED), ("ə z", C.P_MED)],
    "had": [("h æ d", 0), ("h ə d", C.P_MED), ("ə d", C.P_MED)],
    "was": [("w ɑ z", 0), ("w ə z", C.P_COMMON), ("w ʌ z", C.P_MED)],
    "were": [("w ɜ˞", 0), ("w ə˞", C.P_COMMON)],
    "you": [("j u", 0), ("j ə", C.P_MED)],
    "your": [("j ɔ ɹ", 0), ("j ɜ˞", C.P_COMMON), ("j ʊ ɹ", C.P_MED)],
    "them": [("ð ɛ m", 0), ("ð ə m", C.P_COMMON), ("ə m", C.P_LOW)],
    "some": [("s ʌ m", 0), ("s ə m", C.P_COMMON)],
    "than": [("ð æ n", 0), ("ð ə n", C.P_COMMON)],
    "that": [("ð æ t", 0), ("ð ə t", C.P_MED)],
    "but": [("b ʌ t", 0), ("b ə t", C.P_COMMON)],
    "or": [("ɔ ɹ", 0), ("ɜ˞", C.P_MED)],
    "are": [("ɑ ɹ", 0), ("ɜ˞", C.P_MED)],
    "does": [("d ʌ z", 0), ("d ə z", C.P_COMMON)],
    "just": [("dʒ ʌ s t", 0), ("dʒ ə s t", C.P_COMMON)],
    "there": [("ð ɛ ɹ", 0), ("ð ɜ˞", C.P_MED)],
    "he": [("h i", 0), ("i", C.P_MED)],
    "him": [("h ɪ m", 0), ("ɪ m", C.P_MED), ("ə m", C.P_LOW)],
    "his": [("h ɪ z", 0), ("ɪ z", C.P_MED)],
    "her": [("h ɜ˞", 0), ("h ə˞", C.P_COMMON), ("ɜ˞", C.P_MED)],
    "us": [("ʌ s", 0), ("ə s", C.P_MED)],
    "into": [("ɪ n t u", 0), ("ɪ n t ə", C.P_COMMON)],
    "by": [("b aɪ", 0), ("b ə", C.P_LOW)],
    "my": [("m aɪ", 0), ("m ə", C.P_LOW)],
}
FUNCTION = set(WEAK) | {
    "in", "on", "is", "it", "its", "it's", "with", "be", "been", "this", "these", "those", "they",
    "we", "she", "not", "if", "so", "no", "do", "did", "will", "would", "should", "could", "shall",
    "may", "might", "must", "who", "whom", "which", "what", "when", "than", "then", "up", "out",
    "all", "any", "each", "our", "their", "me", "nor", "off", "per", "via", "am", "upon", "own",
    "such", "how", "where", "why", "both", "while", "about", "over", "after", "before", "under",
}


ER_TO_STRESSED = True
H_WEAK = {"he", "him", "his", "her", "have", "has", "had", "hers", "himself", "herself"}
VOWEL_LIKE = lambda ipa: C.is_nucleus_ipa(ipa)  # noqa: E731
SONORANT_CODA = {"ɹ", "ɜ˞", "l", "l̴", "n"}


class English(EuG2P):
    lang = "en"
    lex_name = "en"
    decimal_comma = False
    thousand_space = False
    ordinals = "st|nd|rd|th|s|'s"
    num_lang = "en"
    letters = LETTERS
    symbols = {"%": "percent", "&": "and", "$": "dollars", "€": "euros", "£": "pounds", "+": "plus",
               "=": "equals", "@": "at", "°": "degrees", "§": "section"}
    abbrev = {"mr": "mister", "mrs": "missus", "ms": "miz", "dr": "doctor", "st": "saint", "vs": "versus",
              "etc": "et cetera", "jr": "junior", "sr": "senior", "prof": "professor", "mt": "mount",
              "no": "number", "approx": "approximately", "e.g": "for example", "i.e": "that is",
              "km": "kilometers", "kg": "kilograms", "cm": "centimeters", "mm": "millimeters"}

    def rewrite(self, phones, word):
        """ZIPA conventions measured on FLEURS en_us: stressed ʌ is heard as ə (62 %),
        r-colored schwa as ɜ˞ (54 %), coda l as velarised l̴ (~55 %)."""
        out = []
        n = len(phones)
        for i, t in enumerate(phones):
            ipa, st = C.parse_phone(t)
            if ipa == "ʌ":
                ipa = "ə"
            elif ipa == "ə˞" and ER_TO_STRESSED:
                ipa = "ɜ˞"
            elif ipa == "l" and (i == n - 1 or not C.is_nucleus_ipa(C.parse_phone(phones[i + 1])[0])):
                ipa = "l̴"
            out.append(ipa + ("" if st is None else str(st)))
        return out

    def lookup(self, w: str):
        if w in WEAK:
            return [WEAK[w][0][0].split()]
        r = super().lookup(w)
        if r is None and w.endswith("'s") and len(w) > 2:
            b = super().lookup(w[:-2])
            if b:
                last = C.parse_phone(b[0][-1])[0]
                suf = ["ɪ", "z"] if last in ("s", "z", "ʃ", "ʒ", "tʃ", "dʒ") else (
                    ["s"] if last in ("p", "t", "k", "f", "θ") else ["z"])
                return [b[0] + suf]
        if r is None and w.endswith("'") and len(w) > 1:
            return super().lookup(w[:-1])
        return r

    def build_word(self, tok, items, warnings):
        w = super().build_word(tok, items, warnings)
        if len(items) == 1 and items[0] in FUNCTION and len(w.syls) == 1:
            w.syls = [(a, b, 0) for a, b, _ in w.syls]
        if not w.phones:
            return w
        if len(items) == 1 and items[0] in WEAK:
            for pron, prior in WEAK[items[0]][1:]:
                C.add_word_variant(w, self.rewrite(pron.split(), items[0]), prior)
        ph = w.phones
        stress_of = {}
        for (a, b, st) in w.syls:
            for k in range(a, b):
                stress_of[k] = st
                p = ph[k]
                # convention alternatives: both realisations are native
                if p.ipa == "l̴":
                    p.add_alt("l", C.P_COMMON)
                elif p.ipa == "l":
                    p.add_alt("l̴", C.P_COMMON)
                elif p.ipa == "ɜ˞":
                    p.add_alt("ə˞", C.P_COMMON)
                elif p.ipa == "ə" and st:
                    p.add_alt("ʌ", C.P_MED)
                elif p.ipa == "ɔ":                      # cot-caught merger
                    p.add_alt("ɑ", C.P_MED)
        # syllabic consonants: unstressed ə before word-final n / l / m (button, bottle)
        if len(ph) >= 3 and ph[-2].ipa == "ə" and ph[-1].ipa in ("n", "l̴", "l", "m") \
                and not ph[-3].nucleus and w.syls[-1][2] == 0:
            ph[-2].add_alt("", C.P_COMMON)
        # -ing -> -in'
        if len(items) == 1 and items[0].endswith("ing") and len(ph) >= 3 and ph[-1].ipa == "ŋ" \
                and ph[-2].ipa == "ɪ":
            ph[-1].add_alt("n", C.P_LOW)
        # flapping inside the word: t/d between a vowel (or r) and an unstressed vowel
        for k in range(1, len(ph) - 1):
            if ph[k].ipa in ("t", "d") and (ph[k - 1].nucleus or ph[k - 1].ipa in ("ɹ", "n")) \
                    and ph[k + 1].nucleus and not stress_of.get(k + 1):
                ph[k].add_alt("ɾ", C.P_MED)
                if ph[k].ipa == "t":
                    ph[k].add_alt("d", C.P_MED)
        return w

    def connected(self, words):
        for i, w in enumerate(words):
            nxt = words[i + 1] if i + 1 < len(words) else None
            last = w.phones[-1]
            before_v = nxt is not None and nxt.phones[0].nucleus
            before_c = nxt is not None and not nxt.phones[0].nucleus
            # flapping across the boundary: "get it", "what a"
            if last.ipa in ("t", "d") and len(w.phones) > 1 and before_v and \
                    (w.phones[-2].nucleus or w.phones[-2].ipa in ("ɹ", "n")):
                last.add_alt("ɾ", C.P_MED)
                if last.ipa == "t":
                    last.add_alt("d", C.P_MED)
            # t/d dropping in final clusters before a consonant ("mos(t) people") / at the end
            if last.ipa in ("t", "d") and len(w.phones) > 1 and not w.phones[-2].nucleus \
                    and w.phones[-2].ipa != "ɹ":
                last.add_alt("", C.P_MED if before_c else C.P_LOW)
            # glottal(ised) t before a consonant
            if last.ipa == "t" and before_c:
                last.add_alt("ʔ", C.P_LOW)
            items = w.info.get("items", [])
            # h-dropping in weak pronouns / auxiliaries (not utterance-initial)
            if i > 0 and len(items) == 1 and items[0] in H_WEAK and w.phones[0].ipa == "h":
                w.phones[0].add_alt("", C.P_COMMON)
            # "the" before a vowel: [ði]
            if items == ["the"] and before_v:
                w.phones[-1].add_alt("i", C.P_COMMON)


def build() -> English:
    return English()
