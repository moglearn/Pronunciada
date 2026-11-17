"""Italian text front-end.

Lexicon: WikiPron Italian (Wiktionary, CC BY-SA), stress added offline: written final accent,
else the syllable with ɛ/ɔ (only possible under stress), else a suffix model trained on those
words (~90 % on words of known stress). OOV: learned letter-context table + the same suffix model.
"""
from __future__ import annotations

from . import _eu_common as C
from ._eu_base import EuG2P

# elided clitics (written with an apostrophe) -> phones
ELIDED = {
    "l": "l", "dell": "d e l l", "all": "a l l", "dall": "d a l l", "nell": "n e l l", "sull": "s u l l",
    "coll": "k o l l", "un": "u n", "quest": "k w e s t", "quell": "k w e l l", "c": "tʃ", "d": "d",
    "s": "s", "m": "m", "t": "t", "v": "v", "n": "n", "anch": "a n k", "cent": "tʃ e n t",
    "sant": "s a n t", "bell": "b ɛ l l", "tutt": "t u t t", "nessun": "n e s s u n", "buon": "b w ɔ n",
    "grand": "g r a n d", "senz": "s ɛ n ts", "com": "k o m", "dov": "d o v", "ch": "k", "gl": "ʎ",
    "qual": "k w a l", "po": "p ɔ1", "mo": "m ɔ1", "be": "b ɛ1", "di": "d i", "de": "d e",
}
FUNC = {"il", "lo", "la", "i", "gli", "le", "l'", "un", "uno", "una", "di", "a", "da", "in", "con", "su",
        "per", "tra", "fra", "e", "ed", "o", "od", "ma", "che", "se", "non", "mi", "ti", "si", "ci", "vi",
        "ne", "del", "al", "dal", "nel", "sul", "col", "dei", "ai", "dai", "nei", "sui", "coi", "degli",
        "agli", "dagli", "negli", "sugli", "ad"}
# words triggering raddoppiamento sintattico (plus every word with final stress)
RS_TRIGGER = {"a", "e", "è", "da", "che", "tra", "fra", "più", "già", "può", "ma", "o", "se", "so", "sto",
              "va", "fa", "fu", "qua", "là", "lì", "sì", "né", "chi", "tre", "re", "blu", "su", "come",
              "dove", "qualche", "sopra", "contra"}
OPEN = {"e": "ɛ", "o": "ɔ", "ɛ": "e", "ɔ": "o"}

# Stress (number of nuclei from the end: 0 final, 1 penult, 2 antepenult) for words without
# lexical evidence. Default penultimate; curated suffixes that are (almost) always antepenultimate.
ANTEPENULT_SUFFIXES = (
    "abile", "abili", "ibile", "ibili", "evole", "evoli", "esimo", "esima", "esimi", "esime",
    "issimo", "issima", "issimi", "issime", "errimo", "errima", "ologo", "ologa", "ologi", "ologhe",
    "ofono", "ofona", "ofoni", "ografo", "ografa", "ografi", "agine", "agini", "igine", "igini",
    "udine", "udini", "aggine", "aggini", "ico", "ica", "ici", "iche", "ichi", "ide", "idi",
    "ifero", "ifera", "ivoro", "ivora", "metro", "metri", "ssero", "ssimo", "rono", "ttero",
)
PENULT_EXCEPT = {"amico", "amica", "amici", "amiche", "antico", "antica", "antichi", "antiche", "fatica",
                 "fatiche", "formica", "formiche", "pudico", "mendico", "lombrico", "ombelico", "panico",
                 "chimico", "arrivederci", "dieci", "felice", "radice", "provincia", "provincie",
                 "nemico", "nemica", "nemici", "nemiche", "tedesco", "ventidue", "ventitré"}
ANTEPENULT_EXCEPT = {"panico", "chimico"}


def stress_from_end(word: str, n_nuclei: int) -> int:
    if n_nuclei <= 1:
        return 0
    if word[-1:] in "àèéìòóù":
        return 0
    if word in ANTEPENULT_EXCEPT and n_nuclei >= 3:
        return 2
    if word in PENULT_EXCEPT:
        return 1
    if n_nuclei >= 3:
        for suf in ANTEPENULT_SUFFIXES:
            if word.endswith(suf):
                before = word[-len(suf) - 1: -len(suf)]
                if suf.startswith("ic") and (not before or before in "aeiouc"):
                    continue          # -aico, -eico (laico), -cico are not reliable
                return 2
    return 1


FUNC_PRON = {
    "la": "l a", "lo": "l o", "le": "l e", "il": "i l", "i": "i", "gli": "ʎ i", "un": "u n", "uno": "u1 n o",
    "di": "d i", "da": "d a", "in": "i n", "con": "k o n", "su": "s u", "per": "p e r", "tra": "t r a",
    "fra": "f r a", "e": "ɛ", "o": "o", "ma": "m a", "che": "k e", "se": "s e", "non": "n o n",
    "mi": "m i", "ti": "t i", "si": "s i", "ci": "tʃ i", "vi": "v i", "ne": "n e", "ed": "e d",
    "ad": "a d", "è": "ɛ1",
}
for _p, _q in (("d", "d e"), ("a", "a"), ("da", "d a"), ("n", "n e"), ("s", "s u")):
    _b = {"d": "de", "a": "a", "da": "da", "n": "ne", "s": "su"}[_p]
    _ph = {"d": "d e", "a": "a", "da": "d a", "n": "n e", "s": "s u"}[_p]
    FUNC_PRON.update({_b + "l": _ph + " l", _b + "lla": _ph + " l1 l a".replace("l1", "l"),
                      _b + "llo": _ph + " l l o", _b + "lle": _ph + " l l e", _b + "i": _ph + " i",
                      _b + "gli": _ph + " ʎ i", _b + "ll'": _ph + " l l"})
FUNC_PRON.update({"col": "k o l", "coi": "k o i", "sull'": "s u l l", "nell'": "n e l l"})
# ZIPA hears the articulated prepositions de- / ne- with open ɛ (FLEURS it_it)
for _k in [k for k in FUNC_PRON if k[:2] in ("de", "ne") and k not in ("di",)]:
    FUNC_PRON[_k] = FUNC_PRON[_k].replace(" e", " ɛ", 1)
# stress on the first vowel of the two-syllable articulated prepositions (della, dello ...)
for _k, _v in list(FUNC_PRON.items()):
    _t = _v.split()
    _nuc = [i for i, x in enumerate(_t) if x[0] in "aeiouɛɔ"]
    if len(_nuc) == 2 and _k not in ("uno",):
        _t[_nuc[0]] += "1"
        FUNC_PRON[_k] = " ".join(_t)


class Italian(EuG2P):
    lang = "it"
    measured_mid = False           # round 16: see the open / close mid vowels in build_word
    lex_name = "it"
    decimal_comma = True
    thousand_space = True
    ordinals = "º|ª|°"
    num_lang = "it"
    letters = {"a": "a", "b": "bi", "c": "ci", "d": "di", "e": "e", "f": "effe", "g": "gi", "h": "acca",
               "i": "i", "j": "i lunga", "k": "cappa", "l": "elle", "m": "emme", "n": "enne", "o": "o",
               "p": "pi", "q": "cu", "r": "erre", "s": "esse", "t": "ti", "u": "u", "v": "vu",
               "w": "doppia vu", "x": "ics", "y": "ipsilon", "z": "zeta"}
    symbols = {"%": "per cento", "&": "e", "$": "dollari", "€": "euro", "+": "più", "=": "uguale",
               "@": "chiocciola", "°": "gradi"}
    abbrev = {"sig": "signor", "sigg": "signori", "sig.ra": "signora", "dott": "dottor", "prof": "professor",
              "ecc": "eccetera", "km": "chilometri", "kg": "chilogrammi", "cm": "centimetri",
              "n": "numero", "pag": "pagina", "st": "san", "s": "san", "usa": "u esse a"}

    def expand_token(self, tok, warnings):
        items = super().expand_token(tok, warnings)
        if items and len(items) == 1 and "'" in items[0] and items[0] not in self.lex:
            head, _, tail = items[0].partition("'")
            return [head + "'"] + ([tail] if tail else [])
        return items

    inflect_finals = (("i", ("o", "e", "a")), ("e", ("a", "o")), ("a", ("o", "e")), ("o", ("a", "e")),
                      ("he", ("a",)), ("hi", ("o",)))
    inflect_phones = {"a": ["a"], "o": ["o"], "e": ["e"], "i": ["i"], "he": ["e"], "hi": ["i"]}

    def fix_stress(self, phones, word=""):
        out = [C.parse_phone(p)[0] for p in phones]
        nuc = [i for i, p in enumerate(out) if C.is_nucleus_ipa(p)]
        if nuc:
            out[nuc[len(nuc) - 1 - min(len(nuc) - 1, stress_from_end(word, len(nuc)))]] += "1"
        return out

    def rewrite(self, phones, word):
        """ZIPA conventions on FLEURS it_it: ʎ ɲ ʃ (always long between vowels) are written
        single, geminate affricates as one affricate."""
        out = []
        for t in phones:
            ipa, st = C.parse_phone(t)
            if ipa == "ɔ":                       # ZIPA writes o for 62 % of open ɔ
                t = "o" + ("" if st is None else str(st))
            if out and ipa in ("ʎ", "ɲ", "ʃ") and C.parse_phone(out[-1])[0] == ipa:
                continue
            if out and ipa in ("ts", "dz", "tʃ", "dʒ") and C.parse_phone(out[-1])[0] == ipa[0]:
                out.pop()
            out.append(t)
        return out

    def lookup(self, w):
        if w in FUNC_PRON:
            return [FUNC_PRON[w].split()]
        if w.endswith("'") and len(w) > 1:
            head = w[:-1]
            if head in ELIDED:
                return [ELIDED[head].split()]
            for v in ("", "o", "a", "e", "i"):      # elided form of a known word: drop final vowel
                r = super().lookup(head + v)
                if r:
                    ph = [C.parse_phone(x)[0] for x in r[0]]
                    if v and C.is_nucleus_ipa(ph[-1]):
                        ph = ph[:-1]
                    if any(C.is_nucleus_ipa(x) for x in ph):
                        return [ph]
            return None
        return super().lookup(w)

    def build_word(self, tok, items, warnings):
        w = super().build_word(tok, items, warnings)
        if not w.phones:
            return w
        if len(items) == 1 and items[0] in FUNC and len(w.syls) == 1:
            w.syls = [(a, b, 0) for a, b, _ in w.syls]
        ph = w.phones
        stressed = set()
        for a, b, st in w.syls:
            if st == 1:
                stressed.update(range(a, b))
        for k, p in enumerate(ph):
            # open/closed e o vary regionally in stressed syllables
            # Round 16 (MEASURED_MID): natives use the partner for stressed e 26 %, ɛ 32-40 %,
            # unstressed e 4 %, stressed o 13 % (bench/calib/alt_audit.py, FLEURS validation);
            # alternatives under 15 % (the round-8 bar) are dropped.
            if p.nucleus and p.ipa[0] in OPEN and (k in stressed or p.ipa[0] in "eɛ"):
                if not (self.measured_mid and ((p.ipa[0] == "e" and k not in stressed) or p.ipa[0] == "o")):
                    p.add_alt(OPEN[p.ipa[0]] + p.ipa[1:], C.P_COMMON if k in stressed else C.P_MED)
            # intervocalic s voicing varies (north z, centre-south s)
            if p.ipa in ("s", "z") and 0 < k < len(ph) - 1 and ph[k - 1].nucleus and ph[k + 1].nucleus:
                p.add_alt("z" if p.ipa == "s" else "s", C.P_COMMON)
            # z: ts ~ dz
            if p.ipa in ("ts", "dz"):
                p.add_alt("dz" if p.ipa == "ts" else "ts", C.P_MED)
            # geminates may be realised short in fast speech
            if k + 1 < len(ph) and not p.nucleus and ph[k + 1].ipa == p.ipa:
                p.add_alt("", C.P_LOW)
        w.info["final_stress"] = bool(w.syls) and w.syls[-1][2] == 1 and len(w.syls) > 1
        return w

    def connected(self, words):
        for i in range(len(words) - 1):
            w, nx = words[i], words[i + 1]
            items = w.info.get("items", [])
            first = nx.phones[0]
            # raddoppiamento sintattico: optional lengthening of the next word's initial consonant
            if (items and items[-1] in RS_TRIGGER or w.info.get("final_stress")) and not first.nucleus \
                    and len(nx.phones) > 1 and (nx.phones[1].nucleus or nx.phones[1].ipa in ("r", "l", "j", "w")) \
                    and first.ipa not in ("j", "w", "ʎ", "ɲ", "ʃ"):
                first.add_alt(first.ipa[0] + first.ipa, C.P_MED)
            # vowel sandhi: final unstressed vowel before a vowel may be elided
            last = w.phones[-1]
            if last.nucleus and first.nucleus and w.syls[-1][2] != 1:
                last.add_alt("", C.P_MED)


def build() -> Italian:
    return Italian()
