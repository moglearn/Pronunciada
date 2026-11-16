"""French text front-end (no word stress: RefSyllable.stress = None).

Lexicon: ipa-dict fr_FR (MIT). OOV: learned letter-context table. Elided clitics (l' d' qu' ...)
are part of the following word. Liaison: obligatory contexts in the canonical form, optional ones
as a liaison consonant with a "" alternative; schwa elision as "" alternatives.
"""
from __future__ import annotations

from . import _eu_common as C
from ._eu_base import EuG2P

ELIDED = {"l": "l", "d": "d", "j": "ʒ", "m": "m", "t": "t", "s": "s", "n": "n", "c": "s", "ç": "s",
          "qu": "k", "jusqu": "ʒ y s k", "lorsqu": "l ɔ ʁ s k", "puisqu": "p ɥ i s k", "quoiqu": "k w a k",
          "presqu": "p ʁ ɛ s k", "quelqu": "k ɛ l k"}
# homographs / clitics. ZIPA's French labels follow the first reading of CharsiuG2P "fra"
# (FLEURS fr_fr: "est" heard [ɛst] 77 %, "un" [yn] 84 %); the natural forms are alternatives.
FUNC_PRON = {"est": "ɛ s t", "plus": "p l y", "tous": "t u", "fils": "f i s", "as": "a", "couvent": "k u v",
             "t": "t", "-t-": "t", "ce": "s ə", "que": "k ə", "de": "d ə", "le": "l ə", "je": "ʒ ə",
             "me": "m ə", "te": "t ə", "se": "s ə", "ne": "n ə", "un": "y n", "en": "ɑ̃", "dix": "d i s",
             "six": "s i s", "huit": "ɥ i t", "cent": "s ɑ̃", "vingt": "v ɛ̃", "neuf": "n œ f", "os": "ɔ s"}
LATENT = {"s": "z", "x": "z", "z": "z", "t": "t", "d": "t", "n": "n", "p": "p", "r": "ʁ"}
OBLIGATORY = {"les", "des", "ces", "mes", "tes", "ses", "nos", "vos", "leurs", "aux", "un", "aucun", "mon",
              "ton", "son", "quels", "quelles", "certains", "certaines", "plusieurs", "deux", "trois",
              "six", "dix", "nous", "vous", "ils", "elles", "on", "en", "dans", "chez", "sans", "sous",
              "très", "tout", "quand", "dont", "bien", "rien", "aucuns", "toutes", "tous", "grands",
              "petits", "grand", "petit", "gros", "premier", "dernier", "vingt", "cent", "c'est", "lesquels"}
OPTIONAL = {"est", "sont", "ont", "était", "étaient", "avait", "avaient", "sera", "seront", "pas", "plus",
            "mais", "toujours", "jamais", "après", "avant", "depuis", "assez", "trop", "beaucoup", "souvent",
            "peu", "font", "vont", "doit", "peut", "fait", "étant", "comment", "alors", "pendant", "devant"}
H_ASPIRE = {"haut", "haute", "hauts", "hautes", "héros", "hibou", "hache", "haine", "hall", "hameau", "hamster",
            "hanche", "handicap", "hangar", "hareng", "haricot", "hasard", "hâte", "hausse", "hollande",
            "hongrois", "honte", "hors", "huit", "huitième", "hurler", "hockey", "hiérarchie", "hamac", "harpe",
            "hérisson", "héron", "hêtre", "hibou", "homard", "hublot", "huer", "hutte", "hongrie", "hauteur",
            "hollandais", "halte", "hareng", "honteux", "haïr", "harceler", "hardi"}
LETTER_PRON = {"#a": "a", "#bé": "b e", "#cé": "s e", "#dé": "d e", "#e": "ə", "#effe": "ɛ f", "#gé": "ʒ e",
               "#ache": "a ʃ", "#i": "i", "#ji": "ʒ i", "#ka": "k a", "#elle": "ɛ l", "#emme": "ɛ m", "#enne": "ɛ n",
               "#o": "o", "#pé": "p e", "#qu": "k y", "#erre": "ɛ ʁ", "#esse": "ɛ s", "#té": "t e", "#u": "y",
               "#vé": "v e", "#double": "d u b l", "#ixe": "i k s", "#grec": "g ʁ ɛ k", "#zède": "z ɛ d"}
CLITIC_SCHWA = {"le", "de", "ce", "je", "me", "te", "se", "ne", "que"}
VOWEL_LETTERS = set("aeiouyéèêëàâîïôûùœæ")


class French(EuG2P):
    lang = "fr"
    lex_name = "fr"
    default_stress = None
    decimal_comma = True
    thousand_space = True
    ordinals = "er|re|ère|e|ème|es|èmes"
    num_lang = "fr"
    letters = {"a": "#a", "b": "#bé", "c": "#cé", "d": "#dé", "e": "#e", "f": "#effe", "g": "#gé", "h": "#ache",
               "i": "#i", "j": "#ji", "k": "#ka", "l": "#elle", "m": "#emme", "n": "#enne", "o": "#o", "p": "#pé",
               "q": "#qu", "r": "#erre", "s": "#esse", "t": "#té", "u": "#u", "v": "#vé", "w": "#double #vé",
               "x": "#ixe", "y": "#i #grec", "z": "#zède"}
    symbols = {"%": "pour cent", "&": "et", "$": "dollars", "€": "euros", "+": "plus", "=": "égale",
               "@": "arobase", "°": "degrés"}
    abbrev = {"m": "monsieur", "mme": "madame", "mlle": "mademoiselle", "dr": "docteur", "st": "saint",
              "ste": "sainte", "etc": "et cetera", "km": "kilomètres", "kg": "kilogrammes", "cm": "centimètres",
              "env": "environ", "av": "avenue", "bd": "boulevard", "n°": "numéro"}

    def expand_token(self, tok, warnings):
        items = super().expand_token(tok, warnings)
        if items and len(items) == 1 and "'" in items[0] and items[0] not in self.lex:
            head, _, tail = items[0].partition("'")
            if head in ELIDED and tail:
                return [head + "'", tail]
        return items

    def lookup(self, w):
        if w in LETTER_PRON:
            return [LETTER_PRON[w].split()]
        if w in FUNC_PRON:
            return [FUNC_PRON[w].split()]
        if w.endswith("'") and w[:-1] in ELIDED:
            return [ELIDED[w[:-1]].split()]
        return super().lookup(w)

    def build_word(self, tok, items, warnings):
        w = super().build_word(tok, items, warnings)
        if not w.phones:
            return w
        ph = w.phones
        n = len(ph)
        if items == ["est"]:
            ph[1].add_alt("", C.P_COMMON)
            ph[2].add_alt("", C.P_COMMON)
        if items == ["un"]:
            ph[0].add_alt("œ̃", C.P_COMMON)
            ph[0].add_alt("ɛ̃", C.P_COMMON)
            ph[1].add_alt("", C.P_COMMON)
        # final orthographic -e / -es after a pronounced consonant: optional final schwa
        last_it = items[-1] if items else ""
        if last_it.endswith(("e", "es")) and not last_it.endswith(("ée", "ées", "ie", "ies", "ue", "ues")) \
                and n > 1 and not ph[-1].nucleus:
            ph[-1].add_alt(ph[-1].ipa + "ə", C.P_MED)
            ph[-1].add_alt(ph[-1].ipa + "ɛ", C.P_MED)
        for k, p in enumerate(ph):
            if p.ipa == "ə":
                # schwa elision (le, de, que ...; samedi, petit)
                last = items[-1] if items else ""
                p.add_alt("", C.P_COMMON if (len(items) == 1 and last in CLITIC_SCHWA) else C.P_MED)
            if p.ipa == "ʁ":
                p.add_alt("ʀ", C.P_COMMON)
                if k == n - 1 and k > 0 and not ph[k - 1].nucleus:
                    p.add_alt("", C.P_MED)                 # final cluster -tre, -bre simplification
            if p.ipa == "l" and k == n - 1 and k > 0 and not ph[k - 1].nucleus:
                p.add_alt("", C.P_MED)                     # table, possible
            alt = {"œ̃": "ɛ̃", "e": "ɛ", "ɛ": "e", "o": "ɔ", "ɔ": "o", "ø": "œ", "œ": "ø", "a": "ɑ",
                   "ɑ": "a"}.get(p.ipa)
            if alt:
                p.add_alt(alt, C.P_COMMON)
        # latent final consonant for liaison (orthography vs pronunciation)
        last_item = items[-1] if items else ""
        latent = None
        if last_item and last_item[-1] in LATENT:
            cons = LATENT[last_item[-1]]
            last_ph = ph[-1].ipa
            if last_ph != cons and (ph[-1].nucleus or last_item[-1] in "sxz"):
                if not (last_item[-1] == "n" and "̃" not in last_ph):
                    if not (last_item[-1] == "r" and last_item not in ("premier", "dernier", "léger")):
                        latent = cons
        w.info["latent"] = latent
        w.info["last_item"] = last_item
        return w

    def connected(self, words):
        for i in range(len(words) - 1):
            w, nx = words[i], words[i + 1]
            latent = w.info.get("latent")
            if not latent:
                continue
            first_item = (nx.info.get("items") or [""])[0]
            if first_item in H_ASPIRE or not first_item:
                continue
            f0 = nx.phones[0]
            starts_v = f0.nucleus or (f0.ipa in ("j", "w", "ɥ") and first_item[:1] in VOWEL_LETTERS)
            if not starts_v or first_item[:1] in ("y",) and f0.ipa == "j" and first_item != "y":
                continue
            word = w.info.get("last_item", "")
            if word in OBLIGATORY:
                prior = C.P_LOW          # obligatory: in the canonical form, dropping is rare
            elif word in OPTIONAL:
                prior = C.P_COMMON       # optional: consonant may or may not be pronounced
            else:
                continue
            last = w.phones[-1]
            p = C.P(latent, False)
            p.add_alt("", prior)
            w.phones.append(p)
            # the liaison consonant joins the last syllable of its word (enchaînement is free)
            a, b, st = w.syls[-1]
            w.syls[-1] = (a, b + 1, st)
            if latent == "n" and "̃" in last.ipa:
                last.add_alt(last.ipa.replace("̃", ""), C.P_MED)   # bon ami: denasalised


def build() -> French:
    return French()
