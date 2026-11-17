"""Brazilian Portuguese text front-end.

Pronunciation: orthographic rules (stress from written accents / word endings, BR vowel and
consonant rules) + a lexicon of per-word corrections (open ɛ/ɔ in the stressed syllable and
irregular x) extracted offline from WikiPron por (bz) (Wiktionary, CC BY-SA). The same word list
is used for the "oov:" warning. Stress: orthographic rules (reliable for Portuguese).
"""
from __future__ import annotations

import os
import gzip
import unicodedata

from . import _eu_common as C
from ._eu_base import EuG2P

V_ACUTE = {"á": "a", "é": "ɛ", "í": "i", "ó": "ɔ", "ú": "u", "à": "a"}
V_CIRC = {"â": "ɐ", "ê": "e", "ô": "o"}
V_TILDE = {"ã": "ɐ̃", "õ": "õ"}
VOWEL_LETTERS = set("aeiouáéíóúâêôãõàü")
STRONG = set("aeoáéóâêôãõàíú")
NASAL = {"a": "ɐ̃", "e": "ẽ", "i": "ĩ", "o": "õ", "u": "ũ", "ɛ": "ẽ", "ɔ": "õ", "ɐ": "ɐ̃"}
VOICED = set("bdgvzʒmnɲlʎrɾ")
FUNC = {"o", "a", "os", "as", "um", "uns", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
        "e", "ou", "que", "se", "me", "te", "lhe", "lhes", "por", "com", "sem", "ao", "aos", "à", "às",
        "num", "numa", "pra", "pro", "mas", "nem", "the"}


def _units(w: str):
    """letters -> list of [kind, value, letter]; kind V (vowel letter) / C (consonant phone) / N
    (nasal coda marker)."""
    out = []
    n = len(w)
    i = 0
    at = lambda k: w[k] if 0 <= k < n else ""  # noqa: E731
    while i < n:
        c, c2 = w[i], w[i:i + 2]
        nx = at(i + 1)
        if c in VOWEL_LETTERS:
            if c == "u" and at(i - 1) in "qg" and nx in ("e", "i", "é", "í", "ê"):
                i += 1
                continue                       # que, qui, gue, gui: silent u
            out.append(["V", "u" if c == "ü" else c, c])
            i += 1
            continue
        if c in "qg" and nx in ("u", "ü") and at(i + 2) in ("a", "o", "ã", "õ", "á", "ó", "â", "ô"):
            out.append(["C", "k" if c == "q" else "g", c]); out.append(["C", "w", "u"]); i += 2; continue
        if c in "qg" and nx == "ü":
            out.append(["C", "k" if c == "q" else "g", c]); out.append(["C", "w", "u"]); i += 2; continue
        if c2 in ("ch", "lh", "nh"):
            out.append(["C", {"ch": "ʃ", "lh": "ʎ", "nh": "ɲ"}[c2], c2]); i += 2; continue
        if c2 == "rr":
            out.append(["C", "R", c2]); i += 2; continue
        if c2 in ("ss", "sç") or (c2 == "sc" and at(i + 2) in ("e", "i", "é", "í", "ê")) \
                or (c2 == "xc" and at(i + 2) in ("e", "i", "é", "í", "ê")):
            out.append(["C", "s", c2]); i += 2; continue
        if c == "ç":
            out.append(["C", "s", c]); i += 1; continue
        if c == "c":
            out.append(["C", "s" if nx in ("e", "i", "é", "í", "ê") else "k", c]); i += 1; continue
        if c == "g":
            out.append(["C", "ʒ" if nx in ("e", "i", "é", "í", "ê") else "g", c]); i += 1; continue
        if c == "q":
            out.append(["C", "k", c]); i += 1; continue
        if c == "h":
            i += 1; continue
        if c == "j":
            out.append(["C", "ʒ", c]); i += 1; continue
        if c == "x":
            if i > 0 and w[:i] in ("e", "ê", "ine", "he") and nx in VOWEL_LETTERS:
                out.append(["C", "z", c])          # exame, exemplo, inexato
            elif i > 0 and at(i - 1) in "eê" and nx and nx not in VOWEL_LETTERS:
                out.append(["C", "s", c])          # explicar, texto
            else:
                out.append(["C", "ʃ", c])
            i += 1
            continue
        if c in "mn":
            # coda nasal: before a consonant (not h) or at the end -> nasalises the vowel
            if out and out[-1][0] == "V" and (i == n - 1 or (nx not in VOWEL_LETTERS and nx != "h")):
                out.append(["N", "", c]); i += 1; continue
            out.append(["C", c, c]); i += 1; continue
        if c == "r":
            strong = i == 0 or (out and out[-1][0] in ("C", "N") and out[-1][2] in ("n", "l", "s", "m"))
            out.append(["C", "R" if strong else "ɾ", c]); i += 1; continue
        if c == "s":
            between = out and out[-1][0] in ("V",) and nx in VOWEL_LETTERS
            out.append(["C", "z" if between else "s", c]); i += 1; continue
        if c == "z":
            out.append(["C", "z", c]); i += 1; continue
        m = {"b": "b", "d": "d", "f": "f", "k": "k", "l": "l", "p": "p", "t": "t", "v": "v", "w": "w", "y": "i"}.get(c)
        if m == "i":
            out.append(["V", "i", c]); i += 1; continue
        if m:
            if out and out[-1][0] == "C" and out[-1][1] == m:
                i += 1; continue
            out.append(["C", m, c])
        i += 1
    return out


QU_W = ("frequen", "sequen", "consequen", "cinquen", "tranquil", "delinquen", "eloquen", "aguent",
        "linguiç", "pinguim", "bilíngu", "unguent", "quinquen", "arguir", "equestr", "equino")


def rules(w: str, open_vowel: str | None = None) -> list[str]:
    for q in QU_W:
        if q in w:
            w = w.replace(q, q.replace("qu", "qü").replace("gu", "gü"), 1)
    """Orthographic BR Portuguese -> phone tokens with a stress digit. `open_vowel` = "ɛ"/"ɔ"/"e"/"o"
    forces the quality of a stressed e/o (from the lexicon)."""
    u = _units(w)
    # nuclei groups over consecutive vowel letters
    groups: list[list[int]] = []
    prev_v = False
    for k, (kind, v, _) in enumerate(u):
        if kind != "V":
            prev_v = False
            continue
        if prev_v:
            g = groups[-1]
            last = u[g[-1]][1]
            weak = v in ("i", "u")
            # falling diphthong V + unaccented i/u; "ui", "iu"; else hiatus
            if weak and len(g) == 1 and (last in STRONG or (last in ("i", "u") and last != v)) \
                    and not (k + 1 < len(u) and u[k + 1][1] in ("ɲ",)) \
                    and not (v == "i" and k + 1 < len(u) and u[k + 1][0] == "N" and k + 1 == len(u) - 1):
                g.append(k)
            elif last == "ã" and v in ("o", "e"):
                g.append(k)                    # ão, ãe
            elif last == "õ" and v == "e":
                g.append(k)                    # õe
            else:
                groups.append([k])
        else:
            groups.append([k])
        prev_v = True
    if not groups:
        return [x[1] for x in u if x[1]]
    # stress
    acc = [gi for gi, g in enumerate(groups) if any(u[k][1] in V_ACUTE or u[k][1] in V_CIRC for k in g)]
    til = [gi for gi, g in enumerate(groups) if any(u[k][1] in V_TILDE for k in g)]
    if acc:
        sg = acc[-1]
    elif til:
        sg = til[-1]
    elif len(groups) == 1:
        sg = 0
    else:
        e = w
        if e.endswith(("a", "e", "o", "as", "es", "os", "am", "em", "ens")):
            sg = len(groups) - 2
        else:
            sg = len(groups) - 1
    # post-tonic final -ia/-io/-ua/-ue... (história, água, série): rising diphthong [ja] [wa]
    if len(groups) >= 2 and sg <= len(groups) - 3 + 1 and sg != len(groups) - 1:
        a, b = groups[-2], groups[-1]
        if sg == len(groups) - 3 and len(a) == 1 and u[a[0]][1] in ("i", "u") and len(b) == 1 \
                and a[0] + 1 == b[0]:
            u[a[0]] = ["C", "j" if u[a[0]][1] == "i" else "w", u[a[0]][2]]
            groups = groups[:-2] + [b]
    gidx = {k: gi for gi, g in enumerate(groups) for k in g}
    out: list[str] = []
    k = 0
    while k < len(u):
        kind, v, letter = u[k]
        if kind == "C":
            out.append(v)
            k += 1
            continue
        if kind == "N":
            k += 1
            continue
        gi = gidx[k]
        g = groups[gi]
        stressed = gi == sg
        final = gi == len(groups) - 1
        nasal_next = g[-1] + 1 < len(u) and u[g[-1] + 1][0] == "N"
        before_nasal_c = g[-1] + 1 < len(u) and u[g[-1] + 1][0] == "C" and u[g[-1] + 1][1] in ("m", "n", "ɲ")
        vs = []
        for kk in g:
            x = u[kk][1]
            q = V_ACUTE.get(x) or V_CIRC.get(x) or V_TILDE.get(x) or x
            vs.append(q)
        nuc = vs[0]
        rest = vs[1:]
        # vowel quality of the nucleus
        if nuc == "ɛ" and nasal_next:
            nuc = "e"                                    # também, ninguém
        if stressed and nuc in ("e", "o") and open_vowel:
            nuc = {"e": {"ɛ": "ɛ", "e": "e"}, "o": {"ɔ": "ɔ", "o": "o"}}[nuc].get(open_vowel, nuc)
        if not stressed and final and len(g) == 1 and not nasal_next:
            tail = "".join(x[1] for x in u[g[-1] + 1:])
            if tail in ("", "s", "z"):
                nuc = {"e": "i"}.get(nuc, nuc)      # ZIPA keeps final -o as o (83 %), -e as i
        # ZIPA writes BR nasal vowels as oral vowel + n/m (53 %+); a before a nasal onset as a
        ph = nuc
        for x in rest:
            if x in ("o",) and vs[0] == "ɐ̃":
                ph = "o"                                 # -ão / -am: ZIPA [on] (w mostly absent)
            elif x in ("e",) and vs[0] in ("ɐ̃", "õ"):
                ph = ("a" if vs[0] == "ɐ̃" else "o") + "j"
            elif x in ("i", "e"):
                ph += "j"
            elif x in ("u", "o"):
                ph += "w"
        if ph == "ɐ̃":
            ph = "a"
        elif ph == "õ":
            ph = "o"
        nasal_coda = nasal_next or (len(g) > 1 and vs[0] in ("ɐ̃", "õ")) or (len(g) == 1 and vs[0] in ("ɐ̃", "õ")
                                                                            and not before_nasal_c)
        if nasal_next and len(g) == 1:
            tail = "".join(x[2] for x in u[g[-1] + 1:])
            if final and vs[0] == "a" and not stressed and tail == "m":
                ph = "o"                                 # foram, ficam -> [on]

        out.append(ph + ("1" if stressed else ""))
        if nasal_coda:
            nxt_letter = next((x[2] for x in u[g[-1] + 1:] if x[0] == "C"), "")
            out.append("m" if nxt_letter[:1] in ("p", "b") else "n")
            out.append("N")                              # marker: preceding n/m is a nasal coda
        k = g[-1] + 1
    # consonant context rules
    res: list[str] = []
    out = [t for t in out if t != "N"]
    for j, t in enumerate(out):
        ipa = t.rstrip("1")
        nxt = out[j + 1].rstrip("1") if j + 1 < len(out) else ""
        coda = not nxt or not C.is_nucleus_ipa(nxt)
        if ipa in ("t", "d") and nxt[:1] in ("i", "ĩ"):
            t = "tʃ" if ipa == "t" else "dʒ"
        elif ipa == "R":
            t = "ʁ" if j > 0 and C.is_nucleus_ipa(out[j - 1].rstrip("1")) else "ɾ"   # ZIPA: rr ʁ, #r ɾ
        elif ipa == "ɾ" and coda:
            t = "r"
        elif ipa in ("s", "z") and coda and j > 0:
            t = "z" if nxt and nxt[0] in VOICED else "s"
        res.append(unicodedata.normalize("NFD", t))
    return res


# weak forms of clitics: canonical + alternatives (prior)
WEAK = {
    "de": ("d i", [("d e", C.P_COMMON), ("d ɛ", C.P_COMMON), ("dʒ i", C.P_COMMON)]),
    "e": ("i", [("e", C.P_MED), ("ɛ", C.P_MED)]),
    "que": ("k e", [("k i", C.P_COMMON), ("k ɛ", C.P_COMMON)]),
    "se": ("s i", [("s e", C.P_MED), ("s ɛ", C.P_MED)]),
    "me": ("m i", [("m e", C.P_MED)]),
    "te": ("tʃ i", [("t e", C.P_MED), ("t i", C.P_MED)]),
    "lhe": ("l i", [("ʎ i", C.P_COMMON), ("ʎ e", C.P_MED)]),
    "em": ("i n", [("e n", C.P_COMMON), ("ɛ n", C.P_COMMON), ("ej n", C.P_COMMON)]),
    "não": ("n ow1", [("n o n", C.P_COMMON), ("n ow n", C.P_COMMON), ("n ɐ̃w̃", C.P_COMMON)]),
    "o": ("o", [("u", C.P_MED)]), "os": ("o s", [("u s", C.P_MED)]),
    "do": ("d o", [("d u", C.P_MED)]), "dos": ("d o s", [("d u s", C.P_MED)]),
    "no": ("n o", [("n u", C.P_MED)]), "nos": ("n o s", [("n u s", C.P_MED)]),
    "como": ("k o1 m o", [("k o m u", C.P_MED)]), "para": ("p a1 ɾ a", [("p ɾ a", C.P_MED)]),
    "a": ("a", []), "as": ("a s", []), "da": ("d a", []), "das": ("d a s", []), "na": ("n a", []),
    "nas": ("n a s", []), "um": ("u n", [("ũ", C.P_MED)]), "com": ("k o n", [("k õ", C.P_MED), ("k u n", C.P_MED)]),
    "por": ("p o r", [("p u r", C.P_MED), ("p o", C.P_MED)]), "ao": ("a w", [("a o", C.P_MED), ("a u", C.P_MED)]),
}


class Portuguese(EuG2P):
    lang = "pt"
    measured_nasal = True          # round 16 (adopted): see the nasal coda in build_word
    lex_name = ""
    use_lts = False
    decimal_comma = True
    thousand_space = True
    ordinals = "º|ª|°|o|a"
    num_lang = "pt_BR"
    letters = {"a": "á", "b": "bê", "c": "cê", "d": "dê", "e": "é", "f": "efe", "g": "gê", "h": "agá",
               "i": "i", "j": "jota", "k": "cá", "l": "ele", "m": "eme", "n": "ene", "o": "ó", "p": "pê",
               "q": "quê", "r": "erre", "s": "esse", "t": "tê", "u": "u", "v": "vê", "w": "dáblio",
               "x": "xis", "y": "ípsilon", "z": "zê"}
    symbols = {"%": "por cento", "&": "e", "$": "dólares", "€": "euros", "+": "mais", "=": "igual",
               "@": "arroba", "°": "graus"}
    abbrev = {"sr": "senhor", "sra": "senhora", "dr": "doutor", "dra": "doutora", "etc": "etcétera",
              "km": "quilômetros", "kg": "quilos", "cm": "centímetros", "av": "avenida", "pág": "página",
              "nº": "número", "eua": "estados unidos"}
    onsets = frozenset(tuple(x.split()) for x in (
        "p ɾ", "p l", "b ɾ", "b l", "t ɾ", "d ɾ", "k ɾ", "k l", "g ɾ", "g l", "f ɾ", "f l", "v ɾ", "t l"))

    def __init__(self):
        super().__init__()
        self.known: dict[str, str] = {}
        with C.lex_path("pt.tsv.gz").open("rb") as fb, gzip.open(fb, "rt", encoding="utf-8") as f:
            for line in f:
                k, _, v = line.rstrip("\n").partition("\t")
                if k and k[0] != "#":
                    self.known[k] = v

    def lookup(self, w):
        if w in WEAK:
            return [WEAK[w][0].split()]
        v = self.known.get(w)
        if v is None:
            # inflected forms: plural / feminine of a listed word (only affects the oov flag
            # and the stressed-vowel quality; the pronunciation comes from the rules)
            for a, b in (("s", ""), ("es", ""), ("ões", "ão"), ("ães", "ão"), ("ais", "al"), ("éis", "el"),
                         ("eis", "el"), ("óis", "ol"), ("is", "il"), ("ns", "m"), ("a", "o"), ("as", "o"),
                         ("as", "a"), ("os", "o"), ("ram", "r"), ("m", ""), ("da", "do"), ("das", "do")):
                if w.endswith(a) and len(w) > len(a) + 2 and (w[: len(w) - len(a)] + b) in self.known:
                    v = self.known[w[: len(w) - len(a)] + b]
                    v = v if v in ("ɛ", "ɔ") and a in ("s", "es", "a", "as", "os") else ""
                    break
        if v is None:
            return None
        if v in ("ɛ", "ɔ", "e", "o"):
            return [rules(w, v)]
        if v:
            return [v.split()]
        return [rules(w)]

    def oov(self, w):
        return rules(w)

    def fix_stress(self, phones, word=""):
        return phones

    def build_word(self, tok, items, warnings):
        w = super().build_word(tok, items, warnings)
        if not w.phones:
            return w
        if len(items) == 1 and items[0] in FUNC and len(w.syls) == 1:
            w.syls = [(a, b, 0) for a, b, _ in w.syls]
        if len(items) == 1 and items[0] in WEAK:
            for alt, prior in WEAK[items[0]][1]:
                C.add_word_variant(w, alt.split(), prior)
        ph = w.phones
        last_syl_stress = w.syls[-1][2]
        n = len(ph)
        for k, p in enumerate(ph):
            nxt = ph[k + 1] if k + 1 < n else None
            coda = nxt is None or not nxt.nucleus
            if p.ipa == "ʎ":                           # ZIPA writes lh as l (60 %)
                p.ipa = "l"
                for a in ("ʎ", "lj", "j"):
                    p.add_alt(a, C.P_COMMON if a == "ʎ" else C.P_MED)
            elif p.ipa == "ɲ":
                p.add_alt("n", C.P_MED)
                p.add_alt("j̃", C.P_LOW)
            elif p.ipa in ("tʃ", "dʒ"):                 # non-palatalising accents (NE Brazil), -tes
                p.add_alt(p.ipa[0], C.P_MED if k >= n - 3 else C.P_LOW)
            elif p.ipa in ("ʁ", "x") or (p.ipa == "ɾ" and k == 0):     # strong r
                for a in ("ʁ", "x", "h", "r", "ɾ"):
                    p.add_alt(a, C.P_MED)
            elif p.ipa == "ɾ":
                p.add_alt("r", C.P_COMMON)
            elif p.ipa == "r" and coda:                  # coda r: r ɾ x h ɹ, final r often dropped
                for a in ("ɾ", "x", "h", "ɹ"):
                    p.add_alt(a, C.P_COMMON if a == "ɾ" else C.P_MED)
                if k == n - 1:
                    p.add_alt("", C.P_COMMON)
            elif p.ipa == "s" and coda and k > 0:
                p.add_alt("ʃ", C.P_MED)                # carioca coda s
            elif p.ipa == "z" and coda and k > 0:
                p.add_alt("ʒ", C.P_MED)
            elif p.ipa == "l" and coda and k > 0:
                p.add_alt("w", C.P_COMMON)             # l vocalisation
            elif p.ipa in ("n", "m") and coda and k > 0 and ph[k - 1].nucleus:
                # nasal coda: ZIPA writes V+n; a nasal vowel without consonant is the same sound.
                # Round 16: natives drop the coda in 7.5 % of word-medial and 16 % of word-final
                # cases (bench/calib/alt_audit.py, FLEURS validation): with MEASURED_NASAL the
                # deletion stays accepted word-finally only.
                if not (self.measured_nasal and k < n - 1 and os.environ.get("PRONUNCIADA_PT_NASAL", "1") != "0"):
                    p.add_alt("", C.P_MED)
                p.add_alt("ŋ", C.P_LOW)
                v = ph[k - 1]
                base = v.ipa
                if base in ("ow", "o") and k == n - 1:
                    v.add_alt("ɐ̃w̃", C.P_MED); v.add_alt("ɐw", C.P_MED); v.add_alt("aw", C.P_LOW)
                    v.add_alt("õ", C.P_MED)
                elif base == "e" and k == n - 1:
                    v.add_alt("ej", C.P_COMMON); v.add_alt("ẽj̃", C.P_MED); v.add_alt("ɛj", C.P_MED)
                elif base in ("aj", "oj"):
                    v.add_alt(base[0] + "\u0303j\u0303" if base[0] == "o" else "ɐ̃j̃", C.P_MED)
                elif len(base) == 1 and base in "aeiou":
                    v.add_alt(("ɐ" if base == "a" else base) + "\u0303", C.P_MED)
        # final unstressed vowel reduction / raising (canonical i u a; full vowels and loss as alts)
        if last_syl_stress == 0 and len(w.syls) > 1:
            a, b, _ = w.syls[-1]
            for k in range(a, b):
                p = ph[k]
                if p.nucleus and p.ipa in ("i", "o", "a"):
                    for a in {"i": ("e", "ɛ"), "o": ("u",), "a": ("ɐ",)}[p.ipa]:
                        p.add_alt(a, C.P_COMMON)
                    if p.ipa == "i":
                        p.add_alt("", C.P_LOW)
        return w

    def connected(self, words):
        # vowel sandhi: final unstressed vowel + initial vowel -> elision / glide
        for i in range(len(words) - 1):
            w, nx = words[i], words[i + 1]
            a, b = w.phones[-1], nx.phones[0]
            if a.nucleus and b.nucleus and w.syls[-1][2] != 1:
                a.add_alt("", C.P_MED)
                if a.ipa in ("i", "u"):
                    a.add_alt("j" if a.ipa == "i" else "w", C.P_MED)


def build() -> Portuguese:
    return Portuguese()
