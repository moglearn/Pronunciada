"""Latin American Spanish text front-end (seseo, yeísmo, no θ).

Pronunciation: orthographic rules (Spanish spelling is regular) + a small exception list.
Vocabulary check (for "oov:" warnings only): word list of ipa-dict es_MX (MIT).
Stress: orthographic rules (written accent, else penultimate for words ending in a vowel / n / s,
else final) - reliable for native words; monosyllabic function words get stress 0.
"""
from __future__ import annotations


from . import _eu_common as C
from ._eu_base import EuG2P

STRONG = set("aeoáéíóú")
VOW = set("aeiouáéíóúü")
ACC = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
FUNC = {"el", "la", "lo", "los", "las", "un", "de", "del", "en", "y", "e", "o", "u", "a", "al", "que",
        "se", "su", "sus", "mi", "mis", "tu", "tus", "me", "te", "le", "les", "nos", "por", "con",
        "sin", "ni", "si", "pues", "tan"}
# words whose pronunciation departs from the spelling rules (phones space separated, digits=stress)
EXCEPT = {
    "méxico": "m e1 x i k o", "mexicano": "m e x i k a1 n o", "mexicana": "m e x i k a1 n a",
    "mexicanos": "m e x i k a1 n o s", "mexicanas": "m e x i k a1 n a s", "texas": "t e1 x a s",
    "oaxaca": "w a x a1 k a", "xalapa": "x a l a1 p a", "show": "tʃ o1 w", "web": "w e1 b",
    "wifi": "w a1 i f i", "jazz": "dʒ a1 s", "rock": "r o1 k", "hockey": "x o1 k i",
}


def _letters_to_segments(w: str):
    """-> list of (kind, value): kind 'C' consonant phone, 'V' vowel letter."""
    out = []
    i, n = 0, len(w)
    nxt = lambda k: w[k] if k < n else ""  # noqa: E731
    while i < n:
        c = w[i]
        c2 = w[i:i + 2]
        if c in VOW:
            if c == "u" and i > 0 and w[i - 1] in "qg" and nxt(i + 1) in ("e", "i", "é", "í"):
                i += 1          # silent u in que/qui/gue/gui
                continue
            out.append(("V", "u" if c == "ü" else c))
            i += 1
            continue
        if c == "y":
            # vowel when final after a vowel / standalone / before a consonant
            if (i == n - 1) or (nxt(i + 1) and nxt(i + 1) not in VOW):
                out.append(("V", "i"))
            else:
                out.append(("C", "j"))
            i += 1
            continue
        if c2 == "ch":
            out.append(("C", "tʃ")); i += 2; continue
        if c2 == "ll":
            out.append(("C", "ʎ")); i += 2; continue
        if c2 == "rr":
            out.append(("C", "r")); i += 2; continue
        if c2 == "qu":
            out.append(("C", "k")); i += 1; continue  # u handled as silent above
        if c == "c":
            out.append(("C", "s" if nxt(i + 1) in ("e", "i", "é", "í") else "k")); i += 1; continue
        if c == "g":
            out.append(("C", "x" if nxt(i + 1) in ("e", "i", "é", "í") else "g")); i += 1; continue
        if c == "h":
            i += 1; continue
        if c == "x":
            out.append(("C", "k")); out.append(("C", "s")); i += 1; continue
        if c == "r":
            initial = i == 0 or (out and out[-1][0] == "C" and out[-1][1] in ("n", "l", "s"))
            out.append(("C", "r" if initial else "ɾ")); i += 1; continue
        m = {"b": "b", "v": "b", "d": "d", "f": "f", "j": "x", "k": "k", "l": "l", "m": "m", "n": "n",
             "ñ": "ɲ", "p": "p", "q": "k", "s": "s", "t": "t", "w": "w", "z": "s"}.get(c)
        if m:
            if out and out[-1] == ("C", m) and m not in ("n", "l"):   # doubled letters in loans
                i += 1
                continue
            out.append(("C", m))
        i += 1
    return out


def rules(w: str) -> list[str]:
    seg = _letters_to_segments(w)
    # group vowels into nuclei
    groups: list[list[int]] = []      # indices into seg of vowels in each nucleus group
    prev_v = False
    for k, (kind, v) in enumerate(seg):
        if kind != "V":
            prev_v = False
            continue
        if prev_v:
            g = groups[-1]
            last = seg[g[-1]][1]
            if (last in STRONG and v in STRONG) or len(g) >= 3:
                groups.append([k])
            else:
                g.append(k)
        else:
            groups.append([k])
        prev_v = True
    if not groups:
        return [v for _, v in seg]
    # nucleus of each group
    nuc_of = {}
    for gi, g in enumerate(groups):
        strong = [k for k in g if seg[k][1] in STRONG]
        nuc_of[gi] = strong[0] if strong else g[-1]
    # stress
    acc = [gi for gi, g in enumerate(groups) if any(seg[k][1] in ACC for k in g)]
    if acc:
        sg = acc[-1]
    elif len(groups) == 1:
        sg = 0
    else:
        last = w[-1]
        sg = len(groups) - 2 if (last in VOW or last in "ns") else len(groups) - 1
    out: list[str] = []
    gidx = {k: gi for gi, g in enumerate(groups) for k in g}
    k = 0
    while k < len(seg):
        kind, v = seg[k]
        if kind == "C":
            out.append(v)
            k += 1
            continue
        gi = gidx[k]
        g = groups[gi]
        pend = ""
        nk = nuc_of[gi]
        st = "1" if gi == sg else ""
        for kk in g:
            vv = ACC.get(seg[kk][1], seg[kk][1])
            if kk < nk:
                if vv == "i" and out and not C.is_nucleus_ipa(out[-1].rstrip("1")) and kk == g[0] \
                        and k > 0:
                    pend = "i"      # rising diphthong after a consonant: one phone "ie", "ia" ...
                    continue
                out.append("j" if vv == "i" else "w")
            elif kk == nk:
                out.append(pend + vv + st)
                pend = ""
            else:  # falling diphthong: glide glued to the nucleus phone
                out[-1] = out[-1].rstrip("1") + vv + st if st else out[-1] + vv
        k = g[-1] + 1
    return out


VOICED_STOP_WEAK = {"b": "β", "d": "ð", "g": "ɣ"}


class Spanish(EuG2P):
    lang = "es"
    lex_name = ""          # pronunciation by rules; word list only for OOV detection
    use_lts = False
    decimal_comma = False  # es_419: "3.5" and "1,000" are both common; default to point
    thousand_space = True
    ordinals = "º|ª|°|o|a|er|ra|do|da|to|ta|vo|va|no|na|mo|ma"
    num_lang = "es"
    letters = {"a": "a", "b": "be", "c": "ce", "d": "de", "e": "e", "f": "efe", "g": "ge", "h": "hache",
               "i": "i", "j": "jota", "k": "ka", "l": "ele", "m": "eme", "n": "ene", "ñ": "eñe", "o": "o",
               "p": "pe", "q": "cu", "r": "erre", "s": "ese", "t": "te", "u": "u", "v": "uve",
               "w": "doble uve", "x": "equis", "y": "ye", "z": "zeta"}
    symbols = {"%": "por ciento", "&": "y", "$": "dólares", "€": "euros", "+": "más", "=": "igual",
               "@": "arroba", "°": "grados"}
    abbrev = {"sr": "señor", "sra": "señora", "srta": "señorita", "dr": "doctor", "dra": "doctora",
              "ee.uu": "estados unidos", "eeuu": "estados unidos", "etc": "etcétera", "km": "kilómetros",
              "kg": "kilogramos", "cm": "centímetros", "pág": "página", "núm": "número", "aprox": "aproximadamente",
              "ud": "usted", "uds": "ustedes", "av": "avenida"}
    onsets = frozenset(
        [tuple(x.split()) for x in ("p ɾ", "p l", "b ɾ", "b l", "t ɾ", "d ɾ", "k ɾ", "k l", "g ɾ", "g l",
                                     "f ɾ", "f l", "t l", "p r", "b r", "t r", "d r", "k r", "g r", "f r")]
        + [(c, "w") for c in "pbtdkgfsxmnlrɾʎɲjʝ"] + [("tʃ", "w")]
        + [(a, b, "w") for a, b in (("p", "r"), ("b", "r"), ("t", "r"), ("d", "r"), ("k", "r"), ("g", "r"),
                                    ("f", "r"), ("k", "l"), ("g", "l"), ("f", "l"), ("p", "l"), ("b", "l"))])

    def __init__(self):
        super().__init__()
        # sorted word list (pronunciada/text/_eu_build.py build_es), kept as one bytes object
        self.words_known = C.SortedWords(C.read_gz("es_words.txt.gz"))

    def lookup(self, w):
        if w in EXCEPT:
            return [EXCEPT[w].split()]
        if w in self.words_known or len(w) == 1:
            return [rules(w)]
        return None

    def oov(self, w):
        return rules(w)

    def rewrite(self, phones, word):
        """ZIPA convention on FLEURS es_419: the tap is written r (86 %) and the trill ɾ (71 %)."""
        return [{"ɾ": "r", "r": "ɾ"}.get(p, p) for p in phones]

    def fix_stress(self, phones, word=""):
        return phones

    def build_word(self, tok, items, warnings):
        w = super().build_word(tok, items, warnings)
        if len(items) == 1 and len(w.syls) == 1 and items[0] in FUNC:
            w.syls = [(a, b, 0) for a, b, _ in w.syls]
        return w

    def connected(self, words):
        flat = [(wi, k, p) for wi, w in enumerate(words) for k, p in enumerate(w.phones)]
        from ..r20 import enabled

        if enabled("es_glide", "es"):
            # round 20 (on by default, pronunciada.r20.DEFAULTS; PRONUNCIADA_R20_ES_GLIDE=0 switches it off): the
            # recogniser writes the glides of rising diphthongs as vowels (muy, cuando: w ~ u); a post-consonantal
            # j likewise (j ~ i)
            for j, (wi, k, p) in enumerate(flat):
                prev = flat[j - 1][2] if j > 0 else None
                if p.ipa == "w":
                    p.add_alt("u", C.P_COMMON)
                elif p.ipa == "j" and prev is not None and not prev.nucleus:
                    p.add_alt("i", C.P_COMMON)
        for j, (wi, k, p) in enumerate(flat):
            prev = flat[j - 1][2] if j > 0 else None
            nxt = flat[j + 1][2] if j + 1 < len(flat) else None
            # b d g -> β ð ɣ after a vowel or a continuant (not after a nasal; d also not after l)
            if p.ipa in VOICED_STOP_WEAK and prev is not None and not (
                    prev.ipa in ("m", "n", "ɲ", "ŋ") or (p.ipa == "d" and prev.ipa == "l")):
                p.add_alt(VOICED_STOP_WEAK[p.ipa], C.P_COMMON)
            # word-final d (verdad, ciudad) may be weak or dropped
            if p.ipa == "d" and k == len(words[wi].phones) - 1:
                p.add_alt("", C.P_MED)
            # syllable-final s: aspiration / loss (Caribbean-type accents), low prior
            if p.ipa == "s" and (nxt is None or not nxt.nucleus):
                p.add_alt("h", C.P_RARE)
                p.add_alt("", C.P_RARE)
            if p.ipa == "x":
                p.add_alt("h", C.P_LOW)
            if p.ipa in ("j", "ʎ") and nxt is not None and nxt.nucleus and (
                    prev is None or prev.nucleus or k == 0):
                # yeísmo: ll and y are the same sound; ZIPA writes ll mostly as ʎ, y as j
                for a in ("j", "ʎ", "ʝ", "ʒ", "ʃ", "dʒ"):
                    p.add_alt(a, C.P_COMMON if a in ("j", "ʎ") else C.P_LOW)
            if p.ipa[0] == "e" and p.nucleus:                    # ZIPA: e ~ ɛ (55/40 everywhere)
                p.add_alt("ɛ" + p.ipa[1:], C.P_COMMON)
            if len(p.ipa) == 2 and p.ipa[0] == "i" and p.nucleus:  # rising diphthong ie ~ je
                p.add_alt("j" + p.ipa[1], C.P_COMMON)
                if p.ipa[1] == "e":
                    p.add_alt("jɛ", C.P_COMMON); p.add_alt("iɛ", C.P_COMMON)
        # synalepha / vowel merging across words
        for wi in range(len(words) - 1):
            a, b = words[wi].phones[-1], words[wi + 1].phones[0]
            if a.nucleus and b.nucleus:
                a.add_alt("", C.P_COMMON if a.ipa[0] == b.ipa[0] else C.P_MED)
                if a.ipa in ("e", "o") or a.ipa in ("i", "u"):
                    a.add_alt({"e": "j", "i": "j", "o": "w", "u": "w"}[a.ipa], C.P_LOW)


def build() -> Spanish:
    return Spanish()
