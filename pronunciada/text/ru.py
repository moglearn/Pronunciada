"""Russian text front-end.

Pronunciation: rule-based (palatalisation, vowel reduction, voicing assimilation, special
clusters) driven by the stress position. Stress source, in this order: an acute accent or ё in the
prompt; the stress lexicon (stressed-vowel index per word form, derived offline from WikiPron
Russian = Wiktionary, CC BY-SA: the unreduced vowel of the transcription, majority over variants);
a suffix model trained on that lexicon for OOV words (~75 % on held-out forms, see notes).
Irregular words (что, его, солнце ...) come from the same lexicon as full transcriptions.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import unicodedata

from . import _eu_common as C
from ._eu_base import EuG2P

VOWELS = "аеёиоуыэюя"
SOFT_V = "еёиюя"
HARD_ONLY = set("жшц")
SOFT_ONLY = set("чщй")
CONS = {"б": "b", "в": "v", "г": "g", "д": "d", "ж": "ʐ", "з": "z", "й": "j", "к": "k", "л": "l", "м": "m",
        "н": "n", "п": "p", "р": "r", "с": "s", "т": "t", "ф": "f", "х": "x", "ц": "ts", "ч": "tɕ",
        "ш": "ʂ", "щ": "ɕː"}
VOICED_PAIR = {"b": "p", "d": "t", "g": "k", "v": "f", "z": "s", "ʐ": "ʂ", "ʑː": "ɕː", "dz": "ts", "dʑ": "tɕ"}
VOICELESS_PAIR = {v: k for k, v in VOICED_PAIR.items()}
SONORANT = set("mnlrj")
ACUTE = "́"

# unstressed function words (proclitics / enclitics)
CLITICS = {"в", "во", "к", "ко", "с", "со", "без", "до", "из", "на", "над", "о", "об", "обо", "от", "по", "под",
           "при", "про", "у", "за", "не", "ни", "же", "ли", "бы", "и", "а", "но", "да", "что", "как", "из-за",
           "для", "или", "то", "ведь", "уж", "чтобы", "через", "перед", "между", "как", "так"}
PROCLITIC_NO_VOWEL = {"в", "к", "с"}


def _base(p: str) -> str:
    return p.replace("ʲ", "").replace("ː", "")


def rules(word: str, stress: int | None, zipa: bool = False) -> list[str]:
    """Cyrillic word (lower case, may contain ё) + stressed vowel index -> phones (Wiktionary-like
    notation: ɐ ə ɪ ʊ ɨ for reduced vowels, ʲ palatalisation, ɕː, tɕ)."""
    w = word.replace("-", "")
    letters = [c for c in w if c in VOWELS or c in CONS or c in "ьъ"]
    vowel_pos = [i for i, c in enumerate(letters) if c in VOWELS]
    if stress is None:
        stress = 0
    nv = len(vowel_pos)
    out: list[str] = []
    vi = -1
    n = len(letters)
    for i, c in enumerate(letters):
        prev = letters[i - 1] if i > 0 else ""
        nxt = letters[i + 1] if i + 1 < n else ""
        if c in "ьъ":
            continue
        if c in VOWELS:
            vi += 1
            stressed = vi == stress
            after_soft = bool(out) and (out[-1].endswith("ʲ") or out[-1] in ("tɕ", "ɕː", "j"))
            after_hard_only = prev in HARD_ONLY
            iot = c in "еёюя" and (i == 0 or prev in VOWELS or prev in "ьъ")
            if c in "еёюя" and (iot or prev in "ьъ"):
                out.append("j")
                after_soft = True
            elif c in SOFT_V and out and prev in CONS and prev not in HARD_ONLY and prev not in SOFT_ONLY:
                if not out[-1].endswith("ʲ"):
                    out[-1] = out[-1] + "ʲ"
                after_soft = True
            if c == "и" and after_hard_only:
                base = "ɨ"
            else:
                base = {"а": "a", "о": "o", "у": "u", "ы": "ɨ", "и": "i", "э": "e", "е": "e", "ё": "o",
                        "ю": "u", "я": "a"}[c]
            if stressed:
                v = base
                if c == "е" and after_hard_only:
                    v = "e"
            else:
                pretonic1 = vi == stress - 1
                initial = i == 0 or (out == ["j"] and c in "еёюя")
                final = vi == nv - 1
                if base in ("a", "o", "e"):
                    if after_hard_only and base == "e":
                        v = "ɨ"
                    elif after_soft:
                        v = "ə" if (final and base == "a") else "ɪ"
                    elif after_hard_only and base == "a" and not pretonic1:
                        v = "ə"
                    elif pretonic1 or (i == 0):
                        v = "ɐ"
                    else:
                        v = "ə"
                elif base == "u":
                    v = "ʊ"
                elif base == "i":
                    v = "ɪ"
                else:
                    v = "ɨ"
                if initial and base in ("a", "o") and not out:
                    v = "ɐ"
            if zipa:
                # ZIPA conventions (FLEURS ru_ru): weak reduction - ɐ/ə heard as a (55-70 %),
                # unstressed е as ɛ, и as i, у as u; stressed е as ɛ (75 %)
                if stressed and v == "e":
                    v = "ɛ"
                elif not stressed:
                    v = {"ɐ": "a", "ə": "a", "ʊ": "u"}.get(v, v)
                    if v == "ɪ":
                        v = "ɛ" if c in "еэ" else ("a" if c == "я" else "i")
            out.append(v + ("1" if stressed else ""))
            continue
        # consonant
        ph = CONS[c]
        if c == "г" and i + 2 <= n and w.endswith(("ого", "его")) and i == n - 2:
            ph = "v"                                     # -ого / -его genitive
        soft = False
        if c not in HARD_ONLY and c not in SOFT_ONLY:
            if nxt == "ь" or (nxt and nxt in SOFT_V):
                soft = True
        if soft:
            ph += "ʲ"
        if c == "й":
            ph = "j"
        out.append(ph)
    # cluster simplifications on the phone string
    s = " ".join(out)
    for a, b in (("tʲ sʲ ɐ", "ts ə"), ("t sʲ ɐ", "ts ə"), ("tʲ s ə", "ts ə"), ("t s ə", "ts ə"),
                 ("tʲ sʲ ə", "ts ə"), ("t sʲ ə", "ts ə"),
                 ("s tɕ", "ɕː"), ("z tɕ", "ɕː"), ("sʲ tɕ", "ɕː"), ("s t n", "s n"), ("z d n", "z n"),
                 ("l n ts", "n ts"), ("v s t v", "s t v"), ("t s", "ts"), ("d s", "ts"), ("tɕ n", "tɕ n")):
        s = (" " + s + " ").replace(" " + a + " ", " " + b + " ").strip()
    out = s.split()
    # final devoicing and regressive voicing assimilation (right to left)
    res = out[:]
    state = "end"
    for j in range(len(res) - 1, -1, -1):
        core = res[j].rstrip("1")
        b = core.replace("ʲ", "")
        soft = "ʲ" if "ʲ" in core else ""
        if C.is_nucleus_ipa(core) or b in SONORANT:
            state = "son"
            continue
        if b == "v":                       # в devoices but does not trigger voicing
            if state in ("end", "voiceless"):
                res[j] = "f" + soft
                state = "voiceless"
            else:
                state = "son"
            continue
        if state in ("end", "voiceless") and b in VOICED_PAIR:
            res[j] = VOICED_PAIR[b] + soft
        elif state == "voiced" and b in VOICELESS_PAIR and not zipa:
            # ZIPA does not write regressive voicing inside words (также [takʐa], когда [kakda])
            res[j] = VOICELESS_PAIR[b] + soft
        nb = res[j].replace("ʲ", "")
        state = "voiced" if nb in VOICED_PAIR else "voiceless"
    return res


EXC_VOWEL = {"ɐ": "a", "ə": "a", "ɪ": "i", "ʊ": "u", "e": "ɛ", "ɵ": "o", "æ": "a", "ʉ": "u"}
# ZIPA consonant conventions (FLEURS ru_ru) -> canonical, alternatives
CONS_ZIPA = {
    "l": ("ɭ", ["l"]), "lʲ": ("ɭʲ", ["lʲ", "ɭ"]), "rʲ": ("r", ["rʲ"]), "mʲ": ("m", ["mʲ"]),
    "tʲ": ("ts", ["tsʲ", "tʲ", "t"]), "dʲ": ("dz", ["dzʲ", "dʲ", "zʲ", "d"]),
    "tɕ": ("ʈʂ", ["tʂ", "tɕ", "tʃ", "tsʲ"]), "ɕː": ("ʂʲ", ["ʂ", "ɕ", "ɕː", "sʲ", "ʃ"]),
    "ʑː": ("ʐʲ", ["ʐ", "ʑ"]),
}
PLAIN_BEFORE_V = set("kgxf")
VOWEL_ALTS = {"a": ["ɐ", "ə", "ɑ", "ʌ"], "i": ["ɪ", "ɛ", "ɨ"], "ɛ": ["ɪ", "i", "e"], "u": ["ʊ"],
              "ɨ": ["ɪ", "a"]}

# Round 8: reduction-aware targets. pronunciada/data/lex/ru_vowels.json holds, per position class,
# how native FLEURS-ru validation speech is realised according to the acoustic model
# (bench/calib/ru_round8.py audit): vowels keyed (letter, position, context), soft / plain t and d
# keyed (origin, next). For words whose stress is known (lexicon, accent mark, ё, one vowel,
# unstressed clitic) the primary symbol becomes the most frequent realisation and an alternative
# is kept only if natives produce it in at least VOWEL_MIN of the cases (log prior = log of its
# share relative to the primary's). Words with predicted stress keep the broad rule-based alts.
#
# On by default since round 8 together with the adapter retrained on these targets and the Russian
# sound bias chosen by the detection-aware rule (results/round8_ru.md). `measured_targets = False`
# on a G2P instance gives the round-7 references (the shipped adapter expects the measured ones).
MEASURED_TARGETS = True
VOWEL_MIN = 0.15
CLASS_MIN_N = 30
KNOWN_STRESS = frozenset({"lex", "accent", "yo", "mono", "clitic", "extra", "fix"})
# Round 10: stress (and restored ё) for word forms missing from ru.tsv.gz, derived OFFLINE with
# silero-stress (MIT) by bench/calib/ru_stress_lex.py (pronunciada/data/lex/ru_stress_extra.tsv.gz;
# torch is not needed at runtime). Two variants = homograph: both readings, broad alternatives.
EXTRA_STRESS = True
# Issue #6: stress corrections to ru.tsv.gz from the same offline silero-stress votes in context
# (pronunciada/data/lex/ru_stress_fix.tsv.gz, bench/calib/ru_stress_lex.py fix): the lexicon took a rare
# Wiktionary homograph for common forms (како́й read as ка́кой, она́ as о́на, лю́ди as люди́). A word is
# corrected when >= 75 % of silero's readings in context agree on another vowel.
STRESS_FIX = os.environ.get("PRONUNCIADA_RU_STRESS_FIX", "1") != "0"     # 0 = off, for A/B runs


def vowel_keys(word: str, stressed: int | None) -> list[tuple[str, str, str]]:
    """One (letter, position, context) per vowel letter of a Cyrillic word.
    position: S stressed, P1 first pretonic, P2 earlier pretonic, T post-tonic non-final,
    F post-tonic word-final, C unstressed word (stressed is None);
    context: V word-initial / after a vowel / after ь ъ, H after ж ш ц, C after another consonant."""
    letters = [c for c in word.lower() if c in VOWELS or c in CONS or c in "ьъ"]
    vpos = [i for i, c in enumerate(letters) if c in VOWELS]
    out = []
    for k, i in enumerate(vpos):
        if stressed is None:
            pos = "C"
        elif k == stressed:
            pos = "S"
        elif k == stressed - 1:
            pos = "P1"
        elif k < stressed:
            pos = "P2"
        elif k == len(vpos) - 1:
            pos = "F"
        else:
            pos = "T"
        prev = letters[i - 1] if i else ""
        ctx = "V" if (not prev or prev in VOWELS or prev in "ьъ") else ("H" if prev in HARD_ONLY else "C")
        out.append((letters[i], pos, ctx))
    return out


def cons_key(origin: str, nxt: str | None, nxt_nucleus: bool = False) -> tuple[str, str]:
    """origin: 't' 'tʲ' 'd' 'dʲ' (before the ZIPA mapping); next: Vi before и-type vowels,
    V other vowel, C consonant, # word end."""
    if nxt is None:
        return (origin, "#")
    if nxt_nucleus:
        return (origin, "Vi" if nxt[:1] in "iɪ" else "V")
    return (origin, "C")


def load_vowel_table() -> dict:
    p = C.lex_path("ru_vowels.json")
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {tuple(e["key"]): (int(e["n"]), dict(e["dist"])) for e in raw["classes"]}


class Russian(EuG2P):
    lang = "ru"
    lex_name = ""
    use_lts = False
    decimal_comma = True
    thousand_space = True
    ordinals = "-?(?:й|го|му|м|я|е|х|ми|ю|ое|ая|ой)"
    num_lang = "ru"
    stress_name = "ru_stress"
    letters = {"а": "а", "б": "бэ", "в": "вэ", "г": "гэ", "д": "дэ", "е": "е", "ж": "жэ", "з": "зэ", "и": "и",
               "к": "ка", "л": "эль", "м": "эм", "н": "эн", "о": "о", "п": "пэ", "р": "эр", "с": "эс",
               "т": "тэ", "у": "у", "ф": "эф", "х": "ха", "ц": "цэ", "ч": "чэ", "ш": "ша", "э": "э", "ю": "ю",
               "я": "я"}
    symbols = {"%": "процентов", "&": "и", "$": "долларов", "€": "евро", "+": "плюс", "=": "равно",
               "№": "номер", "°": "градусов"}
    abbrev = {"т.е": "то есть", "т.д": "так далее", "т.п": "тому подобное", "г": "год", "гг": "годы",
              "км": "километров", "кг": "килограммов", "см": "сантиметров", "млн": "миллионов",
              "млрд": "миллиардов", "тыс": "тысяч", "др": "другие", "им": "имени", "ул": "улица"}
    onsets = frozenset()  # filled from the stress lexicon header

    def __init__(self):
        super().__init__()
        # the sorted stress lexicon as one buffer (pronunciada.text._eu_common.SortedLexicon: same
        # entries as the dict, about a third of the memory); its keys in order serve the stem search
        lex = C.SortedLexicon.from_tsv(C.read_gz("ru.tsv.gz"))
        if lex is not None:
            self.stress = lex
            if "onsets" in lex.meta:
                self.onsets = C.parse_onsets(lex.meta["onsets"])
            self._keys = lex.sorted_keys()
        else:
            self.stress = {}
            with C.lex_path("ru.tsv.gz").open("rb") as fb, gzip.open(fb, "rt", encoding="utf-8") as f:
                for line in f:
                    k, _, v = line.rstrip("\n").partition("\t")
                    if k[:1] == "#":
                        if k == "#onsets":
                            self.onsets = C.parse_onsets(v)
                        continue
                    self.stress[k] = v
            self._keys = sorted(self.stress)
        self._src: dict[str, str] = {}               # word -> where its stress came from
        self.vowel_table = load_vowel_table()
        self.vowel_min = VOWEL_MIN
        self.extra_stress: dict[str, list[tuple[int, str | None]]] = {}
        p = C.lex_path("ru_stress_extra.tsv.gz")
        if p.exists():
            with p.open("rb") as fb, gzip.open(fb, "rt", encoding="utf-8") as f:
                for line in f:
                    if line[:1] == "#" or "\t" not in line:
                        continue
                    k, _, v = line.rstrip("\n").partition("\t")
                    self.extra_stress[k] = [(int(a), b or None) for a, _, b in (x.partition(":") for x in v.split("|"))]
        self.use_extra_stress = EXTRA_STRESS
        self.stress_fix = self._read_variants("ru_stress_fix.tsv.gz")
        self.use_stress_fix = STRESS_FIX
        self._resp: dict[str, str] = {}               # word -> spelling with ё restored (extra lexicon)
        self.measured_targets = MEASURED_TARGETS

    def normalise_word(self, w):
        w = unicodedata.normalize("NFC", C.fold_apos(w)).lower()
        return w

    def _strip_accent(self, w: str) -> tuple[str, int | None]:
        c = unicodedata.normalize("NFC", w)
        if ACUTE not in c:
            return w, None
        vi, idx = -1, None
        for ch in c:
            if ch.lower() in VOWELS:
                vi += 1
            elif ch == ACUTE and vi >= 0:
                idx = vi
        clean = unicodedata.normalize("NFC", c.replace(ACUTE, ""))
        return clean, idx

    def lookup(self, w):
        w2, marked = self._strip_accent(w)
        v = self.stress.get(w2)
        self._src[w] = "lex" if v is not None else "stem"
        if marked is not None:
            self._src[w] = "accent"
            if v is not None and not v.isdigit():
                ph = v.split()
                nuc = [i for i, x in enumerate(ph) if C.is_nucleus_ipa(x.rstrip("1"))]
                if marked < len(nuc) and ph[nuc[marked]].endswith("1"):
                    return [[EXC_VOWEL.get(x.rstrip("1"), x.rstrip("1")) + ("1" if x.endswith("1") else "")
                             for x in ph]]
            return [rules(w2, marked, True)]
        if w2 in CLITICS and (v is None or v.isdigit()):
            self._src[w] = "clitic"
            return [rules(w2, -1, True)]               # unstressed: reduced vowels
        fx = self.stress_fix.get(w2) if self.use_stress_fix and (v is None or v.isdigit()) else None
        if fx:
            self._src[w] = "fix" if len(fx) == 1 else "homograph"
            if fx[0][1]:
                self._resp[w] = fx[0][1]
            return [rules(resp or w2, idx, True) for idx, resp in fx]
        if v is not None:
            return [rules(w2, int(v), True)] if v.isdigit() else [[EXC_VOWEL.get(x.rstrip("1"), x.rstrip("1")) +
                                                                     ("1" if x.endswith("1") else "")
                                                                     for x in v.split()]]
        vow = [c for c in w2 if c in VOWELS]
        if "ё" in vow:
            self._src[w] = "yo"
            return [rules(w2, vow.index("ё"), True)]
        if len(vow) <= 1:
            self._src[w] = "mono"
            return [rules(w2, 0, True)]
        ex = self.extra_stress.get(w2) if self.use_extra_stress else None
        if ex:
            self._src[w] = "extra" if len(ex) == 1 else "homograph"
            if ex[0][1]:
                self._resp[w] = ex[0][1]
            return [rules(resp or w2, idx, True) for idx, resp in ex]
        k = self._stem_stress(w2, len(vow))
        if k is not None:
            return [rules(w2, k, True)]
        return None

    @staticmethod
    def _read_variants(name: str) -> dict[str, list[tuple[int, str | None]]]:
        """word -> [(stressed vowel index, spelling with ё restored or None)] from a lex/ TSV."""
        out: dict[str, list[tuple[int, str | None]]] = {}
        p = C.lex_path(name)
        if p.exists():
            with p.open("rb") as fb, gzip.open(fb, "rt", encoding="utf-8") as f:
                for line in f:
                    if line[:1] == "#" or "\t" not in line:
                        continue
                    k, _, v = line.rstrip("\n").partition("\t")
                    out[k] = [(int(a), b or None) for a, _, b in (x.partition(":") for x in v.split("|"))]
        return out

    def set_stress_fix(self, on: bool) -> None:
        """Switch the issue-#6 lexicon corrections (clears the pronunciation cache)."""
        self.use_stress_fix = bool(on)
        self._src.clear()
        self._resp.clear()
        self._cache.cache_clear()

    def set_extra_stress(self, on: bool) -> None:
        """Switch the round-10 offline stress lexicon (clears the pronunciation cache)."""
        self.use_extra_stress = bool(on)
        self._src.clear()
        self._resp.clear()
        self._cache.cache_clear()

    def _stem_stress(self, w: str, nv: int) -> int | None:
        """Inflected form missing from the lexicon: reuse the stress of forms sharing the stem
        when it falls on the stem (fixed stem stress)."""
        import bisect
        for cut in (1, 2, 3):
            if len(w) - cut < 3:
                break
            stem = w[:-cut]
            sv = sum(c in VOWELS for c in stem)
            if sv == 0:
                continue
            i = bisect.bisect_left(self._keys, stem)
            votes: dict[int, int] = {}
            while i < len(self._keys) and self._keys[i].startswith(stem):
                k = self._keys[i]
                v = self.stress[k]
                if len(k) - len(stem) <= 4 and v.isdigit():
                    votes[int(v)] = votes.get(int(v), 0) + 1
                i += 1
            if votes:
                best = max(votes, key=votes.get)
                if best < sv and votes[best] >= 0.7 * sum(votes.values()):
                    return best
                return None
        return None

    def oov(self, w):
        self._src[w] = "oov"
        w2, _ = self._strip_accent(w)
        if "ё" in w2:
            return rules(w2, [c for c in w2 if c in VOWELS].index("ё"), True)
        nv = sum(c in VOWELS for c in w2)
        if nv == 0:
            return rules(w2, 0, True)
        k = self.stress_model.predict(w2, nv) if self.stress_model else nv - 1
        return rules(w2, k, True)

    def fix_stress(self, phones, word=""):
        return phones

    def rewrite(self, phones, word):
        return phones

    def build_word(self, tok, items, warnings):
        w = super().build_word(tok, items, warnings)
        if not w.phones:
            return w
        if len(items) == 1 and items[0] in CLITICS and len(w.syls) <= 2:
            w.syls = [(a, b, 0) for a, b, _ in w.syls]
            for p in w.phones:
                if p.ipa == "a":
                    p.ipa = "ɑ"                         # ZIPA: на, за, как, так, а -> ɑ (60-77 %)
                    p.add_alt("a", C.P_COMMON)
        stressed = set()
        for a, b, st in w.syls:
            if st == 1:
                stressed.update(range(a, b))
        n = len(w.phones)
        origin = {k: p.ipa for k, p in enumerate(w.phones) if p.ipa in ("t", "tʲ", "d", "dʲ")}
        for k, p in enumerate(w.phones):
            if not p.nucleus and ("ˠ" in p.ipa or ("ː" in p.ipa and p.ipa != "ɕː")):
                base = p.ipa.replace("ˠ", "").replace("ː", "")
                p.ipa = base
                p.add_alt(base + base if len(base) == 1 else base, C.P_MED)
            nxt = w.phones[k + 1] if k + 1 < n else None
            before_c = nxt is None or not nxt.nucleus
            before_i = nxt is not None and nxt.nucleus and nxt.ipa[0] in "iɪ"
            if p.ipa.endswith("ʲ") and not p.nucleus and p.ipa not in ("ɕː", "ʑː"):
                # ZIPA marks palatalisation before e/a/o/u vowels (written ʲ), not before i
                # and rarely before a consonant
                soft = p.ipa
                plain = soft[:-1]
                mapped = {"t": "ts", "d": "dz", "l": "ɭ", "tɕ": "ʈʂ"}.get(plain, plain)
                p.ipa = mapped if (before_c or before_i or plain in PLAIN_BEFORE_V) else mapped + "ʲ"
                for x in dict.fromkeys((soft, mapped, mapped + "ʲ", plain)):  # ordered: a set gave hash-seed order
                    p.add_alt(x, C.P_COMMON)
                continue
            if p.ipa in CONS_ZIPA:
                can, alts = CONS_ZIPA[p.ipa]
                p.ipa = can
                for x in alts:
                    p.add_alt(x, C.P_COMMON)
            elif p.nucleus and k not in stressed and p.ipa in VOWEL_ALTS:
                for x in VOWEL_ALTS[p.ipa]:
                    p.add_alt(x, C.P_COMMON)
            elif p.nucleus and p.ipa in ("ɛ", "o", "a"):
                p.add_alt({"ɛ": "e", "o": "ʌ", "a": "ɑ"}[p.ipa], C.P_COMMON)
            elif p.ipa == "j" and 0 < k < len(w.phones) - 1 and w.phones[k - 1].nucleus and w.phones[k + 1].nucleus:
                p.add_alt("", C.P_MED)
        if self.measured_targets and self.vowel_table and len(items) == 1 and self._src.get(items[0]) in KNOWN_STRESS:
            self._measured_targets(w, self._resp.get(items[0], items[0]), origin)
        return w

    def _set_target(self, p, key) -> bool:
        e = self.vowel_table.get(key)
        if e is None or e[0] < CLASS_MIN_N:
            return False
        dist = e[1]
        prim = max(dist, key=dist.get)
        top = dist[prim]
        keep_drop = [(a, w) for a, w in p.alts if a == ""]
        p.ipa = prim
        p.alts = [(a, math.log(s / top)) for a, s in sorted(dist.items(), key=lambda kv: -kv[1])
                  if a != prim and a and s >= self.vowel_min] + keep_drop
        return True

    def _measured_targets(self, w, word: str, origin: dict) -> None:
        """Round 8: measured primary + alternatives for vowels and t/d (see VOWEL_MIN)."""
        st = [i for i, (_, _, s) in enumerate(w.syls) if s == 1]
        nuc = [p for p in w.phones if p.nucleus]
        keys = vowel_keys(word, st[0] if st else None)
        if len(keys) == len(nuc) == len(w.syls):
            for p, key in zip(nuc, keys):
                self._set_target(p, key)
        n = len(w.phones)
        for k, o in origin.items():
            nx = w.phones[k + 1] if k + 1 < n else None
            self._set_target(w.phones[k], cons_key(o, nx.ipa if nx else None, bool(nx and nx.nucleus)))

    def connected(self, words):
        for i in range(len(words) - 1):
            w, nx = words[i], words[i + 1]
            last, first = w.phones[-1], nx.phones[0]
            fb = first.ipa.replace("ʲ", "")
            lb = last.ipa.replace("ʲ", "")
            soft = "ʲ" if "ʲ" in last.ipa else ""
            # voicing assimilation across the word boundary
            if fb in VOICED_PAIR and fb != "v" and lb in VOICELESS_PAIR:
                last.add_alt(VOICELESS_PAIR[lb] + soft, C.P_COMMON if len(w.phones) <= 2 else C.P_MED)
            if fb in VOICELESS_PAIR and lb in VOICED_PAIR:
                last.add_alt(VOICED_PAIR[lb] + soft, C.P_MED)
            # one-consonant prepositions take the voicing of the next word
            it = w.info.get("items")
            if it in (["в"], ["к"], ["с"]) and len(w.phones) == 1:
                voiced = {"в": "v", "к": "g", "с": "z"}[it[0]]
                voiceless = {"в": "f", "к": "k", "с": "s"}[it[0]]
                if fb in VOICELESS_PAIR or fb in ("x", "ts", "ʈʂ", "ʂʲ", "tɕ", "ɕː", "dz") and fb != "dz":
                    can = voiceless
                elif fb in VOICED_PAIR or fb == "dz":
                    can = voiced if it[0] != "в" else "v"
                else:
                    can = {"в": "v", "к": "k", "с": "s"}[it[0]]
                if last.ipa != can:
                    old = last.ipa
                    last.ipa = can
                    last.alts = [(a, p) for a, p in last.alts if a != can]
                    last.add_alt(old, C.P_MED)
        for w in words:
            ph = w.phones
            for k in range(len(ph) - 1):
                # optional assimilative softening (сн, ст, зд, нт ... before a soft consonant)
                a, b = ph[k], ph[k + 1]
                if a.ipa in ("s", "z", "n", "t", "d") and b.ipa.endswith("ʲ") and not b.nucleus:
                    a.add_alt(a.ipa + "ʲ", C.P_MED)


def build() -> Russian:
    return Russian()
