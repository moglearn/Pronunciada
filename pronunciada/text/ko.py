"""Korean text front-end: Hangul -> jamo -> standard pronunciation rules -> ZIPA-convention IPA.

Implemented in-house (no external analyser): the standard pronunciation rules (표준 발음법) that
can be applied from spelling alone. Within an eojeol (space-separated word) they are canonical;
across eojeol boundaries inside a phrase they become alternatives (fluent connected speech).
"""
from __future__ import annotations

import unicodedata

from ..types import Reference
from ._cjk_common import Ph, Syl, Wd, assemble, is_latin, nfkc_digit
from .base import G2P

L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
T = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
     "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

# clusters: (stays in coda, moves to next onset before a vowel)
CLUSTER = {"ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"), "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"),
           "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("ㄹ", "ㅎ"),
           "ㅄ": ("ㅂ", "ㅅ")}
# coda neutralisation (7 representative codas)
NEUTRAL = {"ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㄳ": "ㄱ", "ㄺ": "ㄱ", "ㄴ": "ㄴ", "ㄵ": "ㄴ", "ㄶ": "ㄴ",
           "ㄷ": "ㄷ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ", "ㄹ": "ㄹ",
           "ㄼ": "ㄹ", "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㅀ": "ㄹ", "ㅁ": "ㅁ", "ㄻ": "ㅁ", "ㅂ": "ㅂ", "ㅍ": "ㅂ",
           "ㅄ": "ㅂ", "ㄿ": "ㅂ", "ㅇ": "ㅇ"}
TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
LAX_TENSE = frozenset(TENSE) | frozenset(TENSE.values())
ASPIRATE = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}
NASAL = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}
OBSTRUENT_CODA = set("ㄱㄷㅂ")

# ---------------------------------------------------------------- ZIPA conventions
# ZIPA writes tense consonants as plain unaspirated stops (63-79 % on FLEURS) and ㄹ as [r].
ONSET_IPA = {"ㄱ": "k", "ㄲ": "k", "ㄴ": "n", "ㄷ": "t", "ㄸ": "t", "ㄹ": "r", "ㅁ": "m", "ㅂ": "p", "ㅃ": "p",
             "ㅅ": "s", "ㅆ": "s", "ㅈ": "tʃ", "ㅉ": "tʃ", "ㅊ": "tʃʰ", "ㅋ": "kʰ", "ㅌ": "tʰ", "ㅍ": "pʰ", "ㅎ": "h"}
VOICED_ONSET = {"ㄱ": "g", "ㄷ": "d", "ㅂ": "b", "ㅈ": "dʒ"}
# lax stops between voiced sounds: ZIPA ㄷ d 40 % / t 42-55 %, ㅂ b 61 %, ㄱ g 27 %, ㅈ dʒ ~30 %
MEDIAL_LAX = {"ㄱ": "k", "ㄷ": "t", "ㅂ": "b", "ㅈ": "tʃ"}
CODA_IPA = {"ㄱ": "k", "ㄴ": "n", "ㄷ": "t", "ㄹ": "l", "ㅁ": "m", "ㅂ": "p", "ㅇ": "ŋ"}
# vowel -> (glide, nucleus). ZIPA: ㅓ -> ɔ (43 %, o 17 %, a 15 %); ㅐ/ㅔ merged -> e (62 % / 67 %);
# ㅗ o 43 % / u 37 %; ㅡ is never written ɯ (see _eu below).
VOWEL_IPA = {"ㅏ": ("", "a"), "ㅐ": ("", "e"), "ㅑ": ("j", "a"), "ㅒ": ("j", "e"), "ㅓ": ("", "ɔ"),
             "ㅔ": ("", "e"), "ㅕ": ("j", "ɔ"), "ㅖ": ("j", "e"), "ㅗ": ("", "o"), "ㅘ": ("w", "a"),
             "ㅙ": ("w", "e"), "ㅚ": ("w", "e"), "ㅛ": ("j", "o"), "ㅜ": ("", "u"), "ㅝ": ("w", "ɔ"),
             "ㅞ": ("w", "e"), "ㅟ": ("w", "i"), "ㅠ": ("j", "u"), "ㅡ": ("", "ɯ"), "ㅢ": ("", "ɯi"),
             "ㅣ": ("", "i")}

SINO = "영일이삼사오육칠팔구"


def decompose(ch: str) -> tuple[str, str, str] | None:
    o = ord(ch) - 0xAC00
    if not 0 <= o < 11172:
        return None
    return L[o // 588], V[(o % 588) // 28], T[o % 28]


def _sino_int(n: int) -> str:
    if n == 0:
        return "영"
    out = ""
    for u, name in ((10 ** 12, "조"), (10 ** 8, "억"), (10 ** 4, "만")):
        if n >= u:
            hi, n = divmod(n, u)
            out += ("" if (hi == 1 and name == "만") else _sino_int(hi)) + name
    for u, name in ((1000, "천"), (100, "백"), (10, "십"), (1, "")):
        d, n = divmod(n, u)
        if d:
            out += ("" if d == 1 and name else SINO[d]) + name
    return out


def number_reading(tok: str) -> str:
    pct = tok.endswith("%")
    tok = tok.rstrip("%").replace(",", "")
    if "." in tok:
        ip, fp = tok.split(".", 1)
        s = _sino_int(int(ip or "0")) + "점" + "".join(SINO[int(c)] for c in fp if c.isdigit())
    elif len(tok) > 1 and tok[0] == "0":
        s = "".join("공" if c == "0" else SINO[int(c)] for c in tok)
    else:
        s = _sino_int(int(tok)) if len(tok) <= 16 else "".join(SINO[int(c)] for c in tok)
    return s + ("퍼센트" if pct else "")


class KoG2P(G2P):
    lang = "ko"

    def __init__(self) -> None:
        self("국물이 같이 좋다. 신라 읽다 123")

    def __call__(self, text: str) -> Reference:
        warnings: list[str] = []
        phrases = self._scan(text, warnings)
        words: list[Wd] = []
        for ph in phrases:
            words.extend(self._phrase(ph, text))
        return assemble("ko", text, words, warnings)

    # ------------------------------------------------------------ scanning
    def _scan(self, text: str, warnings: list[str]):
        """phrases -> words -> list of (block_text, (L, V, T)) plus char span."""
        phrases: list[list[tuple[tuple[int, int], list[tuple[str, tuple[str, str, str]]]]]] = []
        cur_phrase: list = []
        cur_word: list = []
        wstart = None
        i, n = 0, len(text)

        def end_word(pos):
            nonlocal cur_word, wstart
            if cur_word:
                cur_phrase.append(((wstart, pos), cur_word))
            cur_word, wstart = [], None

        def end_phrase(pos):
            nonlocal cur_phrase
            end_word(pos)
            if cur_phrase:
                phrases.append(cur_phrase)
            cur_phrase = []

        while i < n:
            ch = text[i]
            d = decompose(ch)
            if d is not None:
                if wstart is None:
                    wstart = i
                cur_word.append((ch, d))
                i += 1
            elif nfkc_digit(ch) is not None:
                j = i
                while j < n and (nfkc_digit(text[j]) is not None or (
                        unicodedata.normalize("NFKC", text[j]) in ".," and j + 1 < n and nfkc_digit(text[j + 1]) is not None)):
                    j += 1
                if j < n and unicodedata.normalize("NFKC", text[j]) == "%":
                    j += 1
                tok = unicodedata.normalize("NFKC", text[i:j])
                if wstart is None:
                    wstart = i
                for c in number_reading(tok):
                    cur_word.append((c, decompose(c)))
                i = j
            elif is_latin(ch):
                j = i
                while j < n and (is_latin(text[j]) or text[j] in "'-"):
                    j += 1
                warnings.append(f"skipped non-Korean text: {text[i:j]}")
                end_phrase(i)
                i = j
            elif ch.isspace():
                end_word(i)
                i += 1
            else:
                cat = unicodedata.category(ch)
                if cat[0] in "PS":
                    end_phrase(i)
                elif 0x3131 <= ord(ch) <= 0x318E or 0x1100 <= ord(ch) <= 0x11FF:
                    warnings.append(f"skipped isolated jamo: {ch}")
                elif cat[0] not in "ZC":
                    warnings.append(f"skipped unsupported character: {ch}")
                    end_phrase(i)
                i += 1
        end_phrase(n)
        return phrases

    # ------------------------------------------------------------ rules
    @staticmethod
    def _pair(c: str, o: str, v2: str, within: bool) -> tuple[str, str, str]:
        """Apply the boundary rules to (coda c of syllable i, onset o of syllable i+1, vowel v2 of i+1).
        o == 'ㅇ' means no onset. Returns (new coda, new onset, tag)."""
        tag = ""
        # --- ㅎ coda
        if c in ("ㅎ", "ㄶ", "ㅀ"):
            rest = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[c]
            if o in ASPIRATE:
                return rest, ASPIRATE[o], "asp"
            if o == "ㅅ":
                return rest, "ㅆ", "asp"
            if o == "ㅇ":
                if rest:
                    return "", rest if rest != "ㄹ" else "ㄹ", "liaison"
                return "", "ㅇ", "hdrop"
            if o == "ㄴ":
                return ("ㄴ" if c == "ㅎ" else rest), "ㄴ" if rest != "ㄹ" else "ㄹ", "nasal"
            c = rest or "ㄷ"
        # --- obstruent coda + ㅎ -> aspiration
        if o == "ㅎ" and c:
            if c in CLUSTER and CLUSTER[c][1] in ASPIRATE:
                return CLUSTER[c][0], ASPIRATE[CLUSTER[c][1]], "asp"
            nc = NEUTRAL.get(c, c)
            if nc in ASPIRATE and c not in ("ㄹ",):
                if nc == "ㄷ" and v2 == "ㅣ" and c in ("ㄷ", "ㅌ"):
                    return "", "ㅊ", "pal"
                if c in ("ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅌ", "ㄷ"):
                    return "", "ㅌ" if c != "ㅈ" else "ㅊ", "asp"
                return "", ASPIRATE[nc], "asp"
        # --- liaison (vowel-initial next syllable)
        if o == "ㅇ" and c and c != "ㅇ":
            if v2 == "ㅣ" and c in ("ㄷ", "ㅌ", "ㄾ"):
                stay = "ㄹ" if c == "ㄾ" else ""
                return stay, "ㅈ" if c == "ㄷ" else "ㅊ", "pal"
            if c in CLUSTER:
                stay, mv = CLUSTER[c]
                if mv == "ㅅ":
                    mv = "ㅆ"            # 값이 -> 갑씨
                return stay, mv, "liaison"
            if not within:
                # across words: neutralise first, then link (옷 안 -> 오단)
                return "", NEUTRAL[c], "liaison"
            return "", c, "liaison"
        if not c:
            return c, o, tag
        # --- neutralise remaining coda (with cluster choice)
        if c == "ㄺ":
            nc = "ㄹ" if o == "ㄱ" else "ㄱ"
        else:
            nc = NEUTRAL[c]
        tense_trigger = nc in OBSTRUENT_CODA or c in ("ㄵ", "ㄻ", "ㄼ", "ㄾ")
        # --- ㄹ rules
        if o == "ㄹ":
            if nc == "ㄴ":
                return "ㄹ", "ㄹ", "lat"      # 신라 -> 실라
            if nc == "ㄹ":
                return "ㄹ", "ㄹ", ""
            if nc in ("ㅁ", "ㅇ"):
                return nc, "ㄴ", "nasal"      # 침략 -> 침냑
            if nc in ("ㄱ", "ㅂ", "ㄷ"):
                return NASAL[nc], "ㄴ", "nasal"   # 국립 -> 궁닙
        if nc == "ㄹ" and o == "ㄴ":
            return "ㄹ", "ㄹ", "lat"          # 설날 -> 설랄
        # --- nasalisation
        if nc in NASAL and o in ("ㄴ", "ㅁ"):
            return NASAL[nc], o, "nasal"
        # --- tensification
        if tense_trigger and o in TENSE:
            return nc, TENSE[o], "tense"
        return nc, o, tag

    def _apply(self, syls: list[list[str]], within: bool = True) -> list[str]:
        """In-place rule application on [L, V, T] lists of one eojeol. Returns tags per boundary."""
        tags = []
        for k in range(len(syls) - 1):
            a, b = syls[k], syls[k + 1]
            nc, no, tag = self._pair(a[2], b[0], b[1], within)
            a[2], b[0] = nc, no
            tags.append(tag)
        if syls:
            last = syls[-1]
            if last[2]:
                last[2] = {"ㄺ": "ㄱ", "ㄼ": "ㄹ"}.get(last[2], NEUTRAL[last[2]])
        return tags

    # ------------------------------------------------------------ build
    def _phrase(self, phrase, text: str) -> list[Wd]:
        # canonical: rules inside each eojeol only
        canon = []
        for span, blocks in phrase:
            sy = [list(d) for _, d in blocks]
            for k, (ch, _) in enumerate(blocks):
                if ch == "밟" and (k + 1 == len(sy) or sy[k + 1][0] != "ㅇ"):
                    sy[k][2] = "ㅂ"            # 밟다 -> 밥따 (exception to ㄼ -> ㄹ)
            self._apply(sy)
            canon.append(sy)
        out: list[Wd] = []
        prev_last: list[Ph] | None = None
        prev_raw = None
        for wi, ((span, blocks), sy) in enumerate(zip(phrase, canon)):
            wd = Wd(text=text[span[0]:span[1]], span=span)
            for si, ((orig, raw), (l, v, t)) in enumerate(zip(blocks, sy)):
                prev_t = sy[si - 1][2] if si > 0 else (canon[wi - 1][-1][2] if wi > 0 else None)
                first_of_word = si == 0
                phones = self._syl(l, v, t, raw, prev_t, first_of_word, wi == 0 and si == 0,
                                   last_of_word=si == len(blocks) - 1)
                wd.syls.append(Syl(text=orig, phones=phones))
            if wi > 0 and prev_last is not None and wd.syls:
                self._cross(prev_raw, canon[wi - 1][-1], blocks[0][1], sy[0], prev_last, wd.syls[0].phones)
            if wd.syls:
                prev_last = wd.syls[-1].phones
                prev_raw = phrase[wi][1][-1][1]
            out.append(wd)
        return out

    def _syl(self, l: str, v: str, t: str, raw, prev_t, first_of_word: bool, utt_initial: bool,
             last_of_word: bool) -> list[Ph]:
        out: list[Ph] = []
        voiced_ctx = prev_t is not None and (prev_t in ("", "ㄴ", "ㄹ", "ㅁ", "ㅇ"))
        if l != "ㅇ":
            ipa = ONSET_IPA[l]
            if l == "ㄹ" and prev_t == "ㄹ":
                ipa = "l"
            elif l in MEDIAL_LAX and voiced_ctx and not first_of_word:
                ipa = MEDIAL_LAX[l]
            elif l in ("ㅅ", "ㅆ") and v in ("ㅣ", "ㅟ") or (l in ("ㅅ", "ㅆ") and VOWEL_IPA[v][0] == "j"):
                ipa = "ʃ"                             # ZIPA: 시 -> ʃi (52 %)
            p = Ph(ipa)
            if l in LAX_TENSE:                        # round 18: lax / tense marker (same symbol)
                p.tense = l in TENSE.values()
            if l in VOICED_ONSET:
                p.alt(VOICED_ONSET[l] if ipa != VOICED_ONSET[l] else ONSET_IPA[l], -0.3)
                if l == "ㅈ":
                    p.alt("ts", -0.7)
            if l == "ㄹ" and ipa == "r":
                p.alt("ɾ", -0.3)
                p.alt("l", -0.7)
            if ipa == "ʃ":
                p.alt("s", -0.3)
            if l == "ㅎ":
                p.alt("", -0.5 if voiced_ctx else -1.0)   # h-weakening: 전화 -> 저놔
            if l == "ㄴ" and prev_t == "ㅁ":
                p.alt("", -0.5)                       # 습니다 -> [sɯmida]
            if l in ("ㅁ", "ㄴ") and first_of_word:
                p.alt("b" if l == "ㅁ" else "d", -0.5)    # word-initial denasalisation (ZIPA b 55 %)
            if l in ("ㅊ", "ㅋ", "ㅌ", "ㅍ"):
                p.alt(ONSET_IPA[l].rstrip("ʰ"), -1.0)     # weak aspiration (ZIPA drops ʰ 30-50 %)
            out.append(p)
        glide, nuc = VOWEL_IPA[v]
        if v == "ㅢ":
            if l != "ㅇ":
                glide, nuc = "", "i"                   # 희 -> hi
            elif not first_of_word:
                nuc = "i"                             # 주의 -> 주이 (ɯi also standard)
        if glide:
            out.append(Ph(glide))
        if v == "ㅡ":
            nuc = self._eu(l)
        n = Ph(nuc, nucleus=True)
        if nuc == "e":
            n.alt("ɛ", -0.3)                           # ㅐ/ㅔ merger
        if v in ("ㅓ", "ㅕ", "ㅝ"):
            n.alt("ʌ", -0.3)
            n.alt("o", -0.7)
        if v in ("ㅗ", "ㅛ"):
            if t == "ㅇ":
                n.ipa = "u"                            # ZIPA: 공 통 동 -> kuŋ (67-84 %)
                n.alt("o", -0.3)
            else:
                n.alt("u", -0.3)                       # ㅗ raising
        if v == "ㅡ":
            for a in ("ɯ", "ə", "u", "ɨ", "e"):
                n.alt(a, -0.3 if a in ("ɯ", "ə") else -0.7)
            if l in ("ㅅ", "ㅆ", "ㅎ", "ㅋ", "ㅌ", "ㅍ", "ㅊ"):
                n.alt("", -1.0)                        # devoiced / deleted after voiceless C
        if v == "ㅢ":
            n.alt("ɯi" if nuc == "i" else "i", -0.5)
            if l == "ㅇ" and last_of_word and not first_of_word:
                n.alt("e", -0.5)                       # genitive 의 -> 에
        out.append(n)
        if t:
            out.append(Ph(CODA_IPA[t]))
        return out

    @staticmethod
    def _eu(l: str) -> str:
        """ZIPA realisation of ㅡ by onset: u after labials, ə after velars and ㅅ, e elsewhere."""
        if l in ("ㅂ", "ㅍ", "ㅃ"):
            return "u"
        if l in ("ㄱ", "ㅋ", "ㄲ", "ㅅ", "ㅆ"):
            return "ə"
        return "e"

    def _cross(self, a_raw, a_can, b_raw, b_can, a_ph: list[Ph], b_ph: list[Ph]) -> None:
        """Fluent-speech alternatives across an eojeol boundary inside a phrase."""
        c = a_raw[2]
        if not c:
            return
        nc, no, tag = self._pair(c, b_raw[0], b_raw[1], within=False)
        coda = a_ph[-1] if not a_ph[-1].nucleus else None
        onset = b_ph[0] if b_can[0] != "ㅇ" else None
        if coda is None:
            return
        if b_raw[0] == "ㅇ" and b_raw[1] in ("ㅣ", "ㅑ", "ㅕ", "ㅛ", "ㅠ") and a_can[2]:
            # ㄴ-insertion across words in fluent speech: 할 일 -> [할릴], 무슨 일 -> [무슨닐]
            ins = {"ㄹ": "ll", "ㄱ": "ŋn", "ㄷ": "nn", "ㅂ": "mn"}.get(a_can[2], CODA_IPA[a_can[2]] + "n")
            coda.alt(ins, -1.0)
        if tag == "liaison" and no in CODA_IPA or (tag == "liaison" and no in ONSET_IPA):
            # coda linked to the next vowel: voiced / flapped realisation of the same coda slot
            linked = {"ㄱ": "g", "ㄷ": "d", "ㅂ": "b", "ㄹ": "ɾ"}.get(no, ONSET_IPA.get(no, coda.ipa))
            coda.alt(linked, -0.5)
            return
        if tag in ("nasal", "lat", "asp", "tense", "pal"):
            if nc:
                coda.alt(CODA_IPA.get(NEUTRAL.get(nc, nc), coda.ipa), -0.7)
            else:
                coda.alt("", -0.7)
            if onset is not None and no != "ㅇ":
                ipa = ONSET_IPA[no]
                if no == "ㄹ" and nc == "ㄹ":
                    ipa = "l"
                onset.alt(ipa, -0.7)


def build() -> G2P:
    return KoG2P()
