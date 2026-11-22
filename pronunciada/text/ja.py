"""Japanese text front-end: kanji/kana -> readings (pyopenjtalk-plus / Open JTalk) -> moras ->
ZIPA-convention phones. (Pitch accent is not scored; the full-context labels are read for the
mora segmentation and devoicing only.)

Unit structure: one RefSyllable per mora, except that ZIPA writes geminates as a doubled
consonant and long vowels as V+ː, so a っ mora is merged into the following mora ("ッカ" ->
[kk a]) and a long-vowel mora ー into the preceding one ("トー" -> [t oː]). ん is its own unit
whose nucleus is the (place-assimilated) nasal.
"""
from __future__ import annotations

import re
import threading
import unicodedata

from ..types import Reference
from ._cjk_common import Ph, Syl, Wd, assemble, is_latin
from .base import G2P

# Open JTalk phoneme -> ZIPA consonant
CONS = {
    "k": "k", "g": "g", "s": "s", "sh": "ʃ", "z": "z", "j": "dʒ", "t": "t", "ch": "tʃ", "ts": "ts",
    "d": "d", "n": "n", "h": "h", "f": "f", "b": "b", "p": "p", "m": "m", "y": "j", "r": "r", "w": "w",
    "ky": "kj", "gy": "gj", "ny": "ɲ", "hy": "hj", "by": "bj", "py": "pj", "my": "mj", "ry": "rj",
    "v": "v", "dy": "dj", "ty": "tj",
}
# multi-token consonants are split into (consonant phone, glide phone)
SPLIT = {"hj": ("h", "j"), "kj": ("k", "j"), "gj": ("g", "j"), "bj": ("b", "j"), "pj": ("p", "j"), "mj": ("m", "j"),
         "rj": ("r", "j"), "dj": ("d", "j"), "tj": ("t", "j")}
VOW = {"a": "a", "i": "i", "u": "u", "e": "e", "o": "o"}
VOICELESS = {"k", "s", "sh", "t", "ch", "ts", "h", "f", "p", "ky", "hy", "py", "ty"}

_SMALL = set("ャュョァィゥェォヮ")
_LAB = re.compile(r"\^(.+?)-(.+?)\+(.+?)=")
_ACC = re.compile(r"/A:([\-\d]+|xx)\+(\d+|xx)\+(\d+|xx)")


def kana_moras(pron: str) -> list[str]:
    """'キョーワ' -> ['キョ', 'ー', 'ワ'] (devoicing marks removed)."""
    pron = pron.replace("’", "").replace("'", "")
    out: list[str] = []
    for ch in pron:
        if ch in _SMALL and out:
            out[-1] += ch
        elif "ァ" <= ch <= "ヺ" or ch == "ー":
            out.append(ch)
    return out


class JaG2P(G2P):
    lang = "ja"

    def __init__(self) -> None:
        import pyopenjtalk

        self._pj = pyopenjtalk
        self._lock = threading.Lock()   # Open JTalk instance is not re-entrant
        self("今日は良い天気ですね。東京に行きました、3人で。")

    # ------------------------------------------------------------ main
    def __call__(self, text: str) -> Reference:
        warnings: list[str] = []
        words: list[Wd] = []
        for seg_start, seg in self._segments(text, warnings):
            words.extend(self._segment(seg, seg_start, text, warnings))
        return assemble("ja", text, words, warnings)

    @staticmethod
    def _segments(text: str, warnings: list[str]):
        """Cut out Latin-letter runs (skipped with a warning); yield (offset, chunk)."""
        i, n, start = 0, len(text), 0
        while i < n:
            if is_latin(text[i]):
                j = i
                while j < n and (is_latin(text[j]) or text[j] in "'-. "):
                    j += 1
                warnings.append(f"skipped non-Japanese text: {text[i:j].strip()}")
                if i > start:
                    yield start, text[start:i]
                start = i = j
            else:
                i += 1
        if start < n:
            yield start, text[start:n]

    def _segment(self, seg: str, off: int, text: str, warnings: list[str]) -> list[Wd]:
        if not seg.strip():
            return []
        with self._lock:
            njd = self._pj.run_frontend(seg)
            labels = self._pj.make_label(njd) if any(x["mora_size"] > 0 for x in njd) else []
        # --- moras from the labels: (consonant, vowel/N/cl, a1, a2, devoiced)
        lab_moras: list[list] = []
        cur_c = ""
        for lab in labels:
            m = _LAB.search(lab)
            if not m:
                continue
            ph = m.group(2)
            if ph in ("sil", "pau"):
                cur_c = ""
                continue
            a = _ACC.search(lab)
            a1 = int(a.group(1)) if a and a.group(1) != "xx" else 0
            a2 = int(a.group(2)) if a and a.group(2) != "xx" else 1
            if ph in ("a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "cl"):
                lab_moras.append([cur_c, ph.lower() if ph not in ("N", "cl") else ph, a1, a2, ph in "AIUEO"])
                cur_c = ""
            else:
                cur_c = ph
        # --- words
        wl, pause_before, brk = [], [], True
        for x in njd:
            if x["mora_size"] > 0 and kana_moras(x["pron"]):
                wl.append(x)
                pause_before.append(brk)
                brk = False
            else:
                brk = True
        total = sum(len(kana_moras(x["pron"])) for x in wl)
        if total != len(lab_moras):
            warnings.append("full-context labels out of sync; kana readings used for: " + seg.strip())
            lab_moras = []
        spans = self._spans([x["string"] for x in wl], seg, off)
        # --- build mora records over the whole segment
        recs: list[dict] = []
        k = 0
        for wi, x in enumerate(wl):
            km = kana_moras(x["pron"])
            read = x.get("read", "")
            for j, kana in enumerate(km):
                lm = lab_moras[k] if lab_moras else None
                k += 1
                recs.append({"w": wi, "kana": kana, "lab": lm, "read": read, "pos": x["pos"],
                             "first": j == 0, "string": x["string"], "pause": j == 0 and pause_before[wi]})
        return self._build(recs, wl, spans, text)

    @staticmethod
    def _spans(strings: list[str], seg: str, off: int) -> list[tuple[int, int]]:
        """Map analyser word strings back to offsets in the original text. Words the analyser
        rewrote (numbers -> kanji) get the gap between their found neighbours."""
        norm = [unicodedata.normalize("NFKC", c) for c in seg]
        flat = "".join(norm)
        idx = []                       # flat position -> seg index
        for i, c in enumerate(norm):
            idx.extend([i] * len(c))
        idx.append(len(seg))
        found: list[tuple[int, int] | None] = []
        pos = 0
        for s in strings:
            s2 = unicodedata.normalize("NFKC", s)
            j = flat.find(s2, pos) if s2 else -1
            gap = flat[pos:j] if j >= 0 else ""
            if j >= 0 and (j - pos <= 12 or not re.search(r"[\w]", gap)):
                found.append((idx[j], idx[j + len(s2)]))
                pos = j + len(s2)
            else:
                found.append(None)
        # fill gaps
        out: list[tuple[int, int]] = []
        for i, f in enumerate(found):
            if f is not None:
                out.append((f[0] + off, f[1] + off))
                continue
            prev_end = next((found[q][1] for q in range(i - 1, -1, -1) if found[q] is not None), 0)
            nxt = next((found[q][0] for q in range(i + 1, len(found)) if found[q] is not None), len(seg))
            while prev_end < nxt and seg[prev_end].isspace():
                prev_end += 1
            out.append((prev_end + off, max(prev_end, nxt) + off))
        return out

    # ------------------------------------------------------------ phones
    def _build(self, recs: list[dict], wl, spans, text: str) -> list[Wd]:
        words: list[Wd] = []
        wmap: dict[tuple[int, int], Wd] = {}
        widx: dict[int, Wd] = {}
        for wi in range(len(wl)):
            sp = spans[wi]
            if sp in wmap and sp[1] > sp[0]:
                widx[wi] = wmap[sp]     # analyser split one written token (e.g. digits): one word
                continue
            wd = Wd(text=text[sp[0]:sp[1]] or wl[wi]["string"], span=sp)
            wmap[sp] = widx[wi] = wd
            words.append(wd)
        from ..ja_round19 import on as _r19

        merge_on = _r19("merge")
        n = len(recs)
        i = 0
        pending_gem: dict | None = None
        last_syl: Syl | None = None
        last_rec: dict | None = None
        while i < n:
            r = recs[i]
            lab = r["lab"]
            kana = r["kana"]
            cons, vow = (lab[0], lab[1]) if lab else self._kana_ph(kana)
            nxt = recs[i + 1] if i + 1 < n and not recs[i + 1]["pause"] else None
            nxt_cons = None
            if nxt is not None:
                nxt_cons = (nxt["lab"][0] if nxt["lab"] else self._kana_ph(nxt["kana"])[0])
            wd = widx[r["w"]]
            # っ: merge into the following mora (doubled consonant) when it has a consonant
            if vow == "cl":
                if nxt is not None and nxt_cons:
                    pending_gem = {"kana": kana}
                    i += 1
                    continue
                # word/phrase-final っ (あっ): glottal stop closing the previous mora
                if last_syl is not None:
                    last_syl.text += kana
                    p = Ph("ʔ")
                    p.alt("", -0.5)
                    last_syl.phones.append(p)
                i += 1
                continue
            # round 19: a bare vowel mora repeating the previous mora's vowel inside the same word
            # is said as one long vowel (場合 バアイ [baːi], 思惟 シイ [ʃiː], 大きい オーキイ
            # [oːkiː]); written as ー would be, so the length channel judges it as long
            rep = (merge_on and not cons and vow in VOW and pending_gem is None and last_syl is not None
                   and last_rec is not None and last_rec["w"] == r["w"] and not r["pause"]
                   and not (lab is not None and lab[4])
                   and next((p.ipa for p in last_syl.phones if p.nucleus), "") == VOW[vow])
            # ー (long vowel): merge into the previous mora's nucleus
            if (kana == "ー" or rep) and last_syl is not None and last_rec is not None and not r["pause"]:
                nuc = next(p for p in last_syl.phones if p.nucleus)
                if not nuc.ipa.endswith("ː"):
                    base = nuc.ipa
                    nuc.ipa = base + "ː"
                    nuc.alts = [(a + "ː" if a and a[-1] not in "ː" else a, pr) for a, pr in nuc.alts
                                if a and not (rep and a[0] not in "aiueoɯ")]
                    nuc.alt(base, -0.7)                  # shortened in fast speech
                    spelled = None if rep else self._spelled_long(last_rec, base)
                    if spelled:
                        nuc.alt(spelled, -1.0)           # spelling-pronunciation おう -> [ou], えい -> [ei]
                last_syl.text += kana
                i += 1
                continue
            phones = self._mora_phones(cons, vow, r, nxt_cons, lab, i, recs)
            text_m = kana
            if pending_gem is not None:
                text_m = pending_gem["kana"] + kana
                c0 = phones[0]
                if not c0.nucleus:
                    single = c0.ipa
                    c0.ipa = self._geminate(single)
                    c0.alts = [(self._geminate(a), p) for a, p in c0.alts if a]
                    c0.alt(single, -1.0)                 # weak / missing gemination
                pending_gem = None
            syl = Syl(text=text_m, phones=phones)
            wd.syls.append(syl)
            last_syl, last_rec = syl, r
            i += 1
        return words

    @staticmethod
    def _geminate(c: str) -> str:
        """ZIPA writes geminates as a doubled consonant: kk, tt, ss, ʃʃ; affricates ttʃ, tts."""
        if c in ("tʃ", "ts", "dʒ", "dz"):
            return "t" + c if c[0] == "t" else "d" + c
        return c[0] + c

    @staticmethod
    def _spelled_long(rec: dict, vowel: str) -> str | None:
        read = rec.get("read", "")
        if vowel == "o" and "ウ" in read:
            return "ou"
        if vowel == "e" and "イ" in read:
            return "ei"
        return None

    @staticmethod
    def _kana_ph(kana: str) -> tuple[str, str]:
        """Fallback when labels are unavailable: rough kana -> (consonant, vowel)."""
        if kana == "ン":
            return "", "N"
        if kana == "ッ":
            return "", "cl"
        if kana == "ー":
            return "", "a"
        try:
            name = unicodedata.name(kana[0])
        except ValueError:
            return "", "a"
        rom = name.split()[-1].lower()
        v = rom[-1] if rom[-1] in "aiueo" else "a"
        c = rom[:-1]
        return {"si": "sh", "ti": "ch", "tu": "ts", "hu": "f", "zi": "j"}.get(rom, c), v

    def _mora_phones(self, cons: str, vow: str, r: dict, nxt_cons, lab, i: int, recs) -> list[Ph]:
        out: list[Ph] = []
        if vow == "N":
            nas = self._moraic_nasal(nxt_cons, i + 1 < len(recs))
            p = Ph(nas, nucleus=True)
            for a in ("n", "m", "ŋ", "ɴ"):
                p.alt(a, -0.7)
            out.append(p)
            return out
        prev_vowel = i > 0 and recs[i - 1]["lab"] is not None and recs[i - 1]["lab"][1] in "aiueo"
        if cons:
            c = CONS.get(cons, cons)
            # ZIPA never writes ç / ɸ for Japanese: ひ -> h (49 %), ふ -> f (33 %, h 23 %)
            if cons == "h" and vow == "u":
                c = "f"
            if c in SPLIT:
                c0, g = SPLIT[c]
                p = Ph(c0)
                out.append(p)
                out.append(Ph(g))
            else:
                p = Ph(c)
                if cons == "g" and not r["first"]:
                    p.alt("ŋ", -0.7)                    # nasal g, word-medial
                if cons == "r":
                    p.alt("ɾ", -0.3)
                    p.alt("l", -0.7)
                    p.alt("d", -1.5)
                if cons == "z":
                    p.alt("dz", -0.5)
                if cons == "j":
                    p.alt("ʒ", -0.5)
                if c == "f":
                    p.alt("ɸ", -0.3)
                    p.alt("h", -0.5)
                if cons == "h" and vow == "i":
                    p.alt("ç", -0.3)
                if cons == "w":
                    p.alt("v", -0.5)                    # ZIPA hears わ / particle は as [va] 29 %
                out.append(p)
        v = Ph(VOW.get(vow, vow), nucleus=True)
        if vow == "u":
            v.alt("ɯ", -0.3)
        if lab is not None and lab[4]:
            v.alt("", -0.2)                             # devoiced vowel (desu, mashita)
        elif vow in ("i", "u") and cons in VOICELESS and nxt_cons in VOICELESS:
            v.alt("", -0.7)                             # devoicing environment
        if not cons and vow in ("i", "e") and prev_vowel:
            v.alt("j", -0.3)                            # ai -> [aj], ae -> [aj]
        if not cons and vow == "o" and r["kana"] == "ヲ":
            v.alt("wo", -1.0)
        out.append(v)
        return out

    @staticmethod
    def _moraic_nasal(nxt: str | None, has_next: bool) -> str:
        if not has_next or nxt is None:
            return "ŋ"
        if nxt in ("p", "b", "m", "py", "by", "my"):
            return "m"
        if nxt in ("k", "g", "ky", "gy"):
            return "ŋ"
        if nxt in ("t", "d", "n", "ts", "ch", "j", "z", "r", "ny", "ty", "dy", "ry"):
            return "n"
        return "ŋ"


def build() -> G2P:
    return JaG2P()
