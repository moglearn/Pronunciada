"""Mandarin Chinese text front-end (Simplified and Traditional input).

Pipeline: original text -> (Traditional -> Simplified with OpenCC, length-preserving) ->
jieba words -> pypinyin (phrase-aware) -> lexical tones -> neutral tone / 一 / 不 / third-tone
sandhi -> ZIPA-convention phones (initial, glide, nucleus, coda) + connected-speech alternatives.

Everything after the script conversion runs on Simplified text, so equivalent Simplified and
Traditional prompts give identical phones and tones; only `text` fields keep the user's script.
"""
from __future__ import annotations

import re
import threading
import unicodedata

from ..types import Reference
from ._cjk_common import Ph, Syl, Wd, assemble, data_lines, is_han, is_latin, nfkc_digit
from .base import G2P

# --------------------------------------------------------------------------- pinyin -> ZIPA

INITIALS = {
    "b": "p", "p": "pʰ", "m": "m", "f": "f", "d": "t", "t": "tʰ", "n": "n", "l": "l",
    "g": "k", "k": "kʰ", "h": "x", "j": "tɕ", "q": "tɕʰ", "x": "ɕ",
    "zh": "ʈʂ", "ch": "ʈʂʰ", "sh": "ʂ", "r": "ʐ", "z": "ts", "c": "tsʰ", "s": "s",
}
_INIT_RE = re.compile(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*)$")

# final (after y/w/ü/iu/ui/un normalisation) -> (glide, nucleus, coda) in ZIPA convention
FINALS: dict[str, tuple[str, str, str]] = {
    "a": ("", "a", ""), "o": ("", "ɔ", ""), "e": ("", "ɤ", ""), "ê": ("", "ɛ", ""),
    "ai": ("", "aɪ", ""), "ei": ("", "eɪ", ""), "ao": ("", "ɑʊ", ""), "ou": ("", "oʊ", ""),
    "an": ("", "a", "n"), "en": ("", "ə", "n"), "ang": ("", "ɑ", "ŋ"), "eng": ("", "ɤ", "ŋ"),
    "ong": ("", "ʊ", "ŋ"), "er": ("", "ɑɻ", ""),
    "i": ("", "i", ""), "ia": ("j", "a", ""), "ie": ("j", "ɛ", ""), "iao": ("j", "ɑʊ", ""),
    "iou": ("j", "oʊ", ""), "ian": ("j", "ɛ", "n"), "in": ("", "i", "n"), "iang": ("j", "ɑ", "ŋ"),
    "ing": ("", "i", "ŋ"), "iong": ("j", "ʊ", "ŋ"), "io": ("j", "ɔ", ""),
    "u": ("", "u", ""), "ua": ("w", "a", ""), "uo": ("w", "ɔ", ""), "uai": ("w", "aɪ", ""),
    "uei": ("w", "eɪ", ""), "uan": ("w", "a", "n"), "uen": ("w", "ə", "n"), "uang": ("w", "ɑ", "ŋ"),
    "ueng": ("w", "ə", "ŋ"),
    "ü": ("", "y", ""), "üe": ("ɥ", "œ", ""), "üan": ("ɥ", "œ", "n"), "ün": ("", "y", "n"),
    "-i": ("", "ɯ", ""),      # apical vowel after z c s (ZIPA: ɯ ~70-90 %)
    "-ir": ("", "ɨ", ""),     # apical vowel after zh ch sh r
    # syllabic nasals / interjections
    "m": ("", "m", ""), "n": ("", "n", ""), "ng": ("", "ŋ", ""), "hm": ("", "m", ""), "hng": ("", "ŋ", ""),
}
_Y = {"yi": "i", "ya": "ia", "ye": "ie", "yao": "iao", "you": "iou", "yan": "ian", "yin": "in",
      "yang": "iang", "ying": "ing", "yong": "iong", "yo": "io", "yu": "ü", "yue": "üe", "yuan": "üan",
      "yun": "ün"}
_W = {"wu": "u", "wa": "ua", "wo": "uo", "wai": "uai", "wei": "uei", "wan": "uan", "wen": "uen",
      "wang": "uang", "weng": "ueng"}


def parse_pinyin(base: str) -> tuple[str, str] | None:
    """'zhong' -> ('zh', 'ong'); 'yu' -> ('', 'ü'); returns (initial, normalised final)."""
    base = base.replace("v", "ü").replace("u:", "ü")
    if base in ("m", "n", "ng", "hm", "hng"):
        return "", base
    if base in ("e", "ê", "eh"):
        return "", "ê" if base != "e" else "e"
    if base in _Y:
        return "", _Y[base]
    if base in _W:
        return "", _W[base]
    m = _INIT_RE.match(base)
    if not m:
        return None
    ini, fin = m.group(1), m.group(2)
    if ini in ("y", "w"):
        return None
    if ini in ("j", "q", "x") and fin.startswith("u"):
        fin = "ü" + fin[1:]
    if ini and fin == "iu":
        fin = "iou"
    elif ini and fin == "ui":
        fin = "uei"
    elif ini and fin == "un":
        fin = "uen" if ini not in ("j", "q", "x") else "ün"
    if fin == "i" and ini in ("z", "c", "s"):
        fin = "-i"
    elif fin == "i" and ini in ("zh", "ch", "sh", "r"):
        fin = "-ir"
    if ini in ("b", "p", "m", "f") and fin == "o":
        fin = "uo"           # bo po mo fo are [pwɔ] etc.
    if fin not in FINALS:
        return None
    return ini, fin


# --------------------------------------------------------------------------- numbers

_DIG = "零一二三四五六七八九"
_MEASURE = set("个只件位名本条张次天岁种家口人辆台把双对项场倍份座艘架头匹间所棵枝支颗粒块片层趟遍回声句首部群批组周")


def _int_reading(n: int) -> str:
    if n == 0:
        return "零"
    units = [(10 ** 12, "万亿"), (10 ** 8, "亿"), (10 ** 4, "万")]
    for u, name in units:
        if n >= u:
            hi, lo = divmod(n, u)
            s = _int_reading(hi) + name
            if lo:
                s += ("零" if lo < u // 10 else "") + _int_reading(lo)
            return s
    s, zero = "", False
    for u, name in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d, n = divmod(n, u)
        if d:
            if zero and s:
                s += "零"
            s += _DIG[d] + name
            zero = False
        elif s:
            zero = True
    return s


def number_reading(tok: str, nxt: str, prev: str) -> str:
    """Chinese reading (Simplified) of a digit token like '2019', '3.5', '7,000', '45%'."""
    pct = tok.endswith("%")
    tok = tok.rstrip("%")
    tok_nc = tok.replace(",", "")
    if "." in tok_nc:
        ip, fp = tok_nc.split(".", 1)
        s = _int_reading(int(ip or "0")) + "点" + "".join(_DIG[int(c)] for c in fp if c.isdigit())
    elif (len(tok_nc) == 4 and nxt == "年") or len(tok_nc) > 12 or (len(tok_nc) > 1 and tok_nc[0] == "0"):
        s = "".join(_DIG[int(c)] for c in tok_nc)          # years, codes, phone numbers
    else:
        n = int(tok_nc)
        s = _int_reading(n)
        if 10 <= n < 20 and s.startswith("一十"):
            s = s[1:]
        if n == 2 and prev != "第" and (nxt in _MEASURE or nxt == "年"):
            s = "两"
        s = re.sub(r"^二(?=[千万亿])", "两", s)
    if pct:
        s = "百分之" + s
    return s


# --------------------------------------------------------------------------- tone tables

PARTICLES = set("的地得了着过吗呢吧啊呀嘛啦哦么们哇哪喽呗咯")
_ZI_FULL = set("原电分质中离光粒量因孔老庄孟荀墨韩君男女天太王弟学游才赤甲臣公仙骄棋种瓜莲")  # X子 with full tone 3
_ERHUA_NOT = {"女儿", "婴儿", "儿女", "幼儿", "孤儿", "健儿", "宠儿", "胎儿", "少儿", "男儿", "患儿",
              "育儿", "孙儿", "混血儿", "新生儿", "儿", "托儿", "产儿", "病儿", "犬儿", "侄儿"}
_ZHU = {"著名", "著作", "显著", "著称", "著述", "卓著", "土著", "编著", "名著", "原著", "巨著", "专著", "论著",
        "著者", "昭著", "著录", "拙著", "遗著", "新著", "译著", "合著", "所著", "大著",
        "彰著", "撰著", "杰著", "近著", "著译"}
_REDUP_SANDHI = {"想想", "走走", "等等", "写写", "洗洗", "讲讲", "找找", "比比", "哪里", "小姐", "打扫"}


class ZhG2P(G2P):
    lang = "zh"

    def __init__(self) -> None:
        import logging

        import jieba
        from opencc import OpenCC
        from pypinyin import Style, lazy_pinyin

        logging.getLogger("jieba").setLevel(logging.ERROR)
        jieba.setLogLevel(logging.ERROR)
        jieba.initialize()
        self._jieba = jieba
        self._freq = jieba.dt.FREQ
        self._lazy = lazy_pinyin
        self._style = Style.TONE3
        self._t2s = OpenCC("t2s")
        self._s2t = OpenCC("s2t")
        self._lock = threading.Lock()   # jieba / opencc keep small mutable caches
        self.neutral: dict[str, list[int] | None] = {}
        for parts in data_lines("zh_neutral.txt"):
            if parts[0].startswith("="):
                self.neutral[parts[0][1:]] = [int(x) for x in parts[1:]]
            else:
                for w in parts:
                    self.neutral.setdefault(w, None)
        self.tw: dict[tuple[str, str], int] = {}
        for ch, py, t in data_lines("zh_tw_tones.txt"):
            self.tw[(ch, py)] = int(t)
        self._cit: dict[str, tuple[str, int]] = {}
        for w in self.neutral:              # register neutral-tone words with jieba so they stay whole
            if w not in self._freq:
                jieba.add_word(w, 50)
        self("你好，今天天氣很好。我們一起去公園吧！2019年")   # warm-up

    # ----------------------------------------------------------------- helpers
    def citation(self, ch: str) -> tuple[str, int]:
        """Most common single-character reading (base, tone)."""
        r = self._cit.get(ch)
        if r is None:
            p = self._lazy(ch, style=self._style, neutral_tone_with_five=True)[0]
            r = (p[:-1], int(p[-1])) if p[-1:].isdigit() else (p, 5)
            self._cit[ch] = r
        return r

    def to_simplified(self, chunk: str) -> str:
        s = self._t2s.convert(chunk)
        if len(s) != len(chunk):
            s = "".join(self._t2s.convert(c)[:1] or c for c in chunk)
        if "著" in s:
            # Taiwan-standard text writes the particle/verb 着 as 著; OpenCC t2s keeps 著.
            # Keep 著 (zhù) only inside its own words (著名, 显著, 土著 ...).
            s = "".join("着" if c == "著" and s[max(0, i - 1):i + 1] not in _ZHU and s[i:i + 2] not in _ZHU
                        else c for i, c in enumerate(s))
        return s

    def _segment(self, simp: str) -> list[str]:
        out: list[str] = []
        for w in self._jieba.lcut(simp, HMM=True):
            # 慢慢地走 -> jieba sometimes glues the adverbial 地 to the next word
            if (len(w) > 1 and w[0] in "地的" and out and len(out[-1]) >= 2 and out[-1][-1] == out[-1][-2]):
                out.append(w[0])
                w = w[1:]
            out.append(w)
        return out

    # ----------------------------------------------------------------- main
    def __call__(self, text: str) -> Reference:
        warnings: list[str] = []
        with self._lock:
            phrases, trad = self._scan(text, warnings)
            words: list[Wd] = []
            for ph in phrases:
                words.extend(self._phrase(ph, text, trad, warnings))
        return assemble("zh", text, words, warnings)

    def _scan(self, text: str, warnings: list[str]):
        """Split into phrases (between punctuation); each phrase is a list of segments:
        ('han', simplified, [(orig_start, orig_end, shown_char)]) or ('num', simplified, span)."""
        trad = self.to_simplified(text) != text
        phrases: list[list] = []
        cur: list = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if is_han(ch):
                j = i
                while j < n and is_han(text[j]):
                    j += 1
                chunk = text[i:j]
                simp = self.to_simplified(chunk)
                cur.append(("han", simp, [(i + k, i + k + 1, chunk[k]) for k in range(len(chunk))]))
                i = j
            elif nfkc_digit(ch) is not None:
                j = i
                while j < n and (nfkc_digit(text[j]) is not None or (
                        unicodedata.normalize("NFKC", text[j]) in ".,%" and j + 1 <= n and (
                        unicodedata.normalize("NFKC", text[j]) == "%" or
                        (j + 1 < n and nfkc_digit(text[j + 1]) is not None)))):
                    if unicodedata.normalize("NFKC", text[j]) == "%":
                        j += 1
                        break
                    j += 1
                tok = unicodedata.normalize("NFKC", text[i:j])
                jn = j
                while jn < n and text[jn].isspace():
                    jn += 1
                nxt = self.to_simplified(text[jn]) if jn < n else ""
                prev = self.to_simplified(text[i - 1]) if i > 0 else ""
                if "," in tok and not re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?%?", tok):
                    # a list separator, not a thousands separator: read the parts separately
                    parts = re.split(r"(,)", tok)
                    pos = i
                    for p in parts:
                        if p == ",":
                            phrases.append(cur) if cur else None
                            cur = []
                        elif p:
                            cur.append(("num", number_reading(p, nxt, prev), (pos, pos + len(p))))
                        pos += len(p)
                else:
                    cur.append(("num", number_reading(tok, nxt, prev), (i, j)))
                i = j
            elif is_latin(ch):
                j = i
                while j < n and (is_latin(text[j]) or text[j] in "'-"):
                    j += 1
                warnings.append(f"skipped non-Chinese text: {text[i:j]}")
                if cur:
                    phrases.append(cur)
                cur = []
                i = j
            else:
                cat = unicodedata.category(ch)
                if not ch.isspace() and cat[0] in "PSZ" or ch in "\n\r\t":
                    if cur:
                        phrases.append(cur)
                    cur = []
                elif not ch.isspace() and cat[0] not in "PSZC":
                    warnings.append(f"skipped unsupported character: {ch}")
                i += 1
        if cur:
            phrases.append(cur)
        return phrases, trad

    def _phrase(self, segs, text: str, trad: bool, warnings: list[str]) -> list[Wd]:
        # 1. words: (simplified word, [(start, end, shown)] per char, is_number)
        wl: list[tuple[str, list[tuple[int, int, str]], bool]] = []
        for kind, simp, info in segs:
            if kind == "han":
                pos = 0
                for w in self._segment(simp):
                    wl.append((w, info[pos:pos + len(w)], False))
                    pos += len(w)
            else:
                shown = self._s2t.convert(simp) if trad else simp
                if len(shown) != len(simp):
                    shown = simp
                s, e = info
                wl.append((simp, [(s, e, c) for c in shown], True))
        # 2. syllables with pinyin and lexical tones
        syls: list[dict] = []
        pys = self._lazy([w for w, _, _ in wl], style=self._style, neutral_tone_with_five=True,
                         errors=lambda x: ["?"] * len(x))
        k = 0
        for wi, (w, chars, isnum) in enumerate(wl):
            for ci, ch in enumerate(w):
                p = pys[k] if k < len(pys) else None
                k += 1
                if p is None or not p[-1:].isdigit():
                    if is_han(ch):
                        warnings.append(f"no reading for character: {chars[ci][2]}")
                    continue
                base, tone = p[:-1], int(p[-1])
                if ch == "一":
                    base, tone = "yi", 1
                elif ch == "不":
                    base, tone = "bu", 4
                syls.append({"ch": ch, "w": wi, "ci": ci, "wl": len(w), "word": w, "base": base, "lex": tone,
                             "tone": tone, "alts": [], "span": chars[ci][:2], "shown": chars[ci][2]})
            if len(pys) < k:
                break
        if not syls:
            return []
        self._tones(syls)
        return self._build(syls, wl, text, trad)

    # ----------------------------------------------------------------- tones
    def _tones(self, S: list[dict]) -> None:
        # lexical tone of neutral-toned dictionary readings: citation tone of the same syllable
        for s in S:
            if s["lex"] == 5:
                cb, ct = self.citation(s["ch"])
                if cb == s["base"] and ct != 5:
                    s["lex"] = ct
        # neutral tone (surface)
        n = len(S)
        for i, s in enumerate(S):
            w, ci, wl, ch = s["word"], s["ci"], s["wl"], s["ch"]
            neutral = None
            if w in self.neutral:
                pat = self.neutral[w]
                if pat is None:
                    neutral = ci == wl - 1
                else:
                    t = pat[ci] if ci < len(pat) else s["lex"]
                    if t == 5:
                        neutral = True
                    else:
                        if ch not in "一不":
                            s["tone"] = t
                        neutral = False
                        s["fixed"] = True
            elif s["tone"] == 5:
                neutral = True
                if ch == "子" and wl >= 2 and ci == wl - 1 and w[ci - 1] in _ZI_FULL:
                    neutral = False
                    s["tone"] = s["lex"] = 3
            elif ch == "们" or (ch == "子" and ci == wl - 1 and wl >= 2 and w[ci - 1] not in _ZI_FULL
                                and s["base"] == "zi"):
                neutral = True
            elif ch == "得" and s["base"] == "de" and 0 < ci < wl - 1:
                neutral = True            # V得C complement
            elif ch == "不" and wl == 3 and ci == 1:
                neutral = True            # V不C potential complement / A不A
            elif ch == "一" and wl == 3 and ci == 1 and w[0] == w[2]:
                neutral = True            # 看一看
            elif (ch == "地" and wl == 1 and 0 < i < n - 1 and S[i - 1]["w"] == s["w"] - 1
                  and S[i - 1]["wl"] >= 2 and S[i + 1]["w"] == s["w"] + 1):
                s["base"], neutral = "de", True   # adverbial marker: 有效地 寻找
            elif ch == "过" and ci == 0 and wl == 1 and i > 0 and S[i - 1]["w"] == s["w"] - 1:
                s["alts"].append(5)       # aspect particle 过 (guo) vs verb guò
            if ch == "不" and wl == 1 and 0 < i < n - 1 and S[i - 1]["ch"] == S[i + 1]["ch"]:
                neutral = True            # 是不是 / 好不好 (separate words)
            if ch == "一" and wl == 1 and 0 < i < n - 1 and S[i - 1]["ch"] == S[i + 1]["ch"]:
                neutral = True
            if neutral:
                s["tone"] = 5
                s["neutral"] = True
        # 一 and 不 sandhi
        for i, s in enumerate(S):
            if s["ch"] not in "一不" or s.get("neutral") or s.get("fixed"):
                continue
            nx = S[i + 1] if i + 1 < n else None
            nt = None
            if nx is not None:
                nt = nx["lex"] if nx["tone"] == 5 else nx["tone"]
            if s["ch"] == "不":
                if nt == 4:
                    s["tone"] = 2
                continue
            prev = S[i - 1]["ch"] if i > 0 else ""
            if nx is None or (s["ci"] == s["wl"] - 1 and s["wl"] > 1):
                continue                  # 统一, 唯一, 之一, phrase-final
            if nx["ch"] in "百千万亿":
                if prev and prev in "十零一二三四五六七八九〇":
                    continue              # 二十一万
            elif prev and prev in "第零一二三四五六七八九十百千万亿两〇":
                continue                  # numbers, ordinals: 第一 十一 一百一十
            elif nx["ch"] in "零一二三四五六七八九十〇":
                continue                  # digit strings 一九九五
            if nx["ch"] in "月号":
                s["alts"].append(2 if nt == 4 else 4)
                continue
            s["tone"] = 2 if nt == 4 else 4
        # third-tone sandhi on maximal chains of consecutive tone-3 syllables
        i = 0
        while i < n:
            if S[i]["tone"] != 3 or S[i].get("fixed"):
                i += 1
                continue
            j = i
            while j < n and S[j]["tone"] == 3 and not S[j].get("fixed"):
                j += 1
            if j - i >= 2:
                self._t3_chain(S[i:j])
            i = j
        # 3 + neutral syllable of underlying tone 3 (想想 xiángxiang, 姐姐 jiějie)
        for i in range(n - 1):
            a, b = S[i], S[i + 1]
            if a["tone"] == 3 and b["tone"] == 5 and b["lex"] == 3 and a["w"] == b["w"]:
                if a["word"] in _REDUP_SANDHI:
                    a["tone"], alt = 2, 3
                else:
                    alt = 2
                a["alts"].append(alt)
        # Taiwan usage: full tone instead of lexical neutral tone; per-character reading tones
        for s in S:
            if s["tone"] == 5 and s["ch"] not in PARTICLES and s["lex"] != 5:
                s["alts"].append(s["lex"])
            tw = self.tw.get((s["ch"], f"{s['base']}{s['lex']}"))
            if tw is not None and s["tone"] == s["lex"]:
                s["alts"].append(tw)

    def _t3_chain(self, C: list[dict]) -> None:
        """Third-tone sandhi on a chain of tone-3 syllables; word structure decides grouping."""
        segs: list[list[dict]] = []
        for s in C:
            if segs and segs[-1][0]["w"] == s["w"]:
                segs[-1].append(s)
            else:
                segs.append([s])
        for k, seg in enumerate(segs):
            last = k == len(segs) - 1
            m = len(seg)
            if m == 1:
                tones, alts = [3], [[]]
            elif m == 2:
                tones, alts = [2, 3], [[], []]
            elif m == 3:
                w = "".join(x["ch"] for x in seg)
                if self._freq.get(w[1:], 0) > self._freq.get(w[:2], 0):     # A+BC: 小老虎
                    tones, alts = [3, 2, 3], [[2], [], []]
                else:                                                      # AB+C: 展览馆
                    tones, alts = [2, 2, 3], [[], [], []]
            else:
                tones = [2] * (m - 1) + [3]
                alts = [[3] if q % 2 == 1 else [] for q in range(m - 1)] + [[]]
            if not last:
                tones[-1] = 2
                alts[-1] = alts[-1] + [3]
            for s, t, a in zip(seg, tones, alts):
                s["tone"] = t
                s["alts"].extend(a)

    # ----------------------------------------------------------------- phones
    def _build(self, S: list[dict], wl, text: str, trad: bool) -> list[Wd]:
        words: dict[int, Wd] = {}
        prev_syl: Syl | None = None
        prev_info = None
        out: list[Wd] = []
        for idx, s in enumerate(S):
            wi = s["w"]
            if wi not in words:
                chars = wl[wi][1]
                span = (min(c[0] for c in chars), max(c[1] for c in chars))
                words[wi] = Wd(text=text[span[0]:span[1]], span=span)
                out.append(words[wi])
            # erhua: word-final 儿 merges into the previous syllable of the same word
            if (s["ch"] == "儿" and s["ci"] == s["wl"] - 1 and s["wl"] >= 2 and s["word"] not in _ERHUA_NOT
                    and prev_syl is not None and prev_info is not None and prev_info["w"] == wi
                    and s["base"] == "er"):
                prev_syl.text += s["shown"]
                last = prev_syl.phones[-1]
                if not last.nucleus and last.ipa in ("n", "j"):
                    last.alt("", -0.3)
                er = Ph("ɻ")
                er.alt("ɑɻ", -0.7)
                er.alt("ə˞", -1.0)
                prev_syl.phones.append(er)
                continue
            parsed = parse_pinyin(s["base"])
            if parsed is None:
                continue
            phones = self._syl_phones(*parsed, s)
            attrs = {"tone": s["tone"], "tone_lexical": s["lex"],
                     "tone_alts": sorted({t for t in s["alts"] if t != s["tone"]})}
            syl = Syl(text=s["shown"], phones=phones, attrs=attrs)
            words[wi].syls.append(syl)
            # cross-syllable coda alternatives (assimilation to the next initial)
            if prev_syl is not None:
                self._coda_assim(prev_syl, phones)
            prev_syl, prev_info = syl, s
        return out

    def _syl_phones(self, ini: str, fin: str, s: dict) -> list[Ph]:
        glide, nuc, coda = FINALS[fin]
        out: list[Ph] = []
        if ini:
            p = Ph(INITIALS[ini])
            if ini == "r":
                p.alt("ɻ", -0.7)
            if ini == "h":
                p.alt("h", -1.0)
            out.append(p)
        if glide == "j" and not ini and fin in ("iou", "iong"):
            glide = "y"                # ZIPA writes you/yong as [yoʊ] [yʊŋ] (~95 %)
        if glide:
            g = Ph(glide)
            if glide == "y":
                g.alt("j", -0.5)
            elif glide == "j" and ini in ("j", "q", "x"):
                g.alt("", -1.0)        # glide absorbed by the palatal initial (千 tɕʰɛn)
            out.append(g)
        if fin == "eng" and ini in ("sh", "r", "h"):
            nuc = "ə"                  # ZIPA: ʂəŋ ʐəŋ xəŋ
        n = Ph(nuc, nucleus=True)
        if fin == "eng":
            n.alt("ɤ" if nuc == "ə" else "ə", -0.3)
        if fin in ("-i", "-ir"):
            n.alt("", -1.5)            # apical vowel swallowed in fast speech (是 → ʂ)
            n.alt("ɨ" if fin == "-i" else "ɯ", -0.5)
        if s["tone"] == 5 and nuc in ("ɤ", "a", "ɔ", "oʊ", "ə", "ɛ"):
            n.alt("ə", -0.7)          # neutral-tone vowel reduction
        out.append(n)
        if fin == "er":
            n.alt("ɑ", -1.0)
            n.alt("ə˞", -1.0)
        if coda:
            c = Ph(coda)
            c.alt("", -2.0)            # nasal coda realised as vowel nasalisation only
            if fin == "in":
                c.alt("ŋ", -1.0)       # -in / -ing merger, very common in native speech
            out.append(c)
        return out

    @staticmethod
    def _coda_assim(prev: Syl, nxt: list[Ph]) -> None:
        last = prev.phones[-1]
        if last.nucleus or last.ipa != "n" or not nxt:
            return
        first = nxt[0].ipa
        if first[0] in "pm":
            last.alt("m", -1.0)
        elif first[0] in "kx":
            last.alt("ŋ", -1.0)


def build() -> G2P:
    return ZhG2P()
