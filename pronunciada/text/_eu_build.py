"""OFFLINE lexicon builder for the European front-ends (not imported at runtime).

    .venv/bin/python -m pronunciada.text._eu_build <lang> <source files...>

Writes pronunciada/data/lex/<lang>.tsv.gz (word \t phones[\t phones2]) and <lang>_lts.tsv.gz.
Sources and licences: see docs/text_eu_notes.md.
"""
from __future__ import annotations

import collections
import gzip
import os
import re
import sys
import unicodedata

from .ipa import is_valid, normalize
from . import _eu_lts

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "lex")


def onsets_of(lex: dict[str, list[list[str]]], min_count: int = 5, extra=()) -> str:
    """Legal onsets = consonant clusters (2-3 phones) seen word-initially >= min_count times."""
    from .ipa import VOWELS
    c = collections.Counter()
    for prons in lex.values():
        p = [x.rstrip("012") for x in prons[0]]
        k = 0
        while k < len(p) and not (p[k][0] in VOWELS or "\u0329" in p[k]):
            k += 1
        if 2 <= k <= 3 and k < len(p):
            c[tuple(p[:k])] += 1
            if k == 3:
                c[tuple(p[1:3])] += 1
    keep = sorted({"+".join(x) for x, n in c.items() if n >= min_count} | set(extra))
    return " ".join(keep)


def write_lex(lang: str, lex: dict[str, list[list[str]]], meta: dict | None = None) -> None:
    path = os.path.join(OUT, f"{lang}.tsv.gz")
    os.makedirs(OUT, exist_ok=True)
    meta = dict(meta or {})
    meta.setdefault("onsets", onsets_of(lex))
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        for k, v in meta.items():
            f.write(f"#{k}\t{v}\n")
        for w in sorted(lex):
            prons = []
            for p in lex[w]:
                s = " ".join(p)
                if s not in prons:
                    prons.append(s)
            f.write(w + "\t" + "\t".join(prons[:2]) + "\n")
    print(f"wrote {path} {os.path.getsize(path) / 1e6:.2f} MB, {len(lex)} entries")


def write_lts(lang: str, lex: dict[str, list[list[str]]], max_words: int = 200000) -> None:
    pairs = [(w, p[0]) for w, p in lex.items() if " " not in w and "-" not in w and "'" not in w][:max_words]
    m = _eu_lts.train(pairs)
    path = os.path.join(OUT, f"{lang}_lts.tsv.gz")
    m.save(path)
    # quick self-check on a held-in sample
    ok = sum(m.predict(w) == p for w, p in pairs[::50])
    print(f"wrote {path} {os.path.getsize(path) / 1e6:.2f} MB, {len(m.table)} contexts; "
          f"train-sample word acc {ok / len(pairs[::50]):.2f}")


# --------------------------------------------------------------------------- ipa helpers

TIES = ("\u0361", "\u035c")
NONSYLL = "\u032f"
GLIDE = {"i": "j", "ɪ": "j", "u": "w", "ʊ": "w", "y": "ɥ", "e": "j", "o": "w", "ɐ": "ɐ"}


def ipa_phones(s: str, multi: tuple[str, ...] = (), glide_attach: bool = True) -> list[str] | None:
    """Split a dictionary IPA string into ZIPA phones. Affricates written with a tie bar and
    vowel + non-syllabic vowel (diphthongs) become one phone; `multi` lists extra units to merge
    (normalised spelling, e.g. "ts", "aɪ"). Stress marks become a digit on the next nucleus.
    Returns None if a symbol cannot be mapped."""
    from .ipa import MODIFIERS, VOWELS
    # NB: ipa.normalize() NFD-decomposes ç (U+00E7) into c + cedilla and drops the cedilla;
    # protect it with a placeholder so the ZIPA token ç survives.
    s = unicodedata.normalize("NFC", s.strip().strip("/[]")).replace("ç", "\uE000")
    s = unicodedata.normalize("NFD", s).replace("c\u0327", "\uE000")
    s = s.replace("⁽ʲ⁾", "ʲ").replace("⁽", "").replace("⁾", "")
    units: list[list] = []          # [text, nonsyll, tie_to_next, stress_before]
    stress = None
    tie = False
    for ch in s:
        if ch in "ˈˌ":
            stress = 1 if ch == "ˈ" else 2
            continue
        if ch in TIES or ch == "‿":
            tie = True
            continue
        if ch in ". -":
            continue
        if ch == NONSYLL:
            if units:
                units[-1][1] = True
            continue
        n = "ç" if ch == "\uE000" else normalize(ch)
        if not n:
            if unicodedata.combining(ch) or ch in "ːˑ":
                continue
            return None
        if n[0] in MODIFIERS and units:
            units[-1][0] += n
            continue
        if tie and units:
            units[-1][0] += n
            tie = False
            continue
        units.append([n, False, False, stress])
        stress = None
    if not units:
        return None
    merged: list[list] = []
    for u in units:
        if u[1] and u[0][0] in VOWELS:
            if glide_attach and merged and merged[-1][0][0] in VOWELS and not merged[-1][1]:
                merged[-1][0] += u[0]
                continue
            g = GLIDE.get(u[0][0])
            if g is None:
                return None
            if g != "ɐ":
                u = [g + u[0][1:].replace("\u0303", ""), True, False, u[3]]
        merged.append(u)
    out: list[list] = []
    j = 0
    while j < len(merged):
        for mu in multi:
            acc, k = "", j
            while k < len(merged) and len(acc) < len(mu):
                acc += merged[k][0]
                k += 1
            if acc == mu and k - j > 1:
                st = next((x[3] for x in merged[j:k] if x[3] is not None), None)
                out.append([acc, False, False, st])
                j = k
                break
        else:
            out.append(merged[j])
            j += 1
    res: list[str] = []
    carry = None
    for text, nonsyll, _, st in out:
        if st is not None:
            carry = st
        if (text[0] in VOWELS and not nonsyll) or "\u0329" in text:
            res.append(text + (str(carry) if carry is not None else ""))
            carry = None
        else:
            res.append(text)
    if not all(is_valid(p.rstrip("012")) for p in res):
        return None
    return res


# --------------------------------------------------------------------------- English

ARPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɜ˞",
    "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "g", "HH": "h", "JH": "dʒ",
    "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}


def arpa_to_ipa(ph: list[str]) -> list[str]:
    out = []
    for p in ph:
        st = p[-1] if p[-1] in "012" else ""
        base = p.rstrip("012")
        ipa = ARPA[base]
        if base == "AH" and st == "0":
            ipa = "ə"
        if base == "ER" and st == "0":
            ipa = "ə˞"
        out.append(ipa + ("" if st == "" else ("" if st == "0" else st)))
    return out


def build_en(cmudict: str) -> None:
    lex: dict[str, list[list[str]]] = collections.defaultdict(list)
    for line in open(cmudict, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        w, *ph = line.split()
        w = re.sub(r"\(\d+\)$", "", w).lower()
        lex[w].append(arpa_to_ipa(ph))
    write_lex("en", lex)
    write_lts("en", lex)




# --------------------------------------------------------------------------- Spanish (word list)

def build_es(ipadict_es_mx: str) -> None:
    words = set()
    for line in open(ipadict_es_mx, encoding="utf-8"):
        w = line.split("\t")[0].strip()
        if w and " " not in w:
            words.add(unicodedata.normalize("NFC", w).lower())
    path = os.path.join(OUT, "es_words.txt.gz")
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("\n".join(sorted(words)))
    print(f"wrote {path} {os.path.getsize(path) / 1e6:.2f} MB, {len(words)} words")


# --------------------------------------------------------------------------- WikiPron helpers

def read_tsv(path: str, multi=(), lower=True, max_prons=2) -> dict[str, list[list[str]]]:
    """Read a word<TAB>ipa list. When lower-casing, entries written in lower case win over
    Capitalised ones, which win over ALL-CAPS ones (abbreviations: IN /iˈʔɛn/ vs in /ɪn/)."""
    lex: dict[str, list[list[str]]] = collections.defaultdict(list)
    rank: dict[str, int] = {}
    bad = 0
    rows = []
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            orig = unicodedata.normalize("NFC", parts[0].strip())
            r = 0 if orig.islower() or not lower else (2 if orig.isupper() and len(orig) > 1 else 1)
            rows.append((r, orig, parts))
    rows.sort(key=lambda x: x[0])
    for r, orig, parts in rows:
        w = orig
        if lower:
            w = w.lower()
            if w in rank and rank[w] < r:
                continue
            rank[w] = r
        if not w or " " in w:
            continue
        for raw in parts[1].split(", "):
            ph = ipa_phones(raw, multi)
            if ph is None:
                bad += 1
                continue
            if len(lex[w]) < max_prons and ph not in lex[w]:
                lex[w].append(ph)
    print(f"read {path}: {len(lex)} words, {bad} unparsable prons")
    return {k: v for k, v in lex.items() if v}


def nuclei(ph: list[str]) -> list[int]:
    from .ipa import VOWELS
    return [i for i, p in enumerate(ph) if p[0] in VOWELS or "\u0329" in p]


# --------------------------------------------------------------------------- Italian

IT_ACC = "àèéìòóù"


def build_it(wikipron_ita: str) -> None:
    lex = read_tsv(wikipron_ita, multi=("ts", "dz", "tʃ", "dʒ"))
    from .it import stress_from_end
    for w in ("unito", "finito", "pratico", "musica", "telefono", "parlano", "credere", "vedere",
              "amico", "politica", "economia", "italia", "possibile", "ventesimo", "macchina", "piano"):
        if w in lex:
            nu = nuclei(lex[w][0])
            print("   ", w, " ".join(lex[w][0]), "-> from end", stress_from_end(w, len(nu)))
    out = {}
    for w, prons in lex.items():
        res = []
        for ph in prons:
            nu = nuclei(ph)
            if not nu:
                continue
            if w[-1] in IT_ACC:
                si = len(nu) - 1
            else:
                open_ = [k for k, i in enumerate(nu) if ph[i][0] in "ɛɔ"]
                si = open_[0] if len(open_) == 1 else len(nu) - 1 - stress_from_end(w, len(nu))
            ph = list(ph)
            ph[nu[si]] += "1"
            res.append(ph)
        if res:
            out[w] = res
    write_lex("it", out)
    write_lts("it", out)


# --------------------------------------------------------------------------- Portuguese

def build_pt(wikipron_por_bz: str) -> None:
    """Word list + stressed-vowel quality (ɛ/ɔ vs e/o) from WikiPron por (bz)."""
    from .pt import rules
    from ._eu_common import align_alt
    lex = read_tsv(wikipron_por_bz, max_prons=6)
    out: dict[str, str] = {}
    n_open = 0
    for w, prons in lex.items():
        if not w.isalpha():
            continue
        mine = rules(w)
        si = next((i for i, x in enumerate(mine) if x.endswith("1")), None)
        flag = ""
        if si is not None and mine[si].rstrip("1")[0] in "eo":
            votes = collections.Counter()
            for p in prons:
                al = align_alt([x.rstrip("1") for x in mine], [x.rstrip("012") for x in p])
                if al is None:
                    votes[mine[si][0]] += 1
                    continue
                q = al[si][:1]
                if q in ("ɛ", "ɔ", "e", "o"):
                    votes[q] += 1
            if votes:
                best = votes.most_common(1)[0][0]
                if best in ("ɛ", "ɔ"):
                    flag = best
                    n_open += 1
        out[w] = flag
    path = os.path.join(OUT, "pt.tsv.gz")
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        for w in sorted(out):
            f.write(f"{w}\t{out[w]}\n")
    print(f"wrote {path} {os.path.getsize(path) / 1e6:.2f} MB, {len(out)} words, {n_open} open stressed e/o")


# --------------------------------------------------------------------------- Russian

RU_FULL = set("aoeiuæɵɛɔ")


def build_ru(wikipron_rus_narrow: str) -> None:
    import random
    from ._eu_lts import StressModel
    from .ru import VOWELS as RV, rules
    lex = read_tsv(wikipron_rus_narrow, multi=("ts", "tɕ"), max_prons=12)
    out: dict[str, str] = {}
    n_exc = 0
    pairs = []
    for w, prons in lex.items():
        if not all(c in RV or "а" <= c <= "я" or c in "ё-" for c in w):
            continue
        vow = [c for c in w if c in RV]
        nv = len(vow)
        if nv == 0:
            continue
        votes = collections.Counter()
        good = []
        for p in prons:
            nu = nuclei(p)
            if len(nu) != nv:
                continue
            base = [p[i].rstrip("012")[0] for i in nu]
            cand = [k for k, b in enumerate(base) if b in RU_FULL]
            if len(cand) != 1:
                cand2 = [k for k, b in enumerate(base) if b in "ɨʉ"]
                cand = cand2 if (not cand and len(cand2) == 1) else ([] if len(cand) != 1 else cand)
            if nv == 1:
                cand = [0]
            if len(cand) == 1:
                votes[cand[0]] += 1
                good.append((cand[0], p))
        if "ё" in vow:
            si = vow.index("ё")
        elif votes:
            si = votes.most_common(1)[0][0]
        else:
            continue
        mine = rules(w, si)
        skel = lambda ph: tuple(x.replace("ˠ", "").replace("ː", "") for x in ph if not C_is_nuc(x))  # noqa: E731
        sk = collections.Counter(skel([x.rstrip("012") for x in p]) for st, p in good if st == si)
        val = str(si)
        my_sk = skel([x.rstrip("1") for x in mine])
        others = [(c, k) for k, c in sk.items() if k != my_sk]
        if others:
            cnt, best = max(others)
            mine_n = sk.get(my_sk, 0)
            # a tie only counts for substitutions (что ʂto), not for casual consonant loss (когда, его)
            if cnt > mine_n or (cnt == mine_n and len(best) == len(my_sk)):
                cands = [p for st, p in good if st == si and skel([x.rstrip("012") for x in p]) == best]
                p = [x.rstrip("012") for x in cands[0]]
                nu = nuclei(p)
                p[nu[si]] += "1"
                val = " ".join(p)
                n_exc += 1
        out[w] = val
        if nv > 1:
            pairs.append((w, nv - 1 - si))
    # keys without ё for texts that write е
    for w in list(out):
        if "ё" in w:
            k = w.replace("ё", "е")
            if k not in out:
                out[k] = out[w] if not out[w].isdigit() else out[w]
    random.seed(0)
    random.shuffle(pairs)
    hold = pairs[: len(pairs) // 20]
    sm = StressModel.train(pairs[len(pairs) // 20:], dflt=1, min_count=3)
    acc = sum(sm.predict(w, len([c for c in w if c in RV])) == len([c for c in w if c in RV]) - 1 - r
              for w, r in hold) / max(1, len(hold))
    print(f"ru: {len(out)} words, {n_exc} full-transcription exceptions; suffix stress model "
          f"held-out acc {acc:.3f} ({len(sm.table)} suffixes)")
    sm = StressModel.train(pairs, dflt=1, min_count=3)
    sm.save(os.path.join(OUT, "ru_stress.tsv.gz"))
    sample = {w: [rules(w, int(v)) if v.isdigit() else v.split()] for w, v in list(out.items())[::5]}
    onsets = onsets_of(sample, min_count=3)
    path = os.path.join(OUT, "ru.tsv.gz")
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write(f"#onsets\t{onsets}\n")
        for w in sorted(out):
            f.write(f"{w}\t{out[w]}\n")
    print(f"wrote {path} {os.path.getsize(path) / 1e6:.2f} MB")


def C_is_nuc(x: str) -> bool:
    from .ipa import VOWELS
    return x[:1] in VOWELS


# --------------------------------------------------------------------------- German / French (ipa-dict)

DE_MULTI = ("aɪ", "aʊ", "ɔʏ", "ɔɪ")   # affricates carry tie bars in ipa-dict


def build_de(ipadict_de: str, max_len: int = 18) -> None:
    lex = read_tsv(ipadict_de, multi=DE_MULTI)
    lex = {w: p for w, p in lex.items() if w.isalpha() and len(w) <= max_len}
    write_lex("de", lex)
    write_lts("de", lex, max_words=150000)


def build_fr(ipadict_fr: str) -> None:
    lex = read_tsv(ipadict_fr, multi=())
    lex = {w: p for w, p in lex.items() if all(c.isalpha() or c in "'-" for c in w)}
    write_lex("fr", lex)
    write_lts("fr", lex, max_words=150000)


def _main(argv):
    lang, *srcs = argv
    globals()[f"build_{lang}"](*srcs)


if __name__ == "__main__":
    _main(sys.argv[1:])
