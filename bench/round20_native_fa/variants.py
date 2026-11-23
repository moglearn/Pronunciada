"""Alternative pronunciations of one prompt word, and re-scoring the word against each of them through
the real pipeline (Scorer.score on a Reference whose word has the alternative as its canonical form).

Sources (all Apache-2.0 compatible; recorded in the write-up):
  lex    other readings of the word in the shipped lexicon (CMUdict BSD-2-Clause-style for en,
         ipa-dict MIT for de and fr); the front-end uses only the first reading (the second one only
         as per-sound alternatives when the lengths agree)
  brit   Britfone 3.0.1 (MIT), British English, en only; IPA converted to the en front-end's
         conventions (length marks dropped, the front-end's own rewrite applied)
  rule:* written optional phonological rules of the language (below), each applied at all its sites
  alt    the aligner's own choice: every sound of the word for which the lattice picked one of the
         front-end's alternatives is given that alternative as its canonical symbol (the old canonical
         becomes the alternative, with the same prior). This asks whether a realisation the lattice
         already accepts is still scored as worse than the canonical one.
Construction: the word's sounds are replaced; sounds that align to an unchanged canonical sound keep
the front-end's per-sound alternatives, changed sounds get none (a conservative variant). European
languages are re-syllabified with the front-end's onsets and syllabifier; Korean sounds keep their
syllable (Hangul syllable) by alignment. Stress: from the source reading (lex, brit) or carried over
from the canonical syllables (rule, alt). Nothing is fitted.
"""
from __future__ import annotations

import copy
import csv
from functools import lru_cache
from pathlib import Path

from bench.round20_native_fa.rebuild import ROOT

EXT = ROOT / "ext"


# ----------------------------------------------------------------------------- alignment

def align_pairs(a: list[str], b: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment of two symbol lists: [(i or None, j or None)] in order."""
    n, m = len(a), len(b)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1.2))
    out, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1.2):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


# ----------------------------------------------------------------------------- word view

def word_view(ref, wi: int) -> list[dict]:
    w = ref.words[wi]
    out = []
    for i in range(*w.phones):
        p = ref.phones[i]
        out.append({"ipa": p.ipa, "nuc": bool(p.is_nucleus), "stress": ref.syllables[p.syllable].stress if p.is_nucleus
                    else None, "syl": p.syllable - w.syllables[0], "src": i - w.phones[0]})
    return out


def _tokens(view: list[dict]) -> list[tuple[str, int | None]]:
    return [(v["ipa"], v["stress"] if v["nuc"] else None) for v in view]


# ----------------------------------------------------------------------------- sources

def _norm_word(text: str) -> str:
    import re
    import unicodedata

    t = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    return re.sub(r"^\W+|\W+$", "", t)


def lex_variants(g2p, text: str) -> list[list[tuple[str, int | None]]]:
    from pronunciada.text import _eu_common as C

    if not hasattr(g2p, "lex"):
        return []
    norm = _norm_word(text)
    raw = g2p.lex.get(norm)
    if raw is None:
        return []
    prons = C.split_prons(raw)
    out = []
    for p in prons[1:]:
        toks = g2p.rewrite(list(p), norm)
        out.append([C.parse_phone(t) for t in toks])
    return out


@lru_cache(maxsize=1)
def _britfone() -> dict[str, list[str]]:
    p = EXT / "britfone" / "britfone.main.3.0.1.csv"
    out: dict[str, list[str]] = {}
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            w = row[0].strip()
            if "(" in w:
                w = w[: w.index("(")]
            out.setdefault(w.lower(), []).append(row[1].strip())
    return out


_BRIT_MAP = {"ɐ": "ʌ", "iː": "i", "uː": "u", "ɑː": "ɑ", "ɔː": "ɔ", "ɜː": "ɜ", "ɡ": "g", "r": "ɹ", "əʊ": "əʊ", "eə": "ɛə",
             "ɪə": "ɪə", "ʊə": "ʊə", "tʃ": "tʃ", "dʒ": "dʒ", "aɪ": "aɪ", "eɪ": "eɪ", "aʊ": "aʊ", "ɔɪ": "ɔɪ"}
# Britfone writes STRUT as ɐ; the en front-end writes it ʌ (then ə by its ZIPA rewrite): same phoneme


def brit_variants(g2p, text: str) -> list[list[tuple[str, int | None]]]:
    from pronunciada.text import _eu_common as C
    from pronunciada.text.ipa import is_valid

    norm = _norm_word(text)
    out = []
    for ipa in _britfone().get(norm, []):
        toks = []
        for t in ipa.split():
            st = None
            if t.startswith("ˈ"):
                st, t = 1, t[1:]
            elif t.startswith("ˌ"):
                st, t = 2, t[1:]
            t = _BRIT_MAP.get(t, t)
            if not is_valid(t):
                toks = None
                break
            if C.is_nucleus_ipa(t):
                toks.append(t + str(st if st is not None else 0))
            else:
                toks.append(t)
        if toks:
            toks = g2p.rewrite(toks, norm)
            out.append([C.parse_phone(t) for t in toks])
    return out


# --- written optional rules (each returns a new token list or None when it does not apply) ---------

def _apply_sites(toks, fn):
    new = fn(list(toks))
    return None if new is None or new == toks else new


def rules(lang: str) -> dict:
    """name -> function(tokens) -> tokens | None. tokens: [(ipa, stress|None)]."""
    V = lambda t: t[0] and t[0][0] in "aeiouyæøœɐɑɒɔɘəɛɜɞɤɨɪɯɵɶʉʊʌʏ"  # noqa: E731

    def drop(pred):
        def f(t):
            out = [x for k, x in enumerate(t) if not pred(t, k)]
            return out if len(out) != len(t) and any(V(x) for x in out) else None
        return f

    def sub(pred, new):
        def f(t):
            out = [((new(t, k), x[1]) if pred(t, k) else x) for k, x in enumerate(t)]
            return out
        return f

    if lang == "en":
        CONS = lambda t, k: k + 1 < len(t) and not V(t[k + 1])  # noqa: E731

        def nonrhotic(t):
            out = []
            for k, (p, s) in enumerate(t):
                if p == "ɹ" and (k == len(t) - 1 or not V(t[k + 1])) and k > 0 and V(t[k - 1]):
                    continue
                out.append(({"ɜ˞": "ɜ", "ə˞": "ə"}.get(p, p), s))
            return out

        return {"nonrhotic": nonrhotic,
                "t_glottal": sub(lambda t, k: t[k][0] == "t" and k > 0 and V(t[k - 1]) and (k == len(t) - 1 or CONS(t, k)),
                                 lambda t, k: "ʔ"),
                "schwa_drop": drop(lambda t, k: t[k][0] == "ə" and not t[k][1] and 0 < k < len(t) - 1
                                   and not V(t[k - 1]) and t[k + 1][0] in ("ɹ", "l", "n", "l̴"))}
    if lang == "de":
        def ig(t):
            if len(t) >= 2 and t[-2][0] == "ɪ" and t[-1][0] == "ç":
                return t[:-1] + [("k", None)]
            return None

        def r_voc(t):           # post-vocalic r vocalised: Vʀ -> Vɐ (a diphthong in ZIPA writing)
            out, ch = [], False
            for k, (p, s) in enumerate(t):
                if p == "ʀ" and k > 0 and V(t[k - 1]) and (k == len(t) - 1 or not V(t[k + 1])):
                    q, qs = out[-1]
                    out[-1] = ("ɐ", qs) if q == "ə" else (q + "ɐ", qs)
                    ch = True
                    continue
                out.append((p, s))
            return out if ch else None

        return {"ig_k": ig,
                "schwa_sonorant": drop(lambda t, k: t[k][0] == "ə" and k + 1 < len(t) and t[k + 1][0] in ("n", "m", "l")
                                       and (k + 2 == len(t) or not V(t[k + 2])) and k > 0),
                "r_vocalised": r_voc,
                "final_e_schwa": sub(lambda t, k: k == len(t) - 1 and t[k][0] == "eː" and not t[k][1], lambda t, k: "ə"),
                "short_unstressed": sub(lambda t, k: V(t[k]) and t[k][0].endswith("ː") and not t[k][1]
                                        and t[k][0][0] in "ieouyø", lambda t, k: t[k][0][:-1])}
    if lang == "es":
        return {"seseo": sub(lambda t, k: t[k][0] == "θ", lambda t, k: "s"),
                "distincion": sub(lambda t, k: t[k][0] == "s" and k + 1 < len(t) and t[k + 1][0][:1] in "ie", lambda t, k: "θ"),
                "yeismo": sub(lambda t, k: t[k][0] in ("ʎ", "j") and k + 1 < len(t) and V(t[k + 1]), lambda t, k: "ʝ"),
                "ado_d_drop": drop(lambda t, k: t[k][0] in ("d", "ð") and 0 < k == len(t) - 2 and V(t[k - 1]) and t[k + 1][0] == "o"),
                "coda_s_h": sub(lambda t, k: t[k][0] == "s" and (k == len(t) - 1 or not V(t[k + 1])), lambda t, k: "h"),
                "lenition": sub(lambda t, k: t[k][0] in ("b", "d", "g") and k > 0 and V(t[k - 1]),
                                lambda t, k: {"b": "β", "d": "ð", "g": "ɣ"}[t[k][0]])}
    if lang == "fr":
        return {"schwa_drop": drop(lambda t, k: t[k][0] == "ə"),
                "mid_open": sub(lambda t, k: t[k][0] in ("e", "o", "ø"), lambda t, k: {"e": "ɛ", "o": "ɔ", "ø": "œ"}[t[k][0]]),
                "mid_close": sub(lambda t, k: t[k][0] in ("ɛ", "ɔ", "œ"), lambda t, k: {"ɛ": "e", "ɔ": "o", "œ": "ø"}[t[k][0]]),
                "final_cluster": drop(lambda t, k: k == len(t) - 1 and t[k][0] in ("ʁ", "l") and k > 0 and not V(t[k - 1])),
                "a_back": sub(lambda t, k: t[k][0] == "a", lambda t, k: "ɑ")}
    if lang == "ko":
        LAB, VEL = ("p", "pʰ", "b", "m"), ("k", "kʰ", "g")

        def place(t):
            out = list(t)
            ch = False
            for k in range(len(t) - 1):
                p, q = t[k][0], t[k + 1][0]
                if k > 0 and V(t[k - 1]) and not V(t[k]):
                    new = None
                    if p == "n" and q in LAB:
                        new = "m"
                    elif p in ("n", "m") and q in VEL:
                        new = "ŋ"
                    elif p == "t" and q in LAB:
                        new = "p"
                    elif p in ("t", "p") and q in VEL:
                        new = "k"
                    if new:
                        out[k] = (new, None)
                        ch = True
            return out if ch else None

        def n_insert(t):
            for k in range(1, len(t) - 1):
                if not V(t[k]) and V(t[k - 1]) and (t[k + 1][0] in ("j",) or t[k + 1][0] == "i"):
                    ins = "l" if t[k][0] == "l" else "n"
                    return t[:k + 1] + [(ins, None)] + t[k + 1:]
            return None

        def lax_voicing(t):
            out = list(t)
            ch = False
            for k in range(1, len(t) - 1):
                p = t[k][0]
                if p in ("k", "t", "p", "tʃ") and V(t[k - 1]) and V(t[k + 1]):
                    out[k] = ({"k": "g", "t": "d", "p": "b", "tʃ": "dʒ"}[p], None)
                    ch = True
            return out if ch else None

        return {"place_assim": place, "n_insertion": n_insert,
                "h_drop": drop(lambda t, k: t[k][0] == "h" and k > 0),
                "ui_i": sub(lambda t, k: t[k][0] == "ɯi", lambda t, k: "i"),
                "lax_voicing": lax_voicing,
                "eo_ʌ": sub(lambda t, k: t[k][0] == "ɔ", lambda t, k: "ʌ")}
    return {}


def alt_variant(ref, wi: int, ev) -> tuple[list[tuple[str, int | None]], dict] | None:
    """Tokens with the aligner's chosen alternatives as canonical, and {position: (old canonical, prior)}."""
    view = word_view(ref, wi)
    a = ref.words[wi].phones[0]
    toks, swaps, ch = [], {}, False
    for v in view:
        i = a + v["src"]
        k = ev.slots[i].alt_index
        p = ref.phones[i]
        if k == 0:
            toks.append((v["ipa"], v["stress"]))
        elif k < 0:
            ch = True
        else:
            alt, prior = p.alts[k - 1]
            if alt == "":
                ch = True
                continue
            swaps[len(toks)] = (p.ipa, prior, k - 1)
            toks.append((alt, v["stress"]))
            ch = True
    if not ch or not any(t[1] is not None or _is_nuc(t[0]) for t in toks):
        return None
    return toks, swaps


def _is_nuc(ipa: str) -> bool:
    from pronunciada.text import _eu_common as C

    return C.is_nucleus_ipa(ipa)


def all_variants(lang: str, g2p, ref, wi: int, ev=None) -> list[tuple[str, list, dict | None]]:
    """[(kind, tokens, swaps)] distinct from the canonical form and from each other."""
    view = word_view(ref, wi)
    canon = _tokens(view)
    seen = {tuple(p for p, _ in canon)}
    out = []

    def add(kind, toks, swaps=None):
        key = tuple(p for p, _ in toks)
        if key in seen or not toks:
            return
        seen.add(key)
        out.append((kind, toks, swaps))

    text = ref.words[wi].text
    if lang in ("en", "de", "fr"):
        for t in lex_variants(g2p, text):
            add("lex", t)
    if lang == "en":
        for t in brit_variants(g2p, text):
            add("brit", t)
    for name, fn in rules(lang).items():
        try:
            t = fn(list(canon))
        except Exception:  # noqa: BLE001
            t = None
        if t:
            add("rule:" + name, t)
    if ev is not None:
        r = alt_variant(ref, wi, ev)
        if r:
            add("alt", r[0], r[1])
    return out


# ----------------------------------------------------------------------------- splice

def splice(lang: str, g2p, ref, wi: int, toks: list[tuple[str, int | None]], swaps: dict | None = None):
    """A copy of `ref` whose word wi has `toks` as its sounds (see the module docstring)."""
    from bench.corrupt import _flatten, _nest, _P, _S
    from pronunciada.text import _eu_common as C
    from pronunciada.types import RefPhone, RefSyllable

    nest = _nest(ref)
    w = nest[wi]
    old = [p.obj for s in w.syls for p in s.phones]
    old_syl = [si for si, s in enumerate(w.syls) for _ in s.phones]
    pairs = align_pairs([p.ipa for p in old], [t[0] for t in toks])
    src = {j: i for i, j in pairs if j is not None and i is not None}
    new_ph = []
    for j, (ipa, st) in enumerate(toks):
        i = src.get(j)
        same = i is not None and old[i].ipa == ipa
        alts = copy.deepcopy(old[i].alts) if same else []
        if swaps and j in swaps:
            canon_old, prior, k = swaps[j]
            alts = [a for n, a in enumerate(copy.deepcopy(old[i].alts)) if n != k] + [(canon_old, prior)]
        rp = RefPhone(ipa, wi, 0, C.is_nucleus_ipa(ipa) or ("̩" in ipa), alts,
                      tense=getattr(old[i], "tense", None) if i is not None else None,
                      length=old[i].length if same else None)
        new_ph.append((rp, st, i))
    if lang in ("ko", "ja", "zh"):
        # keep each sound in its old syllable; an inserted sound joins the next old sound's syllable
        syl_of = []
        for j, (rp, st, i) in enumerate(new_ph):
            if i is not None:
                syl_of.append(old_syl[i])
            else:
                nxt = next((old_syl[x[2]] for x in new_ph[j + 1:] if x[2] is not None), old_syl[-1])
                syl_of.append(nxt)
        syls = []
        for si, s in enumerate(w.syls):
            ps = [_P(rp) for (rp, _, _), so in zip(new_ph, syl_of) if so == si]
            if not ps:
                continue
            if sum(p.obj.is_nucleus for p in ps) != 1:
                return None
            syls.append(_S(copy.deepcopy(s.obj), ps))
        w.syls = syls
    else:
        P = [C.P(rp.ipa, rp.is_nucleus, stress=st) for rp, st, _ in new_ph]
        if not any(p.nucleus for p in P):
            return None
        sy = C.syllabify(P, getattr(g2p, "onsets", frozenset()), default_stress=getattr(g2p, "default_stress", 0))
        for p, (rp, _, _) in zip(P, new_ph):
            rp.is_nucleus = p.nucleus
        old_stress = [s.obj.stress for s in w.syls]
        if all(st is None for _, st, _ in new_ph) or (len(sy) == len(old_stress) and all(
                x == 0 for x in old_stress)):
            stresses = old_stress if len(sy) == len(old_stress) else [b[2] for b in sy]
        else:
            stresses = [b[2] for b in sy]
        syls = []
        for (a, b, _), st in zip(sy, stresses):
            syls.append(_S(RefSyllable(word=wi, phones=(0, 0), stress=st), [_P(new_ph[k][0]) for k in range(a, b)]))
        w.syls = syls
    out, _, _ = _flatten(ref.lang, ref.text, ref.warnings, nest)
    out.validate()
    return out
