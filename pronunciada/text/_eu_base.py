"""Common G2P driver for the European front-ends. A language module subclasses EuG2P and fills
in: lexicon name, onsets, number settings, letter names, symbol words, abbreviations, and the
hooks `rewrite` (ZIPA convention fixes on a word's phones), `oov` (rules / LTS) and
`connected` (cross-word optional processes)."""
from __future__ import annotations

import functools
import unicodedata

from ..types import Reference
from . import _eu_common as C
from ._eu_lts import LTS, StressModel
from .base import G2P


class EuG2P(G2P):
    lang = ""
    lex_name = ""
    use_lts = True
    decimal_comma = True
    thousand_space = True
    ordinals = ""
    num_lang = ""
    letters: dict[str, str] = {}       # letter -> spelled word(s) for acronyms
    symbols: dict[str, str] = {}       # symbol -> word(s)
    abbrev: dict[str, str] = {}        # lower-case abbreviation (no dot) -> expansion
    onsets: frozenset = frozenset()
    default_stress: int | None = 0     # None for French
    stress_name = ""                   # suffix stress model file (without .tsv.gz)

    def __init__(self) -> None:
        self.lex, meta = C.load_lexicon(f"{self.lex_name}.tsv.gz") if self.lex_name else ({}, {})
        if not self.onsets and "onsets" in meta:
            self.onsets = C.parse_onsets(meta["onsets"])
        self.lts = None
        if self.use_lts and self.lex_name:
            with C.lex_path(f"{self.lex_name}_lts.tsv.gz").open("rb") as f:
                self.lts = LTS.load(f)
        self.stress_model = None
        if self.stress_name:
            with C.lex_path(f"{self.stress_name}.tsv.gz").open("rb") as f:
                self.stress_model = StressModel.load(f)
        # onsets come from the lexicon notation; add their ZIPA-convention rewrites
        extra = set()
        for o in self.onsets:
            r = tuple(C.parse_phone(x)[0] for x in self.rewrite(list(o), ""))
            if len(r) == len(o):
                extra.add(r)
        self.onsets = frozenset(self.onsets) | extra
        self.rx = C.make_token_re(self.thousand_space, self.ordinals)
        self._abbr_dot = frozenset(self.abbrev)
        self._number_words: set[str] = set()
        self._cache = functools.lru_cache(maxsize=50000)(self._pron_uncached)

    # ----------------------------------------------------------------- hooks
    def normalise_word(self, w: str) -> str:
        return unicodedata.normalize("NFC", C.fold_apos(w)).lower()

    def lookup(self, w: str) -> list[list[str]] | None:
        raw = self.lex.get(w)
        return C.split_prons(raw) if raw is not None else None

    def oov(self, w: str) -> list[str]:
        return self.lts.predict(w) if self.lts else []

    def rewrite(self, phones: list[str], word: str) -> list[str]:
        """ZIPA-convention fixes on one word (phones carry stress digits)."""
        return phones

    def fix_stress(self, phones: list[str], word: str = "") -> list[str]:
        """Exactly one primary stress per word for rule/LTS output. With a suffix stress model
        (it, ru, pt) the model decides; otherwise keep the first predicted primary."""
        nuc = [i for i, p in enumerate(phones) if C.is_nucleus_ipa(C.parse_phone(p)[0])]
        if not nuc or self.default_stress is None:
            return phones
        if self.stress_model is not None:
            out = [C.parse_phone(p)[0] for p in phones]
            k = self.stress_model.predict(word, len(nuc))
            out[nuc[k]] += "1"
            return out
        prim = [i for i in nuc if phones[i].endswith("1")]
        out = list(phones)
        if not prim:
            i = nuc[0]
            out[i] = C.parse_phone(out[i])[0] + "1"
        for i in prim[1:]:
            out[i] = C.parse_phone(out[i])[0] + "2"
        return out

    def connected(self, words: list[C.W]) -> None:
        """Cross-word optional processes: add alts in place."""

    def word_alts(self, w: C.W, prons: list[list[str]]) -> None:
        """Homographs: second reading as per-phone alts when lengths align."""
        if len(prons) > 1:
            a = [C.parse_phone(x)[0] for x in prons[1]]
            if len(a) == len(w.phones):
                for p, x in zip(w.phones, a):
                    if x != p.ipa and C.is_valid(x):
                        p.add_alt(x, C.P_MED)

    # ----------------------------------------------------------------- core
    def _pron_uncached(self, norm: str) -> tuple[tuple[tuple[str, ...], ...], bool, str]:
        """normalised word -> (prons (each a tuple of phone tokens), oov flag, how)."""
        prons = self.lookup(norm)
        if prons:
            return tuple(tuple(self.rewrite(p, norm)) for p in prons), False, "lex"
        prons = self.lookup_compound(norm)
        if prons:
            return (tuple(self.rewrite(prons, norm)),), False, "compound"
        prons = self.lookup_inflected(norm)
        if prons:
            return (tuple(self.rewrite(prons, norm)),), False, "inflected"
        if len(norm) >= self.split_min_len or norm in self._number_words:
            prons = self.split_compound(norm)
            if prons:
                return (tuple(self.rewrite(prons, norm)),), False, "split"
        ph = self.oov(norm)
        ph = [p for p in ph if p and C.is_valid(C.parse_phone(p)[0])]
        if not ph:
            return (), True, "none"
        return (tuple(self.rewrite(self.fix_stress(ph, norm), norm)),), True, "rules"

    def lookup_compound(self, w: str) -> list[str] | None:
        """Hyphenated / apostrophe pieces, all in the lexicon."""
        for sep in ("-", "'"):
            if sep in w:
                parts = [p for p in w.split(sep) if p]
                out: list[str] = []
                for p in parts:
                    r = self.lookup(p)
                    if not r:
                        return None
                    out.extend(r[0])
                return out
        return None

    # inflection fallback: word differs from a lexicon word only in its final vowel (it, pt ...)
    inflect_finals: tuple[str, ...] = ()
    split_min_len = 99           # DP compound splitting for OOV words at least this long (de)
    split_min_piece = 3
    split_links: tuple[str, ...] = ()

    def lookup_inflected(self, w: str) -> list[str] | None:
        for old, news in self.inflect_finals:
            if w.endswith(old) and len(w) > len(old) + 2:
                stem = w[: len(w) - len(old)]
                for new in news:
                    r = self.lookup(stem + new)
                    if r:
                        ph = list(r[0])
                        newp, oldp = self.inflect_phones.get(new), self.inflect_phones.get(old)
                        if newp is None or oldp is None:
                            continue
                        n = len(newp)
                        tail = [C.parse_phone(x)[0] for x in ph[len(ph) - n:]] if n else []
                        if tail != newp:
                            continue
                        stem_ph = ph[: len(ph) - n]
                        if not any(C.is_nucleus_ipa(C.parse_phone(x)[0]) and x[-1] == "1" for x in stem_ph):
                            continue            # stress was on the replaced ending: unsafe
                        return stem_ph + list(oldp)
        return None

    inflect_phones: dict[str, list[str]] = {}

    def split_compound(self, w: str) -> list[str] | None:
        """Min-pieces DP segmentation into lexicon words (optionally with linking letters);
        the last piece keeps primary stress, earlier pieces get secondary."""
        n = len(w)
        best: list = [None] * (n + 1)
        best[0] = (0, [])
        for i in range(n):
            if best[i] is None:
                continue
            for j in range(i + self.split_min_piece, n + 1):
                piece = w[i:j]
                if piece in self.lex:
                    cand = (best[i][0] + 1, best[i][1] + [piece])
                    if best[j] is None or cand[0] < best[j][0]:
                        best[j] = cand
                    for link in self.split_links:
                        k = j + len(link)
                        if k < n and w[j:k] == link:
                            cand2 = (best[i][0] + 1.1, best[i][1] + [piece, link])
                            if best[k] is None or cand2[0] < best[k][0]:
                                best[k] = cand2
        if best[n] is None or len(best[n][1]) < 2:
            return None
        out: list[str] = []
        pieces = best[n][1]
        for pi, piece in enumerate(pieces):
            if piece in self.split_links:
                out.extend(self.link_phones[piece])
                continue
            ph = self.lookup(piece)[0]
            if self.split_primary == "first":
                keep = all(pp in self.split_links for pp in pieces[:pi])
            else:
                keep = all(pp in self.split_links for pp in pieces[pi + 1:])
            for x in ph:
                ipa, st = C.parse_phone(x)
                if st == 1 and not keep:
                    st = 2
                out.append(ipa + ("" if st is None else str(st)))
        return out

    link_phones: dict[str, list[str]] = {}
    split_primary = "last"       # which compound piece keeps the primary stress

    def expand_token(self, tok: C.Tok, warnings: list[str]) -> list[str] | None:
        """Token -> list of normalised lexical items (None = drop)."""
        if tok.kind == "num":
            ws = C.number_words(tok, self.num_lang or self.lang, self.decimal_comma, warnings)
            if ws:
                ws = [self.normalise_word(x) for x in ws]
                self._number_words.update(x for x in ws if x not in self.lex)
            return ws
        if tok.kind == "sym":
            s = self.symbols.get(tok.text)
            return s.split() if s else None
        t = tok.text
        low = self.normalise_word(t.rstrip("."))
        if low in self.abbrev and (t.endswith(".") or low not in self.lex):
            return self.abbrev[low].split()
        if C.is_acronym(t) and (("." in t) or self.lookup(low) is None) and self.letters:
            return [self.letters.get(ch.lower(), ch.lower()) for ch in t.replace(".", "")]
        return [low.rstrip(".")]

    def build_word(self, tok: C.Tok, items: list[str], warnings: list[str]) -> C.W:
        phones_all: list[str] = []
        prons_first: list[list[str]] = []
        oov_any = False
        for it in items:
            for sub in it.split():
                prons, oov, _ = self._cache(sub)
                if not prons:
                    warnings.append(f"no_pron:{sub}")
                    continue
                if oov:
                    oov_any = True
                    warnings.append(f"oov:{sub}")
                phones_all.extend(prons[0])
                prons_first.append([list(p) for p in prons])
        w = C.word_from_prons(tok.text, tok.span, phones_all, self.onsets, default_stress=self.default_stress) \
            if phones_all else C.W(tok.text, tok.span)
        w.norm = " ".join(items)
        w.oov = oov_any
        w.info["items"] = items
        if len(prons_first) == 1 and phones_all:
            self.word_alts(w, prons_first[0])
        return w

    def words(self, text: str, warnings: list[str]) -> list[C.W]:
        out: list[C.W] = []
        for tok in C.tokenize(text, self.rx, self._abbr_dot):
            items = self.expand_token(tok, warnings)
            if not items:
                if tok.kind != "sym":
                    warnings.append(f"skipped:{tok.text}")
                continue
            w = self.build_word(tok, items, warnings)
            if w.phones:
                out.append(w)
        return out

    def __call__(self, text: str) -> Reference:
        warnings: list[str] = []
        ws = self.words(text, warnings)
        self.connected(ws)
        return C.assemble(self.lang, text, ws, warnings)
