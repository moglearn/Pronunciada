"""Shared helpers for the zh / ja / ko text front-ends (Reference assembly, char classes)."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from ..types import RefPhone, RefSyllable, RefWord, Reference
from .ipa import is_valid


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cjk"


def data_text(name: str) -> str:
    """Read a text resource from pronunciada/data/cjk/."""
    return (DATA_DIR / name).read_text(encoding="utf-8")


def data_lines(name: str):
    """Non-empty, non-comment lines of a pronunciada/data/cjk/ resource, split on whitespace."""
    for line in data_text(name).splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield line.split()


@dataclass
class Ph:
    """A phone under construction: ipa, nucleus flag, alternatives (ipa, log-prior)."""
    ipa: str
    nucleus: bool = False
    alts: list[tuple[str, float]] = field(default_factory=list)

    def alt(self, ipa: str, prior: float) -> None:
        """Add an alternative unless it equals the canonical form or is already present
        (keeps the better prior)."""
        if ipa == self.ipa:
            return
        for i, (a, p) in enumerate(self.alts):
            if a == ipa:
                if prior > p:
                    self.alts[i] = (a, prior)
                return
        self.alts.append((ipa, min(0.0, prior)))


@dataclass
class Syl:
    text: str
    phones: list[Ph]
    attrs: dict = field(default_factory=dict)


@dataclass
class Wd:
    text: str
    span: tuple[int, int]
    syls: list[Syl] = field(default_factory=list)


def assemble(lang: str, text: str, words: list[Wd], warnings: list[str]) -> Reference:
    """Turn Wd/Syl/Ph trees into a Reference. Words without syllables are dropped, alternatives
    that are not valid ZIPA strings are dropped (defensive; tables are checked by tests)."""
    ref = Reference(lang=lang, text=text, warnings=list(dict.fromkeys(warnings)))
    for w in words:
        syls = [s for s in w.syls if s.phones]
        if not syls:
            continue
        wi = len(ref.words)
        s0, p0 = len(ref.syllables), len(ref.phones)
        for s in syls:
            si = len(ref.syllables)
            q0 = len(ref.phones)
            for ph in s.phones:
                alts = [(a, float(p)) for a, p in ph.alts if a == "" or is_valid(a)]
                ref.phones.append(RefPhone(ipa=ph.ipa, word=wi, syllable=si, is_nucleus=ph.nucleus, alts=alts))
            ref.syllables.append(RefSyllable(word=wi, phones=(q0, len(ref.phones)), text=s.text, **s.attrs))
        ref.words.append(RefWord(text=w.text, syllables=(s0, len(ref.syllables)), phones=(p0, len(ref.phones)),
                                 char_span=w.span))
    return ref


def is_han(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2FA1F
            or 0xF900 <= o <= 0xFAFF or ch == "〇")


def is_latin(ch: str) -> bool:
    c = unicodedata.normalize("NFKC", ch)
    return len(c) == 1 and ("a" <= c.lower() <= "z" or unicodedata.name(c, "").startswith("LATIN"))


def nfkc_digit(ch: str) -> str | None:
    c = unicodedata.normalize("NFKC", ch)
    return c if len(c) == 1 and "0" <= c <= "9" else None
