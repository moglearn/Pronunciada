"""IPA <-> ZIPA token conventions. Shared by every language module (owned by the architect:
language-specific rewrites belong in the language module, not here)."""
from __future__ import annotations

import unicodedata
from functools import lru_cache
from importlib import resources

SPECIAL = ("<blk>", "<sos/eos>", "<unk>", "▁")


@lru_cache(maxsize=None)
def load_tokens() -> tuple[str, ...]:
    """Token list of the default ZIPA models, index = token id."""
    txt = resources.files("pronunciada").joinpath("data/zipa_tokens.txt").read_text(encoding="utf-8")
    toks: list[str] = []
    for line in txt.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        tok, idx = parts[0], int(parts[1])
        assert idx == len(toks), "token ids must be contiguous"
        toks.append(tok)
    return tuple(toks)


@lru_cache(maxsize=None)
def token_ids() -> dict[str, int]:
    return {t: i for i, t in enumerate(load_tokens())}


@lru_cache(maxsize=None)
def symbols() -> frozenset[str]:
    return frozenset(t for t in load_tokens() if t not in SPECIAL)


# Symbols that attach to the preceding base symbol to form one phone.
MODIFIERS = frozenset("ʰʲʷʼː˞ˠˤ" + "̴̥̩̪̺̃̚")

# Universal, language-independent folds into the ZIPA inventory.
FOLD = {
    "ɡ": "g", "ɚ": "ə˞", "ɝ": "ɜ˞", "ʧ": "tʃ", "ʤ": "dʒ", "ʦ": "ts", "ʣ": "dz", "ʨ": "tɕ", "ʥ": "dʑ",
    "ɫ": "l", "ε": "ɛ", "ǝ": "ə", "ꞵ": "β", "ɩ": "ɪ", "ɷ": "ʊ", "ʱ": "ʰ", "ˑ": "", "'": "", "’": "",
    "͡": "", "͜": "",          # tie bars
    "ˈ": "", "ˌ": "", ".": "", " ": "", "‿": "", "|": "", "‖": "",
    "˥": "", "˦": "", "˧": "", "˨": "", "˩": "",
    "̯": "",                        # non-syllabic mark
    "̆": "",                        # extra-short
    "̞": "", "̝": "", "̟": "", "̠": "", "̈": "",  # lowered/raised/advanced/retracted/centralised
    "̹": "", "̜": "", "̘": "", "̙": "", "̻": "", "̼": "",
    "̤": "", "̰": "", "̬": "",                              # breathy/creaky/voiced
    "ːː": "ː",
}


def normalize(ipa: str, dropped: list[str] | None = None) -> str:
    """Map an arbitrary IPA string onto ZIPA symbols. Unknown symbols are dropped
    (and appended to `dropped` when given, so callers can warn)."""
    # NFD splits "ç" into c + cedilla; ç is a ZIPA symbol in its own right, so put it back.
    s = unicodedata.normalize("NFD", ipa).replace("ç", "ç")
    out: list[str] = []
    syms = symbols()
    for ch in s:
        ch = FOLD.get(ch, ch)
        for c in ch:
            if c in syms:
                out.append(c)
            elif dropped is not None:
                dropped.append(c)
    return "".join(out)


def is_valid(ipa: str) -> bool:
    syms = symbols()
    return bool(ipa) and all(c in syms for c in ipa) and ipa[0] not in MODIFIERS


def to_ids(ipa: str) -> list[int]:
    ids = token_ids()
    return [ids[c] for c in ipa]


def segment(ipa: str) -> list[str]:
    """Split a normalized IPA string into phones = base symbol + trailing modifiers.
    Does NOT merge affricates or diphthongs; language modules decide that."""
    phones: list[str] = []
    for c in ipa:
        if c in MODIFIERS and phones:
            phones[-1] += c
        else:
            phones.append(c)
    return phones


VOWELS = frozenset("aeiouyæøœɐɑɒɔɘəɛɜɞɤɨɪɯɵɶʉʊʌʏ")


def is_vowel(phone: str) -> bool:
    return bool(phone) and phone[0] in VOWELS
