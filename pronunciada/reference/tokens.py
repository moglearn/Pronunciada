"""Token file and target-symbol checks (port of the parts of pronunciada/text/ipa.py the
alignment needs).

Token file format (zipa_tokens.txt, UTF-8, 127 lines): "<symbol> <id>" per line, ids 0..126 in
order. Ids 0..3 are <blk>, <sos/eos>, <unk> and the word marker; 4..126 are single Unicode code
points (IPA letters and combining / spacing modifiers). A target phone is a string of one or more
of these code points (for example "tʃ", "aː", "kʰ"), never starting with a modifier.
"""
from __future__ import annotations

import os

SPECIAL = ("<blk>", "<sos/eos>", "<unk>", "▁")
# code points that attach to the preceding symbol (a phone may not start with one)
MODIFIERS = frozenset("ʰʲʷʼː˞ˠˤ" + "̴̥̩̪̺̃̚")


def default_token_file() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "zipa_tokens.txt")


def load_tokens(path: str | None = None) -> list:
    """Token list, index = id."""
    toks = []
    with open(path or default_token_file(), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if int(parts[1]) != len(toks):
                raise ValueError("token ids must be contiguous from 0")
            toks.append(parts[0])
    return toks


def is_valid_phone(ipa: str, tokens) -> bool:
    syms = set(t for t in tokens if t not in SPECIAL)
    if not ipa or ipa[0] in MODIFIERS:
        return False
    for c in ipa:
        if c not in syms:
            return False
    return True
