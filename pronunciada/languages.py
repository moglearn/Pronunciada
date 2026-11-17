"""Supported languages and the optional extras their text front-ends need (no model load).

    from pronunciada.languages import languages
    languages()   # [{"code": "zh", "name": "Mandarin Chinese", "extra": "zh", "ready": True, ...}, ...]
"""
from __future__ import annotations

from importlib.util import find_spec

from .text.base import LANGS

NAMES = {"en": "English", "zh": "Mandarin Chinese", "ja": "Japanese", "es": "Spanish", "fr": "French",
         "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ko": "Korean"}
# language -> (pip extra, import names it needs); languages not listed need only the core install
EXTRAS = {"zh": ("zh", ("pypinyin", "jieba", "opencc")), "ja": ("ja", ("pyopenjtalk",))}


def _installed(mod: str) -> bool:
    try:
        return find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def languages() -> list[dict]:
    out = []
    for code, aliases in LANGS.items():
        extra, mods = EXTRAS.get(code, (None, ()))
        missing = [m for m in mods if not _installed(m)]
        out.append({"code": code, "name": NAMES.get(code, code), "extra": extra,
                    "install": f"pip install 'pronunciada[{extra}]'" if extra else None,
                    "ready": not missing, "missing": missing, "aliases": list(aliases)})
    return out
