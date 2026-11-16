"""Text front-end registry: prompt text -> Reference (words, syllables, target sounds, tones, stress)."""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

from ..types import Reference

# canonical code -> accepted aliases (lower-case, '-' or '_' both fine)
LANGS: dict[str, tuple[str, ...]] = {
    "en": ("en", "en-us", "en-gb", "eng", "english"),
    "zh": ("zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant", "cmn", "cmn-hans-cn", "cmn-hant-tw", "mandarin"),
    "ja": ("ja", "ja-jp", "jpn", "japanese"),
    "es": ("es", "es-419", "es-mx", "es-us", "es-co", "es-ar", "spa", "spanish"),
    "fr": ("fr", "fr-fr", "fra", "french"),
    "de": ("de", "de-de", "deu", "german"),
    "it": ("it", "it-it", "ita", "italian"),
    "pt": ("pt", "pt-br", "por", "portuguese"),
    "ru": ("ru", "ru-ru", "rus", "russian"),
    "ko": ("ko", "ko-kr", "kor", "korean"),
}
_ALIASES = {a: k for k, v in LANGS.items() for a in v}


def normalize_lang(code: str) -> str:
    key = code.strip().lower().replace("_", "-")
    if key not in _ALIASES:
        raise ValueError(f"unsupported language {code!r}; supported: {sorted(LANGS)}")
    return _ALIASES[key]


class G2P(ABC):
    """One instance per language. Must be cheap to call (<5 ms per sentence after warm-up),
    thread-safe after construction, and must return a Reference that passes .validate()."""

    lang: str

    @abstractmethod
    def __call__(self, text: str) -> Reference: ...


_CACHE: dict[str, G2P] = {}


def get_g2p(lang: str) -> G2P:
    """Each language lives in pronunciada/text/<code>.py and exposes `build() -> G2P`."""
    code = normalize_lang(lang)
    if code not in _CACHE:
        mod = importlib.import_module(f"pronunciada.text.{code}")
        _CACHE[code] = mod.build()
    return _CACHE[code]
