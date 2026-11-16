"""Native statistics and learned variants (DESIGN section 3), pure Python/numpy at runtime.

Tables: pronunciada/data/native/<lang>.json, produced by bench/pipeline/native_stats.py from native
FLEURS-train speech (with that language's output adapter applied, if it ships one):

* stats: per canonical phone, per (phone, position), per broad class and global, for each feature
  in FEATURES: count, mean, std and the 1/5/25/50/75/95 % quantiles.
* variants: per phone, learned alternative realisations [(ipa, log_prior, count, rate)].
  ``""`` = the sound may be dropped.
* variants_enabled: whether the evaluation recommended turning variants on for this language.

    NativeStats.load("en").z("t", "final", "llr_sub", -2.3)
    apply_variants(ref, "en")
"""
from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from importlib import resources
from pathlib import Path

FEATURES = ("llr_sub", "llr_del", "n_frames", "peak_post")
POSITIONS = ("initial", "medial", "final", "only")
QUANTILES = (1, 5, 25, 50, 75, 95)
MIN_COUNT = 30
CLASSES = ("vowel", "stop", "fricative", "affricate", "nasal", "liquid-glide")

_VOWELS = frozenset("aeiouyæøœɐɑɒɔɘəɛɜɞɤɨɪɯɵɶʉʊʌʏ")
_MODIFIERS = frozenset("ʰʲʷʼː˞ˠˤ" + "̴̥̩̪̺̃̚")
_STOPS = frozenset("pbtdʈɖcɟkgqɢʔɓɗʄɠʛᶑǀǁǂǃʘ")
_FRICS = frozenset("ɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɕʑɧʍ")
_NASALS = frozenset("mɱnɳɲŋɴ")
_LIQ = frozenset("lɭʎʟɬɮrɾɹɻɽɺʀʙⱱjwɥɰʋ")
_AFFR = frozenset({("t", "s"), ("d", "z"), ("t", "ʃ"), ("d", "ʒ"), ("t", "ɕ"), ("d", "ʑ"),
                   ("ʈ", "ʂ"), ("ɖ", "ʐ"), ("p", "f"), ("t", "θ"), ("k", "x"), ("c", "ç")})


def broad_class(ipa: str) -> str:
    """vowel / stop / fricative / affricate / nasal / liquid-glide of a ZIPA phone string."""
    if ipa and ipa[0] in _VOWELS:
        return "vowel"
    base = [c for c in ipa if c not in _MODIFIERS]
    if len(base) >= 2 and (base[0], base[1]) in _AFFR:
        return "affricate"
    c = base[0] if base else (ipa[:1] or "?")
    if c in _STOPS:
        return "stop"
    if c in _FRICS:
        return "fricative"
    if c in _NASALS:
        return "nasal"
    if c in _LIQ:
        return "liquid-glide"
    return "vowel" if any(x in _VOWELS for x in base) else "fricative"


def phone_position(ref, i: int) -> str:
    """initial / medial / final / only (single-phone word) of phone i within its word."""
    p0, p1 = ref.words[ref.phones[i].word].phones
    if p1 - p0 == 1:
        return "only"
    if i == p0:
        return "initial"
    if i == p1 - 1:
        return "final"
    return "medial"


def table_path(lang: str) -> Path:
    return Path(str(resources.files("pronunciada").joinpath(f"data/native/{lang}.json")))


@lru_cache(maxsize=None)
def _table(lang: str) -> dict | None:
    p = table_path(lang)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def reload() -> None:
    _table.cache_clear()


class NativeStats:
    def __init__(self, table: dict):
        self.t = table
        self.lang = table.get("lang", "")

    @classmethod
    def load(cls, lang: str) -> "NativeStats | None":
        t = _table(lang)
        return None if t is None else cls(t)

    def stats(self, phone: str, position: str | None, feature: str) -> tuple[dict, str]:
        """(stats dict, level) after back-off phone+position -> phone -> broad class -> global,
        taking the first level with count >= MIN_COUNT."""
        s = self.t["stats"]
        if position is not None:
            d = s["phone_pos"].get(phone, {}).get(position, {}).get(feature)
            if d and d["count"] >= MIN_COUNT:
                return d, "phone_pos"
        d = s["phone"].get(phone, {}).get(feature)
        if d and d["count"] >= MIN_COUNT:
            return d, "phone"
        d = s["class"].get(broad_class(phone), {}).get(feature)
        if d and d["count"] >= MIN_COUNT:
            return d, "class"
        return s["global"][feature], "global"

    def z(self, phone: str, position: str | None, feature: str, value: float) -> float:
        d, _ = self.stats(phone, position, feature)
        return (float(value) - d["mean"]) / max(d["std"], 1e-3)

    def variants(self, phone: str) -> list[tuple[str, float]]:
        return [(a, float(w)) for a, w, *_ in self.t.get("variants", {}).get(phone, [])]

    @property
    def variants_enabled(self) -> bool:
        return bool(self.t.get("variants_enabled", False))


def learned_alts(lang: str, phone: str) -> list[tuple[str, float]]:
    ns = NativeStats.load(lang)
    return [] if ns is None else ns.variants(phone)


def apply_variants(ref, lang: str | None = None, only_if_enabled: bool = False):
    """Deep copy of `ref` with each phone's learned alternatives appended to RefPhone.alts
    (never duplicating the canonical phone or an existing alternative)."""
    lang = lang or ref.lang
    out = copy.deepcopy(ref)
    ns = NativeStats.load(lang)
    if ns is None or (only_if_enabled and not ns.variants_enabled):
        return out
    for p in out.phones:
        have = {p.ipa} | {a for a, _ in p.alts}
        for a, w in ns.variants(p.ipa):
            if a not in have:
                p.alts.append((a, w))
                have.add(a)
    return out


def strip_variants(ref, lang: str | None = None):
    """Inverse of apply_variants: deep copy without the learned alternatives (matched on
    ipa and log-prior)."""
    lang = lang or ref.lang
    out = copy.deepcopy(ref)
    ns = NativeStats.load(lang)
    if ns is None:
        return out
    for p in out.phones:
        learned = ns.variants(p.ipa)
        p.alts = [(a, w) for a, w in p.alts
                  if not any(a == la and math.isclose(w, lw, abs_tol=1e-9) for la, lw in learned)]
    return out
