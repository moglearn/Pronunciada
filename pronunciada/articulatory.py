"""Articulatory feature table for every sound symbol Pronunciada can expect or hear.

Plain data plus two small pure functions, no dependencies, deterministic, meant to be ported
as-is (the two dicts below are the whole table; `features()` / `diff()` are ~80 lines).

    features("tʰ")  -> {"type": "consonant", "place": "alveolar", "manner": "stop",
                        "voicing": "voiceless", "aspirated": True, ...}
    diff("tʰ", "d") -> ["aspirated", "voicing"]

Symbols are ZIPA tokens (pronunciada/data/zipa_tokens.txt). A *phone* is a string of tokens: one
base symbol plus trailing modifiers ("tʰ", "aː", "ɔ̃"), or a sequence the language front-ends treat
as one sound:
  * stop + fricative            -> one affricate ("ts", "tʃ", "dʒ", "pf", "tɕ", "ʈʂ")
  * the same consonant twice    -> a long (geminate) consonant ("tt", "ss"; "tts" = long "ts")
  * two or more vowels          -> a diphthong: the first vowel's features plus `glide`
                                   (the last vowel; "ai", "aʊ", "ieu"); a repeated vowel = long
  * vowel + "ɐ" / "ɻ"           -> German vocalised r ("eːɐ") / Mandarin erhua ("ɑɻ") = rhotic
  * anything else (rare: "la", "tʃɛ", "kw", "ɯi") -> the first sound's features plus `next`
                                   (the rest of the sequence)

Feature names (all phones carry every key of their type; booleans default False):
  consonant: place, manner, voicing, aspirated, palatalised, labialised, velarised,
             pharyngealised, ejective, unreleased, syllabic, long, laminal_dental (dental
             diacritic), apical, nasal_release, lateral, rhotic (r-colored), glide, next
  vowel:     height (close, near-close, close-mid, mid, open-mid, near-open, open),
             backness (front, central, back), rounded, nasal, long, rhotic, devoiced, glide, next
Affricates keep the fricative's place (tʃ = postalveolar affricate); "lateral" is a separate
boolean so l / ɮ / ɬ / ʎ / ɭ / ʟ / ɺ differ from their central counterparts in one feature.
"""
from __future__ import annotations

from functools import lru_cache

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- base symbols
# consonants: symbol -> (place, manner, voicing[, lateral])
_C = {
    # stops
    "p": ("bilabial", "stop", "voiceless"), "b": ("bilabial", "stop", "voiced"),
    "t": ("alveolar", "stop", "voiceless"), "d": ("alveolar", "stop", "voiced"),
    "ʈ": ("retroflex", "stop", "voiceless"), "ɖ": ("retroflex", "stop", "voiced"),
    "c": ("palatal", "stop", "voiceless"), "ɟ": ("palatal", "stop", "voiced"),
    "k": ("velar", "stop", "voiceless"), "g": ("velar", "stop", "voiced"),
    "q": ("uvular", "stop", "voiceless"), "ɢ": ("uvular", "stop", "voiced"),
    "ʔ": ("glottal", "stop", "voiceless"), "ʡ": ("epiglottal", "stop", "voiceless"),
    # implosives
    "ɓ": ("bilabial", "implosive", "voiced"), "ɗ": ("alveolar", "implosive", "voiced"),
    "ᶑ": ("retroflex", "implosive", "voiced"), "ʄ": ("palatal", "implosive", "voiced"),
    "ɠ": ("velar", "implosive", "voiced"), "ʛ": ("uvular", "implosive", "voiced"),
    # clicks
    "ʘ": ("bilabial", "click", "voiceless"), "ǀ": ("dental", "click", "voiceless"),
    "ǃ": ("alveolar", "click", "voiceless"), "ǂ": ("palatal", "click", "voiceless"),
    "ǁ": ("alveolar", "click", "voiceless", True),
    # nasals
    "m": ("bilabial", "nasal", "voiced"), "ɱ": ("labiodental", "nasal", "voiced"),
    "n": ("alveolar", "nasal", "voiced"), "ɳ": ("retroflex", "nasal", "voiced"),
    "ɲ": ("palatal", "nasal", "voiced"), "ŋ": ("velar", "nasal", "voiced"),
    "ɴ": ("uvular", "nasal", "voiced"),
    # trills, taps / flaps
    "ʙ": ("bilabial", "trill", "voiced"), "r": ("alveolar", "trill", "voiced"),
    "ʀ": ("uvular", "trill", "voiced"),
    "ⱱ": ("labiodental", "flap", "voiced"), "ɾ": ("alveolar", "flap", "voiced"),
    "ɽ": ("retroflex", "flap", "voiced"), "ɺ": ("alveolar", "flap", "voiced", True),
    # fricatives
    "ɸ": ("bilabial", "fricative", "voiceless"), "β": ("bilabial", "fricative", "voiced"),
    "f": ("labiodental", "fricative", "voiceless"), "v": ("labiodental", "fricative", "voiced"),
    "θ": ("dental", "fricative", "voiceless"), "ð": ("dental", "fricative", "voiced"),
    "s": ("alveolar", "fricative", "voiceless"), "z": ("alveolar", "fricative", "voiced"),
    "ʃ": ("postalveolar", "fricative", "voiceless"), "ʒ": ("postalveolar", "fricative", "voiced"),
    "ʂ": ("retroflex", "fricative", "voiceless"), "ʐ": ("retroflex", "fricative", "voiced"),
    "ɕ": ("alveolopalatal", "fricative", "voiceless"), "ʑ": ("alveolopalatal", "fricative", "voiced"),
    "ç": ("palatal", "fricative", "voiceless"), "ʝ": ("palatal", "fricative", "voiced"),
    "x": ("velar", "fricative", "voiceless"), "ɣ": ("velar", "fricative", "voiced"),
    "χ": ("uvular", "fricative", "voiceless"), "ʁ": ("uvular", "fricative", "voiced"),
    "ħ": ("pharyngeal", "fricative", "voiceless"), "ʕ": ("pharyngeal", "fricative", "voiced"),
    "ʜ": ("epiglottal", "fricative", "voiceless"), "ʢ": ("epiglottal", "fricative", "voiced"),
    "h": ("glottal", "fricative", "voiceless"), "ɦ": ("glottal", "fricative", "voiced"),
    "ɧ": ("postalveolar-velar", "fricative", "voiceless"),
    "ɬ": ("alveolar", "fricative", "voiceless", True), "ɮ": ("alveolar", "fricative", "voiced", True),
    # approximants
    "ʋ": ("labiodental", "approximant", "voiced"), "ɹ": ("alveolar", "approximant", "voiced"),
    "ɻ": ("retroflex", "approximant", "voiced"), "j": ("palatal", "approximant", "voiced"),
    "ɰ": ("velar", "approximant", "voiced"), "w": ("labial-velar", "approximant", "voiced"),
    "ʍ": ("labial-velar", "approximant", "voiceless"), "ɥ": ("labial-palatal", "approximant", "voiced"),
    "l": ("alveolar", "approximant", "voiced", True), "ɭ": ("retroflex", "approximant", "voiced", True),
    "ʎ": ("palatal", "approximant", "voiced", True), "ʟ": ("velar", "approximant", "voiced", True),
}

# vowels: symbol -> (height, backness, rounded)
_V = {
    "i": ("close", "front", False), "y": ("close", "front", True),
    "ɨ": ("close", "central", False), "ʉ": ("close", "central", True),
    "ɯ": ("close", "back", False), "u": ("close", "back", True),
    "ɪ": ("near-close", "front", False), "ʏ": ("near-close", "front", True),
    "ʊ": ("near-close", "back", True),
    "e": ("close-mid", "front", False), "ø": ("close-mid", "front", True),
    "ɘ": ("close-mid", "central", False), "ɵ": ("close-mid", "central", True),
    "ɤ": ("close-mid", "back", False), "o": ("close-mid", "back", True),
    "ə": ("mid", "central", False),
    "ɛ": ("open-mid", "front", False), "œ": ("open-mid", "front", True),
    "ɜ": ("open-mid", "central", False), "ɞ": ("open-mid", "central", True),
    "ʌ": ("open-mid", "back", False), "ɔ": ("open-mid", "back", True),
    "æ": ("near-open", "front", False), "ɐ": ("near-open", "central", False),
    "a": ("open", "front", False), "ɶ": ("open", "front", True),
    "ɑ": ("open", "back", False), "ɒ": ("open", "back", True),
}

# modifiers (ZIPA tokens that attach to the preceding symbol): token -> (feature, value)
MODIFIERS = {
    "ʰ": ("aspirated", True), "ʲ": ("palatalised", True), "ʷ": ("labialised", True),
    "ʼ": ("ejective", True), "ː": ("long", True), "˞": ("rhotic", True),
    "ˠ": ("velarised", True), "ˤ": ("pharyngealised", True), "̃": ("nasal", True),
    "̚": ("unreleased", True), "̥": ("voiceless", True), "̩": ("syllabic", True),
    "̪": ("laminal_dental", True), "̴": ("velarised", True), "̺": ("apical", True),
}

HEIGHTS = ("close", "near-close", "close-mid", "mid", "open-mid", "near-open", "open")
BACKNESS = ("front", "central", "back")
PLACES = ("bilabial", "labiodental", "dental", "alveolar", "postalveolar", "retroflex",
          "alveolopalatal", "palatal", "labial-palatal", "velar", "labial-velar",
          "postalveolar-velar", "uvular", "pharyngeal", "epiglottal", "glottal")

CONSONANT_KEYS = ("place", "manner", "voicing", "lateral", "aspirated", "palatalised",
                  "labialised", "velarised", "pharyngealised", "ejective", "unreleased",
                  "syllabic", "long", "laminal_dental", "apical", "nasal_release", "rhotic",
                  "glide", "next")
VOWEL_KEYS = ("height", "backness", "rounded", "nasal", "long", "rhotic", "devoiced", "glide",
              "next")
# features compared by diff() (glide / next are compared by value: the glide target vowel)
_STOP_FRIC = {("bilabial", "labiodental"), ("alveolar", "alveolar"), ("alveolar", "postalveolar"),
              ("alveolar", "alveolopalatal"), ("retroflex", "retroflex"), ("palatal", "palatal"),
              ("velar", "velar"), ("uvular", "uvular"), ("alveolar", "dental"),
              ("bilabial", "bilabial"), ("alveolar", "retroflex"), ("velar", "uvular")}


def symbols() -> set[str]:
    """Every base symbol and modifier the table describes."""
    return set(_C) | set(_V) | set(MODIFIERS)


def _units(phone: str) -> list[str]:
    out: list[str] = []
    for ch in phone:
        if ch in MODIFIERS and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def _unit(u: str) -> dict:
    base, mods = u[0], u[1:]
    if base in _C:
        c = _C[base]
        d = {"type": "consonant", "place": c[0], "manner": c[1], "voicing": c[2],
             "lateral": bool(len(c) > 3 and c[3])}
        for k in CONSONANT_KEYS[4:]:
            d.setdefault(k, None if k in ("glide", "next") else False)
        for m in mods:
            k, v = MODIFIERS[m]
            if k == "voiceless":
                d["voicing"] = "voiceless"
            elif k == "nasal":
                d["nasal_release"] = True
            else:
                d[k] = v
        return d
    if base in _V:
        v = _V[base]
        d = {"type": "vowel", "height": v[0], "backness": v[1], "rounded": v[2], "nasal": False,
             "long": False, "rhotic": False, "devoiced": False, "glide": None, "next": None}
        for m in mods:
            k, val = MODIFIERS[m]
            if k == "voiceless":
                d["devoiced"] = True
            elif k in d:
                d[k] = val
        return d
    raise KeyError(f"no articulatory features for {u!r}")


def _strip(u: str) -> str:
    return u[0]


@lru_cache(maxsize=4096)
def _features(phone: str) -> tuple:
    us = _units(phone)
    if not us:
        raise KeyError("empty phone")
    fs = [_unit(u) for u in us]
    head = dict(fs[0])
    rest = us[1:]
    if not rest:
        return tuple(sorted(head.items()))
    if head["type"] == "consonant":
        f1 = fs[1]
        # geminate: same consonant repeated ("tt", "ss", "tts" = long ts handled below)
        if len(us) >= 2 and _strip(us[0]) == _strip(us[1]) and f1["type"] == "consonant":
            g = _features(phone[len(us[0]):])
            d = dict(g)
            d["long"] = True
            return tuple(sorted(d.items()))
        # affricate: stop + homorganic-ish fricative
        if (head["manner"] == "stop" and f1["type"] == "consonant"
                and f1["manner"] == "fricative" and (head["place"], f1["place"]) in _STOP_FRIC):
            d = dict(f1)
            d["manner"] = "affricate"
            d["voicing"] = "voiced" if (head["voicing"] == "voiced" or f1["voicing"] == "voiced") else "voiceless"
            for k in ("aspirated", "palatalised", "labialised", "ejective", "long"):
                d[k] = bool(head[k] or f1[k])
            if len(us) > 2:                                       # "tʃɛ": affricate + rest
                d["next"] = "".join(us[2:])
            return tuple(sorted(d.items()))
        if len(us) == 2 and us[1] in ("w", "j"):              # "kw", "pj": secondary articulation
            head["labialised" if us[1] == "w" else "palatalised"] = True
            return tuple(sorted(head.items()))
        head["next"] = "".join(rest)
        return tuple(sorted(head.items()))
    # vowel first
    if all(f["type"] == "vowel" for f in fs):
        if all(_strip(u) == _strip(us[0]) for u in us):          # "ii": a long vowel
            head["long"] = True
            return tuple(sorted(head.items()))
        if len(us) == 2 and _strip(us[1]) == "ɐ" and _strip(us[0]) != "ɐ":   # German vocalised r
            head["rhotic"] = True
            return tuple(sorted(head.items()))
        head["glide"] = _strip(us[-1])
        return tuple(sorted(head.items()))
    if len(us) == 2 and us[1] in ("j", "w"):                     # Portuguese "aj", "ew"
        head["glide"] = "i" if us[1] == "j" else "u"
        return tuple(sorted(head.items()))
    if len(us) == 2 and fs[1]["type"] == "consonant" and _strip(us[1]) in ("ɻ", "ɹ"):  # erhua
        head["rhotic"] = True
        return tuple(sorted(head.items()))
    head["next"] = "".join(rest)
    return tuple(sorted(head.items()))


def features(phone: str) -> dict:
    """Feature dict of one phone (see module docstring); KeyError for an unknown symbol."""
    return dict(_features(phone))


def covered(phone: str) -> bool:
    try:
        features(phone)
        return True
    except KeyError:
        return False


def diff(expected: str, heard: str) -> list[str]:
    """Names of the features that differ between two phones, in a fixed order. Consonant vs
    vowel -> ["type"]. Identical phones -> []."""
    if expected == heard:
        return []
    a, b = features(expected), features(heard)
    if a["type"] != b["type"]:
        return ["type"]
    keys = CONSONANT_KEYS if a["type"] == "consonant" else VOWEL_KEYS
    return [k for k in keys if a.get(k) != b.get(k)]


def height_steps(a: str, b: str) -> int:
    """Signed vowel-height difference b - a in steps (positive = b is more open)."""
    return HEIGHTS.index(b) - HEIGHTS.index(a)


def describe(phone: str) -> str:
    """Short English description, e.g. 'voiceless alveolar stop (aspirated)'."""
    f = features(phone)
    if f["type"] == "consonant":
        mods = [k for k in ("aspirated", "palatalised", "labialised", "velarised",
                            "pharyngealised", "ejective", "long", "syllabic") if f[k]]
        s = f"{f['voicing']} {f['place']} {'lateral ' if f['lateral'] else ''}{f['manner']}"
    else:
        mods = [k for k in ("nasal", "long", "rhotic", "devoiced") if f[k]]
        s = f"{f['height']} {f['backness']} {'rounded' if f['rounded'] else 'unrounded'} vowel"
        if f["glide"]:
            s += f" gliding to {f['glide']}"
    return s + (f" ({', '.join(mods)})" if mods else "")
