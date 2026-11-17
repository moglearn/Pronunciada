"""Controlled-error protocol.

Take the correct Reference for a native recording, return a corrupted deep copy plus labels
saying where it is now wrong. Scoring the (correct) native audio against the corrupted reference
simulates a learner error at a known place:

kind            what changes in the reference                learner error it simulates   flagged
phone_sub_near  one phone -> articulatorily close phone       substitution (subtle)        that phone
phone_sub_far   one phone -> distant phone of the same class  substitution (gross)         that phone
phone_ins       a consonant is inserted into a syllable       learner DELETED that sound   the inserted phone
phone_del       a non-nucleus phone is removed                learner INSERTED a sound     no phone (word only;
                                                                                           `gap` gives position)
word_swap       a word's syllables/phones <- another word's   wrong word / misreading      word + differing phones
tone            one syllable's tone -> another tone           wrong tone                   syllable (tone)
stress          primary stress moved within a word            misplaced stress             2 syllables (stress)

Every flagged item also flags its word. The result always passes Reference.validate().
Similarity of phones: panphon articulatory feature distance (bench-only dependency) between
segments, DTW over segments for multi-segment phones (affricates, diphthongs); vowels are only
replaced by vowels and consonants by consonants. Distances are cached in
$PRONUNCIADA_CACHE/bench/phone_dist.json.
"""
from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pronunciada.text.ipa import is_vowel
from pronunciada.types import RefPhone, RefSyllable, RefWord, Reference

KINDS = ("phone_sub_near", "phone_sub_far", "phone_del", "phone_ins", "word_swap", "tone",
         "stress")


class NotApplicable(ValueError):
    """The requested corruption cannot be applied to this reference."""


# ----------------------------------------------------------------------------- phone distance

def _cache_path() -> Path:
    base = os.environ.get("PRONUNCIADA_CACHE")
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "pronunciada"
    return root / "bench" / "phone_dist.json"


class PhoneDistance:
    """Articulatory distance between two ZIPA phones (strings), symmetric, 0 for identical."""

    UNKNOWN = 50.0   # distance for phones panphon cannot parse (never "near")

    def __init__(self, cache: Path | None = None):
        self.path = cache or _cache_path()
        self.table: dict[str, float] = {}
        if self.path.exists():
            try:
                self.table = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                self.table = {}
        self._dirty = False
        self._pp = None

    def _panphon(self):
        if self._pp is None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import panphon
                import panphon.distance
            self._pp = (panphon.FeatureTable(), panphon.distance.Distance())
        return self._pp

    def _seg_dist(self, a: str, b: str) -> float:
        return float(self._panphon()[1].weighted_feature_edit_distance(a, b))

    def _compute(self, a: str, b: str) -> float:
        ft, _ = self._panphon()
        sa, sb = ft.ipa_segs(a), ft.ipa_segs(b)
        if not sa or not sb or "".join(sa) != a or "".join(sb) != b:
            return self.UNKNOWN
        # DTW over segments with substitution cost only, normalised by path length
        n, m = len(sa), len(sb)
        D = np.full((n + 1, m + 1), np.inf)
        L = np.zeros((n + 1, m + 1))
        D[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                c = self._seg_dist(sa[i - 1], sb[j - 1])
                prev = min(((D[i - 1, j - 1], L[i - 1, j - 1]), (D[i - 1, j], L[i - 1, j]),
                            (D[i, j - 1], L[i, j - 1])), key=lambda t: t[0])
                D[i, j] = prev[0] + c
                L[i, j] = prev[1] + 1
        return float(D[n, m] / L[n, m])

    def __call__(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = "|".join(sorted((a, b)))
        if key not in self.table:
            self.table[key] = self._compute(a, b)
            self._dirty = True
        return self.table[key]

    def save(self) -> None:
        """Write the cache atomically. The temp name is per process (and per thread), so several
        workers can save concurrently; entries already on disk from other processes are merged in."""
        if self._dirty:
            import threading

            self.path.parent.mkdir(parents=True, exist_ok=True)
            merged: dict[str, float] = {}
            try:
                merged = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
            merged.update(self.table)
            tmp = self.path.with_name(f"{self.path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(merged, ensure_ascii=False, sort_keys=True))
            os.replace(tmp, self.path)
            self.table = merged
            self._dirty = False


_DIST: PhoneDistance | None = None


def phone_distance() -> PhoneDistance:
    global _DIST
    if _DIST is None:
        _DIST = PhoneDistance()
    return _DIST


# ----------------------------------------------------------------------------- inventory

def _word_phones(ref: Reference, wi: int) -> tuple[str, ...]:
    p0, p1 = ref.words[wi].phones
    return tuple(ref.phones[i].ipa for i in range(p0, p1))


@dataclass
class WordEntry:
    text: str
    phones: tuple[str, ...]
    syllables: list[RefSyllable]          # word index / phone spans relative to the word
    phone_objs: list[RefPhone]


@dataclass
class Inventory:
    """Per-language phone inventory and word pool, built from a list of References."""
    lang: str
    phones: Counter = field(default_factory=Counter)
    words: list[WordEntry] = field(default_factory=list)
    min_count: int = 2                       # phones seen fewer times are not used as replacements

    @classmethod
    def from_references(cls, refs: Iterable[Reference], lang: str | None = None,
                        min_count: int = 2) -> "Inventory":
        refs = list(refs)
        lang = lang or (refs[0].lang if refs else "")
        inv = cls(lang=lang, min_count=min_count)
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for r in refs:
            if r.lang != lang:
                continue
            for p in r.phones:
                inv.phones[p.ipa] += 1
            for wi, w in enumerate(r.words):
                key = (w.text.lower(), _word_phones(r, wi))
                if key in seen:
                    continue
                seen.add(key)
                inv.words.append(_extract_word(r, wi))
        return inv

    def candidates(self, vowel: bool) -> list[str]:
        return sorted(p for p, c in self.phones.items() if c >= self.min_count and is_vowel(p) == vowel)

    def ranked_substitutes(self, ipa: str, exclude: set[str]) -> list[tuple[float, str]]:
        dist = phone_distance()
        cands = [c for c in self.candidates(is_vowel(ipa)) if c != ipa and c not in exclude]
        ranked = sorted(((dist(ipa, c), c) for c in cands), key=lambda t: (t[0], t[1]))
        dist.save()
        return [(d, c) for d, c in ranked if d < PhoneDistance.UNKNOWN] or ranked


def _extract_word(r: Reference, wi: int) -> WordEntry:
    w = r.words[wi]
    s0, s1 = w.syllables
    p0, _ = w.phones
    syls = []
    for si in range(s0, s1):
        s = copy.deepcopy(r.syllables[si])
        s.phones = (s.phones[0] - p0, s.phones[1] - p0)
        syls.append(s)
    phones = [copy.deepcopy(r.phones[i]) for i in range(*w.phones)]
    return WordEntry(w.text, _word_phones(r, wi), syls, phones)


# ----------------------------------------------------------------------------- nested working copy

@dataclass
class _P:
    obj: RefPhone
    wrong: bool = False


@dataclass
class _S:
    obj: RefSyllable
    phones: list[_P]
    flags: set = field(default_factory=set)       # "tone", "stress", "phones"


@dataclass
class _W:
    obj: RefWord
    syls: list[_S]
    flags: set = field(default_factory=set)       # kinds applied to this word


def _nest(ref: Reference) -> list[_W]:
    words = []
    for w in ref.words:
        syls = []
        for si in range(*w.syllables):
            s = ref.syllables[si]
            syls.append(_S(copy.deepcopy(s), [_P(copy.deepcopy(ref.phones[i])) for i in range(*s.phones)]))
        words.append(_W(copy.deepcopy(w), syls))
    return words


def _flatten(lang: str, text: str, warnings: list[str], words: list[_W]):
    ref = Reference(lang=lang, text=text, warnings=list(warnings))
    phone_wrong: list[bool] = []
    syl_flags: list[list[str]] = []
    for wi, w in enumerate(words):
        s_start, p_start = len(ref.syllables), len(ref.phones)
        for s in w.syls:
            si = len(ref.syllables)
            ps = len(ref.phones)
            for p in s.phones:
                p.obj.word, p.obj.syllable = wi, si
                ref.phones.append(p.obj)
                phone_wrong.append(p.wrong)
            s.obj.word = wi
            s.obj.phones = (ps, len(ref.phones))
            ref.syllables.append(s.obj)
            syl_flags.append(sorted(s.flags))
        w.obj.syllables = (s_start, len(ref.syllables))
        w.obj.phones = (p_start, len(ref.phones))
        ref.words.append(w.obj)
    return ref, phone_wrong, syl_flags


# ----------------------------------------------------------------------------- result

@dataclass
class Error:
    kind: str
    word: int
    detail: dict


@dataclass
class Corrupted:
    """ref: corrupted Reference.  Labels are aligned with ref's own indices:
    phone_wrong[i]   phone i of ref is not what the audio contains
    syllable_flags[j] subset of {"phones", "tone", "stress"} wrong in syllable j
    word_wrong[k]    word k contains any error (incl. phone_del, which flags no phone)
    word_kinds[k]    kinds applied to word k
    errors           one entry per applied error (kind, word, detail)
    """
    ref: Reference
    original: Reference
    errors: list[Error]
    phone_wrong: list[bool]
    syllable_flags: list[list[str]]
    word_wrong: list[bool]
    word_kinds: list[list[str]]

    def to_dict(self) -> dict:
        return {"errors": [{"kind": e.kind, "word": e.word, **e.detail} for e in self.errors],
                "phone_wrong": self.phone_wrong, "syllable_flags": self.syllable_flags,
                "word_wrong": self.word_wrong, "word_kinds": self.word_kinds}


# ----------------------------------------------------------------------------- the corruptions

def _forbidden(p: RefPhone) -> set[str]:
    return {p.ipa} | {a for a, _ in p.alts}


def _new_phone(ipa: str, like: RefPhone, nucleus: bool | None = None) -> RefPhone:
    return RefPhone(ipa=ipa, word=like.word, syllable=like.syllable,
                    is_nucleus=like.is_nucleus if nucleus is None else nucleus, alts=[])


def _phone_sites(words, used):
    return [(wi, si, pi) for wi, w in enumerate(words) if wi not in used
            for si, s in enumerate(w.syls) for pi, p in enumerate(s.phones) if not p.wrong]


def _sub(words, used, rng, inv: Inventory, far: bool, near_k: int):
    sites = _phone_sites(words, used)
    rng.shuffle(sites)
    for wi, si, pi in sites:
        p = words[wi].syls[si].phones[pi]
        ranked = inv.ranked_substitutes(p.obj.ipa, _forbidden(p.obj))
        if not ranked:
            continue
        if far:
            pool = ranked[len(ranked) // 2:] if len(ranked) > 1 else ranked
        else:
            pool = ranked[:near_k]
        d, new = pool[int(rng.integers(len(pool)))]
        old = p.obj.ipa
        p.obj = _new_phone(new, p.obj)
        p.wrong = True
        words[wi].syls[si].flags.add("phones")
        return wi, {"syllable_local": si, "from": old, "to": new, "distance": round(d, 3)}
    raise NotApplicable("no phone with a same-class substitute")


def _del(words, used, rng, inv):
    sites = [(wi, si, pi) for wi, si, pi in _phone_sites(words, used)
             if not words[wi].syls[si].phones[pi].obj.is_nucleus and len(words[wi].syls[si].phones) > 1]
    if not sites:
        raise NotApplicable("no non-nucleus phone to delete")
    wi, si, pi = sites[int(rng.integers(len(sites)))]
    s = words[wi].syls[si]
    old = s.phones.pop(pi)
    s.flags.add("phones")
    # gap: position in the syllable where the (now unexpected) sound sits
    return wi, {"syllable_local": si, "removed": old.obj.ipa, "gap_in_syllable": pi}


def _ins(words, used, rng, inv: Inventory):
    cons = inv.candidates(vowel=False)
    if not cons:
        raise NotApplicable("inventory has no consonants")
    sites = [(wi, si) for wi, w in enumerate(words) if wi not in used for si in range(len(w.syls))]
    if not sites:
        raise NotApplicable("no word left")
    wi, si = sites[int(rng.integers(len(sites)))]
    s = words[wi].syls[si]
    nuc = next(i for i, p in enumerate(s.phones) if p.obj.is_nucleus)
    # onset (before nucleus) or coda (after), never next to an identical phone
    positions = list(range(0, nuc + 1)) + list(range(nuc + 1, len(s.phones) + 1))
    for _ in range(20):
        pos = positions[int(rng.integers(len(positions)))]
        new = cons[int(rng.integers(len(cons)))]
        left = s.phones[pos - 1].obj.ipa if pos > 0 else None
        right = s.phones[pos].obj.ipa if pos < len(s.phones) else None
        if new not in (left, right):
            break
    else:
        raise NotApplicable("could not find an insertion")
    s.phones.insert(pos, _P(_new_phone(new, s.phones[nuc].obj, nucleus=False), wrong=True))
    s.flags.add("phones")
    return wi, {"syllable_local": si, "inserted": new, "pos_in_syllable": pos}


def _levenshtein_matches(a: Sequence[str], b: Sequence[str]) -> set[int]:
    """Indices j of b aligned to an identical element of a in a min-edit alignment."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=int)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, out = n, m, set()
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and D[i, j] == D[i - 1, j - 1]:
            out.add(j - 1)
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j - 1] + 1:
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def _swap(words, used, rng, inv: Inventory):
    sites = [wi for wi in range(len(words)) if wi not in used]
    rng.shuffle(sites)
    for wi in sites:
        w = words[wi]
        orig = tuple(p.obj.ipa for s in w.syls for p in s.phones)
        pool = [e for e in inv.words if e.phones != orig and e.text.lower() != w.obj.text.lower()]
        if not pool:
            continue
        same = [e for e in pool if len(e.syllables) == len(w.syls)]
        pool = same or pool
        donor = pool[int(rng.integers(len(pool)))]
        keep = _levenshtein_matches(orig, donor.phones)
        new_syls = []
        for s in donor.syllables:
            s2 = copy.deepcopy(s)
            ps = []
            for k in range(*s.phones):
                p = copy.deepcopy(donor.phone_objs[k])
                p.alts = []
                ps.append(_P(p, wrong=k not in keep))
            flags = {"phones"} if any(x.wrong for x in ps) else set()
            new_syls.append(_S(s2, ps, flags))
        w.syls = new_syls
        return wi, {"donor": donor.text, "from": " ".join(orig), "to": " ".join(donor.phones)}
    raise NotApplicable("no donor word with a different pronunciation")


def _tone(words, used, rng, inv, tones=(1, 2, 3, 4)):
    sites = [(wi, si) for wi, w in enumerate(words) if wi not in used
             for si, s in enumerate(w.syls) if s.obj.tone is not None]
    rng.shuffle(sites)
    for wi, si in sites:
        s = words[wi].syls[si]
        choices = [t for t in tones if t != s.obj.tone and t not in s.obj.tone_alts]
        if not choices:
            continue
        old, new = s.obj.tone, choices[int(rng.integers(len(choices)))]
        s.obj.tone = new
        s.obj.tone_lexical = new
        s.obj.tone_alts = []
        s.flags.add("tone")
        return wi, {"syllable_local": si, "from": old, "to": new}
    raise NotApplicable("no syllable with a tone")


def _stress(words, used, rng, inv):
    sites = [wi for wi, w in enumerate(words) if wi not in used and len(w.syls) >= 2
             and any(s.obj.stress == 1 for s in w.syls) and all(s.obj.stress is not None for s in w.syls)]
    if not sites:
        raise NotApplicable("no word with >= 2 syllables and primary stress")
    wi = sites[int(rng.integers(len(sites)))]
    syls = words[wi].syls
    old = next(i for i, s in enumerate(syls) if s.obj.stress == 1)
    others = [i for i in range(len(syls)) if i != old]
    new = others[int(rng.integers(len(others)))]
    syls[old].obj.stress, syls[new].obj.stress = syls[new].obj.stress, 1
    syls[old].flags.add("stress")
    syls[new].flags.add("stress")
    return wi, {"from_syllable_local": old, "to_syllable_local": new}


def applicable_kinds(ref: Reference) -> list[str]:
    out = ["phone_sub_near", "phone_sub_far", "phone_ins", "word_swap"]
    if any(len(range(*s.phones)) > 1 for s in ref.syllables):
        out.append("phone_del")
    if any(s.tone is not None for s in ref.syllables):
        out.append("tone")
    for w in ref.words:
        syl = ref.syllables[slice(*w.syllables)]
        if len(syl) >= 2 and any(s.stress == 1 for s in syl):
            out.append("stress")
            break
    return out


def corrupt(ref: Reference, kinds: str | Sequence[str], inventory: Inventory | None = None,
            n_errors: int = 1, seed: int | np.random.Generator = 0, one_per_word: bool = True,
            near_k: int = 3, strict: bool = True) -> Corrupted:
    """Apply `n_errors` errors, each of a kind drawn uniformly from `kinds`, to a deep copy of ref.
    `inventory` (same language) is needed for phone_sub_*, phone_ins and word_swap.
    With one_per_word, every error lands on a different word. strict=True raises NotApplicable
    if fewer than n_errors could be applied."""
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    for k in kinds:
        if k not in KINDS:
            raise ValueError(f"unknown kind {k!r}")
    if inventory is not None and inventory.lang and inventory.lang != ref.lang:
        raise ValueError("inventory language differs from reference language")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    ref.validate()
    words = _nest(ref)
    used: set[int] = set()
    errors: list[Error] = []
    fns = {"phone_sub_near": lambda: _sub(words, used, rng, inventory, False, near_k),
           "phone_sub_far": lambda: _sub(words, used, rng, inventory, True, near_k),
           "phone_del": lambda: _del(words, used, rng, inventory),
           "phone_ins": lambda: _ins(words, used, rng, inventory),
           "word_swap": lambda: _swap(words, used, rng, inventory),
           "tone": lambda: _tone(words, used, rng, inventory),
           "stress": lambda: _stress(words, used, rng, inventory)}
    needs_inv = {"phone_sub_near", "phone_sub_far", "phone_ins", "word_swap"}
    for _ in range(n_errors):
        order = list(kinds)
        rng.shuffle(order)
        for k in order:
            if k in needs_inv and inventory is None:
                raise ValueError(f"kind {k} needs an Inventory")
            try:
                wi, detail = fns[k]()
            except NotApplicable:
                continue
            words[wi].flags.add(k)
            if one_per_word:
                used.add(wi)
            errors.append(Error(k, wi, detail))
            break
    if strict and len(errors) < n_errors:
        raise NotApplicable(f"applied {len(errors)}/{n_errors} errors of kinds {kinds}")
    new, phone_wrong, syl_flags = _flatten(ref.lang, ref.text, ref.warnings, words)
    new.validate()
    return Corrupted(ref=new, original=ref, errors=errors, phone_wrong=phone_wrong,
                     syllable_flags=syl_flags, word_wrong=[bool(w.flags) for w in words],
                     word_kinds=[sorted(w.flags) for w in words])
