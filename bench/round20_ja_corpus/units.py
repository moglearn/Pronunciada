"""Pronunciada's Japanese reference (pronunciada/text/ja.py) -> mora units and aligner phones.

Each Reference syllable is one mora unit of the front-end: a (C)(C)V mora whose nucleus may be a long
vowel (Vː, two morae), a moraic nasal N (nucleus ŋ/m/n/ɴ), or a mora whose first consonant is a
geminate (kk, ttʃ ...: the っ mora is merged into it). A word-final っ is a trailing ʔ.

Aligner phones (ASCII, one per sound):
    short vowels a i u e o; long vowels aa ii uu ee oo (one phone, one interval);
    N (moraic nasal); cl (the closure of a geminate / word-final っ);
    consonants k g s sh z j t ch ts d n h f b p m y r w ny v dz, palatalised C+j as one phone (ky ...).
A geminate mora becomes [cl, C, V]: the closure interval is the Q mora, C is the release.
"""
from __future__ import annotations

import hashlib

CMAP = {"k": "k", "g": "g", "s": "s", "ʃ": "sh", "z": "z", "dʒ": "j", "t": "t", "tʃ": "ch", "ts": "ts",
        "d": "d", "n": "n", "h": "h", "f": "f", "b": "b", "p": "p", "m": "m", "j": "y", "r": "r", "w": "w",
        "ɲ": "ny", "v": "v", "dz": "dz", "ç": "hy", "ɸ": "f", "ʒ": "j", "ŋ": "g"}
VOWELS = {"a", "i", "u", "e", "o"}
NASAL_NUC = {"ŋ", "m", "n", "ɴ"}
GEM = {"ttʃ": "tʃ", "tts": "ts", "ddʒ": "dʒ", "ddz": "dz"}
LONG = {"aa", "ii", "uu", "ee", "oo"}
PHONES = sorted(set(CMAP.values()) | {c + "y" for c in ("k", "g", "h", "b", "p", "m", "r", "d", "t", "n")}
                | VOWELS | LONG | {"N", "cl"})


class UnitError(ValueError):
    pass


def _single(c: str) -> str | None:
    if c in GEM:
        return GEM[c]
    if len(c) >= 2 and c[0] == c[1]:
        return c[1:]
    return None


def _cons(c: str) -> str:
    if c not in CMAP:
        raise UnitError(f"consonant {c!r}")
    return CMAP[c]


def ref_units(ref) -> list[dict]:
    """Words of a Reference as [{"text", "units": [unit, ...]}]. A unit is one front-end mora unit:
    {"kana", "kind": "V"|"N", "cons": [ipa...], "nuc": ipa, "long", "gem", "final_q", "devoiced",
     "devoice_env", "spelled": "ou"/"ei"/None, "ph": [aligner phones]}."""
    words = []
    for wi, w in enumerate(ref.words):
        units = []
        for si in range(*w.syllables):
            s = ref.syllables[si]
            phs = [ref.phones[p] for p in range(*s.phones)]
            nuc_i = next((k for k, p in enumerate(phs) if p.is_nucleus), None)
            if nuc_i is None:
                raise UnitError(f"no nucleus in {s.text}")
            pre, nuc, post = phs[:nuc_i], phs[nuc_i], phs[nuc_i + 1:]
            u = {"kana": s.text, "cons": [p.ipa for p in pre], "nuc": nuc.ipa, "gem": False, "final_q": False,
                 "devoiced": any(a == "" and abs(pr + 0.2) < 1e-6 for a, pr in nuc.alts),
                 "devoice_env": any(a == "" for a, _ in nuc.alts),
                 "spelled": next((a for a, _ in nuc.alts if a in ("ou", "ei")), None)}
            ph: list[str] = []
            cons = [p.ipa for p in pre]
            if cons:
                single = _single(cons[0])
                if single is not None:
                    u["gem"] = True
                    ph.append("cl")
                    cons[0] = single
                if len(cons) == 2 and cons[1] == "j":
                    ph.append(_cons(cons[0]) + "y" if cons[0] != "ɲ" else "ny")
                elif len(cons) == 1:
                    ph.append(_cons(cons[0]))
                else:
                    raise UnitError(f"consonant cluster {cons}")
            v = nuc.ipa
            if v in NASAL_NUC and not pre:
                u["kind"], u["long"] = "N", False
                ph.append("N")
            elif v.rstrip("ː") in VOWELS:
                u["kind"], u["long"] = "V", v.endswith("ː")
                ph.append(v[0] * 2 if u["long"] else v)
            else:
                raise UnitError(f"nucleus {v!r}")
            for p in post:
                if p.ipa == "ʔ":
                    u["final_q"] = True
                    ph.append("cl")
                else:
                    raise UnitError(f"post-nuclear {p.ipa!r}")
            u["ph"] = ph
            units.append(u)
        words.append({"text": w.text, "units": units})
    return words


def token(phones: list[str]) -> str:
    """Alphanumeric dictionary word for an aligner phone sequence (the aligner sees only these)."""
    return "w" + hashlib.sha1(" ".join(phones).encode()).hexdigest()[:14]


def variants(units: list[dict]) -> list[tuple[str, list[str], float]]:
    """(name, phones, probability) pronunciations of one word for the QC alignment: the canonical
    one, each long vowel shortened, each geminate closure dropped, each ou / ei spelling form."""
    canon = [p for u in units for p in u["ph"]]
    out = [("canon", canon, 1.0)]
    for k, u in enumerate(units):
        def rebuild(new_ph):
            return [p for j, uu in enumerate(units) for p in (new_ph if j == k else uu["ph"])]
        if u.get("long"):
            out.append((f"short@{k}", rebuild([p if p not in LONG else p[0] for p in u["ph"]]), 0.3))
            if u.get("spelled"):
                sp = u["spelled"]
                out.append((f"spelled@{k}", rebuild([p for p in u["ph"] if p not in LONG] + [sp[0], sp[1]]), 0.5))
        if u.get("gem"):
            out.append((f"nogem@{k}", rebuild(u["ph"][1:]), 0.3))
    seen, uniq = set(), []
    for v in out:
        key = tuple(v[1])
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    return uniq
