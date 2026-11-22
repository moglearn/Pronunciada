"""Round 19 (Korean): sound-identity evidence from a head on the acoustic model's internal layers.

Why: on Korean read speech the frame posteriors that the sound probability is built from barely
separate several phonemic contrasts. ZIPA writes ㅓ as ɔ, o or a, ㅗ as o or u, merges the three-way
stop series into plain stops, and an ㄴ / ㄹ / ㅁ / ㅇ swap inside a long eojeol moves the word
probability very little. So 저기 for 자기, 곳 for 것, 대문에 for 때문에 or 그들은 for 그들을 pass as correct.

What: one linear softmax (``data/coach/ko_identity.npz``) on the pooled enc1|enc3|enc5 vector of a
sound plus its neighbours' mean (the round-17 / 18 head input) predicts the Korean sound class that
was said: the 7 vowel classes (a ʌ o u ɯ i e; ㅐ / ㅔ merged; ㅢ is not judged), the glides j w,
18 onset consonants (lax / tense / aspirated distinct) and 7 codas. Fitted on FLEURS ko train natives
(labels = the front-end's own reference, i.e. the standard pronunciation, read by natives).

Evidence: per sound s = log P(expected class) - log P(best other class) ("how much more the expected
class than its strongest rival"); per word the lowest s. The word factor
(``data/calib/ko_identity_word.json``) is the round-18 form: 1 while the word's lowest s is above a
knee (a low native quantile, by strictness), then linearly down to ``floor`` over ``width``.
Sound probabilities and every other output are unchanged. Switch: PRONUNCIADA_KO_IDENTITY=0.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources

import numpy as np

from . import attr_heads as AH

VOWEL = {"ㅏ": "a", "ㅑ": "a", "ㅘ": "a", "ㅓ": "ʌ", "ㅕ": "ʌ", "ㅝ": "ʌ", "ㅗ": "o", "ㅛ": "o", "ㅜ": "u",
         "ㅠ": "u", "ㅡ": "ɯ", "ㅣ": "i", "ㅟ": "i", "ㅐ": "e", "ㅒ": "e", "ㅔ": "e", "ㅖ": "e", "ㅙ": "e",
         "ㅚ": "e", "ㅞ": "e"}
_PLAIN = {"b": "p", "d": "t", "g": "k", "dʒ": "tʃ", "ɾ": "r", "l": "r", "ʃ": "s"}
_CODA = {"k": "k", "n": "n", "t": "t", "l": "l", "m": "m", "p": "p", "ŋ": "ŋ"}


def _vowel_of(syl_text: str) -> str | None:
    from .text.ko import decompose

    d = decompose(syl_text[0]) if syl_text else None
    return VOWEL.get(d[1]) if d else None


def classes(ref) -> list[str | None]:
    """Korean sound class of every reference sound (None = not judged)."""
    out: list[str | None] = []
    for i, p in enumerate(ref.phones):
        syl = ref.syllables[p.syllable]
        a, b = syl.phones
        nuc = next((k for k in range(a, b) if ref.phones[k].is_nucleus), None)
        ipa = p.ipa
        if p.is_nucleus:
            out.append(_vowel_of(syl.text) if len(ipa) == 1 else None)
        elif ipa in ("j", "w"):
            out.append(ipa)
        elif nuc is not None and i > nuc:
            out.append(_CODA.get(ipa) and _CODA[ipa] + "_")
        else:
            asp = ipa.endswith("ʰ")
            base = ipa[:-1] if asp else ipa
            base = _PLAIN.get(base, base)
            if base not in ("k", "t", "p", "tʃ", "s", "h", "n", "m", "r"):
                out.append(None)
                continue
            out.append(base + ("ʰ" if asp else "*" if p.tense else ""))
    return out


@lru_cache(maxsize=1)
def _head():
    p = resources.files("pronunciada").joinpath("data/coach/ko_identity.npz")
    try:
        with p.open("rb") as f:
            z = dict(np.load(f, allow_pickle=False))
    except (FileNotFoundError, OSError):
        return None
    H = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in z.items()}
    H["index"] = {str(c): j for j, c in enumerate(H["classes"].tolist())}
    return H


@lru_cache(maxsize=1)
def _word_params() -> dict | None:
    p = resources.files("pronunciada").joinpath("data/calib/ko_identity_word.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None


def enabled() -> bool:
    return os.environ.get("PRONUNCIADA_KO_IDENTITY", "1") != "0" and _head() is not None \
        and _word_params() is not None


def _mlp(H, X: np.ndarray) -> np.ndarray:
    """Class probabilities from standardised inputs: linear softmax, or one GELU hidden layer."""
    X = (X - H["mu"]) / H["sd"]
    if "W1" in H:
        A = X @ H["W1"] + H["b1"]
        A = 0.5 * A * (1.0 + np.tanh(0.7978845608 * (A + 0.044715 * A ** 3)))
        Z = A @ H["W2"] + H["b2"]
    else:
        Z = X @ H["W"] + H["b"]
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def probs(V: np.ndarray, H=None) -> np.ndarray:
    H = H or _head()
    X = np.concatenate([V, AH.context_vectors(V)], axis=1)
    return _mlp(H, X)


# Minimal-pair rivals ("counterfactual" twins) of each class: the classes a one-feature change of the
# expected sound lands on (vowel neighbours, the stop / affricate / fricative series, nasal and liquid
# neighbours). Used only when the word parameters say "rivals": true.
_SERIES = (("k", "k*", "kʰ"), ("t", "t*", "tʰ"), ("p", "p*", "pʰ"), ("tʃ", "tʃ*", "tʃʰ"), ("s", "s*"))
RIVALS = {"a": ("ʌ", "e", "o"), "ʌ": ("a", "o", "u", "ɯ"), "o": ("ʌ", "u"), "u": ("o", "ɯ"), "ɯ": ("u", "i", "ʌ"),
          "i": ("e", "ɯ"), "e": ("i", "a"), "j": (), "w": (),
          "n": ("r", "m", "t"), "m": ("n", "p"), "r": ("n",), "h": ("k", "kʰ"),
          "n_": ("ŋ_", "m_", "l_"), "ŋ_": ("n_", "m_"), "m_": ("n_", "ŋ_", "p_"), "l_": ("n_",),
          "k_": ("t_", "p_", "ŋ_"), "t_": ("k_", "p_", "n_"), "p_": ("k_", "t_", "m_")}
for _ser in _SERIES:
    for _c in _ser:
        RIVALS[_c] = tuple(x for x in _ser if x != _c) + (("tʃ",) if _c == "s" else ("s",) if _c == "tʃ" else ())


def evidence(ref, V: np.ndarray, H=None, offsets: dict | None = None, rivals: bool | None = None) -> np.ndarray:
    """s per sound (NaN where not judged): log P(expected) - log max P(other), minus the class's
    native offset when the word parameters give one (offsets=None: the shipped ones)."""
    H = H or _head()
    if offsets is None:
        offsets = (_word_params() or {}).get("offset", {})
    if rivals is None:
        rivals = bool((_word_params() or {}).get("rivals", False))
    cls = classes(ref)
    s = np.full(len(cls), np.nan)
    idx = [i for i, c in enumerate(cls) if c is not None and c in H["index"]]
    if not idx:
        return s
    P = probs(np.asarray(V, np.float32), H)[idx]     # context = neighbours' mean over all sounds
    lp = np.log(np.clip(P, 1e-9, 1.0))
    for r, i in enumerate(idx):
        j = H["index"][cls[i]]
        if rivals:
            rv = [H["index"][c] for c in RIVALS.get(cls[i], ()) if c in H["index"]]
            if not rv:
                continue
            best = lp[r, rv].max()
        else:
            best = np.delete(lp[r], j).max()
        s[i] = lp[r, j] - best - float(offsets.get(cls[i], 0.0))
    return s


def knee_at(cal: dict, s: float) -> float:
    k = cal["knee"]
    s = min(max(float(s), 0.0), 1.0)
    return k[0] + (k[1] - k[0]) * 2 * s if s <= 0.5 else k[1] + (k[2] - k[1]) * (2 * s - 1)


def word_factor(cal: dict, s_word, strictness: float, sounds_prob: float | None = None) -> float | None:
    """Factor for one word from its sounds' evidence (NaN = not judged); None when none is judged.
    form "mult": 1 - (1 - floor) t; form "logit": the word's log-odds lowered by slope * width * t
    (returned as the equivalent factor on sounds_prob); t = clip((knee - min s) / width, 0, 1)."""
    if cal is None:
        return None
    v = np.asarray(s_word, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    t = min(max((knee_at(cal, strictness) - float(v.min())) / cal["width"], 0.0), 1.0)
    if cal.get("form", "mult") == "logit" and sounds_prob is not None:
        p = min(max(float(sounds_prob), 1e-6), 1 - 1e-6)
        z = np.log(p / (1 - p)) - cal["slope"] * cal["width"] * t
        return float(1.0 / (1.0 + np.exp(-z)) / p)
    return 1.0 - (1.0 - cal["floor"]) * t
