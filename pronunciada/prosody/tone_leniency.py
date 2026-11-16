"""Reference-side tone leniency: extra acceptable tones where native speakers legitimately depart
from the front-end's surface tone (on top of RefSyllable.tone_alts, which the text front-end
already sets for optional third-tone sandhi, 一/不 and Taiwan readings).

Rules (each keyed on the reference tone, so a changed / swapped tone never triggers them):
  neutral  a neutral-tone syllable (tone 5) may also be realised with its lexical full tone (if
           known) -- the common "full reading" of 们 / 子 / 得 ... in careful or Taiwan speech
  neutral4 a neutral-tone syllable may also be heard as a short fall (tone 4): after tones 1, 2
           and 4 the neutral tone is a mid-to-low fall, which the tone model cannot tell from a
           reduced tone 4 in fast speech
  yibu     一 / 不 carrying the sandhi tone 2 (before a tone 4) may also keep tone 4: in fast
           connected speech the sandhi rise is often flattened into a fall

Enabled set: PRONUNCIADA_TONE_LENIENCY (comma-separated rule names, "none" = no rule), else the
model's JSON "leniency" list, else DEFAULT. Each rule was checked on FLEURS validation natives
(bench.prosody.tone_native_cats, bench.calib.fit_prosody); see results/tone_v3.md.
"""
from __future__ import annotations

import os

RULES = ("neutral", "neutral4", "yibu")
DEFAULT: tuple[str, ...] = ()


def enabled(meta: dict | None = None) -> tuple[str, ...]:
    env = os.environ.get("PRONUNCIADA_TONE_LENIENCY")
    if env is not None:
        names = [x.strip() for x in env.split(",") if x.strip() and x.strip() != "none"]
    elif meta and "leniency" in meta:
        names = list(meta["leniency"])
    else:
        names = list(DEFAULT)
    bad = [n for n in names if n not in RULES]
    if bad:
        raise ValueError(f"unknown tone leniency rule(s) {bad}; known: {RULES}")
    return tuple(names)


def extra_tones(ref, i: int, rules: tuple[str, ...]) -> set[int]:
    """Extra acceptable tones (1-5) for syllable i of `ref` under `rules`."""
    if not rules:
        return set()
    s = ref.syllables[i]
    out: set[int] = set()
    if s.tone == 5:
        if "neutral" in rules and s.tone_lexical and s.tone_lexical != 5:
            out.add(int(s.tone_lexical))
        if "neutral4" in rules:
            prev = ref.syllables[i - 1].tone if i > 0 else None
            if prev in (1, 2, 4, 5):
                out.add(4)
    if "yibu" in rules and s.text in ("一", "不") and s.tone == 2:
        out.add(4)
    return out - {s.tone}
