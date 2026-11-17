"""Presentation of a SentenceResult: the machine-readable payload and the terminal report.

    from pronunciada.report import payload, render
    d = payload(result, coach=True)          # dict: the result fields (+ "coaching" when coach=True)
    print(render(result, color=False))       # compact human report (pronunciada check)

Nothing here changes a score: it only reads the result (and, for coaching, the evidence cached
on a live result). Every optional field may be missing or None; the renderer never raises on a
result rebuilt with SentenceResult.from_dict.
"""
from __future__ import annotations

import os
import shutil
import unicodedata

# word / sound colour bands (probability the word / sound is acceptable)
GOOD, OK = 0.8, 0.5
MAX_WORD_ROWS = 8

AUDIO_ISSUES = {
    "clipped": "the recording is clipped (too loud); move back from the microphone",
    "too_noisy": "there is a lot of background noise; try a quieter place",
    "too_quiet": "the recording is very quiet; move closer to the microphone",
    "too_short": "less speech than the text needs; was the whole sentence recorded?",
    "mostly_silence": "the recording is mostly silence",
}

_SGR = {"bold": "1", "dim": "2", "under": "4", "green": "32", "amber": "33", "red": "31",
        "magenta": "35", "cyan": "36"}


# ----------------------------------------------------------------------------- payload

def coaching_block(result, sensitivity: str = "balanced", max_tips: int = 3) -> dict | None:
    """{"schema_version", "items", "tips"} for a live result (None when the result carries no
    cached evidence, e.g. one rebuilt from JSON)."""
    if getattr(result, "_pronunciada", None) is None:
        return None
    from .coach import SCHEMA_VERSION
    from .tips import tips

    items = result.coaching(sensitivity=sensitivity)
    tp = [{k: v for k, v in t.items() if k != "item"} for t in tips(items, max_tips, result)]
    return {"schema_version": SCHEMA_VERSION, "items": items, "tips": tp}


def payload(result, coach: bool = False, sensitivity: str = "balanced", max_tips: int = 3,
            debug: bool = True, coaching: dict | None = None) -> dict:
    """The JSON document of `pronunciada score --json` / `check --json`: every SentenceResult
    field, plus "coaching" when coach=True. debug=False drops the raw evidence (PhoneResult
    "features", coaching "gate") for compact output."""
    d = result.to_dict()
    if coach:
        cb = coaching if coaching is not None else coaching_block(result, sensitivity, max_tips)
        if cb is not None:
            d["coaching"] = cb
    if not debug:
        for w in d.get("words") or []:
            for p in w.get("phones") or []:
                p.pop("features", None)
        for it in (d.get("coaching") or {}).get("items") or []:
            it.pop("gate", None)
    return d


# ----------------------------------------------------------------------------- terminal

def want_color(stream=None, no_color: bool = False) -> bool:
    """Colour only on a TTY, never with --no-color or a non-empty NO_COLOR (no-color.org);
    FORCE_COLOR (non-empty) turns it on for a non-TTY."""
    if no_color or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty and isatty())
    except (ValueError, OSError):
        return False


def cell_width(s: str) -> int:
    """Terminal columns of `s`: wide / full-width characters count 2, combining marks 0."""
    n = 0
    for ch in s:
        if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
            continue
        n += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return n


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - cell_width(s))


class _Style:
    def __init__(self, color: bool):
        self.color = color

    def __call__(self, text: str, *names: str) -> str:
        if not self.color or not names or not text:
            return text
        return "\x1b[" + ";".join(_SGR[n] for n in names) + "m" + text + "\x1b[0m"

    def band(self, p) -> str:
        return "dim" if p is None else "green" if p >= GOOD else "amber" if p >= OK else "red"


def _pct(p) -> str:
    return "–" if p is None else f"{round(100 * float(p)):d}%"


def _verdict(p: float) -> tuple[str, str]:
    if p >= GOOD:
        return "✓ Good", "green"
    if p >= OK:
        return "~ Close", "amber"
    return "✗ Try again", "red"


def _items_by_word(items) -> dict:
    out: dict = {}
    for it in items or []:
        if it.get("word_index") is not None:
            out.setdefault(it["word_index"], []).append(it)
    return out


def _flagged_sounds(witems) -> dict:
    """phone_index -> heard (None = flagged without a named replacement)."""
    out = {}
    for it in witems:
        if it.get("kind") in ("substitution", "missing", "length") and it.get("phone_index") is not None:
            out[it["phone_index"]] = it.get("heard")
    return out


def _extra_after(items) -> dict:
    """word index -> heard IPA of extra speech after it (-1 = before the first word)."""
    out = {}
    for it in items or []:
        if it.get("kind") == "extra_speech":
            k = it.get("after_word")
            if k is None:
                continue
            out[int(k)] = (it.get("heard") or it.get("heard_ipa") or "").strip()
    return out


def _sentence_rows(words, items, st: _Style, width: int) -> list[str]:
    """The prompt with every word coloured by its probability; skipped words underlined,
    extra speech shown as '+'. Without colour a marker line underneath carries the same
    information (^ low, ~ middling, _ skipped, + extra speech)."""
    extra = _extra_after(items)
    toks = []                                  # (text, style names, marker char)
    if -1 in extra:
        toks.append(("+", ("magenta", "bold"), "+"))
    for wi, w in enumerate(words):
        p = getattr(w, "prob", None)
        if getattr(w, "omitted", False):
            toks.append((w.text, ("red", "under"), "_"))
        else:
            band = st.band(p)
            toks.append((w.text, (band,) if band != "green" else ("green",),
                         {"red": "^", "amber": "~"}.get(band, " ")))
        if wi in extra:
            toks.append(("+", ("magenta", "bold"), "+"))
    rows, line, marks, used = [], [], [], 0
    indent = "  "
    for text, names, mk in toks:
        w = cell_width(text)
        if line and used + 1 + w > width - len(indent):
            rows.append((line, marks))
            line, marks, used = [], [], 0
        if line:
            used += 1
        line.append(st(text, *names))
        marks.append(mk * w if mk.strip() else " " * w)
        used += w
    if line:
        rows.append((line, marks))
    out = []
    for line, marks in rows:
        out.append(indent + " ".join(line))
        if not st.color:
            m = " ".join(marks).rstrip()
            if m:
                out.append(indent + m)
    return out


def _sound_cell(p, i: int, flagged: dict, st: _Style) -> str:
    prob = getattr(p, "prob", None)
    weak = i in flagged or (prob is not None and prob < OK)
    heard = flagged.get(i, getattr(p, "heard", None))
    s = p.ipa
    if weak and heard is not None:
        s += "→" + (heard or "∅")
    s += " " + _pct(prob)
    lp = getattr(p, "length_prob", None)
    if lp is not None and lp < OK:
        s += f" ({getattr(p, 'length_expected', None) or 'length'} {_pct(lp)})"
    if not weak:
        return st(s, "dim")
    return st(s, "red" if prob is None or prob < OK else "amber", "bold") if st.color else f"[{s}]"


def _prosody_notes(w) -> list[str]:
    notes = []
    for k, sy in enumerate(getattr(w, "syllables", None) or []):
        tp = getattr(sy, "tone_prob", None)
        if tp is not None and tp < OK and getattr(sy, "tone_expected", None):
            th = getattr(sy, "tone_heard", None)
            lab = (sy.text or f"syllable {k + 1}")
            notes.append(f"tone {lab}: {sy.tone_expected}" + (f"→{th}" if th and th != sy.tone_expected else "")
                         + f" {_pct(tp)}")
    sp = getattr(w, "stress_prob", None)
    if sp is not None and sp < OK:
        exp = [k for k, sy in enumerate(getattr(w, "syllables", None) or [])
               if getattr(sy, "stress_expected", None) == 1]
        where = f" on syllable {exp[0] + 1}" if exp and len(w.syllables) > 1 else ""
        notes.append(f"stress{where} {_pct(sp)}")
    tn = getattr(w, "tone_prob", None)
    if tn is not None and tn < OK and not any(n.startswith("tone") for n in notes):
        notes.append(f"tones {_pct(tn)}")
    lp = getattr(w, "length_prob", None)
    if lp is not None and lp < OK and not any("long" in n or "short" in n for n in notes):
        notes.append(f"sound length {_pct(lp)}")
    snd = getattr(w, "sounds_prob", None)
    phones = getattr(w, "phones", None) or []
    if snd is not None and snd < OK and not any((getattr(p, "prob", 1.0) or 0.0) < OK for p in phones):
        notes.append(f"the sounds together {_pct(snd)}")
    cp = getattr(w, "contrast_prob", None)
    if cp is not None and cp < OK:
        notes.append(f"sound contrast {_pct(cp)}")
    return notes


def render(result, coaching: dict | None = None, color: bool = False, width: int | None = None,
           show_items: bool = False) -> str:
    """Compact terminal report. `coaching`: a coaching_block() dict (tips and markers come from
    it; None = no tips). color: ANSI escapes. width: wrap width (default: terminal, max 100)."""
    st = _Style(color)
    if width is None:
        width = min(100, shutil.get_terminal_size((80, 24)).columns)
    width = max(40, int(width))
    items = (coaching or {}).get("items") or []
    tips = (coaching or {}).get("tips") or []
    words = list(getattr(result, "words", None) or [])
    L = []

    # 1. verdict line
    prob = float(getattr(result, "prob", 0.0) or 0.0)
    label, col = _verdict(prob)
    sc = getattr(result, "scores", None) or {}
    parts = [f"sentence {_pct(prob)}"]
    acc = sc.get("accuracy")
    flu = sc.get("fluency", getattr(result, "fluency", None))
    if acc is not None:
        parts.append(f"accuracy {10 * acc:.1f}/10")
    if flu is not None:
        parts.append(f"fluency {10 * flu:.1f}/10")
    comp = getattr(result, "completeness", None)
    if comp is not None and comp < 0.999:
        parts.append(f"completeness {_pct(comp)}")
    meta = f"{getattr(result, 'lang', '?')} · strictness {getattr(result, 'strictness', 0.5):g}"
    L.append(st(label, col, "bold") + "  " + " · ".join(parts) + "   " + st(meta, "dim"))
    L.append("")

    # 2. the sentence
    L += _sentence_rows(words, items, st, width)

    # 3. flagged words with their sounds
    by_word = _items_by_word(items)
    rows = []
    for wi, w in enumerate(words):
        p = getattr(w, "prob", None)
        witems = by_word.get(wi, [])
        if not (getattr(w, "omitted", False) or (p is not None and p < GOOD) or
                any(it.get("kind") != "mispronounced_word" for it in witems)):
            continue
        rows.append((wi, w, witems))
    if rows:
        L.append("")
        wcol = min(16, max(cell_width(w.text) for _, w, _ in rows[:MAX_WORD_ROWS]))
        for wi, w, witems in rows[:MAX_WORD_ROWS]:
            p = getattr(w, "prob", None)
            name = _truncate(w.text, wcol)
            head = st(_pad(name, wcol), st.band(p), "bold") + " " + st(f"{_pct(p):>4}", st.band(p))
            if getattr(w, "omitted", False):
                head += "  " + st("not heard", "red")
                L.append("  " + head)
                continue
            flagged = _flagged_sounds(witems)
            cells = [_sound_cell(ph, i, flagged, st) for i, ph in enumerate(getattr(w, "phones", None) or [])]
            line = "  " + head + "  "
            cur = cell_width(_strip(line))
            for c in cells:
                cw = cell_width(_strip(c))
                if cur + cw + 2 > width and cur > wcol + 9:
                    L.append(line.rstrip())
                    line = " " * (wcol + 9)
                    cur = wcol + 9
                line += c + "  "
                cur += cw + 2
            L.append(line.rstrip())
            notes = _prosody_notes(w)
            if notes:
                L.append(" " * (wcol + 9) + st("; ".join(notes), "amber"))
        if len(rows) > MAX_WORD_ROWS:
            L.append(st(f"  … and {len(rows) - MAX_WORD_ROWS} more", "dim"))

    # 4. tips
    if tips:
        L.append("")
        for t in tips[:3]:
            L.append("  " + st("→", "cyan", "bold") + " " + t.get("text", ""))
    if show_items and items:
        L.append("")
        L.append(st("  coaching items", "dim"))
        for it in items:
            h = it.get("heard")
            hs = "?" if h is None else (h or "∅")
            ep = it.get("error_prob")
            L.append(st(f"    {it.get('kind', ''):<18} {str(it.get('word') or ''):<12} "
                        f"{'' if it.get('expected') is None else it.get('expected')!s:>4} → {hs!s:<4}  "
                        f"err {'-' if ep is None else f'{ep:.2f}'}", "dim"))

    # 5. audio quality and warnings
    au = getattr(result, "audio", None) or {}
    issues = au.get("issues") or []
    warn = list(getattr(result, "warnings", None) or [])
    if issues or warn:
        L.append("")
    for k in issues:
        L.append("  " + st("⚠ audio:", "amber", "bold") + " " + AUDIO_ISSUES.get(k, k.replace("_", " ")))
    if au.get("retry_recommended"):
        L.append("    " + st("scores are less reliable; recording again is recommended", "dim"))
    for w in warn:
        L.append("  " + st("⚠", "amber") + " " + str(w))
    return "\n".join(L) + "\n"


def _truncate(s: str, width: int) -> str:
    if cell_width(s) <= width:
        return s
    out = ""
    for ch in s:
        if cell_width(out + ch) > width - 1:
            break
        out += ch
    return out + "…"


def _strip(s: str) -> str:
    """Remove ANSI SGR sequences."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", s)
