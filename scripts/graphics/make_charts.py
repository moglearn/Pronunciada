"""Render the benchmark charts from results/SCOREBOARD.md.

    python scripts/graphics/make_charts.py

Outputs docs/assets/chart-english.png, chart-detection.png and chart-languages.png (920 px wide,
2x, so the lettering stays readable at README width). Every number is read from the scoreboard tables, which `python -m bench.scoreboard`
generates from results/tables/*.json, so the charts cannot drift from the published results.
"""
from __future__ import annotations

import re

from common import INK, ROOT, font_css, render
from make_logo import wordmark

BOARD = (ROOT / "results/SCOREBOARD.md").read_text(encoding="utf-8").split("\n")
W = 920
OURS, THEIRS, DIM, RULE = "#ffffff", "#6f6f76", "rgba(255,255,255,.55)", "rgba(255,255,255,.14)"
NUM = r"(-?\d+\.\d+)(?: \[(-?\d+\.\d+), (-?\d+\.\d+)\])?"


def table(heading: str) -> list[list[str]]:
    """Body rows of the first markdown table after the line that starts with `heading`."""
    i = next(k for k, line in enumerate(BOARD) if line.startswith(heading))
    while not BOARD[i].startswith("|"):
        i += 1
    rows = []
    for line in BOARD[i + 2:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def value(cell: str):
    m = re.match(NUM, cell)
    v, lo, hi = m.groups()
    return float(v), (float(lo) if lo else None), (float(hi) if hi else None)


def pair(label, ours_cell, azure_cell, verdict_cell):
    return {"label": label, "ours": value(ours_cell), "azure": value(azure_cell), "sig": "**" in verdict_cell}


def english():
    names = {"individual sounds": "Phoneme", "words": "Word", "sentence accuracy": "Sentence accuracy",
             "sentence fluency": "Sentence fluency", "sentence prosodic (intonation)": "Sentence prosody",
             "sentence total": "Sentence total"}
    rows = table("### Agreement with expert raters")
    return [pair(names[r[0]], r[1], r[2], r[5]) for r in rows if r[0] in names]


def detection():
    err = table("### Finding real learner mistakes")
    omp = {r[0]: r for r in table("### Real learners: OMPAL test")}
    return [
        pair("English phonemes, L2-ARCTIC", err[0][1], err[0][2], err[0][5]),
        pair("English phonemes, EpaDB", err[1][1], err[1][2], err[1][5]),
        pair("Mandarin words, any error", omp["word: any error"][2], omp["word: any error"][3], omp["word: any error"][6]),
        pair("Mandarin words, tone errors", omp["word: tone-only error"][2], omp["word: tone-only error"][3], omp["word: tone-only error"][6]),
    ]


def languages():
    names = {"en": "English", "zh": "Mandarin", "ja": "Japanese", "es": "Spanish", "fr": "French", "de": "German",
             "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ko": "Korean"}
    out = []
    for r in table("### Swap benchmark"):
        m = re.match(r"(\d\.\d+) / (\d\.\d+) (.*)", r[2])
        out.append({"label": names[r[0]], "ours": (float(m.group(1)), None, None),
                    "azure": (float(m.group(2)), None, None), "sig": "**" in m.group(3)})
    return out


CSS = f"""
*{{box-sizing:border-box;margin:0}}
body{{background:{INK};color:#fff;font-family:'Geist',sans-serif;-webkit-font-smoothing:antialiased}}
.head{{display:flex;align-items:center;justify-content:space-between;padding:36px 40px 0}}
.title{{display:flex;align-items:center;gap:12px;font-size:21px;font-weight:500;letter-spacing:-.02em}}
.title svg{{height:17.5px;display:block}}
.title span{{color:{DIM};font-weight:400}}
.legend{{display:flex;gap:22px;font-size:14px;color:{DIM}}}
.legend i{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:8px;vertical-align:-1px}}
svg.plot{{display:block;margin:26px 40px 0}}
.lab{{font:400 16px 'Geist',sans-serif;fill:#fff}}
.val{{font:400 13.5px 'Geist Mono',monospace}}
.tick{{font:400 12px 'Geist Mono',monospace;fill:{DIM}}}
.foot{{padding:16px 40px 34px;font:400 12px 'Geist Mono',monospace;color:{DIM};line-height:1.6}}
"""


def page(title: str, plot: str, foot: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{CSS}</style></head><body>"
            f"<div class='head'><div class='title'>{wordmark('#ffffff')}<span>{title}</span></div>"
            f"<div class='legend'><div><i style='background:{OURS}'></i>Pronunciada</div>"
            f"<div><i style='background:{THEIRS}'></i>Azure</div></div></div>"
            f"{plot}<div class='foot'>{foot}</div></body></html>")


def bars(rows, lo=0.0, hi=1.0, ticks=(0, .2, .4, .6, .8, 1.0), chance=None, label_w=236):
    pw = W - 80
    x0, x1 = label_w, pw - 70
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    pitch, bh = 64, 17
    h = len(rows) * pitch + 34
    o = [f'<svg class="plot" xmlns="http://www.w3.org/2000/svg" width="{pw}" height="{h}">']
    for t in ticks:
        o.append(f'<line x1="{sx(t):.1f}" y1="0" x2="{sx(t):.1f}" y2="{h - 30}" stroke="{RULE}"/>')
        o.append(f'<text class="tick" x="{sx(t):.1f}" y="{h - 10}" text-anchor="middle">{t:g}</text>')
    if chance is not None:
        o.append(f'<line x1="{sx(chance):.1f}" y1="0" x2="{sx(chance):.1f}" y2="{h - 30}" stroke="rgba(255,255,255,.5)" stroke-dasharray="3 5"/>')
        o.append(f'<text class="tick" x="{sx(chance):.1f}" y="{h - 10}" text-anchor="middle">0.5 (chance)</text>')
    for i, r in enumerate(rows):
        y = i * pitch + 8
        o.append(f'<text class="lab" x="0" y="{y + bh + 7}">{r["label"]}</text>')
        for k, (key, col) in enumerate((("ours", OURS), ("azure", THEIRS))):
            v, clo, chi = r[key]
            by = y + k * (bh + 5)
            o.append(f'<rect x="{x0}" y="{by}" width="{sx(v) - x0:.1f}" height="{bh}" rx="2" fill="{col}"/>')
            end = sx(v)
            if clo is not None:
                cc = "#8a8a90" if key == "ours" else "#c9c9cf"
                o.append(f'<line x1="{sx(clo):.1f}" y1="{by + bh / 2}" x2="{sx(chi):.1f}" y2="{by + bh / 2}" stroke="{cc}" stroke-width="1.5"/>')
                end = sx(chi)
            best = r["sig"] and v == max(r["ours"][0], r["azure"][0])
            fill = "#fff" if key == "ours" else "#b4b4bb"
            o.append(f'<text class="val" x="{end + 9:.1f}" y="{by + bh - 4}" fill="{fill}" font-weight="{700 if best else 400}">{v:.3f}{" *" if best else ""}</text>')
    o.append("</svg>")
    return "".join(o)


def dumbbell(rows, lo=0.86, hi=1.0, ticks=(.86, .88, .90, .92, .94, .96, .98, 1.0), label_w=150):
    pw = W - 80
    x0, x1 = label_w, pw - 64
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    pitch = 38
    h = len(rows) * pitch + 40
    o = [f'<svg class="plot" xmlns="http://www.w3.org/2000/svg" width="{pw}" height="{h}">']
    for t in ticks:
        o.append(f'<line x1="{sx(t):.1f}" y1="0" x2="{sx(t):.1f}" y2="{h - 30}" stroke="{RULE}"/>')
        o.append(f'<text class="tick" x="{sx(t):.1f}" y="{h - 10}" text-anchor="middle">{t:.2f}</text>')
    for i, r in enumerate(rows):
        y = i * pitch + 20
        a, b = r["ours"][0], r["azure"][0]
        o.append(f'<text class="lab" x="0" y="{y + 5}">{r["label"]}</text>')
        o.append(f'<line x1="{sx(min(a, b)):.1f}" y1="{y}" x2="{sx(max(a, b)):.1f}" y2="{y}" stroke="rgba(255,255,255,.35)" stroke-width="2"/>')
        o.append(f'<circle cx="{sx(b):.1f}" cy="{y}" r="7" fill="{THEIRS}"/>')
        o.append(f'<circle cx="{sx(a):.1f}" cy="{y}" r="7" fill="{OURS}"/>')
        left, right = (b, a) if a >= b else (a, b)
        lf, rf = ("#b4b4bb", "#fff") if a >= b else ("#fff", "#b4b4bb")
        star = " *" if r["sig"] else ""
        o.append(f'<text class="val" x="{sx(left) - 14:.1f}" y="{y + 5}" text-anchor="end" fill="{lf}">{left:.3f}{star if a < b else ""}</text>')
        o.append(f'<text class="val" x="{sx(right) + 14:.1f}" y="{y + 5}" fill="{rf}">{right:.3f}{star if a >= b else ""}</text>')
    o.append("</svg>")
    return "".join(o)


def height(n_rows: int, pitch: int, extra: int) -> int:
    return 36 + 26 + 26 + n_rows * pitch + extra


if __name__ == "__main__":
    out = ROOT / "docs/assets"
    star = "* significant, p &lt; 0.05 (paired bootstrap over speakers)."
    e = english()
    render(page("English learners: agreement with experts", bars(e),
                "Pearson correlation with expert scores, higher is better. speechocean762 test set: 2,500 recordings, "
                f"125 speakers. Lines show 95% confidence intervals. {star}"),
           out / "chart-english.png", W, height(len(e), 64, 34 + 112))
    d = detection()
    render(page("Detecting mispronunciations", bars(d, chance=0.5),
                "Area under the ROC curve for the phonemes or words that annotators marked wrong, higher is better. "
                f"Held-out test sets; L2-ARCTIC and EpaDB were not used in training. Lines show 95% confidence intervals. {star}"),
           out / "chart-detection.png", W, height(len(d), 64, 34 + 112))
    l = languages()
    render(page("Ten languages: changed-word detection", dumbbell(l),
                "Native FLEURS test recordings, 150 per language, scored against a reference with one word replaced by a "
                f"word one phoneme away. Area under the ROC curve for the replaced word, higher is better. Not a test on learner speech. {star}"),
           out / "chart-languages.png", W, height(len(l), 38, 40 + 112))
    print("wrote chart-english.png, chart-detection.png, chart-languages.png")
