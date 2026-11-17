"""Fill the generated parts of site/index.html: the key figures and the charts.

    python scripts/graphics/make_site.py

The page is static HTML. Everything between `<!-- gen:NAME -->` and `<!-- /gen:NAME -->` is
rewritten here from the tables in results/SCOREBOARD.md, so the figures on the page cannot
drift from the published results.
"""
from __future__ import annotations

import html
import json
import re

from common import ROOT
from make_charts import detection, english, table, tally, zipa

PAGE = ROOT / "site/index.html"
esc = html.escape
NUMBER = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def stats() -> str:
    """The four figures under the headline, read from the scoreboard."""
    rows = tally()
    total, better, same = (sum(r[k] for r in rows) for k in (1, 2, 3))
    ms = re.match(r"(\d+) ms", next(r for r in table("End-to-end `Scorer.score`") if r[0] == "fp32")[1]).group(1)
    cells = [
        (f"{better + same}<small> of {total}</small>", "comparisons level with or ahead of Azure"),
        (f"{ms}<small> ms</small>", "per sentence on a laptop CPU"),
        (f"{len(rows)}", "languages"),
    ]
    return '<div class="stats">' + "".join(f"<div><b>{n}</b><span>{label}</span></div>" for n, label in cells) + "</div>"


# ---- charts -------------------------------------------------------------------------------

def pairs(title: str, rows: list[dict], other: str, foot: str) -> str:
    """Two bars per metric: Pronunciada and one other system, on a 0 to 1 scale."""
    groups = []
    for r in rows:
        a, b = r["ours"][0], r["azure"][0]
        ns = "" if r["sig"] else '<sup>n.s.</sup>'
        groups.append(
            f'<div class="ch-g"><div class="ch-name">{r["label"]}<small>{r["what"]}</small></div><div class="ch-bars">'
            f'<div class="ch-row own"><span class="ch-track"><i style="width:{a * 100:.1f}%"></i></span><b>{a:.2f}{ns}</b></div>'
            f'<div class="ch-row"><span class="ch-track"><i style="width:{b * 100:.1f}%"></i></span><b>{b:.2f}</b></div>'
            f'</div></div>')
    if all(r["sig"] for r in rows):
        sig = f"All {NUMBER[len(rows)]} differences are statistically significant."
    else:
        sig = "Differences are statistically significant except where marked n.s."
    key = f'<span><i class="own"></i>Pronunciada</span><span><i></i>{other}</span>'
    return (f'<figure class="chart"><div class="ch-head"><figcaption>{title}</figcaption><div class="ch-key">{key}</div></div>'
            f'{"".join(groups)}<p class="ch-foot">{foot} {sig}</p></figure>')


def pick(rows: list[dict], label: str, name: str, what: str) -> dict:
    r = next(r for r in rows if r["label"] == label)
    return {**r, "label": name, "what": what}


def chart_azure() -> str:
    e, d = english(), detection()
    rows = [
        pick(e, "Phoneme", "English sounds", "agreement with expert scores"),
        pick(e, "Sentence accuracy", "English sentences", "agreement with expert scores"),
        pick(d, "English phonemes, L2-ARCTIC", "English mispronunciations", "finding the sounds experts marked"),
        pick(d, "Mandarin words, tone errors", "Mandarin tones", "finding the tones teachers marked"),
        pick(e, "Sentence prosody", "English intonation", "agreement with expert scores"),
    ]
    return pairs("Learner speech scored by experts", rows, "Azure",
                 "Higher is better. Agreement is the correlation with expert scores, and finding errors is the area under the ROC curve. "
                 "Test sets: speechocean762, L2-ARCTIC and OMPAL.")


def chart_zipa() -> str:
    z = zipa()
    rows = [
        pick(z, "English phonemes, expert scores (r)", "English sounds", "agreement with expert scores"),
        pick(z, "English sentences, expert scores (r)", "English sentences", "agreement with expert scores"),
        pick(z, "English errors, L2-ARCTIC (AUC)", "English mispronunciations", "finding the sounds experts marked"),
        pick(z, "Mandarin errors, tone (AUC)", "Mandarin tones", "finding the tones teachers marked"),
    ]
    return pairs("The same tests, against ZIPA alone", rows, "ZIPA alone",
                 "Higher is better. ZIPA alone is the unmodified model with the standard goodness-of-pronunciation score.")


def chart_tally() -> str:
    rows = tally()
    top = max(r[1] for r in rows)
    out = []
    for label, n, ahead, same, behind in rows:
        seg = "".join(f'<i class="{cls}" style="width:{count / top * 100:.2f}%">{count}</i>'
                      for count, cls in ((ahead, "own"), (same, "same"), (behind, "behind")) if count)
        out.append(f'<div class="tl-row"><span class="tl-name">{label}</span><span class="tl-bar">{seg}</span><span class="tl-n">of {n}</span></div>')
    total = [sum(r[k] for r in rows) for k in (1, 2, 3, 4)]
    key = ('<span><i class="own"></i>Pronunciada significantly better</span><span><i class="same"></i>No significant difference</span>'
           '<span><i class="behind"></i>Azure significantly better</span>')
    return (f'<figure class="chart"><div class="ch-head stack"><figcaption>Comparisons with Azure, by language</figcaption><div class="ch-key">{key}</div></div>'
            f'<div class="tl">{"".join(out)}</div><p class="ch-foot">{total[0]} comparisons on held-out public test sets. '
            'Significance is from paired tests at p &lt; 0.05.</p></figure>')


def speed() -> tuple[float, float, float]:
    """English row of the long-clip table: (Pronunciada seconds, Azure seconds, clip length in seconds)."""
    row = next(r for r in table("Longer clips next to Azure") if r[0] == "en")
    ours = re.match(r"(\d+) ms \((\d+\.\d+) s\)", row[1])
    azure = re.match(r"(\d+\.\d+) s", row[2])
    return int(ours.group(1)) / 1000, float(azure.group(1)), float(ours.group(2))


def chart_speed() -> str:
    ours, azure, clip = speed()
    top = max(ours, azure)
    row = lambda cls, who, sub, v: (f'<div class="sp-row {cls}"><span class="sp-who">{who}<small>{sub}</small></span>'
                                    f'<span class="ch-track"><i style="width:{v / top * 100:.1f}%"></i></span><b>{v:.2f} s</b></div>')
    return (f'<figure class="chart"><div class="ch-head"><figcaption>Time to score a {clip:.0f}-second recording</figcaption></div>'
            f'{row("own", "Pronunciada", "on the laptop, one CPU thread", ours)}{row("", "Azure", "request to the cloud and back", azure)}'
            '<p class="ch-foot">Lower is better. Median over English recordings. This is not a controlled comparison: the two sets of '
            'recordings differ, and the Azure time is the whole request from the same laptop, upload included.</p></figure>')


def figure_word() -> str:
    """One word of the learner recording, sound by sound: expected, recognized instead, probability."""
    word = next(w for w in json.loads((ROOT / "site/samples/en-learner-1.json").read_text(encoding="utf-8"))["result"]["words"] if w["prob"] < 0.5)
    phones = word["phones"]
    def heard(p):
        if p["heard"] is None:
            return '<span class="h">–</span>'
        return f'<span class="h sub">{esc(p["heard"]) or "nothing"}</span>'
    rows = ['<span class="l">Expected sound</span>' + "".join(f'<span class="s">{esc(p["ipa"])}</span>' for p in phones),
            '<span class="l">Recognized instead</span>' + "".join(heard(p) for p in phones),
            '<span class="l">Probability a rater accepts it</span>' + "".join(
                f'<span><span class="n{" low" if p["prob"] < 0.5 else ""}">{p["prob"]:.2f}</span></span>' for p in phones)]
    return (f'<figure class="chart wf"><div class="ch-head"><figcaption>How one word is scored</figcaption>'
            f'<div class="ch-key">“{esc(word["text"])}”, said without its “l”</div></div>'
            f'<div class="wf-grid">{"".join(rows)}</div>'
            f'<p class="ch-foot">The speech model heard no “l” and a different vowel, so those two sounds score low and the word as a whole '
            f'gets {word["prob"]:.2f}.</p></figure>')


def readme_images() -> None:
    """Dark renders of the worked example and the languages card for the README, using the page's own styles."""
    from common import INK, font_css, render
    page = PAGE.read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    dark = (":root{--text:#fff;--soft:rgba(255,255,255,.62);--faint:rgba(255,255,255,.5);--rule:rgba(255,255,255,.14);"
            f"--page:{INK};--card:{INK};--pale:#34343a;--accent:linear-gradient(90deg,#7db8ff,#5eead4)}}"
            f"body{{background:{INK}}}.chart{{box-shadow:none;margin:0;border-radius:0;padding:36px 40px 30px}}.langs{{margin:0;border-radius:0}}")
    head = f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{css}{dark}</style></head><body>"
    render(head + figure_word() + "</body></html>", ROOT / "docs/assets/figure-word.png", 920, 334)
    langs = re.search(r'<div class="langs">.*?</div>', page, re.S).group(0)
    render(head.replace("url(assets/gradient.jpg)", f"url('{(ROOT / 'site/assets/gradient.jpg').as_uri()}')") + langs + "</body></html>",
           ROOT / "docs/assets/languages.jpg", 920, 272)


BLOCKS = {"stats": stats, "figure-word": figure_word, "chart-tally": chart_tally, "chart-azure": chart_azure, "chart-speed": chart_speed, "chart-zipa": chart_zipa}


if __name__ == "__main__":
    page = PAGE.read_text(encoding="utf-8")
    for name, make in BLOCKS.items():
        pattern = re.compile(rf"(<!-- gen:{name} -->).*?(<!-- /gen:{name} -->)", re.S)
        assert pattern.search(page), f"marker gen:{name} is missing from site/index.html"
        block = make()
        page = pattern.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}", page)
    PAGE.write_text(page, encoding="utf-8")
    readme_images()
    print(f"filled {len(BLOCKS)} blocks in site/index.html; wrote docs/assets/figure-word.png and languages.jpg")
