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
from make_charts import detection, english, languages, table, tally, zipa

PAGE = ROOT / "site/index.html"
esc = html.escape
NUMBER = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def stats() -> str:
    """The figures under the headline, read from the scoreboard and the timing run."""
    rows = tally()
    total, better, same = (sum(r[k] for r in rows) for k in (1, 2, 3))
    quiet = ROOT / "results/tables/learner_benchmark_cost/quiet/pronunciada-fp32__t4.json"     # the benchmark page's own timing
    ms = round(json.loads(quiet.read_text(encoding="utf-8"))["p50_s"] * 1000)
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
                 "Higher is better. Agreement is the Pearson correlation with expert scores, and finding errors is the area under the ROC curve. "
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
            f'<p class="ch-foot">Lower is better. Medians over English recordings of about {clip:.0f} seconds from the FLEURS test set. The two systems '
            'were timed on different recordings of similar length. The Azure time is the whole request from the same laptop, upload included.</p></figure>')


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


DB_CSS = """
.db{border-top:1px solid var(--rule);padding-top:14px}
.db-row{display:grid;grid-template-columns:112px 1fr 112px;gap:18px;align-items:center;height:38px}
.db-name{font-size:15.5px}
.db-track{position:relative;height:12px}
.db-track:before{content:"";position:absolute;left:0;right:0;top:5.5px;height:1px;background:var(--rule)}
.db-track span{position:absolute;top:4px;height:4px;border-radius:2px;background:var(--pale)}
.db-track i{position:absolute;top:0;width:12px;height:12px;margin-left:-6px;border-radius:50%;background:#b5afa9}
.db-track i.own{background:var(--accent)}
.db-val{font-family:var(--display);font-size:17px;line-height:1;letter-spacing:-.01em;text-align:right;color:var(--faint);font-variant-numeric:tabular-nums;white-space:nowrap}
.db-val b{font-weight:400;color:var(--text);margin-right:10px}
.db-val sup{font-family:var(--sans);font-size:10px;color:var(--soft);margin:0 6px 0 -7px}
.db-axis{height:22px;font-size:12px;color:var(--soft);font-variant-numeric:tabular-nums}
.db-axis .db-track:before{display:none}
.db-axis em{position:absolute;top:2px;font-style:normal;transform:translateX(-50%)}
.ch-key i.dot{background:#b5afa9}
"""


def chart_languages(lo: float = 0.86, hi: float = 1.0) -> str:
    """Ten languages, changed-word detection: one row per language with a dot for each system."""
    pos = lambda v: (v - lo) / (hi - lo) * 100
    rows = []
    for r in languages():
        a, b = r["ours"][0], r["azure"][0]
        ns = "" if r["sig"] else "<sup>n.s.</sup>"
        rows.append(f'<div class="db-row"><span class="db-name">{r["label"]}</span><span class="db-track">'
                    f'<span style="left:{pos(min(a, b)):.2f}%;width:{pos(max(a, b)) - pos(min(a, b)):.2f}%"></span>'
                    f'<i style="left:{pos(b):.2f}%"></i><i class="own" style="left:{pos(a):.2f}%"></i></span>'
                    f'<span class="db-val"><b>{a:.3f}</b>{ns}{b:.3f}</span></div>')
    ticks = "".join(f'<em style="left:{pos(v):.2f}%">{v:.2f}</em>' for v in (0.86, 0.90, 0.94, 0.98))
    axis = f'<div class="db-row db-axis"><span></span><span class="db-track">{ticks}</span><span></span></div>'
    key = '<span><i class="own"></i>Pronunciada</span><span><i class="dot"></i>Azure</span>'
    return ('<figure class="chart"><div class="ch-head"><figcaption>Ten languages: changed-word detection</figcaption>'
            f'<div class="ch-key">{key}</div></div><div class="db">{"".join(rows)}{axis}</div>'
            '<p class="ch-foot">Higher is better. Native FLEURS test recordings, 150 per language, scored against a text in which one word '
            'was replaced by a word one phoneme away. The figure is the area under the ROC curve for the replaced word. This is not a test '
            'on learner speech. Differences are statistically significant except where marked n.s.</p></figure>')


def readme_images() -> None:
    """The README's figures, rendered from the page's own styles so that the repository and the site look alike."""
    from common import INK, font_css, render, render_card
    page = PAGE.read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", page, re.S).group(1) + DB_CSS
    out = ROOT / "docs/assets"
    e, d, z = english(), detection(), zipa()
    agree, find = "agreement with expert scores", "finding the errors annotators marked"
    render_card(chart_tally(), css, out / "chart-tally.png")
    render_card(chart_languages(), css, out / "chart-languages.png")
    render_card(pairs("English learners: agreement with expert scores", [
        pick(e, "Phoneme", "Phonemes", ""), pick(e, "Word", "Words", ""),
        pick(e, "Sentence accuracy", "Sentence accuracy", ""), pick(e, "Sentence fluency", "Sentence fluency", ""),
        pick(e, "Sentence prosody", "Sentence prosody", ""), pick(e, "Sentence total", "Sentence total", "")], "Azure",
        "Higher is better. Pearson correlation with expert scores on the speechocean762 test set: 2,500 recordings, 125 speakers."),
        css, out / "chart-english.png")
    render_card(pairs("Detecting mispronunciations", [
        pick(d, "English phonemes, L2-ARCTIC", "English, L2-ARCTIC", "phonemes marked by annotators"),
        pick(d, "English phonemes, EpaDB", "English, EpaDB", "phonemes marked by annotators"),
        pick(d, "Mandarin words, any error", "Mandarin, any error", "characters marked by teachers"),
        pick(d, "Mandarin words, tone errors", "Mandarin, tone errors", "characters marked by teachers")], "Azure",
        "Higher is better. Area under the ROC curve for the phonemes or words that annotators marked wrong, on held-out test sets. "
        "L2-ARCTIC and EpaDB were not used in training."),
        css, out / "chart-detection.png")
    render_card(pairs("What the project adds to ZIPA", [
        pick(z, "English phonemes, expert scores (r)", "English phonemes", agree),
        pick(z, "English words, expert scores (r)", "English words", agree),
        pick(z, "English sentences, expert scores (r)", "English sentences", agree),
        pick(z, "English errors, L2-ARCTIC (AUC)", "English, L2-ARCTIC", find),
        pick(z, "English errors, EpaDB (AUC)", "English, EpaDB", find),
        pick(z, "Mandarin errors, any kind (AUC)", "Mandarin, any error", find),
        pick(z, "Mandarin errors, tone (AUC)", "Mandarin, tone errors", find)], "ZIPA alone",
        "Higher is better. The same held-out learner tests, scored by Pronunciada and by the unmodified ZIPA model with the standard "
        "goodness-of-pronunciation method. Agreement is the Pearson correlation, and finding errors is the area under the ROC curve."),
        css, out / "chart-zipa.png")
    render_card(chart_speed(), css, out / "chart-speed.png")
    render_card(figure_word(), css, out / "figure-word.png")
    # the languages card is already a dark gradient card on the site; render it as it is
    head = (f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{css}body{{background:{INK}}}"
            ".langs{margin:0;border-radius:0}</style></head><body>")
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
    space = ROOT / "space_static/index.html"          # the Hugging Face Space page repeats the key figures
    text = space.read_text(encoding="utf-8")
    space.write_text(re.sub(r"(<!-- gen:stats -->).*?(<!-- /gen:stats -->)", lambda m: f"{m.group(1)}\n{stats()}\n{m.group(2)}", text, flags=re.S),
                     encoding="utf-8")
    readme_images()
    print(f"filled {len(BLOCKS)} blocks in site/index.html and the figures in space_static/index.html; wrote the README figures in docs/assets")
