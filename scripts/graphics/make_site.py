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


CARD_CSS = """
*{box-sizing:border-box;margin:0}
body{margin:0;font-family:'Geist',sans-serif;color:#fff;-webkit-font-smoothing:antialiased}
.card{position:relative;width:920px;background:linear-gradient(rgba(4,6,10,.40),rgba(4,6,10,.40)),#06080c url(GRADIENT) center/cover}
.arch .head{position:absolute;left:40px;top:34px;display:flex;align-items:center;gap:12px;font-size:21px;letter-spacing:-.02em;color:rgba(255,255,255,.66)}
.arch .head svg{height:17.5px;display:block}
.nd{position:absolute;display:flex;align-items:center;justify-content:center;gap:9px;border-radius:14px;font-weight:500;font-size:14.5px;letter-spacing:-.01em;white-space:nowrap}
.nd.io{border:1px solid rgba(255,255,255,.34);color:rgba(255,255,255,.92);font-weight:400}
.nd.frozen{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.62);font-size:17px}
.nd.tr{background:linear-gradient(120deg,#2f6bff,#14b8a6);box-shadow:0 8px 24px rgba(4,6,10,.28)}
.nd.out{flex-direction:column;gap:0;padding:0;align-items:stretch;justify-content:stretch;border:1px solid rgba(255,255,255,.34);overflow:hidden;font-weight:400}
.nd.out em,.nd.out span{flex:1;display:flex;align-items:center;justify-content:center;font-style:normal;font-size:15px;color:rgba(255,255,255,.92);border-top:1px solid rgba(255,255,255,.2)}
.nd.out em{border-top:0;font:500 11px/1 'Geist Mono',monospace;letter-spacing:.1em;text-transform:uppercase;color:rgba(255,255,255,.6)}
.nd .ic{opacity:.9}
.region{position:absolute;border-radius:22px;background:rgba(4,6,10,.3);border:1px solid rgba(255,255,255,.16)}
.region b{position:absolute;left:20px;top:16px;font:500 11px/1 'Geist Mono',monospace;letter-spacing:.1em;text-transform:uppercase;color:rgba(255,255,255,.6)}
svg.wires{position:absolute;left:0;top:0}
svg.wires path{fill:none;stroke:rgba(255,255,255,.7);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}
.legend{position:absolute;left:40px;bottom:30px;display:flex;align-items:center;gap:10px;font-size:14px;color:rgba(255,255,255,.78)}
.legend .k{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:8px}
.legend .k.frozen{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.62)}
.legend .k.tr{background:linear-gradient(120deg,#2f6bff,#14b8a6);margin-left:18px}
.stats{display:grid;grid-template-columns:1.25fr .8fr 1fr 1.15fr;gap:0 36px;padding:40px 44px 38px}
.stats b{display:block;font-weight:300;font-size:58px;line-height:1;letter-spacing:-.04em;white-space:nowrap;font-variant-numeric:tabular-nums}
.stats b small{font-size:25px;letter-spacing:-.02em;color:rgba(255,255,255,.72);margin-left:5px}
.stats span{display:block;margin-top:14px;font-size:14.5px;line-height:20px;color:rgba(255,255,255,.78)}
"""
def figure_architecture() -> str:
    """The architecture as a graph: ZIPA frozen, Pronunciada's trained parts around it."""
    from make_logo import wordmark

    W, H, NW, NH = 920, 452, 120, 56
    col = [40 + i * 144 for i in range(6)]
    row = [118, 212, 306]
    snow = ('<svg class="ic" width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round">'
            '<path d="M7.5 1v13M1.9 4.25l11.2 6.5M13.1 4.25l-11.2 6.5"/></svg>')
    node = lambda c, r, name, cls, extra="": (f'<div class="nd {cls}" style="left:{col[c]}px;top:{row[r]}px;width:{NW}px;height:{NH}px">'
                                              f'{name}{extra}</div>')
    nodes = (node(0, 1, "Audio", "io") + node(0, 2, "Text", "io") + node(1, 1, "ZIPA", "frozen", snow)
             + node(2, 1, "Adapter", "tr") + node(3, 1, "Lattice scoring", "tr") + node(4, 1, "Calibration", "tr")
             + node(2, 2, "Lexicon + G2P", "tr") + node(3, 0, "Prosody heads", "tr"))
    out_h = 4 * 34
    out = (f'<div class="nd out" style="left:{col[5]}px;top:{row[1] + NH // 2 - out_h // 2}px;width:{NW}px;height:{out_h}px">'
           '<em>Scores</em><span>Phoneme</span><span>Word</span><span>Sentence</span></div>')
    mid = [r + NH // 2 for r in row]
    right = lambda c: col[c] + NW
    head = {"r": "l-6 -5m6 5l-6 5", "u": "l-5 6m5 -6l5 6", "d": "l-5 -6m5 6l5 -6"}
    wire = lambda d, way: f'<path d="{d}"/><path d="M{d.split()[-2]} {d.split()[-1]}{head[way]}"/>'
    xs, xc = col[3] + NW // 2, col[4] + NW // 2            # into Lattice scoring from below, into Calibration from above
    wires = "".join([
        *(wire(f"M{right(c)} {mid[1]} L {col[c + 1] - 3} {mid[1]}", "r") for c in range(5)),
        wire(f"M{right(0)} {mid[2]} L {col[2] - 3} {mid[2]}", "r"),
        wire(f"M{right(2)} {mid[2]} H{xs - 12} Q{xs} {mid[2]} {xs} {mid[2] - 12} L {xs} {row[1] + NH + 3}", "u"),
        wire(f"M{col[1] + NW // 2} {row[1]} V{mid[0] + 12} Q{col[1] + NW // 2} {mid[0]} {col[1] + NW // 2 + 12} {mid[0]} L {col[3] - 3} {mid[0]}", "r"),
        wire(f"M{right(3)} {mid[0]} H{xc - 12} Q{xc} {mid[0]} {xc} {mid[0] + 12} L {xc} {row[1] - 3}", "d"),
    ])
    region = (f'<div class="region" style="left:{col[2] - 18}px;top:{row[0] - 44}px;width:{right(4) - col[2] + 36}px;'
              f'height:{row[2] + NH - row[0] + 62}px"><b>Pronunciada</b></div>')
    legend = (f'<div class="legend"><span class="k frozen">{snow}</span>Frozen<span class="k tr"></span>Added by Pronunciada</div>')
    return (f'<div class="card arch" style="height:{H}px"><div class="head">{wordmark("#ffffff")}<span>Architecture</span></div>'
            f'{region}<svg class="wires" width="{W}" height="{H}">{wires}</svg>{nodes}{out}{legend}</div>')


def figure_stats() -> str:
    """The key figures as one strip."""
    rows = tally()
    ahead, behind = sum(r[2] for r in rows), sum(r[4] for r in rows)
    quiet = json.loads((ROOT / "results/tables/learner_benchmark_cost/quiet/pronunciada-int8__t4.json").read_text(encoding="utf-8"))
    size = round(next(w["mb"] for w in quiet["weight_files"] if w["file"].startswith("model")))
    cells = [(f"{ahead}–{behind}", "significant wins and losses against Azure"), (f"{len(rows)}", "languages"),
             (f"{size}<small>MB</small>", "small model"), (f"{round(quiet['p50_s'] * 1000)}<small>ms</small>", "per sentence on a laptop CPU")]
    return '<div class="card stats">' + "".join(f"<div><b>{n}</b><span>{label}</span></div>" for n, label in cells) + "</div>"


def readme_images() -> None:
    """Dark renders of the worked example and the languages card for the README, using the page's own styles."""
    from common import INK, font_css, render
    page = PAGE.read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    dark = (":root{--text:#fff;--soft:rgba(255,255,255,.62);--faint:rgba(255,255,255,.5);--rule:rgba(255,255,255,.14);"
            f"--page:{INK};--card:{INK};--pale:#34343a;--accent:linear-gradient(90deg,#7db8ff,#5eead4)}}"
            f"body{{background:{INK}}}.chart{{box-shadow:none;margin:0;border-radius:0;padding:36px 40px 30px}}.langs{{margin:0;border-radius:0}}")
    head = f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{css}{dark}</style></head><body>"
    gradient = (ROOT / "site/assets/gradient.jpg").as_uri()
    card = f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{CARD_CSS.replace('GRADIENT', gradient)}</style></head><body>"
    render(card + figure_architecture() + "</body></html>", ROOT / "docs/assets/how-it-works.jpg", 920, 452)
    render(card + figure_stats() + "</body></html>", ROOT / "docs/assets/stats.jpg", 920, 196)
    langs = re.search(r'<div class="langs">.*?</div>', page, re.S).group(0)
    from common import height_of
    card_html = head.replace("url(assets/gradient.jpg)", f"url('{(ROOT / 'site/assets/gradient.jpg').as_uri()}')") + langs + "</body></html>"
    render(card_html, ROOT / "docs/assets/languages.jpg", 920, height_of(card_html, 920))


BLOCKS = {"stats": stats, "figure-word": figure_word, "chart-tally": chart_tally, "chart-azure": chart_azure, "chart-speed": chart_speed, "chart-zipa": chart_zipa}


if __name__ == "__main__":
    page = PAGE.read_text(encoding="utf-8")
    for name, make in BLOCKS.items():
        pattern = re.compile(rf"(<!-- gen:{name} -->).*?(<!-- /gen:{name} -->)", re.S)
        assert pattern.search(page), f"marker gen:{name} is missing from site/index.html"
        block = make()
        page = pattern.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}", page)
    PAGE.write_text(page, encoding="utf-8")
    space = ROOT / "huggingface/space_static/index.html"          # the Hugging Face Space page repeats the key figures
    text = space.read_text(encoding="utf-8")
    space.write_text(re.sub(r"(<!-- gen:stats -->).*?(<!-- /gen:stats -->)", lambda m: f"{m.group(1)}\n{stats()}\n{m.group(2)}", text, flags=re.S),
                     encoding="utf-8")
    readme_images()
    print(f"filled {len(BLOCKS)} blocks in site/index.html and the figures in huggingface/space_static/index.html; wrote docs/assets/how-it-works.jpg, stats.jpg and languages.jpg")
