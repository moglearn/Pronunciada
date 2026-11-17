"""Fill the generated parts of site/index.html: the sample blocks and the charts.

    python scripts/graphics/make_site.py

The page is static HTML. Everything between `<!-- gen:NAME -->` and `<!-- /gen:NAME -->` is
rewritten here: samples from site/samples/*.json (the scorer's own output, with word times),
charts from the tables in results/SCOREBOARD.md. Nothing in those blocks is typed by hand, so
the page cannot drift from the published results.
"""
from __future__ import annotations

import html
import json
import re

import soundfile as sf

from common import ROOT
from make_charts import english, detection, languages, table, tally, zipa

PAGE = ROOT / "site/index.html"
SAMPLES = ROOT / "site/samples"
LANG = {"en": "English", "zh": "Mandarin", "ja": "Japanese", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ko": "Korean"}
WHO = {"native speaker": "native speaker", "adult learner of English": "adult learner",
       "French-speaking learner of Mandarin": "French-speaking learner"}
GROUPS = {
    "sample-intro": ["en-learner-1"],
    "samples-more": ["en-learner-2", "zh-learner-1", "es-native", "ja-native", "ko-native"],
}
# What the corpus annotation says about each learner recording (site/samples/*.json, meta.experts.marked), in words.
EXPERTS = {
    "en-learner-1": ("Expert annotators", "Marked the /l/ in “float” as dropped."),
    "en-learner-2": ("Expert annotators", "Marked the first vowel of “register” as wrong."),
    "zh-learner-1": ("Teachers", "Marked the tone of 口 as wrong."),
}
PLAY = '<svg viewBox="0 0 12 12" width="11" height="11" aria-hidden="true"><path class="i-play" d="M2.5 1.2v9.6L10.6 6z"/><path class="i-pause" d="M2.2 1.5h2.6v9H2.2zM7.2 1.5h2.6v9H7.2z"/></svg>'
esc = html.escape
NUMBER = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def units(result: dict) -> list[dict]:
    """The pieces of the sentence that light up: syllables where they are written characters, else words."""
    out = []
    for w in result["words"]:
        syl = [s for s in w.get("syllables") or [] if s.get("text")]
        if syl and "".join(s["text"] for s in syl) == w["text"]:
            for s in syl:
                tone_low = s.get("tone_prob") is not None and s["tone_prob"] < 0.5
                out.append({"text": s["text"], "start": s["start"], "end": s["end"], "low": tone_low or w["prob"] < 0.5})
        else:
            out.append({"text": w["text"], "start": w["start"], "end": w["end"], "low": w["prob"] < 0.5})
    return out


def marked_up(text: str, pieces: list[dict], tag: str = "span") -> str:
    """The reference text with each piece wrapped in a timed element; punctuation stays outside."""
    out, at = [], 0
    for p in pieces:
        i = text.find(p["text"], at)
        if i < 0:
            i = at
        out.append(esc(text[at:i]))
        cls = ' class="low"' if p["low"] else ""
        out.append(f'<{tag} data-a="{p["start"]:.2f}"{cls}>{esc(p["text"])}</{tag}>')
        at = i + len(p["text"])
    out.append(esc(text[at:]))
    return "".join(out)


def findings(result: dict) -> str:
    """What was flagged, then the sentence: the line shown next to "Pronunciada" for a learner recording."""
    parts = []
    for w in result["words"]:
        if w["prob"] < 0.5:
            parts.append(f'“{esc(w["text"])}” <b class="low">{w["prob"]:.2f}</b>')
        for s in w.get("syllables") or []:
            if s.get("tone_prob") is not None and s["tone_prob"] < 0.5:
                parts.append(f'tone of {esc(s["text"])} <b class="low">{s["tone_prob"]:.2f}</b>')
    parts.append(f'sentence <b>{result["prob"]:.2f}</b>')
    return '<span class="dot"></span>'.join(f"<span>{p}</span>" for p in parts)


def sample(key: str) -> str:
    d = json.loads((SAMPLES / f"{key}.json").read_text(encoding="utf-8"))
    meta, result = d["meta"], d["result"]
    audio, sr = sf.read(str(SAMPLES / f"{key}.mp3"))
    secs = round(len(audio) / sr)
    play = (f'<button class="s-play" type="button" aria-label="Play the recording: {esc(meta["text"])}">'
            f'{PLAY}<time>{secs // 60}:{secs % 60:02d}</time></button>')
    head = (f'<p class="s-label">{LANG[meta["lang"]]}<span>{WHO[meta["speaker"]]}</span></p>'
            f'<p class="s-text" lang="{meta["lang"]}">{marked_up(meta["text"], units(result))}</p>')
    if key not in EXPERTS:        # native speaker: the sentence probability is the whole result
        return (f'<article class="s" data-src="samples/{key}.mp3">{head}<div class="s-foot">{play}'
                f'<p class="s-score">Sentence <b>{result["prob"]:.2f}</b></p></div></article>')
    who, marked = EXPERTS[key]
    tip = next((t["text"] for t in result.get("tips") or [] if t.get("text")), "")
    rows = [("Pronunciada", f'<span class="s-score">{findings(result)}</span>'), (who, esc(marked))]
    if tip:
        rows.append(("Coaching tip", esc(tip)))
    table = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in rows)
    return (f'<article class="s" data-src="samples/{key}.mp3">{head}<div class="s-foot">{play}</div>'
            f'<dl class="s-rows">{table}</dl></article>')


def stats() -> str:
    """The four figures under the headline, read from the scoreboard."""
    rows = tally()
    total, better, same = (sum(r[k] for r in rows) for k in (1, 2, 3))
    ms = re.match(r"(\d+) ms", next(r for r in table("End-to-end `Scorer.score`") if r[0] == "fp32")[1]).group(1)
    ours = next(r for r in english() if r["label"] == "Phoneme")
    alone = zipa()[0]["azure"][0]
    cells = [
        (f"{better + same}<small> of {total}</small>", "benchmark comparisons in which it matches or beats Azure"),
        (f"{ms}<small> ms</small>", "to score a sentence on a laptop CPU, with no network"),
        (f"{len(rows)}", "languages, sharing one acoustic model"),
        (f'{ours["ours"][0]:.2f}', f'agreement with expert phoneme scores. Azure {ours["azure"][0]:.2f}, ZIPA alone {alone:.2f}'),
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
        sig = f"All {NUMBER[len(rows)]} differences are statistically significant (paired bootstrap over speakers, p &lt; 0.05)."
    else:
        sig = "Differences are statistically significant (paired bootstrap over speakers, p &lt; 0.05) except where marked n.s."
    key = f'<span><i class="own"></i>Pronunciada</span><span><i></i>{other}</span>'
    return (f'<figure class="chart"><div class="ch-head"><figcaption>{title}</figcaption><div class="ch-key">{key}</div></div>'
            f'{"".join(groups)}<p class="ch-foot">{foot} {sig}</p></figure>')


def pick(rows: list[dict], label: str, name: str, what: str) -> dict:
    r = next(r for r in rows if r["label"] == label)
    return {**r, "label": name, "what": what}


def chart_azure() -> str:
    e, d = english(), detection()
    rows = [
        pick(e, "Phoneme", "English phonemes", "agreement with expert scores"),
        pick(e, "Sentence accuracy", "English sentences", "agreement with expert accuracy scores"),
        pick(d, "English phonemes, L2-ARCTIC", "English mispronunciations", "detecting the phonemes annotators marked"),
        pick(d, "Mandarin words, tone errors", "Mandarin tones", "detecting the tone errors teachers marked"),
        pick(e, "Sentence prosody", "English intonation", "agreement with expert prosody scores"),
    ]
    return pairs("Learner speech scored by experts", rows, "Azure",
                 "Agreement is the Pearson correlation with the expert score; detection is the area under the ROC curve. "
                 "Higher is better. Test sets: speechocean762, L2-ARCTIC and OMPAL.")


def chart_zipa() -> str:
    z = zipa()
    rows = [
        pick(z, "English phonemes, expert scores (r)", "English phonemes", "agreement with expert scores"),
        pick(z, "English sentences, expert scores (r)", "English sentences", "agreement with expert accuracy scores"),
        pick(z, "English errors, L2-ARCTIC (AUC)", "English mispronunciations", "detecting the phonemes annotators marked"),
        pick(z, "Mandarin errors, tone (AUC)", "Mandarin tones", "detecting the tone errors teachers marked"),
    ]
    return pairs("The same tests, against the model Pronunciada is built on", rows, "ZIPA alone",
                 "ZIPA alone is the unmodified model scored with the standard goodness-of-pronunciation method.")


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
            f'<div class="tl">{"".join(out)}</div><p class="ch-foot">{total[0]} comparisons, each one metric on one held-out public test set: '
            f'{total[1]} significantly better, {total[2]} not significantly different, {total[3]} significantly worse '
            f'(paired tests, p &lt; 0.05).</p></figure>')


def table_languages() -> str:
    rows = []
    for r in languages():
        a, b = r["ours"][0], r["azure"][0]
        fa, fb = (f"<b>{a:.3f}</b>", f"{b:.3f}") if r["sig"] and a > b else (f"{a:.3f}", f"<b>{b:.3f}</b>") if r["sig"] else (f"{a:.3f}", f"{b:.3f}")
        rows.append(f'<tr><td>{r["label"]}</td><td class="n">{fa}</td><td class="n">{fb}</td></tr>')
    return ('<div class="tbl"><table><thead><tr><th>Language</th><th class="n">Pronunciada</th><th class="n">Azure</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


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
            f'{row("own", "Pronunciada", "on the laptop", ours)}{row("", "Azure", "request to the cloud and back", azure)}'
            '<p class="ch-foot">Median over English recordings of the same mean length; lower is better. This is not a controlled comparison: '
            'the two sets of recordings differ, Pronunciada ran on one CPU thread of an Apple M3 Pro, and the Azure figure is the full '
            'REST request to region westus from the same laptop, upload included.</p></figure>')


BLOCKS = {
    **{name: (lambda keys=keys: '<div class="samples">' + "".join(sample(k) for k in keys) + "</div>") for name, keys in GROUPS.items()},
    "stats": stats, "chart-tally": chart_tally, "table-languages": table_languages, "chart-azure": chart_azure,
    "chart-speed": chart_speed, "chart-zipa": chart_zipa,
}


if __name__ == "__main__":
    page = PAGE.read_text(encoding="utf-8")
    for name, make in BLOCKS.items():
        pattern = re.compile(rf"(<!-- gen:{name} -->).*?(<!-- /gen:{name} -->)", re.S)
        assert pattern.search(page), f"marker gen:{name} is missing from site/index.html"
        block = make()
        page = pattern.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}", page)
    PAGE.write_text(page, encoding="utf-8")
    print(f"filled {len(BLOCKS)} blocks in site/index.html")
