"""HTML scenes for the demo video. Each page is laid out for 1280x720 and exposes window.start().

Every scene has the same frame: the wordmark, one sentence that states the scene's point, and
the evidence for it underneath. Recordings play with their words lighting up at the times the
scorer reported; numbers are read from results/SCOREBOARD.md.
"""
from __future__ import annotations

import html
import json
import re

import numpy as np
import soundfile as sf

from common import FLAG, ROOT, font_css
from make_banner import BACKDROP, BAR, GAP
from make_charts import BOARD, english, table, zipa
from make_logo import wordmark
from make_site import LANG, WHO, units

SAMPLES = ROOT / "site/samples"
WIDTH = 1152                # px between the side margins
esc = html.escape

BASE = f"""
*{{box-sizing:border-box;margin:0}}
html{{zoom:1.5}}
body{{width:1280px;height:720px;overflow:hidden;background:#05070b;color:#fff;font-family:'Geist','Noto Sans CJK SC','PingFang SC','Hiragino Sans','Apple SD Gothic Neo',sans-serif;-webkit-font-smoothing:antialiased}}
.bg{{position:absolute;inset:0;background:#05070b;overflow:hidden}}
.bg i{{position:absolute;border-radius:50%;filter:blur(110px)}}
.grain{{position:absolute;inset:0;width:100%;height:100%;mix-blend-mode:overlay;opacity:.55}}
.shade{{position:absolute;inset:0;background:linear-gradient(90deg,rgba(4,6,10,.62),rgba(4,6,10,.4))}}
.mark{{position:absolute;left:64px;top:54px;height:24px}}
.mark svg{{height:100%;display:block}}
.claim{{position:absolute;left:64px;right:64px;top:118px;font-size:52px;font-weight:300;letter-spacing:-.035em;line-height:1.08}}
.dim{{color:rgba(255,255,255,.62)}}
.mono{{font-family:'Geist Mono',monospace}}
.low-on{{color:{FLAG}}}
"""


def page(claim: str, body: str, css: str = "", script: str = "window.start=()=>{}") -> str:
    head = f'<div class="mark">{wordmark("#ffffff")}</div><div class="claim">{claim}</div>' if claim else ""
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{BASE}{css}</style></head><body>"
            f"{BACKDROP}<div class=shade></div>{head}{body}<script>{script}</script></body></html>")


# ---- recordings ---------------------------------------------------------------------------

CLIP_CSS = f"""
.clip{{position:absolute;left:64px;right:64px;top:268px;opacity:0;transition:opacity .35s ease}}
.clip.on{{opacity:1}}
.lab{{font:400 18px 'Geist Mono',monospace;color:rgba(255,255,255,.62);margin-bottom:16px}}
.sent{{font-size:46px;font-weight:300;letter-spacing:-.025em;line-height:1.18;color:rgba(255,255,255,.36)}}
.sent:lang(zh),.sent:lang(ja),.sent:lang(ko){{letter-spacing:0;font-weight:400;font-size:42px}}
.sent span{{transition:color .16s linear}}
.sent .lit{{color:#fff}}
.clip.done .sent{{color:#fff}}
.clip.done .sent .low{{background:linear-gradient(90deg,#7db8ff,#5eead4);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent}}
.wave{{display:block;margin-top:26px}}
.wave rect{{opacity:.26;transition:opacity .1s linear}}
.wave .lit{{opacity:1}}
.res,.tip{{opacity:0;transition:opacity .45s ease}}
.clip.done .res{{opacity:1}}
.clip.done .tip{{opacity:1;transition-delay:.8s}}
.res{{margin-top:24px;font:400 22px 'Geist Mono',monospace}}
.res b{{font-weight:400;color:{FLAG}}}
.res small{{font:400 18px 'Geist',sans-serif;color:rgba(255,255,255,.62);margin-left:14px}}
.tip{{margin-top:14px;font-size:25px;letter-spacing:-.01em}}
.langs{{position:absolute;left:64px;right:64px;bottom:52px;display:flex;gap:22px;font-size:18px;color:rgba(255,255,255,.42)}}
.langs span{{transition:color .3s}}
.langs .cur{{color:#fff}}
"""

CLIP_JS = """
window.start=()=>{const T0=performance.now();
const clips=[...document.querySelectorAll('.clip')].map(e=>({e,d:e.dataset,k:0,st:0,
  els:[...e.querySelectorAll('[data-t]')].map(x=>[+x.dataset.t,x]).sort((a,b)=>a[0]-b[0])}));
function tick(){const t=(performance.now()-T0)/1000;
  for(const c of clips){const at=+c.d.at,lead=+c.d.lead,dur=+c.d.dur,out=+c.d.out;
    if(c.st==0&&t>=at){c.st=1;c.e.classList.add('on');document.querySelectorAll('.langs span').forEach(s=>s.classList.toggle('cur',s.dataset.l==c.d.lang));}
    if(c.st>=1){const ta=t-at-lead+(+c.d.a0);while(c.k<c.els.length&&c.els[c.k][0]<=ta){c.els[c.k][1].classList.add('lit');c.k++;}}
    if(c.st==1&&t>=at+lead+dur+0.15){c.st=2;c.e.classList.add('done');}
    if(c.st==2&&out&&t>=out){c.st=3;c.e.classList.remove('on');}}
  requestAnimationFrame(tick);}
tick();};
"""


def clip(key: str, at: float, hold: float, last: bool, lead: float = 0.45, result=None, tip: str = "") -> tuple[str, float, tuple]:
    """One recording: label, sentence, waveform, then its result. Returns (html, end time, audio cue)."""
    d = json.loads((SAMPLES / f"{key}.json").read_text(encoding="utf-8"))
    meta, res = d["meta"], d["result"]
    pieces = units(res)
    a0 = max(0.0, pieces[0]["start"] - 0.2)
    a1 = pieces[-1]["end"] + 0.2
    dur = a1 - a0
    audio, sr = sf.read(str(SAMPLES / f"{key}.mp3"), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    # sentence
    out, pos, text = [], 0, meta["text"]
    for p in pieces:
        i = text.find(p["text"], pos)
        i = pos if i < 0 else i
        out.append(esc(text[pos:i]))
        out.append(f'<span data-t="{p["start"]:.2f}"{" class=low" if p["low"] else ""}>{esc(p["text"])}</span>')
        pos = i + len(p["text"])
    out.append(esc(text[pos:]))
    # waveform
    n = WIDTH // (BAR + GAP)
    edges = np.linspace(a0, min(a1, len(mono) / sr), n + 1)
    level = np.array([np.abs(mono[int(a * sr):max(int(b * sr), int(a * sr) + 1)]).max() for a, b in zip(edges, edges[1:])])
    level = (level / level.max()) ** 0.7
    h = 56
    bars = "".join(f'<rect data-t="{edges[i]:.3f}" x="{i * (BAR + GAP)}" y="{h / 2 - max(2.0, v * h) / 2:.1f}" width="{BAR}" '
                   f'height="{max(2.0, v * h):.1f}" rx="1.5" fill="#fff"/>' for i, v in enumerate(level))
    result = result if result is not None else f'Sentence {res["prob"]:.2f}'
    end = at + lead + dur + hold
    body = (f'<div class="clip" data-at="{at:.2f}" data-lead="{lead}" data-a0="{a0:.2f}" data-dur="{dur:.2f}" '
            f'data-out="{0 if last else end - 0.4:.2f}" data-lang="{meta["lang"]}">'
            f'<div class="lab">{LANG[meta["lang"]]} · {WHO[meta["speaker"]]}</div>'
            f'<div class="sent" lang="{meta["lang"]}">{"".join(out)}</div>'
            f'<svg class="wave" xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{h}">{bars}</svg>'
            f'<div class="res">{result}</div><div class="tip">{esc(tip)}</div></div>')
    return body, end, (at + lead, SAMPLES / f"{key}.mp3", dur, a0)


def learner():
    """Scene 1. Returns (html, length, audio cues)."""
    d = json.loads((SAMPLES / "en-learner-1.json").read_text(encoding="utf-8"))["result"]
    low = next(w for w in d["words"] if w["prob"] < 0.5)
    result = (f'Sentence {d["prob"]:.2f} · {esc(low["text"])} <b>{low["prob"]:.2f}</b>'
              '<small>probability that a human rater would accept it</small>')
    body, end, cue = clip("en-learner-1", 0.6, 4.3, True, lead=0.8, result=result, tip=d["tips"][0]["text"])
    return page("Pronunciada scores pronunciation.", body, CLIP_CSS, CLIP_JS), end, [cue]


def languages():
    """Scene 2: three native recordings and a Mandarin learner, one after another."""
    zh = json.loads((SAMPLES / "zh-learner-1.json").read_text(encoding="utf-8"))["result"]
    tones = [s for w in zh["words"] for s in w["syllables"] if s["tone_prob"] is not None and s["tone_prob"] < 0.5]
    zh_result = " · ".join(f'tone of {s["text"]} <b>{s["tone_prob"]:.2f}</b>' for s in tones)
    zh_tip = "  ".join(f'{s["text"]}: tone {s["tone_expected"]} expected, tone {s["tone_heard"]} heard.' for s in tones)
    plan = [("es-native", 1.1, {}), ("ja-native", 1.1, {}), ("ko-native", 1.1, {}),
            ("zh-learner-1", 3.6, {"result": zh_result, "tip": zh_tip})]
    at, blocks, cues = 0.5, [], []
    for i, (key, hold, extra) in enumerate(plan):
        body, end, cue = clip(key, at, hold, i == len(plan) - 1, **extra)
        blocks.append(body)
        cues.append(cue)
        at = end
    names = "".join(f'<span data-l="{code}">{name}</span>' for code, name in LANG.items())
    return page("In ten languages.", "".join(blocks) + f'<div class="langs">{names}</div>', CLIP_CSS, CLIP_JS), at, cues


# ---- numbers ------------------------------------------------------------------------------

BARS_CSS = """
.what{position:absolute;left:64px;top:268px;font:400 18px 'Geist Mono',monospace;color:rgba(255,255,255,.62)}
.rows{position:absolute;left:64px;right:64px;top:318px}
.row{display:grid;grid-template-columns:230px 1fr 150px;align-items:center;height:84px}
.who{font-size:28px;letter-spacing:-.015em}
.track i{display:block;height:20px;border-radius:9999px;background:rgba(255,255,255,.3);width:0;transition:width 1.1s cubic-bezier(.2,.7,.2,1)}
.row.own .track i{background:#fff}
.num{font-size:58px;font-weight:300;letter-spacing:-.04em;text-align:right;opacity:0;transition:opacity .5s ease .7s;color:rgba(255,255,255,.62)}
.row.own .num{color:#fff}
.go .track i{width:var(--w)}
.go .num{opacity:1}
.line{position:absolute;left:64px;right:64px;top:540px;font-size:30px;font-weight:300;letter-spacing:-.02em;line-height:1.3;opacity:0;transition:opacity .6s ease}
.line.on{opacity:1}
"""


def bars(claim: str, what: str, rows: list[tuple[str, float]], line: str, length: float = 7.0):
    body = f'<div class="what">{what}</div><div class="rows">' + "".join(
        f'<div class="row{" own" if i == 0 else ""}"><div class="who">{name}</div>'
        f'<div class="track"><i style="--w:{v * 100:.1f}%"></i></div><div class="num">{v:.2f}</div></div>'
        for i, (name, v) in enumerate(rows)) + f'</div><div class="line">{line}</div>'
    script = ("window.start=()=>{setTimeout(()=>document.querySelector('.rows').classList.add('go'),350);"
              "setTimeout(()=>document.querySelector('.line').classList.add('on'),2600);};")
    return page(claim, body, BARS_CSS, script), length


def azure():
    """Scene 3. Phoneme-level agreement with expert scores, and the tally over all comparisons."""
    r = next(r for r in english() if r["label"] == "Phoneme")
    counts = [[int(x) for x in row[1:7]] for row in table("By language:")]
    total, ahead, behind = sum(c[0] for c in counts), sum(c[1] + c[2] for c in counts), sum(c[5] for c in counts)
    return bars("Ahead of Azure on most public tests.", "English learners: agreement with expert phoneme scores",
                [("Pronunciada", r["ours"][0]), ("Azure", r["azure"][0])],
                f'Ahead in {ahead} of {total} comparisons across ten languages.<span class="dim"> Significantly behind in {behind}.</span>')


def baseline():
    """Scene 5. The same test against the unmodified ZIPA model, and the tally over the baseline rows."""
    r = zipa()[0]
    m = next(re.search(r"over (\d+) rows: significantly ahead (\d+), not different (\d+), significantly behind (\d+)", line)
             for line in BOARD if line.startswith("Pronunciada vs gop"))
    n, ahead, _, behind = (int(x) for x in m.groups())
    worse = "Significantly worse on none." if behind == 0 else f"Significantly worse on {behind}."
    return bars("Built on the open ZIPA model, and far more accurate.", "English learners: agreement with expert phoneme scores",
                [("Pronunciada", r["ours"][0]), ("ZIPA alone", r["azure"][0])],
                f'Significantly better on {ahead} of {n} tests.<span class="dim"> {worse}</span>', 6.6)


def end() -> str:
    css = """
.c{position:absolute;left:0;right:0;top:268px;text-align:center}
.c .logo{height:60px;display:inline-block}.c .logo svg{height:100%}
.url{margin-top:48px;font-size:28px;letter-spacing:-.01em}
.sub{margin-top:14px;font:400 18px 'Geist Mono',monospace;color:rgba(255,255,255,.62)}
"""
    return page("", f'<div class="c"><div class="logo">{wordmark("#ffffff")}</div>'
                    '<div class="url">github.com/moglearn/pronunciada</div><div class="sub">Open source · Apache-2.0</div></div>', css)
