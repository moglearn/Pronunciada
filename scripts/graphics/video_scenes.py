"""HTML scenes for the demo video. Each page is laid out for 1280x720 and exposes window.start()."""
from __future__ import annotations

import json

import numpy as np

from common import FLAG, INK, ROOT, font_css
from make_banner import BACKDROP, BAR, GAP, waveform
from make_logo import wordmark

BASE = """
*{box-sizing:border-box;margin:0}
html{zoom:1.5}
body{width:1280px;height:720px;overflow:hidden;background:#05070b;color:#fff;font-family:'Geist','Noto Sans CJK SC','PingFang SC','Hiragino Sans','Apple SD Gothic Neo',sans-serif;-webkit-font-smoothing:antialiased}
.bg{position:absolute;inset:0;background:#05070b;overflow:hidden}
.bg i{position:absolute;border-radius:50%;filter:blur(110px)}
.grain{position:absolute;inset:0;width:100%;height:100%;mix-blend-mode:overlay;opacity:.55}
.mark{position:absolute;left:64px;top:56px;height:26px}
.mark svg{height:100%;display:block}
h1{font-weight:300;letter-spacing:-.035em;line-height:1.04}
h1 span{display:block;color:rgba(255,255,255,.62)}
.fade{opacity:0;transition:opacity .5s ease}
.on{opacity:1}
"""


def page(body: str, css: str = "", script: str = "", backdrop: bool = True) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{BASE}{css}</style></head><body>"
            f"{BACKDROP if backdrop else ''}{body}<script>{script}</script></body></html>")


def title() -> str:
    return page(f"""
<div class="mark" style="height:30px">{wordmark('#ffffff')}</div>
<div style="position:absolute;left:64px;top:250px">
  <h1 style="font-size:64px">Pronunciation assessment<span>for ten languages, on CPU</span></h1>
</div>""", script="window.start=()=>{}")


def listen(lead: float = 0.6):
    """A learner clip plays: bars and words light up in time, then the scores and the tip appear.

    Returns (html, seconds the scene should run, seconds into the scene at which the audio starts).
    """
    words, mono, sr, t0, t1 = waveform()
    meta = json.loads((ROOT / "site/samples/en-learner-1.json").read_text(encoding="utf-8"))
    tip = (meta["result"].get("tips") or [{}])[0].get("text", "")
    width, height = 1152, 200
    n = width // (BAR + GAP)
    edges = np.linspace(t0, t1, n + 1)
    level = np.array([np.abs(mono[int(a * sr):max(int(b * sr), int(a * sr) + 1)]).max() for a, b in zip(edges, edges[1:])])
    level = (level / level.max()) ** 0.7
    px = lambda t: (t - t0) / (t1 - t0) * n * (BAR + GAP)
    bounds = [px(t0)] + [px((a["end"] + b["start"]) / 2) for a, b in zip(words, words[1:])] + [px(t1)]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 80}" viewBox="0 -44 {width} {height + 80}" style="overflow:visible">']
    for w, x0, x1 in zip(words, bounds, bounds[1:]):
        low = w["prob"] < 0.5
        cx = (x0 + x1) / 2
        if low:
            o.append(f'<rect class="fade late" x="{x0 + 2:.1f}" y="-44" width="{x1 - x0 - 4:.1f}" height="{height + 80}" rx="8" fill="rgba(4,6,10,.72)"/>')
        o.append(f'<text class="w" data-t="{w["start"]:.3f}" x="{cx:.1f}" y="-14" text-anchor="middle">{w["text"]}</text>')
        o.append(f'<text class="p fade late" x="{cx:.1f}" y="{height + 26}" text-anchor="middle" fill="{FLAG if low else "rgba(255,255,255,.62)"}">{w["prob"]:.2f}</text>')
    for i, v in enumerate(level):
        h = max(2.0, v * (height - 8))
        o.append(f'<rect class="b" data-t="{edges[i]:.3f}" x="{i * (BAR + GAP):.1f}" y="{height / 2 - h / 2:.1f}" width="{BAR}" height="{h:.1f}" rx="1.5" fill="#fff"/>')
    o.append("</svg>")
    end = t1 + 0.25
    css = f"""
.w{{font:500 24px 'Geist',sans-serif;fill:#fff;opacity:.35;transition:opacity .18s}}
.p{{font:400 16px 'Geist Mono',monospace}}
.b{{opacity:.28}}
.lit{{opacity:1!important}}
.tip{{position:absolute;left:64px;right:64px;top:530px;font-size:26px;font-weight:400;letter-spacing:-.01em;color:#fff}}
"""
    script = f"""
window.start=()=>{{const t0=performance.now()+{lead * 1000};const els=[...document.querySelectorAll('[data-t]')].map(e=>[+e.dataset.t,e]).sort((a,b)=>a[0]-b[0]);let k=0,done=false;
function tick(){{const t=(performance.now()-t0)/1000;while(k<els.length&&els[k][0]<=t){{els[k][1].classList.add('lit');k++;}}
if(!done&&t>{end:.3f}){{done=true;document.querySelectorAll('.late').forEach(e=>e.classList.add('on'));setTimeout(()=>document.querySelector('.tip').classList.add('on'),900);}}
requestAnimationFrame(tick);}}tick();}};"""
    body = f"""
<div class="mark">{wordmark('#ffffff')}</div>
<div style="position:absolute;left:64px;top:196px">{''.join(o)}</div>
<div class="tip fade">{tip}</div>"""
    return page(body, css, script), lead + end + 3.9, lead


def languages(items, each: float = 3.0):
    """items: [(language name, sentence, sentence probability)], shown one after another."""
    blocks = "".join(
        f'<div class="it fade" data-i="{i}"><div class="lang">{name}</div><div class="sent">{text}</div>'
        f'<div class="sc fade">Sentence {prob:.2f}</div></div>' for i, (name, text, prob) in enumerate(items))
    names = "English · Mandarin · Japanese · Spanish · French · German · Italian · Portuguese · Russian · Korean"
    css = """
.it{position:absolute;left:64px;right:64px;top:220px}
.lang{font:400 20px 'Geist Mono',monospace;color:rgba(255,255,255,.62);margin-bottom:18px}
.sent{font-size:44px;font-weight:300;letter-spacing:-.02em;line-height:1.2}
.sc{font:400 22px 'Geist Mono',monospace;margin-top:26px}
.all{position:absolute;left:64px;bottom:120px;font-size:19px;color:rgba(255,255,255,.62)}
"""
    script = f"""
window.start=()=>{{const its=[...document.querySelectorAll('.it')];its.forEach((e,i)=>{{
setTimeout(()=>e.classList.add('on'),i*{each * 1000});setTimeout(()=>e.querySelector('.sc').classList.add('on'),i*{each * 1000}+{(each - 1.0) * 1000});
if(i<its.length-1)setTimeout(()=>e.classList.remove('on'),(i+1)*{each * 1000}-350);}});}};"""
    return page(f'<div class="mark">{wordmark("#ffffff")}</div>{blocks}<div class="all">{names}</div>', css, script), each * len(items) + 0.6


def charts(files, each: float = 4.3):
    imgs = "".join(f'<img class="fade" src="{(ROOT / "docs/assets" / f).as_uri()}">' for f in files)
    css = f"body{{background:{INK}}}img{{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);max-width:1200px;max-height:690px;transition:opacity .45s ease}}"
    script = f"""
window.start=()=>{{const im=[...document.querySelectorAll('img')];im.forEach((e,i)=>{{setTimeout(()=>e.classList.add('on'),i*{each * 1000});
if(i<im.length-1)setTimeout(()=>e.classList.remove('on'),(i+1)*{each * 1000}-250);}});}};"""
    return page(imgs, css, script, backdrop=False), each * len(files)


def end() -> str:
    css = """
.c{position:absolute;left:0;right:0;top:214px;text-align:center}
.c .logo{height:44px;display:inline-block}.c .logo svg{height:100%}
.pill{display:inline-block;margin-top:44px;padding:16px 30px;border-radius:9999px;background:#fff;color:#000;font:500 26px 'Geist Mono',monospace}
.url{margin-top:30px;font-size:24px;color:rgba(255,255,255,.85)}
.sub{margin-top:14px;font-size:19px;color:rgba(255,255,255,.62)}
"""
    return page("""
<div class="c"><div class="logo">%s</div><br><div class="pill">pip install pronunciada</div>
<div class="url">github.com/moglearn/pronunciada</div><div class="sub">Open source, Apache-2.0. Runs on a laptop CPU.</div></div>""" % wordmark("#ffffff"),
                css, "window.start=()=>{}")


def caption(text: str, side: str = "left") -> str:
    """A caption strip rendered on a transparent 1280x720 page and overlaid on a scene."""
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}*{{margin:0}}html{{zoom:1.5}}body{{width:1280px;height:720px;background:transparent;font-family:'Geist',sans-serif}}"
            ".cap{position:absolute;%s:64px;bottom:44px;padding:11px 20px;border-radius:9999px;background:rgba(8,9,12,.82);color:#fff;font-size:20px;letter-spacing:-.005em;box-shadow:0 0 0 1px rgba(255,255,255,.14)}"
            f"</style></head><body><div class='cap'>{text}</div></body></html>").replace("%s:64px", f"{side}:64px")
