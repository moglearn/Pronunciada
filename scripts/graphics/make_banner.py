"""Render the README banner and the link-preview image.

    python scripts/graphics/make_banner.py

The diagram is real output: the waveform of site/samples/en-learner-1.mp3 cut at the word
boundaries in site/samples/en-learner-1.json, with each word's acceptance probability. Words
below 0.5 are boxed. Outputs docs/assets/banner.jpg (1200x400) and docs/assets/social.jpg
(1280x640) at 2x, and the plain gradient site/assets/gradient.jpg.
"""
from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from common import FLAG, ROOT, font_css, render
from make_logo import wordmark

SAMPLE = ROOT / "site/samples/en-learner-1"
BAR, GAP = 3, 3            # px
PAD_S = 0.10               # seconds of audio shown before the first and after the last word


def waveform():
    """Return (words, bars): words carry x-range in bar units; bars are 0..1 peak levels."""
    meta = json.loads(SAMPLE.with_suffix(".json").read_text(encoding="utf-8"))
    words = meta["result"]["words"]
    audio, sr = sf.read(str(SAMPLE.with_suffix(".mp3")), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    t0, t1 = words[0]["start"] - PAD_S, words[-1]["end"] + PAD_S
    return words, mono, sr, t0, t1


def diagram(width: int, height: int) -> str:
    words, mono, sr, t0, t1 = waveform()
    n = width // (BAR + GAP)
    edges = np.linspace(t0, t1, n + 1)
    level = np.array([np.abs(mono[int(a * sr):max(int(b * sr), int(a * sr) + 1)]).max() for a, b in zip(edges, edges[1:])])
    level = (level / level.max()) ** 0.7                        # compress the dynamic range a little
    mid = height / 2
    px = lambda t: (t - t0) / (t1 - t0) * n * (BAR + GAP)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 74}" '
           f'viewBox="0 -40 {width} {height + 74}" style="display:block;overflow:visible">']
    # boundaries halfway between neighboring words
    bounds = [px(t0)] + [px((a["end"] + b["start"]) / 2) for a, b in zip(words, words[1:])] + [px(t1)]
    for w, x0, x1 in zip(words, bounds, bounds[1:]):
        low = w["prob"] < 0.5
        if low:
            out.append(f'<rect x="{x0 + 2:.1f}" y="-40" width="{x1 - x0 - 4:.1f}" height="{height + 74}" rx="6" fill="rgba(4,6,10,.72)"/>')
        cx = (x0 + x1) / 2
        out.append(f'<text x="{cx:.1f}" y="-14" text-anchor="middle" class="w">{w["text"]}</text>')
        out.append(f'<text x="{cx:.1f}" y="{height + 22}" text-anchor="middle" class="p" '
                   f'fill="{FLAG if low else "rgba(255,255,255,.62)"}">{w["prob"]:.2f}</text>')
    for x in bounds[1:-1]:
        out.append(f'<line x1="{x:.1f}" y1="-2" x2="{x:.1f}" y2="{height + 2}" stroke="rgba(255,255,255,.22)" stroke-width="1"/>')
    for i, v in enumerate(level):
        h = max(2.0, v * (height - 8))
        out.append(f'<rect x="{i * (BAR + GAP):.1f}" y="{mid - h / 2:.1f}" width="{BAR}" height="{h:.1f}" rx="1.5" fill="#fff"/>')
    out.append("</svg>")
    return "".join(out)


BACKDROP = """
<div class="bg">
  <i style="left:-12%;top:-70%;width:75%;height:150%;background:#0c1f6b"></i>
  <i style="left:22%;top:8%;width:70%;height:120%;background:#1d4ed8;transform:rotate(-16deg) scaleY(.42)"></i>
  <i style="left:46%;top:-18%;width:62%;height:120%;background:#0ea5a0;transform:rotate(-16deg) scaleY(.34)"></i>
  <i style="left:58%;top:30%;width:58%;height:110%;background:#34d399;transform:rotate(-16deg) scaleY(.26);opacity:.85"></i>
  <i style="left:-20%;top:55%;width:70%;height:90%;background:#020617"></i>
  <svg class="grain" xmlns="http://www.w3.org/2000/svg"><filter id="n"><feTurbulence type="fractalNoise" baseFrequency=".9" numOctaves="2" stitchTiles="stitch"/><feColorMatrix values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .55 0"/></filter><rect width="100%" height="100%" filter="url(#n)"/></svg>
</div>
"""

CSS = """
*{box-sizing:border-box;margin:0}
body{background:#05070b;color:#fff;font-family:'Geist',sans-serif;-webkit-font-smoothing:antialiased;overflow:hidden}
.bg{position:absolute;inset:0;background:#05070b;overflow:hidden}
.bg i{position:absolute;border-radius:50%;filter:blur(110px)}
.grain{position:absolute;inset:0;width:100%;height:100%;mix-blend-mode:overlay;opacity:.55}
.frame{position:absolute;inset:0;display:flex}
.mark{position:absolute;left:56px;top:48px;height:24px}
.mark svg{height:100%;display:block}
h1{font-weight:300;letter-spacing:-.035em;line-height:1.02}
h1 span{display:block;color:rgba(255,255,255,.62)}
.w{font:500 19px 'Geist',sans-serif;fill:#fff;letter-spacing:-.01em}
.p{font:400 14px 'Geist Mono',monospace}
.note{font:400 13px 'Geist Mono',monospace;color:rgba(255,255,255,.5);letter-spacing:.01em}
"""


def page(body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{CSS}</style></head><body>{BACKDROP}{body}</body></html>"


def banner() -> str:
    """README banner, 1200x400: shown at about 880 px wide on GitHub."""
    return page(f"""
<div class="mark" style="left:44px;top:38px">{wordmark('#ffffff')}</div>
<div style="position:absolute;left:44px;bottom:42px;width:500px">
  <h1 style="font-size:34px">On-device pronunciation assessment<span>with cloud-level accuracy</span></h1>
</div>
<div style="position:absolute;right:44px;top:112px">{diagram(588, 150)}</div>
<div class="note" style="position:absolute;right:44px;bottom:26px;font-size:11.5px">learner recording, speechocean762 · probability each word is accepted</div>
""")


def social() -> str:
    """Link preview, 1280x640."""
    return page(f"""
<div class="mark" style="left:50%;top:60px;transform:translateX(-50%)">{wordmark('#ffffff')}</div>
<div style="position:absolute;left:0;right:0;top:132px;text-align:center">
  <h1 style="font-size:56px">On-device pronunciation assessment<span>with cloud-level accuracy</span></h1>
</div>
<div style="position:absolute;left:50%;transform:translateX(-50%);top:352px">{diagram(900, 170)}</div>
""")


def cover() -> str:
    """Poster of the video on the site, 808x455 (16:9): the scored waveform on the gradient."""
    return page(f"""
<div class="note" style="position:absolute;left:44px;top:40px;font-size:13px;color:rgba(255,255,255,.62)">a learner recording · the probability that each word is accepted</div>
<div style="position:absolute;left:44px;top:140px">{diagram(720, 150)}</div>
""")


def backdrop() -> str:
    """The gradient alone, used behind the install card on the site."""
    return page("")


if __name__ == "__main__":
    render(backdrop(), ROOT / "site/assets/gradient.jpg", 1536, 520, scale=1)

    render(banner(), ROOT / "docs/assets/banner.jpg", 1200, 400)
    render(social(), ROOT / "docs/assets/social.jpg", 1280, 640)
    render(cover(), ROOT / "site/assets/cover.jpg", 808, 455)
    print("wrote docs/assets/banner.jpg and docs/assets/social.jpg")
