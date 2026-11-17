"""Record the demo video (docs/assets/demo.mp4, about 30 s, 1920x1080).

    pip install playwright && playwright install ffmpeg     # drives the installed Google Chrome
    python space/app.py               # the demo app, in another terminal (GRADIO_SERVER_PORT=7861)
    python scripts/graphics/make_video.py [http://127.0.0.1:7861]

Three parts, all real: a title frame showing the waveform of site/samples/en-learner-1.mp3 while
that recording plays; a shell session in which `pronunciada check` scores it (recorded by
record_terminal.py and replayed with asciinema-player); and the demo app scoring
site/samples/zh-learner-1.mp3 live. The two recordings are the only sound.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from common import INK, ROOT, font_css, render
from make_banner import video_title

OUT = ROOT / "docs/assets/demo.mp4"
SIZE = {"width": 1920, "height": 1080}      # pages are laid out for 1280x720 and zoomed 1.5x
PLAYER = "https://cdn.jsdelivr.net/npm/asciinema-player@3.8.0/dist/bundle/asciinema-player"
TITLE_S, HOLD_S = 4.6, 4.2

TERMINAL = """<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="%(player)s.css"><style>
%(fonts)s
html{zoom:1.5}
html,body{margin:0;height:100%%;background:%(ink)s}
#wrap{position:absolute;left:56px;right:56px;top:74px;padding:26px 30px 30px;border-radius:18px;background:#111113;box-shadow:0 0 0 1px rgba(255,255,255,.09)}
.ap-wrapper,.ap-player{background:transparent!important}
.asciinema-player-theme-pron{--term-color-foreground:#ececee;--term-color-background:#111113;--term-color-0:#111113;--term-color-1:#ff6b5e;--term-color-2:#4ade80;--term-color-3:#f5b84a;--term-color-4:#7aa2f7;--term-color-5:#c4a7e7;--term-color-6:#5fd3c6;--term-color-7:#ececee;--term-color-8:#77777f;--term-color-9:#ff6b5e;--term-color-10:#4ade80;--term-color-11:#f5b84a;--term-color-12:#7aa2f7;--term-color-13:#c4a7e7;--term-color-14:#5fd3c6;--term-color-15:#ffffff}
</style></head><body><div id="wrap"><div id="term"></div></div><script src="%(player)s.min.js"></script><script>
window.start = () => { window.player = AsciinemaPlayer.create({data: %(cast)s}, document.getElementById('term'),
  {autoPlay: true, controls: false, fit: 'width', theme: 'pron', terminalFontFamily: "'Geist Mono', Menlo, monospace", terminalLineHeight: 1.45}); };
</script></body></html>"""


def record(play, name: str, tmp: Path, run):
    """Open a recorded page, call run(page, clock) and return (video path, marks)."""
    browser = play.chromium.launch(channel="chrome", headless=True, args=["--mute-audio", "--autoplay-policy=no-user-gesture-required"])
    ctx = browser.new_context(viewport=SIZE, color_scheme="light", record_video_dir=str(tmp / name), record_video_size=SIZE)
    t0 = time.monotonic()
    page = ctx.new_page()
    marks = run(page, lambda: time.monotonic() - t0)
    video = page.video.path()
    ctx.close()
    browser.close()
    return Path(video), marks


def terminal(tmp: Path):
    cast = tmp / "cli.cast"
    subprocess.run([sys.executable, str(Path(__file__).with_name("record_terminal.py")), str(cast)], check=True)
    text = cast.read_text(encoding="utf-8")
    length = json.loads(text.strip().split("\n")[-1])[0]
    html = tmp / "terminal.html"
    html.write_text(TERMINAL % {"player": PLAYER, "fonts": font_css(), "ink": INK, "cast": json.dumps(text)}, encoding="utf-8")

    def run(page, clock):
        page.goto(html.as_uri())
        page.wait_for_function("window.AsciinemaPlayer && document.fonts.ready")
        page.wait_for_timeout(400)
        start = clock()
        page.evaluate("window.start()")
        page.wait_for_timeout(int((length + HOLD_S) * 1000))
        return {"start": start, "length": length + HOLD_S}

    return run


def demo(url: str):
    def run(page, clock):
        page.goto(url + "/?__theme=light")
        page.add_style_tag(content="html{zoom:1.5}")
        page.get_by_role("button", name="Score").wait_for()
        page.wait_for_timeout(1200)
        start = clock()
        page.wait_for_timeout(900)
        page.get_by_text("我家有四口人").first.click()
        page.wait_for_timeout(1800)
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(900)
        page.locator("button[aria-label='Play'], button.play-pause-button").first.click()
        play = clock()
        page.wait_for_timeout(3500)
        page.get_by_role("button", name="Score").click()
        page.locator(".pr-gloss").wait_for(timeout=60000)
        page.wait_for_timeout(6000)
        return {"start": start, "play": play - start, "length": clock() - start}

    return run


def assemble(tmp: Path, term_video: Path, term: dict, demo_video: Path, dm: dict) -> None:
    title = tmp / "title.png"
    render(video_title(), title, 1280, 720)
    en, zh = ROOT / "site/samples/en-learner-1.mp3", ROOT / "site/samples/zh-learner-1.mp3"
    fade = 0.35
    v = "scale=1920:1080:flags=lanczos,fps=30,format=yuv420p,setsar=1,settb=AVTB"
    zh_at = TITLE_S - fade + term["length"] - fade + dm["play"]
    total = TITLE_S + term["length"] + dm["length"] - 2 * fade
    graph = (
        f"[0:v]{v}[a];"
        # screen recordings only carry frames when the page changes: hold the last frame to full length
        f"[1:v]fps=30,trim=start={term['start']:.3f},setpts=PTS-STARTPTS,{v},tpad=stop_mode=clone:stop_duration=60,trim=duration={term['length']:.3f}[b];"
        f"[2:v]fps=30,trim=start={dm['start']:.3f},setpts=PTS-STARTPTS,{v},tpad=stop_mode=clone:stop_duration=60,trim=duration={dm['length']:.3f}[c];"
        f"[a][b]xfade=transition=fade:duration={fade}:offset={TITLE_S - fade:.3f}[ab];"
        f"[ab][c]xfade=transition=fade:duration={fade}:offset={TITLE_S - fade + term['length'] - fade:.3f}[v];"
        f"[3:a]adelay=500:all=1[e];[4:a]adelay={int(zh_at * 1000)}:all=1[z];"
        f"[e][z]amix=inputs=2:normalize=0,apad,atrim=duration={total:.3f}[au]"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", f"{TITLE_S}", "-i", str(title),
                    "-i", str(term_video), "-i", str(demo_video), "-i", str(en), "-i", str(zh),
                    "-filter_complex", graph, "-map", "[v]", "-map", "[au]", "-t", f"{total:.3f}",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", str(OUT)], check=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({total:.1f} s)")


if __name__ == "__main__":
    url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7861").rstrip("/")
    with tempfile.TemporaryDirectory() as d, sync_playwright() as play:
        tmp = Path(d)
        term_video, term = record(play, "terminal", tmp, terminal(tmp))
        demo_video, dm = record(play, "demo", tmp, demo(url))
        assemble(tmp, term_video, term, demo_video, dm)
