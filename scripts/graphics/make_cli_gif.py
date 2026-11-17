"""Record the command-line tool scoring a recording as an animated GIF (docs/assets/cli.gif).

    pip install playwright && playwright install ffmpeg     # drives the installed Google Chrome
    python scripts/graphics/make_cli_gif.py

A real shell session is recorded by record_terminal.py, replayed in a page with asciinema-player,
captured with Playwright and converted with ffmpeg. Nothing in the session is edited.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from common import ROOT, font_css

OUT = ROOT / "docs/assets/cli.gif"
PLAYER = "https://cdn.jsdelivr.net/npm/asciinema-player@3.8.0/dist/bundle/asciinema-player"
SIZE = {"width": 1120, "height": 376}
PAGE = """<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="%(player)s.css"><style>%(fonts)s
html,body{margin:0;height:100%%;background:#0b0b0c}
#term{position:absolute;inset:26px 30px}
.ap-wrapper,.ap-player{background:transparent!important}
.asciinema-player-theme-pron{--term-color-foreground:#ececee;--term-color-background:#0b0b0c;--term-color-0:#0b0b0c;--term-color-1:#7db8ff;--term-color-2:#5eead4;--term-color-3:#ececee;--term-color-4:#7aa2f7;--term-color-5:#c4a7e7;--term-color-6:#5fd3c6;--term-color-7:#ececee;--term-color-8:#77777f;--term-color-9:#7db8ff;--term-color-10:#5eead4;--term-color-11:#ffffff;--term-color-12:#7aa2f7;--term-color-13:#c4a7e7;--term-color-14:#5fd3c6;--term-color-15:#ffffff}
</style></head><body><div id="term"></div><script src="%(player)s.min.js"></script><script>
window.start = () => AsciinemaPlayer.create({data: %(cast)s}, document.getElementById('term'),
  {autoPlay: true, controls: false, fit: 'width', theme: 'pron', terminalFontFamily: "'Geist Mono', Menlo, monospace", terminalLineHeight: 1.33});
</script></body></html>"""


def main(hold: float = 4.0) -> None:
    with tempfile.TemporaryDirectory() as d, sync_playwright() as play:
        tmp = Path(d)
        cast = tmp / "cli.cast"
        subprocess.run([sys.executable, str(Path(__file__).with_name("record_terminal.py")), str(cast)], check=True)
        text = cast.read_text(encoding="utf-8")
        length = json.loads(text.strip().split("\n")[-1])[0] + hold
        page_file = tmp / "cli.html"
        page_file.write_text(PAGE % {"player": PLAYER, "fonts": font_css(), "cast": json.dumps(text)}, encoding="utf-8")
        browser = play.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport=SIZE, device_scale_factor=1, record_video_dir=str(tmp), record_video_size=SIZE)
        t0 = time.monotonic()
        page = ctx.new_page()
        page.goto(page_file.as_uri())
        page.wait_for_function("window.AsciinemaPlayer && document.fonts.ready.then(() => true)")
        page.wait_for_timeout(400)
        start = time.monotonic() - t0
        page.evaluate("window.start()")
        page.wait_for_timeout(int(length * 1000))
        video = Path(page.video.path())
        ctx.close()
        browser.close()
        chain = (f"fps=30,trim=start={start:.3f},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=30,trim=duration={length:.3f},"
                 "fps=10,scale=1120:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];[b][p]paletteuse=dither=none")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-filter_complex", chain, "-loop", "0", str(OUT)], check=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({length:.1f} s, {OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
