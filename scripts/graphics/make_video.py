"""Record the demo video (docs/assets/demo.mp4, under a minute, 1920x1080).

    pip install playwright && playwright install ffmpeg     # drives the installed Google Chrome
    python scripts/graphics/make_video.py

Six scenes in one visual style, each making one point: a learner recording being scored
(site/samples/en-learner-1); recordings in Spanish, Japanese, Korean and Mandarin; one bar
chart against Azure; a real shell session (record_terminal.py, replayed with asciinema-player);
one bar chart against the unmodified ZIPA model; the project's name and address. The sample
recordings are the only sound. A copy goes to site/assets/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

import video_scenes as scenes
from common import ROOT, render

OUT = ROOT / "docs/assets/demo.mp4"
SITE = ROOT / "site/assets"                 # gets a copy of the video
SIZE = {"width": 1920, "height": 1080}      # pages are laid out for 1280x720 and zoomed 1.5x
PLAYER = "https://cdn.jsdelivr.net/npm/asciinema-player@3.8.0/dist/bundle/asciinema-player"
FADE = 0.35

TERMINAL_CSS = """
#wrap{position:absolute;left:64px;right:64px;top:262px;padding:24px 28px;border-radius:18px;background:#000;box-shadow:0 0 0 1px rgba(255,255,255,.12)}
.ap-wrapper,.ap-player{background:transparent!important}
.asciinema-player-theme-pron{--term-color-foreground:#ececee;--term-color-background:#090b10;--term-color-0:#090b10;--term-color-1:#7db8ff;--term-color-2:#5eead4;--term-color-3:#ececee;--term-color-4:#7aa2f7;--term-color-5:#c4a7e7;--term-color-6:#5fd3c6;--term-color-7:#ececee;--term-color-8:#77777f;--term-color-9:#7db8ff;--term-color-10:#5eead4;--term-color-11:#ffffff;--term-color-12:#7aa2f7;--term-color-13:#c4a7e7;--term-color-14:#5fd3c6;--term-color-15:#ffffff}
"""
TERMINAL_JS = """window.start = () => { window.player = AsciinemaPlayer.create({data: %s}, document.getElementById('term'),
  {autoPlay: true, controls: false, fit: 'width', theme: 'pron', terminalFontFamily: "'Geist Mono', Menlo, monospace", terminalLineHeight: 1.33}); };"""


def record(play, tmp: Path, name: str, run):
    """Record one page; run(page, clock) returns {'start': s, 'length': s} within the recording."""
    browser = play.chromium.launch(channel="chrome", headless=True, args=["--mute-audio", "--autoplay-policy=no-user-gesture-required"])
    ctx = browser.new_context(viewport=SIZE, color_scheme="light", record_video_dir=str(tmp / name), record_video_size=SIZE)
    t0 = time.monotonic()
    page = ctx.new_page()
    marks = run(page, lambda: time.monotonic() - t0)
    video = Path(page.video.path())
    ctx.close()
    browser.close()
    return {"video": video, **marks}


def scene(tmp: Path, name: str, html: str, length: float, ready: str = "document.fonts.ready.then(() => true)"):
    file = tmp / f"{name}.html"
    file.write_text(html, encoding="utf-8")

    def run(page, clock):
        page.goto(file.as_uri())
        page.wait_for_function(ready)
        page.wait_for_timeout(500)
        start = clock()
        page.evaluate("window.start()")
        page.wait_for_timeout(int(length * 1000))
        return {"start": start, "length": length}

    return run


def terminal(tmp: Path, hold: float = 2.6):
    cast = tmp / "cli.cast"
    subprocess.run([sys.executable, str(Path(__file__).with_name("record_terminal.py")), str(cast)], check=True)
    text = cast.read_text(encoding="utf-8")
    length = json.loads(text.strip().split("\n")[-1])[0] + hold
    body = f'<link rel="stylesheet" href="{PLAYER}.css"><div id="wrap"><div id="term"></div></div><script src="{PLAYER}.min.js"></script>'
    html = scenes.page("On your device. About 50 ms per sentence.", body, TERMINAL_CSS, TERMINAL_JS % json.dumps(text))
    return scene(tmp, "terminal", html, length, "window.AsciinemaPlayer && document.fonts.ready.then(() => true)")


def still(tmp: Path, name: str, html: str, length: float):
    png = tmp / f"{name}.png"
    render(html.replace("html{zoom:1.5}", ""), png, 1280, 720, scale=2)
    return {"image": png, "length": length}


def assemble(parts: list[dict]) -> None:
    """Normalize every part, cross-fade the parts and mix the audio."""
    inputs, chains, sounds = [], [], []
    v = "scale=1920:1080:flags=lanczos,fps=30,format=yuv420p,setsar=1,settb=AVTB"
    at = 0.0
    count = lambda: sum(1 for x in inputs if x == "-i") - 1      # index of the input added last
    for i, p in enumerate(parts):
        if "image" in p:
            inputs += ["-loop", "1", "-t", f"{p['length']:.3f}", "-i", str(p["image"])]
            chains.append(f"[{count()}:v]{v},trim=duration={p['length']:.3f}[s{i}]")
        else:
            inputs += ["-i", str(p["video"])]
            chains.append(f"[{count()}:v]fps=30,trim=start={p['start']:.3f},setpts=PTS-STARTPTS,{v},"
                          f"tpad=stop_mode=clone:stop_duration=60,trim=duration={p['length']:.3f}[s{i}]")
        p["label"], p["at"] = f"s{i}", at
        for offset, path, length, begin in p.get("audio", []):
            sounds.append((at + offset, path, length, begin))
        at += p["length"] - FADE
    total = at + FADE
    prev = parts[0]["label"]
    for i, p in enumerate(parts[1:], 1):
        chains.append(f"[{prev}][{p['label']}]xfade=transition=fade:duration={FADE}:offset={p['at']:.3f}[x{i}]")
        prev = f"x{i}"
    mixes = []
    for j, (when, path, length, begin) in enumerate(sounds):
        inputs += ["-i", str(path)]
        chains.append(f"[{count()}:a]atrim=start={begin:.3f}:end={begin + length:.3f},asetpts=PTS-STARTPTS,"
                      f"afade=t=in:d=0.04,afade=t=out:st={length - 0.2:.3f}:d=0.2,adelay={int(when * 1000)}:all=1[a{j}]")
        mixes.append(f"[a{j}]")
    chains.append(f"{''.join(mixes)}amix=inputs={len(mixes)}:normalize=0:duration=longest,apad=whole_dur={total:.3f}[au]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(chains),
                    "-map", f"[{prev}]", "-map", "[au]", "-t", f"{total:.3f}",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", str(OUT)], check=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({total:.1f} s)")


def main() -> None:
    learner_html, learner_len, learner_audio = scenes.learner()
    lang_html, lang_len, lang_audio = scenes.languages()
    azure_html, azure_len = scenes.azure()
    base_html, base_len = scenes.baseline()
    with tempfile.TemporaryDirectory() as d, sync_playwright() as play:
        tmp = Path(d)
        names = ("learner", "languages", "azure", "terminal", "baseline", "end")
        parts = [
            {**record(play, tmp, "learner", scene(tmp, "learner", learner_html, learner_len)), "audio": learner_audio},
            {**record(play, tmp, "languages", scene(tmp, "languages", lang_html, lang_len)), "audio": lang_audio},
            record(play, tmp, "azure", scene(tmp, "azure", azure_html, azure_len)),
            record(play, tmp, "terminal", terminal(tmp)),
            record(play, tmp, "baseline", scene(tmp, "baseline", base_html, base_len)),
            still(tmp, "end", scenes.end(), 3.6),
        ]
        print("  " + ", ".join(f"{n} {p['length']:.1f} s" for n, p in zip(names, parts)))
        assemble(parts)
    shutil.copy2(OUT, SITE / "demo.mp4")


if __name__ == "__main__":
    main()
