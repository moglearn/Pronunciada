"""Record the demo video (docs/assets/demo.mp4, just under a minute, 1920x1080).

    pip install playwright && playwright install ffmpeg     # drives the installed Google Chrome
    GRADIO_SERVER_PORT=7861 python space/app.py             # the demo app, in another terminal
    python scripts/graphics/make_video.py [http://127.0.0.1:7861]

Scenes: title; a learner recording playing with its word scores (site/samples/en-learner-1);
a real shell session scoring it (record_terminal.py, replayed with asciinema-player); the demo
app scoring a Mandarin learner live (zh-learner-1); three native recordings; the benchmark
charts; an end card. The sample recordings are the only sound.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

import video_scenes as scenes
from common import INK, ROOT, font_css, render

OUT = ROOT / "docs/assets/demo.mp4"
SIZE = {"width": 1920, "height": 1080}      # pages are laid out for 1280x720 and zoomed 1.5x
PLAYER = "https://cdn.jsdelivr.net/npm/asciinema-player@3.8.0/dist/bundle/asciinema-player"
FADE = 0.35
SAMPLES = ROOT / "site/samples"
CHARTS = [f for f in ("chart-tally.png", "chart-english.png", "chart-zipa.png") if (ROOT / "docs/assets" / f).exists()]

TERMINAL = """<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="%(player)s.css"><style>
%(fonts)s
html{zoom:1.5}
html,body{margin:0;height:100%%;background:%(ink)s}
#wrap{position:absolute;left:56px;right:56px;top:64px;padding:26px 30px 30px;border-radius:18px;background:#111113;box-shadow:0 0 0 1px rgba(255,255,255,.09)}
.ap-wrapper,.ap-player{background:transparent!important}
.asciinema-player-theme-pron{--term-color-foreground:#ececee;--term-color-background:#111113;--term-color-0:#111113;--term-color-1:#ff6b5e;--term-color-2:#4ade80;--term-color-3:#f5b84a;--term-color-4:#7aa2f7;--term-color-5:#c4a7e7;--term-color-6:#5fd3c6;--term-color-7:#ececee;--term-color-8:#77777f;--term-color-9:#ff6b5e;--term-color-10:#4ade80;--term-color-11:#f5b84a;--term-color-12:#7aa2f7;--term-color-13:#c4a7e7;--term-color-14:#5fd3c6;--term-color-15:#ffffff}
</style></head><body><div id="wrap"><div id="term"></div></div><script src="%(player)s.min.js"></script><script>
window.start = () => { window.player = AsciinemaPlayer.create({data: %(cast)s}, document.getElementById('term'),
  {autoPlay: true, controls: false, fit: 'width', theme: 'pron', terminalFontFamily: "'Geist Mono', Menlo, monospace", terminalLineHeight: 1.45}); };
</script></body></html>"""


def record(play, tmp: Path, name: str, run):
    """Record one page; run(page, clock) returns {'start': s, 'length': s, ...} within the recording."""
    browser = play.chromium.launch(channel="chrome", headless=True, args=["--mute-audio", "--autoplay-policy=no-user-gesture-required"])
    ctx = browser.new_context(viewport=SIZE, color_scheme="light", record_video_dir=str(tmp / name), record_video_size=SIZE)
    t0 = time.monotonic()
    page = ctx.new_page()
    marks = run(page, lambda: time.monotonic() - t0)
    video = Path(page.video.path())
    ctx.close()
    browser.close()
    return {"video": video, **marks}


def scene(tmp: Path, name: str, html: str, length: float):
    file = tmp / f"{name}.html"
    file.write_text(html, encoding="utf-8")

    def run(page, clock):
        page.goto(file.as_uri())
        page.wait_for_function("document.fonts.ready.then(() => true)")
        page.wait_for_timeout(500)
        start = clock()
        page.evaluate("window.start()")
        page.wait_for_timeout(int(length * 1000))
        return {"start": start, "length": length}

    return run


def terminal(tmp: Path, hold: float = 3.0):
    cast = tmp / "cli.cast"
    subprocess.run([sys.executable, str(Path(__file__).with_name("record_terminal.py")), str(cast)], check=True)
    text = cast.read_text(encoding="utf-8")
    length = json.loads(text.strip().split("\n")[-1])[0] + hold
    html = TERMINAL % {"player": PLAYER, "fonts": font_css(), "ink": INK, "cast": json.dumps(text)}

    def run(page, clock):
        file = tmp / "terminal.html"
        file.write_text(html, encoding="utf-8")
        page.goto(file.as_uri())
        page.wait_for_function("window.AsciinemaPlayer && document.fonts.ready.then(() => true)")
        page.wait_for_timeout(400)
        start = clock()
        page.evaluate("window.start()")
        page.wait_for_timeout(int(length * 1000))
        return {"start": start, "length": length}

    return run


def demo(url: str):
    def run(page, clock):
        page.goto(url + "/?__theme=light")
        page.add_style_tag(content="html{zoom:1.5}")
        page.get_by_role("button", name="Score").wait_for()
        page.wait_for_timeout(1200)
        start = clock()
        page.wait_for_timeout(700)
        page.get_by_text("我家有四口人").first.click()
        page.wait_for_timeout(1500)
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(800)
        page.locator("button[aria-label='Play'], button.play-pause-button").first.click()
        play = clock()
        page.wait_for_timeout(3400)
        page.get_by_role("button", name="Score").click()
        page.locator(".pr-sent").wait_for(timeout=60000)
        page.wait_for_timeout(3600)
        return {"start": start, "audio": [(play - start, SAMPLES / "zh-learner-1.mp3", None)], "length": clock() - start}

    return run


def still(tmp: Path, name: str, html: str, length: float):
    png = tmp / f"{name}.png"
    render(html.replace("html{zoom:1.5}", ""), png, 1280, 720, scale=2)
    return {"image": png, "length": length}


def assemble(tmp: Path, parts: list[dict]) -> None:
    """Normalize every part, overlay its caption, cross-fade the parts and mix the audio."""
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
        label = f"s{i}"
        if p.get("caption"):
            cap = tmp / f"cap{i}.png"
            render(scenes.caption(p["caption"], p.get("side", "left")).replace("html{zoom:1.5}", ""), cap, 1280, 720, scale=2, transparent=True)
            inputs += ["-i", str(cap)]
            chains.append(f"[{count()}:v]scale=1920:1080[c{i}];[{label}][c{i}]overlay=0:0:enable='gte(t,0.5)'[o{i}]")
            label = f"o{i}"
        p["label"], p["at"] = label, at
        for offset, path, clip in p.get("audio", []):
            sounds.append((at + offset, path, clip))
        at += p["length"] - FADE
    total = at + FADE
    prev = parts[0]["label"]
    for i, p in enumerate(parts[1:], 1):
        chains.append(f"[{prev}][{p['label']}]xfade=transition=fade:duration={FADE}:offset={p['at']:.3f}[x{i}]")
        prev = f"x{i}"
    mixes = []
    for j, (when, path, clip) in enumerate(sounds):
        inputs += ["-i", str(path)]
        a = count()
        cut = f"atrim=duration={clip:.3f},afade=t=out:st={clip - 0.35:.3f}:d=0.35," if clip else ""
        chains.append(f"[{a}:a]{cut}adelay={int(when * 1000)}:all=1[a{j}]")
        mixes.append(f"[a{j}]")
    chains.append(f"{''.join(mixes)}amix=inputs={len(mixes)}:normalize=0,apad,atrim=duration={total:.3f}[au]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(chains),
                    "-map", f"[{prev}]", "-map", "[au]", "-t", f"{total:.3f}",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", str(OUT)], check=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({total:.1f} s)")


def main(url: str) -> None:
    natives = []
    for key, name in (("es-native", "Spanish"), ("ja-native", "Japanese"), ("ko-native", "Korean")):
        d = json.loads((SAMPLES / f"{key}.json").read_text(encoding="utf-8"))
        natives.append((name, d["meta"]["text"], d["result"]["prob"], SAMPLES / f"{key}.mp3"))
    each = 3.0
    listen_html, listen_len, lead = scenes.listen()
    lang_html, lang_len = scenes.languages([n[:3] for n in natives], each)
    chart_html, chart_len = scenes.charts(CHARTS)
    with tempfile.TemporaryDirectory() as d, sync_playwright() as play:
        tmp = Path(d)
        parts = [
            still(tmp, "title", scenes.title(), 3.0),
            {**record(play, tmp, "listen", scene(tmp, "listen", listen_html, listen_len)),
             "audio": [(lead, SAMPLES / "en-learner-1.mp3", None)],
             "caption": "A learner reads a sentence. Each word gets the probability that a rater would accept it."},
            {**record(play, tmp, "terminal", terminal(tmp)),
             "caption": "One command. Scoring takes about 50 ms on a laptop CPU, with no network."},
            {**record(play, tmp, "demo", demo(url)), "caption": "Mandarin: a tone score for every syllable.", "side": "right"},
            {**record(play, tmp, "languages", scene(tmp, "languages", lang_html, lang_len)),
             "audio": [(i * each + 0.3, n[3], each - 0.8) for i, n in enumerate(natives)],
             "caption": "Ten languages. Native speakers are not marked down."},
            record(play, tmp, "charts", scene(tmp, "charts", chart_html, chart_len)),
            still(tmp, "end", scenes.end(), 3.8),
        ]
        assemble(tmp, parts)


if __name__ == "__main__":
    main((sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7861").rstrip("/"))
