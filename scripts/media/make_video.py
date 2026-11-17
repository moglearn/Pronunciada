"""Build the ~45 s walkthrough video (docs/assets/walkthrough.mp4) from HTML slides and two sample clips.

    python scripts/media/make_video.py            # needs Google Chrome and ffmpeg

Numbers come from NUMBERS below; keep them equal to results/SCOREBOARD.md.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = ROOT / "docs" / "assets" / "walkthrough.mp4"
W, H = 1920, 1080

NUMBERS = [  # (label, ours, azure, ours significantly ahead?)
    ("English learners: sounds", "0.69", "0.54", True),
    ("English learners: sentence accuracy", "0.75", "0.67", True),
    ("English: flagged sounds experts also marked", "66%", "44%", True),
    ("Mandarin learners: wrong words", "0.79", "0.70", True),
    ("Mandarin learners: tones", "0.78", "0.68", True),
    ("English learners: intonation", "0.77", "0.83", False),
]

CSS = """
html,body{margin:0;width:1920px;height:1080px;background:#0f1115;color:#e9eaee;overflow:hidden;
 font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,"PingFang SC",sans-serif}
.p{padding:96px 120px;height:888px;display:flex;flex-direction:column;justify-content:center}
.b{position:absolute;left:120px;top:64px;font:700 30px ui-monospace,"SF Mono",Menlo,monospace;color:#9aa1ad}
.k{font:600 26px ui-monospace,"SF Mono",Menlo,monospace;color:#9aa1ad;letter-spacing:1px;text-transform:uppercase;margin:0 0 26px}
h1{font-size:104px;line-height:1.03;letter-spacing:-3px;margin:0;font-weight:750;max-width:1500px}
h2{font-size:72px;line-height:1.08;letter-spacing:-2px;margin:0 0 30px;font-weight:720;max-width:1560px}
p{font-size:40px;line-height:1.4;color:#9aa1ad;margin:26px 0 0;max-width:1500px}
.s{font-size:92px;letter-spacing:-1px;display:flex;gap:22px;flex-wrap:wrap;align-items:center;margin:8px 0 34px}
.g{color:#4fc57e}.a{color:#e2b04f;background:#352a12;border-radius:16px;padding:0 20px}.r{color:#f0776b;background:#3a1b18;border-radius:16px;padding:0 20px}
.n{color:#e9eaee}
.row{display:flex;gap:16px;margin:0 0 30px}
.ph{font:500 44px ui-monospace,"SF Mono",Menlo,monospace;border:2px solid #272b35;border-radius:16px;padding:14px 24px;text-align:center;min-width:92px;background:#161920}
.ph small{display:block;font-size:22px;color:#9aa1ad;margin-top:6px}
.ph.r{background:#3a1b18;border-color:#3a1b18}.ph.a{background:#352a12;border-color:#352a12}
ul{margin:0;padding:0;list-style:none} li{font-size:38px;line-height:1.45;margin:10px 0;color:#e9eaee}
li:before{content:"→";color:#7ea2ff;margin-right:18px;font-weight:700}
table{border-collapse:collapse;width:1560px;font-size:38px}
td,th{padding:16px 20px;border-bottom:2px solid #272b35;text-align:left}
th{font:600 24px ui-monospace,"SF Mono",Menlo,monospace;color:#9aa1ad;text-transform:uppercase;letter-spacing:1px}
td.v{font-family:ui-monospace,"SF Mono",Menlo,monospace;width:210px}
.win{color:#4fc57e;font-weight:700}.lose{color:#f0776b}
.chips{display:flex;flex-wrap:wrap;gap:16px;margin-top:34px}
.chips span{font:500 32px ui-monospace,"SF Mono",Menlo,monospace;border:2px solid #272b35;border-radius:14px;padding:10px 20px;background:#161920}
pre{font:500 36px/1.6 ui-monospace,"SF Mono",Menlo,monospace;background:#161920;border:2px solid #272b35;border-radius:22px;padding:34px 40px;margin:0;max-width:1500px}
.c{color:#9aa1ad}.q{color:#4fc57e}
.src{position:absolute;left:120px;bottom:60px;font:500 24px ui-monospace,"SF Mono",Menlo,monospace;color:#5d6470}
.cur{display:inline-block;width:.5em;height:.95em;background:#e9eaee;vertical-align:-.1em;margin-left:6px}
"""


def page(body: str, src: str = "") -> str:
    return (f'<!doctype html><meta charset="utf-8"><style>{CSS}</style><div class="b">pronunciada</div>'
            f'<div class="p">{body}</div>' + (f'<div class="src">{src}</div>' if src else ""))


def slides() -> list[tuple[str, float]]:
    rows = "".join(
        f'<tr><td>{lab}</td><td class="v {"win" if sig else ""}">{ours}</td><td class="v {"" if sig else "win"}">{az}</td></tr>'
        for lab, ours, az, sig in NUMBERS)
    sent = ('<div class="s"><span class="g">the</span><span class="r">ball</span><span class="g">is</span><span class="g">in</span>'
            '<span class="g">their</span><span class="r">court</span><span class="r">now</span></div>')
    return [
        (page('<h1>Pronunciation scoring that runs on the device.<span class="cur"></span></h1>'
              '<p>Open source. Ten languages. Every sound, word and sentence.</p>'), 3.8),
        (page('<div class="k">A learner reads a sentence</div><div class="s"><span class="n">“The ball is in their court now”</span></div>'
              '<p>A real recording of an adult learner of English.</p>', "recording: speechocean762, CC BY 4.0"), 5.6),
        (page('<div class="k">What Pronunciada reports, in about 50 ms</div>' + sent +
              '<div class="row"><div class="ph">k<small>0.90</small></div><div class="ph a">ɔ<small>0.60</small></div>'
              '<div class="ph r">ɹ<small>heard ŋ</small></div><div class="ph r">t<small>0.26</small></div></div>'
              '<ul><li>Keep the air out of your nose for /ɹ/ in “court”; it sounded like /ŋ/.</li>'
              '<li>Listen to the vowel in “ball” and try it again.</li></ul>'), 7.2),
        (page('<div class="k">What expert raters marked on the same recording</div>'
              '<h2>Sentence accuracy 4 out of 10.<br>Words marked wrong: “ball” and “court”.</h2>'
              '<p>The red words are the ones the experts marked.</p>', "expert scores: speechocean762"), 5.2),
        (page('<div class="k">Mandarin tones</div><div class="s"><span class="g">我</span><span class="g">家</span><span class="a">有</span>'
              '<span class="r">四</span><span class="r">口</span><span class="g">人</span></div>'
              '<div class="row"><div class="ph r">四<small>tone 4, heard 1</small></div><div class="ph r">口<small>tone 3, heard 2</small></div></div>'
              '<p>The teachers marked the tone of 口 as wrong. Azure reports no tones at all.</p>', "recording and marks: OMPAL, CC BY 4.0"), 7.2),
        (page('<div class="k">Same held-out recordings, both systems</div>'
              f'<table><tr><th></th><th>Pronunciada</th><th>Azure</th></tr>{rows}</table>'
              '<p style="font-size:30px">Ahead in 60 of 84 comparisons, behind in 24. Every number is reproducible from the repository.</p>'), 8.5),
        (page('<div class="k">One call returns</div><h2>Probabilities for sounds, words and sentences, plus tones, stress, length and coaching tips.</h2>'
              '<div class="chips"><span>English</span><span>中文</span><span>日本語</span><span>Español</span><span>Français</span><span>Deutsch</span>'
              '<span>Italiano</span><span>Português</span><span>Русский</span><span>한국어</span></div>'), 5.2),
        (page('<pre><span class="c">$</span> pip install pronunciada\n<span class="c">$</span> pronunciada check clip.wav <span class="q">"The ball is in their court now"</span></pre>'
              '<p>No audio leaves the device. No fee per request. Apache-2.0.</p>'
              '<p style="color:#e9eaee;font-family:ui-monospace,Menlo,monospace;font-size:36px">github.com/moglearn/pronunciada</p>'), 6.0),
    ]


def main() -> None:
    sl = slides()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        clips, t, starts = [], 0.0, []
        for i, (html_text, dur) in enumerate(sl):
            h, png, mp4 = td / f"s{i}.html", td / f"s{i}.png", td / f"s{i}.mp4"
            h.write_text(html_text, encoding="utf-8")
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                            f"--screenshot={png}", h.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-t", f"{dur}", "-i", str(png), "-r", "30",
                            "-vf", f"fade=t=in:st=0:d=0.35,fade=t=out:st={dur - 0.35:.2f}:d=0.35,format=yuv420p",
                            "-c:v", "libx264", "-crf", "20", "-preset", "medium", str(mp4)], check=True)
            clips.append(mp4); starts.append(t); t += dur
        lst = td / "list.txt"
        lst.write_text("".join(f"file '{c}'\n" for c in clips))
        silent = td / "video.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent)], check=True)
        a1, a2 = ROOT / "site/samples/en-learner-1.mp3", ROOT / "site/samples/zh-learner-1.mp3"
        d1, d2 = int((starts[1] + 1.0) * 1000), int((starts[4] + 1.0) * 1000)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(a1), "-i", str(a2),
                        "-f", "lavfi", "-t", f"{t}", "-i", "anullsrc=r=44100:cl=stereo",
                        "-filter_complex", f"[1:a]adelay={d1}|{d1},volume=1.6[x];[2:a]adelay={d2}|{d2},volume=1.6[y];[3:a][x][y]amix=inputs=3:duration=first:normalize=0[a]",
                        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(OUT)], check=True)
        for i in (2, 5):
            (ROOT / "docs" / "assets" / f"slide{i}.png").write_bytes((td / f"s{i}.png").read_bytes())
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "seconds": round(t, 1), "mb": round(OUT.stat().st_size / 1e6, 2)}))


if __name__ == "__main__":
    sys.exit(main())
