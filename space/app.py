"""Pronunciada demo for Hugging Face Spaces: record or upload a sentence, get scores and tips."""
from __future__ import annotations

import html
import json
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr

import pronunciada
from pronunciada.scorer import Scorer

HERE = Path(__file__).parent
LANGS = {
    "English": "en", "中文 (Mandarin)": "zh", "日本語 (Japanese)": "ja", "Español (Spanish)": "es",
    "Français (French)": "fr", "Deutsch (German)": "de", "Italiano (Italian)": "it",
    "Português (Portuguese)": "pt", "Русский (Russian)": "ru", "한국어 (Korean)": "ko",
}
NO_SPACES = {"zh", "ja"}
SENSITIVITY = {"Careful (fewer, surer tips)": "careful", "Balanced": "balanced", "Eager (more tips)": "eager"}

CSS = """
.pr-wrap{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,'Noto Sans','Noto Sans CJK SC',sans-serif}
.pr-chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
.pr-chip{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;padding:5px 10px;border-radius:8px;border:1px solid var(--border-color-primary);background:var(--background-fill-secondary)}
.pr-sent{font-size:28px;line-height:1.6;margin:0 0 16px}
.pr-w{border-radius:7px;padding:1px 6px;margin:0 1px;display:inline-block}
.pr-good{color:#1f8a4c}.pr-warn{color:#9a6412;background:#fbefd5}.pr-bad{color:#b3261e;background:#fbe1de}
.dark .pr-good{color:#4fc57e}.dark .pr-warn{color:#e2b04f;background:#352a12}.dark .pr-bad{color:#f0776b;background:#3a1b18}
.pr-word{margin:0 0 12px}.pr-word b{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px}
.pr-sounds{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.pr-s{font-family:ui-monospace,Menlo,Consolas,monospace;text-align:center;border:1px solid var(--border-color-primary);border-radius:8px;padding:5px 8px;min-width:44px}
.pr-s i{font-style:normal;font-size:18px;display:block}.pr-s small{font-size:11px;opacity:.75}
.pr-tips li{margin:4px 0}.pr-note{font-size:13px;opacity:.75}
"""


@lru_cache(maxsize=3)
def scorer(lang: str) -> Scorer:
    return Scorer(lang, precision=os.environ.get("PRONUNCIADA_PRECISION", "int8"))


def _cls(p) -> str:
    if p is None:
        return "pr-good"
    return "pr-good" if p >= 0.8 else ("pr-warn" if p >= 0.5 else "pr-bad")


def _f(p) -> str:
    return "–" if p is None else f"{p:.2f}"


def render(res, lang: str, tips: list[dict]) -> str:
    e = html.escape
    sep = "" if lang in NO_SPACES else " "
    out = ['<div class="pr-wrap"><div class="pr-chips">',
           f'<span class="pr-chip">sentence <b>{_f(res.prob)}</b></span>']
    if getattr(res, "fluency", None) is not None:
        out.append(f'<span class="pr-chip">fluency <b>{_f(res.fluency)}</b></span>')
    if getattr(res, "completeness", None) is not None:
        out.append(f'<span class="pr-chip">completeness <b>{_f(res.completeness)}</b></span>')
    out.append('</div><p class="pr-sent">')
    for w in res.words:
        text = w.text if lang in NO_SPACES else w.text.lower()
        out.append(f'<span class="pr-w {_cls(w.prob)}" title="{_f(w.prob)}">{e(text)}</span>{sep}')
    out.append("</p>")
    weak = [w for w in res.words if w.prob is not None and w.prob < 0.8]
    for w in weak[:6]:
        out.append(f'<div class="pr-word"><b>{e(w.text)}</b> <span class="pr-note">word {_f(w.prob)}</span><div class="pr-sounds">')
        for s in w.phones:
            heard = getattr(s, "heard", None)
            note = f"heard {e(heard)}" if heard and heard != s.ipa else ("missing" if heard == "" else _f(s.prob))
            out.append(f'<div class="pr-s {_cls(s.prob)}"><i>{e(s.ipa)}</i><small>{note}</small></div>')
        for y in w.syllables:
            if getattr(y, "tone_expected", None) is not None:
                t = f"tone {y.tone_expected}"
                if y.tone_heard is not None and y.tone_heard != y.tone_expected:
                    t += f", heard {y.tone_heard}"
                out.append(f'<div class="pr-s {_cls(y.tone_prob)}"><i>{e(y.text or "·")}</i><small>{t}</small></div>')
        out.append("</div></div>")
    if tips:
        out.append('<ul class="pr-tips">' + "".join(f"<li>{e(t.get('text') or '')}</li>" for t in tips if t.get("text")) + "</ul>")
    elif not weak:
        out.append("<p>Nothing to correct at this setting.</p>")
    issues = (getattr(res, "audio", None) or {}).get("issues") if isinstance(getattr(res, "audio", None), dict) else None
    if issues:
        out.append(f'<p class="pr-note">Recording check: {e(", ".join(issues))}. Try again in a quieter place or closer to the microphone.</p>')
    out.append('<p class="pr-note">Green: likely accepted by a teacher. Amber: borderline. Red: likely not accepted.</p></div>')
    return "".join(out)


def assess(audio, text, language, strictness, sensitivity):
    if audio is None:
        raise gr.Error("Record or upload a sentence first.")
    if not (text or "").strip():
        raise gr.Error("Type the sentence that was read.")
    lang = LANGS[language]
    try:
        res = scorer(lang).score(audio, text.strip(), strictness=float(strictness))
    except ValueError as err:
        raise gr.Error(f"Could not score this: {err}")
    from pronunciada.tips import tips as make_tips

    try:
        items = res.coaching(max_items=6, sensitivity=SENSITIVITY[sensitivity])
    except TypeError:
        items = res.coaching(max_items=6)
    tips = make_tips(items, max_tips=3, result=res)
    try:
        payload = json.loads(res.to_json())
    except Exception:
        payload = {"prob": res.prob, "words": [{"text": w.text, "prob": w.prob} for w in res.words]}
    return render(res, lang, tips), payload


def examples():
    rows = []
    order = ["en-learner-1", "en-learner-2", "zh-learner-1", "en-clean"]
    paths = sorted((HERE / "samples").glob("*.json"), key=lambda q: (order.index(q.stem) if q.stem in order else len(order), q.stem))
    for meta_path in paths:
        meta = json.loads(meta_path.read_text())["meta"]
        name = next((k for k, v in LANGS.items() if v == meta["lang"]), None)
        if name:
            rows.append([str(meta_path.with_suffix(".mp3")), meta["text"], name, 0.5, "Balanced"])
    return rows


with gr.Blocks(css=CSS, title="Pronunciada") as demo:
    gr.Markdown(
        "# Pronunciada\n"
        "Read a sentence aloud and see how each sound, word and the whole sentence was pronounced. "
        "Ten languages, open source, built to run on the device. "
        "[GitHub](https://github.com/moglearn/pronunciada) · [Results against Azure](https://github.com/moglearn/pronunciada/blob/main/results/SCOREBOARD.md)"
    )
    with gr.Row():
        with gr.Column(scale=5):
            language = gr.Dropdown(list(LANGS), value="English", label="Language")
            text = gr.Textbox(label="Sentence to read", placeholder="Type the sentence, then read it aloud", lines=2)
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Your recording")
            with gr.Accordion("Settings", open=False):
                strictness = gr.Slider(0, 1, value=0.5, step=0.05, label="Strictness (0 lenient, 1 strict)")
                sensitivity = gr.Radio(list(SENSITIVITY), value="Balanced", label="Coaching tips")
            go = gr.Button("Score it", variant="primary")
        with gr.Column(scale=6):
            report = gr.HTML(label="Result")
            with gr.Accordion("Full JSON output", open=False):
                raw = gr.JSON()
    rows = examples()
    if rows:
        gr.Examples(rows, inputs=[audio, text, language, strictness, sensitivity], label="Samples from public research datasets (CC BY 4.0)")
    gr.Markdown(
        "Recordings are scored on this Space's CPU and are not stored. "
        "Sample clips: speechocean762, OMPAL and FLEURS (CC BY 4.0). Speech model: ZIPA (CC BY 4.0)."
    )
    go.click(assess, [audio, text, language, strictness, sensitivity], [report, raw])

if __name__ == "__main__":
    demo.launch()
