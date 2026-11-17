"""Pronunciada demo for Hugging Face Spaces: score a recording against its reference text."""
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
SENSITIVITY = {"Careful (fewer tips)": "careful", "Balanced": "balanced", "Eager (more tips)": "eager"}

CSS = """
.pr-wrap{font-size:15px;line-height:22px}
.pr-gloss{display:flex;flex-wrap:wrap;gap:10px 16px;margin:4px 0 14px}
.pr-gloss span{display:flex;flex-direction:column}
.pr-gloss b{font-weight:400;font-size:20px;line-height:26px}
.pr-gloss i{font-style:normal;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:16px;opacity:.6}
.pr-low b,.pr-low i,.pr-lowp{color:#b42318;opacity:1}
.dark .pr-low b,.dark .pr-low i,.dark .pr-lowp{color:#f97066}
.pr-dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:0 0 12px}
.pr-dl dt{font-weight:500}.pr-dl dd{margin:0}
.pr-ph{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
.pr-note{font-size:13px;opacity:.7}
"""


THEME = gr.themes.Base(
    primary_hue="neutral", neutral_hue="stone", radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#fdfcfc", block_background_fill="#ffffff", block_border_width="1px",
    button_primary_background_fill="#000000", button_primary_background_fill_hover="#262626",
    button_primary_text_color="#ffffff", button_large_radius="9999px", button_small_radius="9999px",
)


@lru_cache(maxsize=3)
def scorer(lang: str) -> Scorer:
    return Scorer(lang, precision=os.environ.get("PRONUNCIADA_PRECISION", "int8"))


def _f(p) -> str:
    return "" if p is None else f"{p:.2f}"


def _low(p) -> bool:
    return p is not None and p < 0.5


def render(res, lang: str, tips: list[dict]) -> str:
    """Words with their acceptance probability underneath; phonemes for words below 0.5."""
    e = html.escape
    out = ['<div class="pr-wrap"><div class="pr-gloss">']
    for w in res.words:
        cls = ' class="pr-low"' if _low(w.prob) else ""
        out.append(f"<span{cls}><b>{e(w.text)}</b><i>{_f(w.prob)}</i></span>")
    out.append('</div><dl class="pr-dl">')
    sentence = _f(res.prob)
    if getattr(res, "fluency", None) is not None:
        sentence += f", fluency {_f(res.fluency)}"
    out.append(f"<dt>Sentence</dt><dd>{sentence}</dd>")
    for w in res.words:
        if not _low(w.prob):
            continue
        parts = []
        for s in w.phones:
            heard = getattr(s, "heard", None)
            t = f"{e(s.ipa)} {_f(s.prob)}" + (f" (recognized as {e(heard)})" if heard and heard != s.ipa else "")
            parts.append(f'<span class="pr-lowp">{t}</span>' if _low(s.prob) else t)
        out.append(f'<dt>{e(w.text)}</dt><dd class="pr-ph">{", ".join(parts)}</dd>')
    tones = []
    for w in res.words:
        for y in w.syllables:
            if getattr(y, "tone_expected", None) is None:
                continue
            t = f"{e(y.text or '')} tone {y.tone_expected}"
            if y.tone_heard is not None and y.tone_heard != y.tone_expected:
                t += f", recognized {y.tone_heard}"
            t += f" {_f(y.tone_prob)}"
            tones.append(f'<span class="pr-lowp">{t}</span>' if _low(y.tone_prob) else t)
    if tones:
        out.append(f'<dt>Tones</dt><dd class="pr-ph">{"; ".join(tones)}</dd>')
    texts = [e(t["text"]) for t in tips if t.get("text")]
    if texts:
        out.append(f'<dt>Tips</dt><dd>{"<br>".join(texts)}</dd>')
    audio = getattr(res, "audio", None)
    issues = audio.get("issues") if isinstance(audio, dict) else None
    if issues:
        out.append(f'<dt>Recording</dt><dd>{e(", ".join(issues))}</dd>')
    out.append("</dl>")
    out.append('<p class="pr-note">The number under each word is the probability that a human rater would accept it. '
               "Words below 0.5 are in red.</p></div>")
    return "".join(out)


def assess(audio, text, language, strictness, sensitivity):
    if audio is None:
        raise gr.Error("Record or upload audio first.")
    if not (text or "").strip():
        raise gr.Error("Enter the reference text.")
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


with gr.Blocks(title="Pronunciada") as demo:
    gr.Markdown(
        "# Pronunciada\n"
        "Pronunciation assessment for ten languages. Record or upload a sentence, enter the text that was read, "
        "and the demo returns the probability that each word and phoneme would be accepted by a human rater. "
        "[Source](https://github.com/moglearn/pronunciada) · "
        "[Benchmarks](https://github.com/moglearn/pronunciada/blob/main/results/SCOREBOARD.md)"
    )
    with gr.Row():
        with gr.Column(scale=5):
            language = gr.Dropdown(list(LANGS), value="English", label="Language")
            text = gr.Textbox(label="Reference text", placeholder="The sentence that is read in the recording", lines=2)
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Recording",
                             waveform_options=gr.WaveformOptions(waveform_color="#b9b4ae", waveform_progress_color="#000000"))
            with gr.Accordion("Settings", open=False):
                strictness = gr.Slider(0, 1, value=0.5, step=0.05, label="Strictness (0 lenient, 1 strict)")
                sensitivity = gr.Radio(list(SENSITIVITY), value="Balanced", label="Coaching tips")
            go = gr.Button("Score", variant="primary")
        with gr.Column(scale=6):
            report = gr.HTML(label="Result")
            with gr.Accordion("Full JSON output", open=False):
                raw = gr.JSON()
    rows = examples()
    if rows:
        gr.Examples(rows, inputs=[audio, text, language, strictness, sensitivity], label="Samples from speechocean762, OMPAL and FLEURS (CC BY 4.0)")
    gr.Markdown(
        "Recordings are scored on this Space's CPU and are not stored. The acoustic model is ZIPA (CC BY 4.0)."
    )
    go.click(assess, [audio, text, language, strictness, sensitivity], [report, raw])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME)
