# Command line

`pronunciada` (installed console script) and `python -m pronunciada` are the same program.

| command | what it does |
|---|---|
| `pronunciada check AUDIO "TEXT"` | score one recording, print a compact report |
| `pronunciada score AUDIO "TEXT" --lang en` | the detailed per-sound table (unchanged from earlier versions) |
| `pronunciada langs` | supported languages and the extra each needs |
| `pronunciada serve` | local HTTP server ([SERVER.md](SERVER.md)) |
| `pronunciada mcp` | Model Context Protocol server on stdio ([MCP.md](MCP.md)) |
| `pronunciada --version` | version |

## `check`

```
pronunciada check clip.wav "The quick brown cat jumps over the lazy dog" --lang en
```

```
✓ Good  sentence 91% · accuracy 8.2/10 · fluency 8.9/10 · completeness 89%   en · strictness 0.5

  The quick brown cat jumps over the lazy dog
  ~~~             ___

  The  56%  [ð 71%]  [ə 76%]
  cat   4%  not heard

  → The word “cat” seemed to be missing; try saying every word.
  → Listen to /ə/ in “The” and try it again; it did not sound quite right.
  → Slow down a little so that every sound is clear.
```

(The recording says "fox", the text says "cat".) How to read it:

- **First line**: a verdict from the sentence probability (Good ≥ 80 %, Close ≥ 50 %, Try again
  below), the sentence probability, the accuracy and fluency scores (0–10), completeness when
  a word was missed, then the language and strictness.
- **The sentence**: each word coloured by its probability: green ≥ 0.8, amber 0.5–0.8, red
  < 0.5; skipped words are underlined; a magenta `+` marks extra speech between words.
  Without colour, a marker line underneath carries the same information: `~` amber, `^` red,
  `_` skipped, `+` extra speech. Wide (CJK) characters take two columns and the markers line
  up with them.
- **Flagged words** (probability < 0.8, skipped, or with a coaching item): every sound in IPA
  with its probability; weak sounds (probability < 0.5 or flagged by coaching) are highlighted
  (in brackets without colour), with `→x` for the sound heard instead when one is named. Tone,
  stress, length and contrast notes appear on the next line when they are low.
- **Tips** (`→`): up to three coaching tips (`--max-tips`), from `pronunciada.tips`.
- **Audio warnings** (`⚠ audio:`): issues from the audio-quality gate, and scorer warnings.

`--strictness 1.0` on the matching text:

```
✓ Good  sentence 94% · accuracy 8.9/10 · fluency 9.3/10   en · strictness 1

  The quick brown fox jumps over the lazy dog
  ^^^

  The  49%  [ð 48%]  [ə 48%]

  → Listen to /ə/ in “The” and try it again; it did not sound quite right.
```

Options:

| option | default | |
|---|---|---|
| `--lang` | `en` | any code or alias from `pronunciada langs` (`en-US`, `zh-CN`, …) |
| `--strictness` | 0.5 | 0 lenient … 1 strict |
| `--coach` | off | also list every coaching item; with `--json`, add the `coaching` block |
| `--sensitivity` | balanced | coaching operating point: careful / balanced / eager |
| `--max-tips` | 3 | |
| `--json` | off | the full result as JSON (below) |
| `--no-color` | off | plain text |
| `--width` | terminal (max 100) | wrap width |
| `--precision`, `--threads` | fp32, auto | ONNX Runtime options |

Colour is used only when stdout is a terminal, `--no-color` is not given and `NO_COLOR` is
unset or empty (see no-color.org); `FORCE_COLOR=1` forces it on. Piped output never contains
escape codes.

Exit status: 0 success, 2 bad input (missing file, unreadable audio, empty text, unsupported
language).

## `--json`

`check --json` and `score --json` print the same document: every `SentenceResult` field
(`pronunciada/types.py`), plus, with `--coach`, `"coaching": {"schema_version", "items",
"tips"}` (docs/COACHING.md).

```
pronunciada check clip.wav "The quick brown cat jumps over the lazy dog" --json --coach
{
 "lang": "en",
 "text": "The quick brown cat jumps over the lazy dog",
 "prob": 0.911779905146861,
 "strictness": 0.5,
 "words": [
  ...
```

## `langs`

```
pronunciada langs
code  language          needs                  status
en    English           core                   ready
zh    Mandarin Chinese  pronunciada[zh]        ready
ja    Japanese          pronunciada[ja]        ready
es    Spanish           core                   ready
...
```

`status` says `missing <module>` when the extra is not installed; `--json` gives the same as
a list of objects (also `pronunciada.languages.languages()`).

## From Python

```python
import pronunciada
from pronunciada.report import render, coaching_block

res = pronunciada.score("clip.wav", "The quick brown fox jumps over the lazy dog")   # lang="en"
print(render(res, coaching_block(res)))          # the check report (color=False by default)
res.to_json()                                    # same fields as --json without coaching
```

`pronunciada.score` keeps one `Scorer` per (language, options); `SentenceResult.to_dict()`,
`to_json()` and `from_dict()` convert results (a rebuilt result has no cached evidence, so it
cannot be rescored or coached).
