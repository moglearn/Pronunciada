# Integration notes: developer entry points

For whoever owns `README.md` / `pyproject.toml` / `NOTICE`. The entry points live in new files
(`pronunciada/report.py`, `languages.py`, `server.py`, `mcp_server.py`, `compat/azure.py`,
`py.typed`, docs `CLI.md`, `SERVER.md`, `AZURE_COMPAT.md`, `MCP.md`, tests
`tests/test_entrypoints.py` + `tests/data/cli/`). Existing files touched: `pronunciada/__main__.py`
(subcommands added; `score` unchanged), `pronunciada/__init__.py` (`score()` with `lang="en"`
default), `pronunciada/types.py` (`SentenceResult.to_dict / to_json / from_dict`). No scoring
code changed. Delete this file once merged.

## pyproject.toml

The console script already exists (`pronunciada = "pronunciada.__main__:main"`); keep it.

```toml
[project.optional-dependencies]
# add:
mcp = ["mcp>=2"]                       # MIT; `pronunciada mcp` only (tested with mcp 2.3.0)
# change:
all = ["pronunciada[zh,ja,numbers,mcp]"]

[project]
classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
    "Topic :: Education",
    # add:
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Typing :: Typed",
]

[tool.setuptools.package-data]
pronunciada = ["data/**/*", "py.typed"]     # add "py.typed"
```

(No license classifier: with a PEP 639 `license` expression setuptools rejects them.)

NOTICE / LICENSING: `mcp` is MIT and optional, not bundled. The Azure adapter contains no
Microsoft code or data (field names only); the docs state it is format-compatible, not
endorsed.

## README sections (ready to paste)

### Quick start: command line

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

Words are coloured by probability in a terminal (green / amber / red, skipped words
underlined); `--json` prints the full result, `--coach` adds the coaching items,
`pronunciada langs` lists the languages and the extras they need. Details:
[docs/CLI.md](docs/CLI.md).

### Quick start: Python

```python
import pronunciada

res = pronunciada.score("clip.wav", "The quick brown fox jumps over the lazy dog", lang="en")
res.prob                     # sentence probability, 0-1
[(w.text, round(w.prob, 2)) for w in res.words]
res.to_json()
```

### Other ways to call it

| | |
|---|---|
| `pronunciada serve` | local HTTP server, `POST /v1/score` ([docs/SERVER.md](docs/SERVER.md)); development use only, no authentication |
| `pronunciada serve --azure-compat` | also answers the Azure Speech short-audio REST path with Azure-*shaped* JSON, so an existing client only changes its base URL ([docs/AZURE_COMPAT.md](docs/AZURE_COMPAT.md)) |
| `pronunciada.compat.azure.to_azure(res)` | the same Azure-shaped dict in Python |
| `pronunciada mcp` | Model Context Protocol server (stdio) with `assess_pronunciation` and `list_languages` tools; `pip install 'pronunciada[mcp]'` ([docs/MCP.md](docs/MCP.md)) |

The Azure-format output is compatible in shape only: the numbers are pronunciada's own, it is
not Azure's output and not endorsed by Microsoft, and thresholds tuned on Azure scores must be
re-chosen.
