# Model Context Protocol server

`pronunciada mcp` runs an MCP server on stdio, so an MCP client (an assistant app, an
agent framework, an IDE) can check pronunciation of local recordings.

```
pip install 'pronunciada[mcp]'      # the official `mcp` Python SDK (optional; core works without it)
pronunciada mcp
```

Without the SDK the command prints one install hint and exits with status 1. Tested with
`mcp` 2.3 (`MCPServer`); on 1.x it falls back to `FastMCP` (not tested).

## Tools

**`assess_pronunciation(audio_path, reference_text, language="en", coach=True, strictness=0.5)`**

- `audio_path`: a local audio file (WAV / FLAC / OGG, any sample rate), read by the server
  process.
- `reference_text`: what the speaker was asked to say. `language`: a code from `list_languages`.
- Returns two text blocks and structured content:
  1. the plain-text `pronunciada check` report ([CLI.md](CLI.md)), which is what a model
     usually needs;
  2. the result as JSON, also sent as `structuredContent`: the `check --json` document
     without the raw debugging evidence (per-sound `features`, coaching `gate`) and
     without `timing_ms`.
- Bad input (missing file, empty text, unsupported language) returns an error result
  (`isError: true`) with a one-line message.

**`list_languages()`**: the supported languages, whether each is ready, and the extra a
missing one needs.

Example call result (first text block) for a recording of "The quick brown fox jumps over
the lazy dog" checked against "...cat...":

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

## Client configuration

Most clients take a JSON block like this (for example a `mcpServers` section in the client's
config file):

```json
{
  "mcpServers": {
    "pronunciada": {
      "command": "pronunciada",
      "args": ["mcp"]
    }
  }
}
```

If `pronunciada` is not on the client's `PATH` (for example it lives in a virtual
environment), give the interpreter instead:

```json
{
  "mcpServers": {
    "pronunciada": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "pronunciada", "mcp"]
    }
  }
}
```

The first `assess_pronunciation` call for a language loads its model (a few seconds); later
calls reuse it. The server reads any local path it is given, with the permissions of the
user who started it.

## From Python, without MCP

The tool functions are plain Python:

```python
from pronunciada.mcp_server import assess_pronunciation, list_languages
text, data = assess_pronunciation("clip.wav", "The quick brown fox jumps over the lazy dog")
```

## Testing

`tests/test_entrypoints.py` lists and calls both tools through the SDK's in-process client,
starts `python -m pronunciada mcp` as a stdio subprocess and calls `list_languages`, and (with
the model cached) runs `assess_pronunciation` on the suite's TTS clip.
