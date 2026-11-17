"""Model Context Protocol server on stdio: `pronunciada mcp`.

Tools
  assess_pronunciation(audio_path, reference_text, language="en", coach=True, strictness=0.5)
      -> a short plain-text report (the `pronunciada check` report without color) plus the
         structured result (pronunciada.report.payload without the raw debug evidence)
  list_languages() -> supported languages and the extras each needs

Needs the optional `mcp` package (pip install 'pronunciada[mcp]'); the core package never imports it.
The tool functions below are plain Python and usable without it.
"""
from __future__ import annotations

import os
import sys

INSTALL_HINT = "the MCP server needs the `mcp` package: pip install 'pronunciada[mcp]'  (or: pip install mcp)"
INSTRUCTIONS = ("Pronunciada checks how well a recording matches a known reference text. Call "
                "assess_pronunciation with a local audio file path (WAV / FLAC / OGG) and the text the "
                "speaker was asked to read. Probabilities are 0-1 (word / sound acceptable); tips are "
                "short coaching sentences.")


def assess_pronunciation(audio_path: str, reference_text: str, language: str = "en", coach: bool = True,
                         strictness: float = 0.5) -> tuple[str, dict]:
    """Score one recording -> (plain-text summary, structured result)."""
    from . import report
    from . import score as _score

    path = os.path.expanduser(str(audio_path))
    if not os.path.isfile(path):
        raise ValueError(f"audio file not found: {audio_path}")
    if not reference_text or not str(reference_text).strip():
        raise ValueError("reference_text is empty")
    s = float(strictness)
    if not 0.0 <= s <= 1.0:
        raise ValueError("strictness must be between 0 and 1")
    res = _score(path, str(reference_text), language, strictness=s)
    cb = report.coaching_block(res) if coach else None
    text = report.render(res, cb, color=False, width=88)
    data = report.payload(res, coach=coach, coaching=cb, debug=False)
    data.pop("timing_ms", None)
    return text, data


def list_languages() -> list[dict]:
    from .languages import languages

    return languages()


def build_server():
    """The MCP server object (mcp >= 2: MCPServer; mcp 1.x: FastMCP). Raises ImportError without mcp."""
    import json

    try:
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:
        from mcp.server.fastmcp import FastMCP as _Server   # mcp 1.x
    from mcp.types import CallToolResult

    from . import __version__

    try:
        srv = _Server("pronunciada", instructions=INSTRUCTIONS, version=__version__)
    except TypeError:
        srv = _Server("pronunciada", instructions=INSTRUCTIONS)

    def _result(text: str, data, error: bool = False):
        content = [{"type": "text", "text": text}]
        d = {"content": content, "isError": error}
        if data is not None:
            content.append({"type": "text", "text": json.dumps(data, ensure_ascii=False)})
            d["structuredContent"] = data
        return CallToolResult.model_validate(d)

    @srv.tool(name="assess_pronunciation")
    def _assess(audio_path: str, reference_text: str, language: str = "en", coach: bool = True,
                strictness: float = 0.5):
        """Check the pronunciation of a recording against the text the speaker meant to say.

        audio_path: local audio file (WAV / FLAC / OGG, any sample rate); reference_text: the
        expected sentence; language: code such as en, zh, ja, es, fr, de, it, pt, ru, ko;
        coach: include coaching items and tips; strictness: 0 lenient .. 1 strict.
        Returns a short text report and the structured result (probabilities 0-1)."""
        import contextlib

        try:
            with contextlib.redirect_stdout(sys.stderr):      # stdout is the protocol channel
                text, data = assess_pronunciation(audio_path, reference_text, language, coach, strictness)
        except (ValueError, ImportError, RuntimeError, OSError) as e:
            return _result(f"error: {e}", None, error=True)
        return _result(text, data)

    @srv.tool(name="list_languages")
    def _langs():
        """List the supported languages, whether each is ready, and the extra it needs."""
        L = list_languages()
        lines = [f"{x['code']}  {x['name']}" + ("" if x["ready"] else f"  (needs {x['install']})") for x in L]
        return _result("\n".join(lines), {"languages": L})

    return srv


def main() -> int:
    try:
        srv = build_server()
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return 1
    srv.run()            # stdio transport
    return 0
