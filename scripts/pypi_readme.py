"""Rewrite README.md in place for PyPI: relative links and images become absolute GitHub URLs.

PyPI renders the README outside the repository, so relative paths break there. The release
workflow runs this on its own checkout just before building; the committed README keeps
relative links.

    python scripts/pypi_readme.py [git-ref]        # default ref: main
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = "moglearn/pronunciada"
README = Path(__file__).resolve().parents[1] / "README.md"


def rewrite(text: str, ref: str = "main") -> str:
    blob = f"https://github.com/{REPO}/blob/{ref}/"
    raw = f"https://raw.githubusercontent.com/{REPO}/{ref}/"

    def is_relative(url: str) -> bool:
        return not re.match(r"^(https?:|mailto:|#)", url)

    # HTML images and links
    text = re.sub(r'src="([^"]+)"', lambda m: f'src="{raw}{m.group(1)}"' if is_relative(m.group(1)) else m.group(0), text)
    text = re.sub(r'href="([^"]+)"', lambda m: f'href="{blob}{m.group(1)}"' if is_relative(m.group(1)) else m.group(0), text)
    # Markdown links: [text](path)
    text = re.sub(r"\]\(([^)\s]+)\)", lambda m: f"]({blob}{m.group(1)})" if is_relative(m.group(1)) else m.group(0), text)
    return text


if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else "main"
    README.write_text(rewrite(README.read_text(encoding="utf-8"), ref), encoding="utf-8")
    print(f"README.md rewritten for PyPI (ref {ref})")
