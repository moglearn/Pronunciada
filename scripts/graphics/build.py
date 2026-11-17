"""Regenerate every graphic and copy the ones the site uses into site/assets/.

    python scripts/graphics/build.py        # needs Google Chrome or Chromium, and ffmpeg
"""
from __future__ import annotations

import runpy
import shutil
from pathlib import Path

from common import ROOT

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    for script in ("make_logo.py", "make_banner.py", "make_charts.py"):
        runpy.run_path(str(HERE / script), run_name="__main__")
    for name in ("social.jpg", "chart-english.png", "chart-detection.png", "chart-languages.png", "chart-tally.png", "chart-zipa.png"):
        shutil.copy2(ROOT / "docs/assets" / name, ROOT / "site/assets" / name)
    print("copied site assets")
