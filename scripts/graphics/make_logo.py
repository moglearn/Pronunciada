"""Write the Pronunciada wordmark and favicon as SVG outlines (no font needed to display them).

    python scripts/graphics/make_logo.py

Outputs docs/assets/logo-light.svg (black, for light backgrounds), docs/assets/logo-dark.svg
(white) and site/assets/favicon.svg. The lettering is Geist at weight 560 with tightened spacing.
"""
from __future__ import annotations

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen

from common import ROOT, fonts

WEIGHT = 560
TRACKING = -0.022      # em


def outline(text: str, weight: float = WEIGHT, tracking: float = TRACKING):
    """Return (svg path data, advance width, ascent, descent) in font units, y down."""
    blob = hb.Blob.from_file_path(str(fonts() / "Geist.ttf"))
    face = hb.Face(blob)
    font = hb.Font(face)
    font.set_variations({"wght": weight})
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": False})
    upem = face.upem
    x = 0.0
    parts = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        pen = SVGPathPen(None)
        font.draw_glyph_with_pen(info.codepoint, pen)
        d = pen.getCommands()
        if d:
            parts.append(f'<path transform="translate({x + pos.x_offset:.1f} 0) scale(1 -1)" d="{d}"/>')
        x += pos.x_advance + tracking * upem
    x -= tracking * upem
    ext = font.get_font_extents("ltr")
    return "".join(parts), x, ext.ascender, -ext.descender, upem


def wordmark(color: str) -> str:
    paths, width, asc, desc, upem = outline("Pronunciada")
    cap = 0.72 * upem                      # trim to cap height plus descender of the "p"
    top, bottom = cap + 0.06 * upem, 0.24 * upem
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {-top:.0f} {width:.0f} {top + bottom:.0f}" '
            f'role="img" aria-label="Pronunciada"><g fill="{color}">{paths}</g></svg>\n')


def favicon() -> str:
    paths, width, asc, desc, upem = outline("P", weight=620, tracking=0)
    size = 1.28 * upem
    dx, dy = (size - width) / 2 + 0.02 * upem, size / 2 + 0.355 * upem
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size:.0f} {size:.0f}">'
            f'<rect width="{size:.0f}" height="{size:.0f}" rx="{0.3 * size:.0f}" fill="#0b0b0c"/>'
            f'<g fill="#fff" transform="translate({dx:.0f} {dy:.0f})">{paths}</g></svg>\n')


if __name__ == "__main__":
    (ROOT / "docs/assets/logo-light.svg").write_text(wordmark("#000000"), encoding="utf-8")
    (ROOT / "docs/assets/logo-dark.svg").write_text(wordmark("#ffffff"), encoding="utf-8")
    (ROOT / "site/assets/favicon.svg").write_text(favicon(), encoding="utf-8")
    (ROOT / "site/assets/logo-light.svg").write_text(wordmark("#000000"), encoding="utf-8")
    print("wrote logo-light.svg, logo-dark.svg, favicon.svg")
