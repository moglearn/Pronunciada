"""Row-by-row difference of two SCOREBOARD.md files (check E of the round 19 merge).

    python bench/round19_merge/scoreboard_diff.py old.md new.md [--json out.json]

Each markdown table row is keyed by (the section heading or bold caption above it, the header
row, the first cell, and the second cell when the first repeats in that table). Prints every row
whose text changed, with both versions, plus rows only in one file.
"""
from __future__ import annotations

import json
import sys


def rows(path):
    out, sec, header, seen = {}, "", None, {}
    for line in open(path, encoding="utf-8"):
        s = line.rstrip("\n")
        if s.startswith("#") or (s.startswith("**") and not s.startswith("|")):
            sec, header = s.strip("# *")[:90], None
            continue
        if not s.startswith("|"):
            header = None if not s.strip() else header
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if header is None:
            header = " | ".join(cells)
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        key0 = (sec, header, cells[0])
        n = seen.get(key0, 0)
        seen[key0] = n + 1
        key = (sec, header, cells[0], "", n)
        out[key] = cells
    return out


def main():
    old, new = rows(sys.argv[1]), rows(sys.argv[2])
    diff = []
    for k in list(old) + [k for k in new if k not in old]:
        a, b = old.get(k), new.get(k)
        if a != b:
            diff.append({"section": k[0], "header": k[1], "row": k[2], "occurrence": k[4], "before": a, "after": b})
    for d in diff:
        print(f"## {d['section']} :: {d['row']} (#{d['occurrence']})\n  before: {d['before']}\n  after:  {d['after']}")
    print(f"{len(diff)} rows changed / added / removed")
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w", encoding="utf-8") as f:
            json.dump(diff, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
