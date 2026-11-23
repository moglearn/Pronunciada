"""Key-by-key difference of the regenerated tables against the committed ones (main), grouped by language.

    python -m bench.round20_fa_rc.table_diff <main results dir> <new results dir> OUT.json

Every numeric or string leaf of every results/tables/*.json that differs is listed with its path; a path is
attributed to a language when one of its components is a language code (or a code-prefixed key such as
es/A, fleurs-test-es, tier1_es). Bookkeeping keys (fingerprints, times, git commit, timings) are counted
separately. Prints the count per language.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

LANGS = ("en", "de", "es", "fr", "ko", "ja", "zh", "it", "pt", "ru")
BOOK = re.compile(r"fingerprint|made_utc|time|git|commit|latency|_ms|rtf|speed|outputs_from|version|date", re.I)


def flat(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{pre}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{pre}[{i}]")
    else:
        yield pre, x


def lang_of(path: str):
    for part in re.split(r"[/\[\]]", path):
        for lg in LANGS:
            if part == lg or re.match(rf"^{lg}([/_\-:]|$)", part) or re.search(rf"[_\-]{lg}$", part):
                return lg
    return None


def main() -> None:
    a_dir, b_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    res = {}
    for pb in sorted((b_dir / "tables").glob("*.json")):
        pa = a_dir / "tables" / pb.name
        if not pa.exists():
            continue
        A, B = dict(flat(json.loads(pa.read_text()))), dict(flat(json.loads(pb.read_text())))
        if A == B:
            continue
        rows = []
        for k in sorted(set(A) | set(B)):
            if A.get(k, "<missing>") != B.get(k, "<missing>"):
                rows.append({"path": k, "lang": "bookkeeping" if BOOK.search(k) else lang_of(k),
                             "main": A.get(k, "<missing>"), "new": B.get(k, "<missing>")})
        res[pb.name] = {"by_lang": dict(Counter(str(r["lang"]) for r in rows)), "rows": rows}
        print(pb.name, res[pb.name]["by_lang"])
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
