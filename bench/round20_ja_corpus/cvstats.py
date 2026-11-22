"""Counts of the Common Voice 22.0 ja metadata (clips, speakers, hours per TSV)."""
from __future__ import annotations

import json
import sys

from bench.round20_ja_corpus.common import CVMETA, cv_durations, read_tsv


def main() -> None:
    dur = cv_durations()
    out = {}
    paths = {}
    for n in ("validated", "train", "dev", "test", "other", "invalidated"):
        rows = list(read_tsv(CVMETA / f"{n}.tsv"))
        paths[n] = {r["path"] for r in rows}
        out[n] = {"clips": len(rows), "speakers": len({r["client_id"] for r in rows}),
                  "hours": round(sum(dur.get(r["path"], 0) for r in rows) / 3600, 2),
                  "sentences": len({r["sentence"] for r in rows})}
    spl = paths["train"] | paths["dev"] | paths["test"]
    rest = paths["validated"] - spl
    out["validated_not_in_splits"] = {"clips": len(rest), "hours": round(sum(dur.get(p, 0) for p in rest) / 3600, 2)}
    out["other_and_validated"] = len(paths["other"] & paths["validated"])
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
