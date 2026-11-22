"""Small QC tables for the repository: exclusions and alignment QC.

    python -m bench.round20_ja_corpus.tables
        -> results/tables/round20_ja_corpus_{exclusions,alignqc}.json
(durations / anchors / xcheck / counts are written by qc.py --table.)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from bench.round20_ja_corpus.common import WORK

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
GROUPS = {"cv22_validated+fleurs": ["valid"], "kokoro_reading_agree": ["kokoro_0", "kokoro_1"],
          "cv22_other": [f"other_{k}" for k in range(8)]}


def main() -> None:
    json.dump(json.load(open(WORK / "exclusions.json")), open(TABLES / "round20_ja_corpus_exclusions.json", "w"), indent=1)
    by, tot = {}, Counter()
    for g, bs in GROUPS.items():
        c = Counter()
        for b in bs:
            c.update(json.load(open(WORK / "aligned" / f"{b}.qc.json")))
        by[g] = dict(c)
        tot.update(c)
    kc: Counter = Counter()
    for line in open(WORK / "records.jsonl", encoding="utf-8"):
        if '"source": "kokoro"' not in line[:300]:
            continue
        ch = json.loads(line)["reading_check"]
        kc["agree" if ch["agree"] else "disagree"] += 1
        if not ch["agree"]:
            for k in ("long", "gem", "N"):
                if ch["kokoro"][k] != ch["frontend"][k]:
                    kc[f"disagree_{k}"] += 1
    out = {"frontend": json.load(open(WORK / "records_counts.json")), "kokoro_reading_check": dict(kc),
           "alignment_by_group": by, "alignment_total": dict(tot),
           "note": "variant:<x> = the aligner preferred a non-canonical pronunciation (short: a long vowel "
                   "shortened; nogem: geminate closure dropped; spelled: ou/ei spelling form) -> utterance dropped; "
                   "alignment_failed = no alignment within beam 100 / retry 400; mismatch = word and phone "
                   "tiers inconsistent. Kokoro clips whose own reading disagrees with the front-end on the "
                   "number of long vowels / geminates / N were not aligned."}
    json.dump(out, open(TABLES / "round20_ja_corpus_alignqc.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
