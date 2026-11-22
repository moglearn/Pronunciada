"""Manifests for OpenPronounce's own `openpronounce-evaluate` command, from our held-out sets that carry a
numeric human score, and the command lines to run it.

    python -m bench.rivals.evaluate_manifests            # writes the CSVs, prints the commands

  cv-real test, one manifest per covered locale (en fr es de it pt): score = up_votes - down_votes of the
           Common Voice validators (accepted clips have net positive votes, rejected ones net negative),
           columns test (A: accepted vs rejected, B: native vs non-native) and side for --group-by.
  speechocean762 test: score = expert total (0-10), group column age_group (OpenPronounce's age rule).
Files go to $PRONUNCIADA_CACHE/round19/rivals/evaluate/ (audio paths absolute).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from bench.rivals.openpronounce import WORK

COVERED = ("en", "fr", "es", "de", "it", "pt")


def age_group(age: int) -> str:
    return "child_12_and_under" if age <= 12 else ("teen_13_17" if age <= 17 else "adult_18_plus")


def main() -> None:
    from bench import eval_cv as EC
    from bench.data import load

    d = WORK / "evaluate"
    d.mkdir(parents=True, exist_ok=True)
    cmds = []
    by = {}
    for r in EC.records():
        if r["lang"] in COVERED and r["test"] in ("A", "B"):
            by.setdefault(r["lang"], []).append(r)
    for lang, rs in sorted(by.items()):
        p = d / f"cv-real-{lang}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["audio", "score", "text", "test", "side"])
            for r in rs:
                w.writerow([r["audio"], int(r["up_votes"]) - int(r["down_votes"]), r["sentence"], r["test"], r["side"]])
        cmds.append((p, lang, ["test", "side"]))
        print(p, len(rs))
    p = d / "speechocean762-test.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio", "score", "text", "age_group"])
        for r in load("speechocean762", split="test"):
            w.writerow([r["audio"], r["labels"]["sentence"]["total"], r["text"], age_group(int(r["labels"]["age"]))])
    cmds.append((p, "en", ["age_group"]))
    print(p, 2500)
    for p, lang, g in cmds:
        print(f"openpronounce-evaluate {p} --lang {lang} " + " ".join(f"--group-by {x}" for x in g)
              + f" --out {p.with_suffix('.results.csv')}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
