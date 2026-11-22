"""Write our OpenPronounce speechocean762 run in the CSV layout of OpenPronounce's own
benchmarks/speechocean762.py, so that its `--report` can be run on all 2,500 test recordings.

    python -m bench.rivals.their_csv --mode default
    <OpenPronounce venv>/bin/python -I <clone>/benchmarks/speechocean762.py --report --out <csv>

Every column is filled the way that script's run() fills it (same call, compare_audio_with_text with
the default options; word metrics with its rule: a word is bad for the raters below 5, a word counts
as flagged when its lower-cased text is among the words in `errors`). wall_time = our per-call seconds.
Their run() also reads the audio from the HF parquet; ours is the same recordings in FLAC.
"""
from __future__ import annotations

import argparse
import csv

from bench.rivals.openpronounce import WORK, read_raw

FIELDS = ["utt", "speaker", "text", "human_total", "human_accuracy", "human_fluency", "human_prosodic",
          "score", "formula_score", "model_accuracy", "model_fluency", "model_prosodic", "sentence_probability",
          "acoustic_distance", "phoneme_error_rate", "word_error_rate",
          "n_words", "n_flagged", "n_human_bad", "n_hits", "word_recall", "word_precision", "wall_time"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="default")
    a = ap.parse_args(argv)
    from bench.data import load

    R = read_raw(a.mode, "speechocean762")
    out = WORK / "their_csv" / f"speechocean762-{a.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rec in load("speechocean762", split="test"):
            o = R[rec["id"]]["out"]
            L = rec["labels"]
            words = [x["text"].lower() for x in L["words"]]
            bad = {i for i, x in enumerate(L["words"]) if x["accuracy"] < 5}
            fw = {e["word"].lower() for e in o["errors"]}
            fl = {i for i, x in enumerate(words) if x in fw}
            hits = bad & fl
            s = L["sentence"]
            w.writerow({"utt": rec["id"].removeprefix("so762-"), "speaker": rec["speaker"], "text": rec["text"],
                        "human_total": s["total"], "human_accuracy": s["accuracy"], "human_fluency": s["fluency"],
                        "human_prosodic": s["prosodic"], "score": o["score"],
                        "formula_score": "" if o["formula_score"] is None else o["formula_score"],
                        "model_accuracy": o["scores"]["accuracy"], "model_fluency": o["scores"]["fluency"],
                        "model_prosodic": o["scores"]["prosodic"],
                        "sentence_probability": (o.get("alignment") or {}).get("sentence_probability", ""),
                        "acoustic_distance": "" if o["acoustic_distance"] is None else o["acoustic_distance"],
                        "phoneme_error_rate": o["phoneme_error_rate"],
                        "word_error_rate": "" if o["word_error_rate"] is None else o["word_error_rate"],
                        "n_words": len(words), "n_flagged": len(fl), "n_human_bad": len(bad), "n_hits": len(hits),
                        "word_recall": round(len(hits) / len(bad), 4) if bad else "",
                        "word_precision": round(len(hits) / len(fl), 4) if fl else "", "wall_time": o["sec"]})
    print(out)


if __name__ == "__main__":
    main()
