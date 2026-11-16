"""One real example per round-3 item kind (FLEURS validation; extra_speech from a prompt with one
word removed), for docs/COACHING.md.

    python -m bench.coach.examples_r3 > results/tables/coaching_examples_r3.json
"""
from __future__ import annotations

import json
import sys

from bench.coach.dump_r3 import remove_word
from bench.pipeline.common import fleurs_validation_records, iter_parquet_audio
from pronunciada.scorer import Scorer
from pronunciada.tips import tip_for

WANT = ("mispronounced_word", "extra_speech", "long_pause", "pause_mid_phrase", "rushed_punctuation", "fast",
        "slow", "flat_pitch")


def main():
    found = {}
    for lang in ("en", "es", "fr", "de"):
        sc = Scorer(lang, threads=2)
        recs = fleurs_validation_records(lang)[:250]
        by_row = {int(r["audio"].rsplit(":", 1)[1]): r for r in recs}
        path = recs[0]["audio"].split(":", 1)[1].rsplit(":", 1)[0]
        for row, wav in iter_parquet_audio(path, set(by_row)):
            rec = by_row[row]
            res = sc.score(wav, rec["text"])
            texts = [(rec["text"], None)]
            ref = res._pronunciada["ref"]  # noqa: SLF001
            if len(ref.words) >= 6 and "extra_speech" not in found:
                for k in range(1, len(ref.words) - 1):
                    texts.append((remove_word(ref, k), ref.words[k].text))
            for txt, removed in texts:
                r = res if removed is None else sc.score(wav, txt)
                words = [w.text for w in r.words]
                for it in r.coaching(delivery="all"):
                    if it["kind"] in WANT and it["kind"] not in found:
                        it.pop("gate", None)
                        found[it["kind"]] = {"lang": lang, "id": rec["id"], "prompt": txt, "removed_word": removed,
                                             "item": it, "tip": tip_for(it, None, words)}
            if len(found) == len(WANT):
                json.dump(found, sys.stdout, ensure_ascii=False, indent=1, default=float)
                return
    json.dump(found, sys.stdout, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
