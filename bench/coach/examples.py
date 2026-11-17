"""Example coaching outputs for docs/COACHING.md: native FLEURS VALIDATION recordings scored
against prompts with planted errors (bench.corrupt), so what "should" be heard is known.

    python -m bench.coach.examples > /tmp/examples.json
"""
from __future__ import annotations

import json
import sys

from bench.corrupt import NotApplicable, corrupt
from bench.pipeline.common import fleurs_validation_records, iter_parquet_audio, stable_seed
from bench.pipeline.evalctl import inventory
from pronunciada import coach
from pronunciada.scorer import Scorer
from pronunciada.tips import tips

PLAN = {"en": ["phone_sub_far", "stress"], "zh": ["tone", "phone_sub_far"], "es": ["phone_sub_far", "phone_sub_near"]}
N_CLIPS = 200


def main():
    out = {}
    for lang, kinds in PLAN.items():
        sc = Scorer(lang, threads=2)
        recs = fleurs_validation_records(lang)[:N_CLIPS]
        by_row = {int(r["audio"].rsplit(":", 1)[1]): r for r in recs}
        path = recs[0]["audio"].split(":", 1)[1].rsplit(":", 1)[0]
        for row, wav in iter_parquet_audio(path, set(by_row)):
            rec = by_row[row]
            if not (4 <= rec["dur"] <= 8):
                continue
            ref = sc.g2p(rec["text"])
            try:
                cor = corrupt(ref, kinds, inventory(lang), n_errors=2, seed=stable_seed(rec["id"], "example"))
            except NotApplicable:
                continue
            res = sc.score(wav, cor.ref)
            items = res.coaching()
            dg = [i for i in items if i["heard"] is not None]
            need = (lambda its: any(i["kind"] == "tone" and i["heard"] for i in its)) if lang == "zh" else \
                (lambda its: sum(i["heard"] is not None for i in its) >= 1 and len(its) <= 6)
            if not need(items):
                continue
            for i in items:
                i.pop("gate")
            out[lang] = {"id": rec["id"], "text": rec["text"],
                         "planted": [{"kind": e.kind, "word": cor.ref.words[e.word].text, **e.detail} for e in cor.errors],
                         "audio": res.audio, "coaching": {"schema_version": coach.SCHEMA_VERSION, "schema_minor": coach.SCHEMA_MINOR, "items": items},
                         "tips": [{k: v for k, v in t.items() if k != "item"} for t in tips(items, 3, res)]}
            break
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
