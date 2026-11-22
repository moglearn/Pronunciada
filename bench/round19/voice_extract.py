"""Round 19: voice cues (pronunciada.sentence_head.voice_cues) for speechocean762, added to
an existing feature cache as voice_<split>.json (feature caches made before the cue existed).

    python -m bench.round19.voice_extract --split train
"""
from __future__ import annotations

import argparse
import json

from bench.round19 import data as RD


def main():
    from bench.pipeline.common import records
    from pronunciada.audio import load_audio
    from pronunciada.sentence_head import voice_cues

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    out = {}
    for r in records("speechocean762"):
        if r["split"] == a.split:
            out[r["id"]] = voice_cues(load_audio(r["audio"]))
    p = RD.feat_dir(a.split).parent / f"voice_speechocean762_{a.split}.json"
    p.write_text(json.dumps(out))
    print(len(out), "->", p)


if __name__ == "__main__":
    main()
