"""Estimate: validated Common Voice ja clips whose audio is only in the train/dev/test shards of OLDER
releases on the same mirror family (fsicoli/common_voice_{15_0,17_0,19_0,21_0}), after the exclusions.
Metadata only (split TSVs under raw/cv_older/<version>/).

    python -m bench.round20_ja_corpus.older_cv
"""
from __future__ import annotations

import json
import sys

from bench.cv_build import norm_sentence
from bench.round20_ja_corpus.common import CVMETA, RAW, WORK, cv_durations, read_tsv
from bench.round20_ja_corpus.exclude import _BAD_TEXT


def main() -> None:
    dur = cv_durations()
    have = {json.loads(x)["path"] for x in open(WORK / "cv_select.jsonl", encoding="utf-8")}
    rep = json.load(open(WORK / "exclusions.json"))
    _ = rep
    bad_sent = set((WORK / "excluded_sentences.txt").read_text(encoding="utf-8").split())
    validated = {r["path"]: r for r in read_tsv(CVMETA / "validated.tsv")}
    # excluded speakers: every client id of a clip we excluded for held-out / practice reasons
    import bench.round20_ja_corpus.exclude as ex

    clips, _s = ex.lingolix()
    clip2spk = {p: r["client_id"] for p, r in validated.items()}
    for n in ("other", "invalidated", "train", "dev", "test"):
        for r in read_tsv(CVMETA / f"{n}.tsv"):
            clip2spk.setdefault(r["path"], r["client_id"])
    bad_spk = {clip2spk[c] for c in clips if c in clip2spk}
    import json as _j
    from pathlib import Path

    repo = Path(ex.__file__).resolve().parents[2]
    for r in map(_j.loads, open(repo / "bench/data/cv_real/ja.jsonl", encoding="utf-8")):
        if r["path"] in clip2spk:
            bad_spk.add(clip2spk[r["path"]])
    for r in map(_j.loads, open(WORK / "r19pool" / "cv_dev.jsonl", encoding="utf-8")):
        p = r["stem"] + ".mp3"
        if p in clip2spk:
            bad_spk.add(clip2spk[p])
    new: dict[str, float] = {}
    per_version = {}
    for v in ("15_0", "17_0", "19_0", "21_0"):
        n_v = 0
        for s in ("train", "dev", "test"):
            p = RAW / "cv_older" / v / f"{s}.tsv"
            if not p.exists():
                continue
            for r in read_tsv(p):
                path = r["path"]
                if path in have or path in new or path not in validated:
                    continue
                vr = validated[path]
                if vr["client_id"] in bad_spk or norm_sentence(vr["sentence"]) in bad_sent:
                    continue
                if int(vr["down_votes"] or 0) > 0 or _BAD_TEXT.search(vr["sentence"]):
                    continue
                d = dur.get(path, 0)
                if 1.0 <= d <= 15.0:
                    new[path] = d
                    n_v += 1
        per_version[v] = n_v
    out = {"new_clips": len(new), "new_hours": round(sum(new.values()) / 3600, 2), "first_seen_in": per_version}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
