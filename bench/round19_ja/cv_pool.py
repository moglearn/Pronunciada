"""Common Voice ja clips for round 19: a dev pool (CV `dev` split) and a fitting pool (CV `train`
split), both away from every held-out set.

    python -m bench.round19_ja.cv_pool select     # metadata only -> <work>/cv_{dev,fit}.jsonl
    python -m bench.round19_ja.cv_pool fetch      # stream the two tar shards, keep the selected clips

Source: Mozilla Common Voice Corpus 22.0 (CC0-1.0), the Hugging Face mirror and revision that
bench/cv_build.py uses. Rules (all pools):
* no speaker of the Lingolix test-1 Japanese items (their 200 clips mapped to client ids through
  validated.tsv: 184 speakers) and no clip with one of their prompts;
* no speaker in the cv-real TEST half (bench.cv_build.speaker_half(client_id) == "test"), whatever
  split the speaker's clips are in;
* accepted clips only (up_votes >= 2, down_votes == 0), 2-8 s, prompts of >= 8 characters, no
  digits / Latin letters (the shape of the benchmark's clips: 2.1-7.9 s, 8-40 characters);
* dev pool: CV `dev` split, at most 3 clips per speaker; fit pool: CV `train` split, at most 6 per
  speaker. CV's own splits are speaker-disjoint; this is checked here again.
Audio: 16 kHz FLAC under $PRONUNCIADA_CACHE/round19/japanese/cv_audio/<split>/.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.cv_build import (HF, active_seconds, is_accepted, norm_sentence, speaker_half, text_rule)
from bench.third_party.lingolix import tests as lx_tests

CACHE = Path.home() / ".cache" / "pronunciada"
WORK = CACHE / "round19" / "japanese"
META_LOCAL = CACHE / "data" / "cv-real" / "meta" / "ja"
SEED = 1919
N = {"dev": 680, "fit": 1800, "fit2": 2400}
PER_SPK = {"dev": 3, "fit": 6, "fit2": 8}
SPLIT = {"dev": "dev", "fit": "train", "fit2": "train"}
# fit2 (round 19, part 3): Common Voice ja train CLIPS not in the fit pool (whose natives trained the
# identity head and the length model), for fitting the word step against a large native pool. The
# train split has no further speakers under these rules (the fit pool already holds all 583), so the
# speakers overlap with the fit pool; the clips do not.


def read_tsv(p: Path):
    csv.field_size_limit(1 << 30)
    with open(p, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def durations() -> dict[str, float]:
    out = {}
    with open(META_LOCAL / "clip_durations.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1].strip().isdigit():
                out[p[0]] = int(p[1]) / 1000.0
    return out


def excluded() -> tuple[set[str], set[str]]:
    """(client ids, normalised prompts) of the Lingolix ja test-1 items."""
    items = lx_tests()["tier1_ja"]
    clips = {it["cv_clip"] for it in items}
    prompts = {norm_sentence(it["prompt"]) for it in items} | {norm_sentence(it["target"]) for it in items}
    spk = set()
    for r in read_tsv(WORK / "cvmeta" / "validated.tsv"):
        if r["path"] in clips:
            spk.add(r["client_id"])
    found = sum(1 for r in read_tsv(WORK / "cvmeta" / "validated.tsv") if r["path"] in clips)
    assert found == len(clips), (found, len(clips))
    return spk, prompts


def select() -> None:
    bad_spk, bad_prompts = excluded()
    dur = durations()
    rng = random.Random(SEED)
    out = {}
    used_fit: set = set()
    for pool, split in SPLIT.items():
        if pool == "fit2":
            used_fit = {x["path"] for x in out["fit"]}
        src = META_LOCAL / "dev.tsv" if split == "dev" else WORK / "cvmeta" / "train.tsv"
        by_spk = defaultdict(list)
        for r in read_tsv(src):
            if r["client_id"] in bad_spk or speaker_half(r["client_id"]) == "test":
                continue
            if pool == "fit2" and r["path"] in used_fit:
                continue
            if norm_sentence(r["sentence"]) in bad_prompts or text_rule(r["sentence"], "ja"):
                continue
            if not is_accepted(int(r["up_votes"] or 0), int(r["down_votes"] or 0)):
                continue
            d = dur.get(r["path"])
            if d is None or not 2.0 <= d <= 8.0 or len(norm_sentence(r["sentence"])) < 8:
                continue
            by_spk[r["client_id"]].append(r)
        rows = []
        for spk in sorted(by_spk):
            rs = by_spk[spk][:]
            rng.shuffle(rs)
            rows += rs[:PER_SPK[pool]]
        rng.shuffle(rows)
        rows = rows[:N[pool]]
        out[pool] = rows
        spks = sorted({r["client_id"] for r in rows})
        smap = {s: f"{pool}_spk{k:04d}" for k, s in enumerate(spks)}
        with open(WORK / f"cv_{pool}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                stem = r["path"].rsplit(".", 1)[0]
                f.write(json.dumps({"id": f"cv-{pool}-{stem}", "stem": stem, "split": split,
                                    "speaker": smap[r["client_id"]], "text": r["sentence"],
                                    "dur": dur[r["path"]],
                                    "audio": str(WORK / "cv_audio" / split / f"{stem}.flac")},
                                   ensure_ascii=False) + "\n")
        print(pool, len(rows), "clips", len(spks), "speakers", file=sys.stderr)
    a = {r["client_id"] for r in out["dev"]}
    b = {r["client_id"] for r in out["fit"]}
    assert not (a & b), "dev and fit pools share speakers"
    c = {r["client_id"] for r in out["fit2"]}
    assert not (c & a), "fit2 shares speakers with dev"
    assert not ({r["path"] for r in out["fit2"]} & {r["path"] for r in out["fit"]})
    assert not ((a | b | c) & bad_spk)


def fetch() -> None:
    import requests
    import soundfile as sf

    from bench.data.common import resample, write_flac

    for pool, split in SPLIT.items():
        rows = [json.loads(x) for x in open(WORK / f"cv_{pool}.jsonl", encoding="utf-8")]
        todo = {r["stem"] for r in rows if not Path(r["audio"]).exists()}
        if not todo:
            continue
        url = f"{HF}/audio/ja/{split}/ja_{split}_0.tar"
        import tarfile

        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            r.raw.decode_content = True
            with tarfile.open(fileobj=r.raw, mode="r|") as t:
                for m in t:
                    if not m.isfile():
                        continue
                    stem = m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                    if stem not in todo:
                        continue
                    data = t.extractfile(m).read()
                    try:
                        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                        x = resample(x.mean(axis=1), sr)
                    except Exception as e:  # noqa: BLE001
                        print("skip", stem, type(e).__name__, file=sys.stderr)
                        todo.discard(stem)
                        continue
                    if active_seconds(x) < 0.3:
                        todo.discard(stem)
                        continue
                    write_flac(WORK / "cv_audio" / split / f"{stem}.flac", x, 16000)
                    todo.discard(stem)
                    if not todo:
                        break
        print(pool, "missing", len(todo), file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "fetch"))
    a = ap.parse_args(argv)
    {"select": select, "fetch": fetch}[a.cmd]()


if __name__ == "__main__":
    main()
