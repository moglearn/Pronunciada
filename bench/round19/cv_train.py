"""Round 19b (German): Common Voice 22.0 de TRAIN natives for the sound-identity head (CC0).

    python -m bench.round19.cv_train select     # rows of train shard 0 (metadata only)
    python -m bench.round19.cv_train fetch      # stream train shard 0, keep the chosen clips (16 kHz FLAC)

Rules: the first SHARD_ROWS rows of train.tsv (= tar shard 0); 1-12 s; accent not marked non-native;
at most 4 clips per speaker. Excluded: every Lingolix German clip, speaker and prompt, and every
speaker of the cv-real German benchmark (both halves, all three sources) and of the round-19 dev
set (CV splits are speaker-disjoint, so dev speakers cannot occur; checked anyway).
"""
from __future__ import annotations

import io
import json
import random
import sys

from bench.round19.devset import CACHE, CV_META, WORK, _norm, lingolix_items, read_tsv

OUT_AUDIO = WORK / "cv_train_audio"
N = 3000
SHARD_ROWS = 40000


def select():
    from bench.cv_build import nativeness

    L = json.loads((WORK / "lingolix_speakers.json").read_text())
    bad_spk = set(L["speakers"]) | set(json.loads((CACHE / "speaker_map" / "de.json").read_text()))
    for name in ("dev.tsv", "test.tsv"):
        bad_spk |= {r["client_id"] for r in read_tsv(CV_META / name)}
    bad_clip = set(L["clips"])
    bad_sent = {_norm(x["prompt"]) for x in lingolix_items()} | {_norm(x["target"]) for x in lingolix_items()}
    durs = {}
    with open(CV_META / "clip_durations.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1].strip().isdigit():
                durs[p[0]] = int(p[1]) / 1000
    rows = []
    for i, r in enumerate(read_tsv(WORK / "cvmeta" / "train.tsv")):
        if i >= SHARD_ROWS:
            break
        if r["client_id"] in bad_spk or r["path"] in bad_clip or _norm(r["sentence"]) in bad_sent:
            continue
        if nativeness("de", r.get("accents") or "") == "nonnative":
            continue
        d = durs.get(r["path"])
        if d is None or not 1.0 <= d <= 12.0:
            continue
        rows.append({"stem": r["path"].rsplit(".", 1)[0], "speaker": r["client_id"], "sentence": r["sentence"], "dur": d})
    rng = random.Random(1919)
    rng.shuffle(rows)
    per, out = {}, []
    for r in rows:
        if per.get(r["speaker"], 0) >= 4:
            continue
        per[r["speaker"]] = per.get(r["speaker"], 0) + 1
        out.append(r)
        if len(out) >= N:
            break
    (WORK / "cv_train_select.json").write_text(json.dumps(out, ensure_ascii=False))
    print("eligible", len(rows), "selected", len(out), "speakers", len(per))


def fetch():
    import tarfile

    import requests
    import soundfile as sf

    from bench.cv_build import shard_url
    from bench.data.common import resample, write_flac

    sel = json.loads((WORK / "cv_train_select.json").read_text())
    todo = {c["stem"] for c in sel if not (OUT_AUDIO / f"{c['stem']}.flac").exists()}
    print("to fetch", len(todo), flush=True)
    with requests.get(shard_url("de", "train", 0), stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=io.BufferedReader(r.raw, 1 << 20), mode="r|") as t:
            for m in t:
                if not m.isfile():
                    continue
                stem = m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                if stem not in todo:
                    continue
                x, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32", always_2d=True)
                write_flac(OUT_AUDIO / f"{stem}.flac", resample(x.mean(axis=1), sr), 16000)
                todo.discard(stem)
                if not todo:
                    break
    print("missing", len(todo))


if __name__ == "__main__":
    {"select": select, "fetch": fetch}[sys.argv[1]]()
