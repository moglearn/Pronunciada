"""Round 19 (Korean): Common Voice ko clips for the dev set, disjoint from every held-out set.

    python -m bench.round19ko.cv_pool select     # metadata only -> $CACHE/round19/korean/cv_pool.json
    python -m bench.round19ko.cv_pool fetch      # stream the needed tar shards, keep only the pool (16 kHz FLAC)

Source: Common Voice Corpus 22.0 (CC0-1.0), the mirror and revision of bench/cv_build.py; the TSVs
already read by cv_build (dev / test / other; invalidated is not used).
Excluded: (1) every speaker (client_id) of the Lingolix Korean benchmark clips (197 of 200 clip ids are
in the TSVs; the 3 others are not in the pool either); (2) every speaker of the cv-real ko test half;
(3) every sentence that is a Lingolix Korean prompt or target. What remains is the dev pool.
Audio goes to $CACHE/round19/korean/cv_audio/<stem>.flac (a new folder; nothing existing is touched).
"""
from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

from bench import cv_build as CB

ROOT = Path(__file__).resolve().parents[2]


def work() -> Path:
    d = CB.cache_root() / "round19" / "korean"
    d.mkdir(parents=True, exist_ok=True)
    return d


def audio_dir() -> Path:
    d = work() / "cv_audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rows():
    meta = CB.cache_root() / "data" / "cv-real" / "meta" / "ko"
    rows = {}
    for f in ("dev", "test", "other", "invalidated"):
        with open(meta / f"{f}.tsv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r["_split"] = f
                rows[r["path"]] = r
    return rows


def select() -> dict:
    from bench.third_party import lingolix as LX

    rows = _rows()
    man = LX.tests()["tier1_ko"]
    bench_spk = {rows[m["cv_clip"]]["client_id"] for m in man if m["cv_clip"] in rows}
    bench_sent = {m["prompt"].strip() for m in man} | {m["target"].strip() for m in man}
    smap = json.loads((CB.cache_root() / "speaker_map" / "ko.json").read_text())
    cvr = [json.loads(line) for line in (ROOT / "bench" / "data" / "cv_real" / "ko.jsonl").open(encoding="utf-8")]
    test_pseudo = {r["speaker"] for r in cvr if r["half"] == "test"}
    test_spk = {cid for cid, ps in smap.items() if ps in test_pseudo}
    durs = {}
    with open(CB.cache_root() / "data" / "cv-real" / "meta" / "ko" / "clip_durations.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            durs[r["clip"]] = int(r["duration[ms]"]) / 1000
    pool, why = [], Counter()
    for r in rows.values():
        if r["_split"] == "invalidated":
            why["invalidated"] += 1
        elif r["client_id"] in bench_spk:
            why["benchmark_speaker"] += 1
        elif r["client_id"] in test_spk:
            why["cvreal_test_speaker"] += 1
        elif r["sentence"].strip() in bench_sent:
            why["benchmark_sentence"] += 1
        elif CB.text_rule(r["sentence"], "ko") or CB.dur_rule(durs.get(r["path"])):
            why["text_or_duration_rule"] += 1
        else:
            pool.append({"stem": r["path"].rsplit(".", 1)[0], "split": r["_split"], "client_id": r["client_id"],
                         "sentence": r["sentence"].strip(), "dur": durs.get(r["path"]),
                         "up": int(r["up_votes"] or 0), "down": int(r["down_votes"] or 0)})
    spk = sorted({p["client_id"] for p in pool})
    lab = {c: f"cvk_{i:03d}" for i, c in enumerate(spk)}
    for p in pool:
        p["speaker"] = lab[p.pop("client_id")]
    out = {"source": f"{CB.REPO}@{CB.REV[:12]}", "excluded": dict(why), "n": len(pool),
           "speakers": len(spk), "pool": sorted(pool, key=lambda p: p["stem"])}
    (work() / "cv_pool.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print(json.dumps({k: v for k, v in out.items() if k != "pool"}), Counter(p["split"] for p in pool))
    return out


def fetch() -> None:
    import requests
    import soundfile as sf

    from bench.data.common import resample, write_flac

    pool = json.loads((work() / "cv_pool.json").read_text())["pool"]
    by = {}
    for p in pool:
        by.setdefault(p["split"], set()).add(p["stem"])
    stats_p = work() / "cv_audio_stats.json"
    stats = json.loads(stats_p.read_text()) if stats_p.exists() else {}
    for split, stems in sorted(by.items()):
        todo = {s for s in stems if not (audio_dir() / f"{s}.flac").exists()}
        if not todo:
            continue
        url = CB.shard_url("ko", split, 0)
        print("streaming", url, len(todo), "clips", flush=True)
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
                        stats[stem] = {"error": type(e).__name__}
                        todo.discard(stem)
                        continue
                    stats[stem] = {"active_s": round(CB.active_seconds(x), 3), "dur": round(len(x) / 16000, 3)}
                    write_flac(audio_dir() / f"{stem}.flac", x, 16000)
                    todo.discard(stem)
                    if not todo:
                        break
        stats_p.write_text(json.dumps(stats))
        print(split, "missing", len(todo), flush=True)


if __name__ == "__main__":
    {"select": select, "fetch": fetch}[sys.argv[1]]()
