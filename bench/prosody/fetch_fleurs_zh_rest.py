"""The rest of FLEURS Mandarin TRAIN (tone v3 training data; never validation or test).

    .venv/bin/python -m bench.prosody.fetch_fleurs_zh_rest
    .venv/bin/python -m bench.prosody.prep fleurs_zh_rest train

bench.data.fleurs caps zh train at the first 2500 clips (in file order). This fetches every other
clip of HF google/fleurs cmn_hans_cn TRAIN (<= 30 s) into ~/.cache/pronunciada/data/fleurs_zh_rest/
(manifest.jsonl + audio/*.flac, records as in bench.data). Sentences that also occur in FLEURS
validation or test (by FLEURS sentence id or by text) are dropped, so dev / test stay unseen.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import soundfile as sf

from bench.data.common import data_dir, decode_audio, open_remote_parquet, read_jsonl, write_flac
from pronunciada.acoustic import cache_root

URL = "https://huggingface.co/api/datasets/google/fleurs/parquet/cmn_hans_cn/train/0.parquet"


def heldout_sentences():
    import pyarrow.parquet as pq

    ids, texts = set(), set()
    val = Path(cache_root()) / "data" / "fleurs" / "cmn_hans_cn" / "validation-0.parquet"
    t = pq.read_table(val, columns=["id", "raw_transcription"]).to_pylist()
    for r in t:
        ids.add(r["id"])
        texts.add(r["raw_transcription"])
    for r in read_jsonl(data_dir("fleurs") / "manifest.jsonl"):
        if r["lang"] == "zh" and r["split"] == "test":
            ids.add(r["labels"]["fleurs_id"])
            texts.add(r["text"])
    return ids, texts


def main():
    have = {r["id"] for r in read_jsonl(data_dir("fleurs") / "manifest.jsonl") if r["lang"] == "zh"}
    bad_ids, bad_texts = heldout_sentences()
    root = data_dir("fleurs_zh_rest")
    (root / "audio").mkdir(parents=True, exist_ok=True)
    pf = open_remote_parquet(URL)
    recs, dropped = [], 0
    cols = ["id", "num_samples", "audio", "raw_transcription", "gender"]
    for rg in range(pf.metadata.num_row_groups):
        for row in pf.read_row_group(rg, columns=cols).to_pylist():
            stem = row["audio"]["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            uid = f"fleurs-zh-train-{stem}"
            if uid in have or row["num_samples"] / 16000 > 30:
                continue
            if row["id"] in bad_ids or row["raw_transcription"] in bad_texts:
                dropped += 1
                continue
            x, sr = decode_audio(row["audio"]["bytes"])
            assert sr == 16000
            out = root / "audio" / f"{stem}.flac"
            dur = write_flac(out, x, sr)
            recs.append({"id": uid, "audio": str(out), "lang": "zh", "text": row["raw_transcription"],
                         "split": "train", "speaker": None, "dur": dur,
                         "labels": {"fleurs_id": row["id"], "config": "cmn_hans_cn"}})
        print(f"row group {rg + 1}/{pf.metadata.num_row_groups}: {len(recs)} new, {dropped} dropped", flush=True)
    with open(root / "manifest.jsonl.tmp", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(root / "manifest.jsonl.tmp", root / "manifest.jsonl")
    print("done", len(recs), "clips", round(sum(r["dur"] for r in recs) / 3600, 2), "h;", dropped, "held-out sentences dropped")


if __name__ == "__main__":
    _ = (io, sf)
    main()
