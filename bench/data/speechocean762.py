"""speechocean762 (Apache-2.0): 5000 English utterances read by 250 Mandarin L1 speakers,
train 2500 / test 2500, scored by 5 experts. Source: HF ``mispeech/speechocean762`` parquet.

    python -m bench.data.speechocean762

labels schema (all human scores kept verbatim from the HF release):
    sentence: {accuracy 0-10 int, fluency 0-10 int, prosodic 0-10 int,
               completeness 0-10 float (share of words read, x10), total 0-10 int}
    words: [ {text (upper case), accuracy 0-10, stress 0-10, total 0-10,
              phones: [ARPAbet with stress digit, canonical, e.g. "AA0"],
              phones_accuracy: [0-2 float per phone, average of experts;
                                2 = correct, 1 = accented/heavy accent, 0 = wrong/missing],
              mispronunciations: [{index: phone index in the word, canonical_phone,
                                   pronounced_phone (ARPAbet, "<DEL>" = deleted, "<unk>" = unclear)}]} ]
    age: int, gender: "m"/"f", phone_set: "arpabet"
"""
from __future__ import annotations

from pathlib import Path

from .common import (PartWriter, check_disk, data_dir, decode_audio, make_record, merge_parts,
                     open_remote_parquet, summarize, write_flac)

NAME = "speechocean762"
URL = "https://huggingface.co/api/datasets/mispeech/speechocean762/parquet/default/{split}/0.parquet"
SPLITS = ("train", "test")


def _labels(row: dict) -> dict:
    words = []
    for w in row["words"]:
        words.append({
            "text": w["text"],
            "accuracy": w["accuracy"],
            "stress": w["stress"],
            "total": w["total"],
            "phones": list(w["phones"]),
            "phones_accuracy": [float(a) for a in w["phones-accuracy"]],
            "mispronunciations": [
                {"index": m["index"], "canonical_phone": m["canonical-phone"],
                 "pronounced_phone": m["pronounced-phone"]}
                for m in (w.get("mispronunciations") or [])
            ],
        })
    return {
        "sentence": {k: row[k] for k in ("accuracy", "fluency", "prosodic", "completeness", "total")},
        "words": words,
        "age": row["age"],
        "gender": row["gender"],
        "phone_set": "arpabet",
    }


def fetch_split(split: str) -> None:
    root = data_dir(NAME)
    pw = PartWriter(root, split)
    if pw.done:
        print(f"{split}: already done ({len(pw.records)})")
        return
    check_disk()
    pf = open_remote_parquet(URL.format(split=split))
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg)
        for row in t.to_pylist():
            stem = Path(row["audio"]["path"]).stem
            uid = f"so762-{stem}"
            if uid in pw.ids:
                continue
            out = root / "audio" / split / f"{stem}.flac"
            x, sr = decode_audio(row["audio"]["bytes"])
            dur = write_flac(out, x, sr)
            pw.add(make_record(uid, out, "en", row["text"], split, row["speaker"], dur, _labels(row)))
        print(f"{split}: row group {rg + 1}/{pf.metadata.num_row_groups} -> {len(pw.records)}")
        del t
    pw.finish()


def main() -> None:
    for s in SPLITS:
        fetch_split(s)
    merge_parts(data_dir(NAME), SPLITS)
    from . import load

    print(summarize(load(NAME)))


if __name__ == "__main__":
    main()
