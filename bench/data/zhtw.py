"""Taiwan Mandarin read speech with Traditional-script prompts: Mozilla Common Voice Scripted
Speech 25.0, zh-TW, official ``test`` split (CC0-1.0). Source: HF mirror
``OpenFormosa/common_voice_25_zh-TW`` (parquet; only the first row group is read).

    python -m bench.data.zhtw

400 utterances, at most 8 per contributor (client_id), clips <= 30 s, in file order.
lang = "zh"; text = the Traditional-script prompt `sentence`.
labels: {client_id, sentence_id, up_votes, down_votes, age, gender, accents, variant, locale}
speaker = client_id (anonymous hash).
Note: Common Voice prompts are read by volunteers with mixed microphones; Taiwan accents are
self-reported (`accents`) and not all speakers are from Taiwan.
"""
from __future__ import annotations

from collections import Counter

from .common import (PartWriter, check_disk, data_dir, decode_audio, make_record, merge_parts,
                     open_remote_parquet, summarize, write_flac)

NAME = "zhtw"
URL = "https://huggingface.co/api/datasets/OpenFormosa/common_voice_25_zh-TW/parquet/default/test/0.parquet"
N = 400
PER_SPEAKER = 8


def main() -> None:
    root = data_dir(NAME)
    pw = PartWriter(root, "test")
    if not pw.done and len(pw.records) < N:
        check_disk()
        per = Counter(r["speaker"] for r in pw.records)
        pf = open_remote_parquet(URL)
        cols = ["audio", "client_id", "path", "sentence_id", "sentence", "up_votes", "down_votes",
                "age", "gender", "accents", "variant", "locale", "duration_ms"]
        for rg in range(pf.metadata.num_row_groups):
            if len(pw.records) >= N:
                break
            for row in pf.read_row_group(rg, columns=cols).to_pylist():
                if len(pw.records) >= N:
                    break
                stem = row["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
                uid = f"zhtw-{stem}"
                if uid in pw.ids or per[row["client_id"]] >= PER_SPEAKER:
                    continue
                if row["duration_ms"] and row["duration_ms"] > 30000:
                    continue
                x, sr = decode_audio(row["audio"]["bytes"])
                out = root / "audio" / "test" / f"{stem}.flac"
                dur = write_flac(out, x, sr)
                labels = {k: row[k] for k in ("client_id", "sentence_id", "up_votes", "down_votes",
                                              "age", "gender", "accents", "variant", "locale")}
                pw.add(make_record(uid, out, "zh", row["sentence"], "test", row["client_id"], dur, labels))
                per[row["client_id"]] += 1
        pw.finish()
    merge_parts(root, ["test"])
    from . import load

    recs = load(NAME)
    print(summarize(recs), "speakers:", len({r["speaker"] for r in recs}))


if __name__ == "__main__":
    main()
