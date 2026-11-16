"""FLEURS (CC-BY-4.0) native read speech, 10 languages. Source: HF ``google/fleurs`` parquet.

    python -m bench.data.fleurs [--langs en ja ...]

Only the ``test`` and ``train`` splits are used (``validation`` is reserved as a dev set and is
never read here). Caps per language: test 400, train 1000 (zh train 2500), in file order,
skipping clips > 30 s. Each remote parquet row group is read only while the cap is not reached.

labels: {fleurs_id: sentence id (shared by all speakers of the same sentence),
         transcription: FLEURS normalised transcription, gender: "male"/"female"/"other",
         config: FLEURS config name}
speaker: null (FLEURS does not publish speaker ids).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

from .common import (PartWriter, check_disk, data_dir, decode_audio, make_record, merge_parts,
                     open_remote_parquet, summarize, write_flac)

NAME = "fleurs"
CONFIGS = {"en": "en_us", "zh": "cmn_hans_cn", "ja": "ja_jp", "es": "es_419", "fr": "fr_fr",
           "de": "de_de", "it": "it_it", "pt": "pt_br", "ru": "ru_ru", "ko": "ko_kr"}
URL = "https://huggingface.co/api/datasets/google/fleurs/parquet/{config}/{split}/0.parquet"
SPLITS = ("test", "train")
MAX_DUR = 30.0
GENDER = {0: "male", 1: "female", 2: "other"}


def cap(lang: str, split: str) -> int:
    if split == "test":
        return 400
    return 2500 if lang == "zh" else 1000


def fetch(lang: str, split: str) -> int:
    assert split in SPLITS  # never "validation"
    root = data_dir(NAME)
    pw = PartWriter(root, f"{lang}.{split}")
    n_cap = cap(lang, split)
    if pw.done or len(pw.records) >= n_cap:
        pw.finish()
        return len(pw.records)
    check_disk()
    config = CONFIGS[lang]
    pf = open_remote_parquet(URL.format(config=config, split=split))
    cols = ["id", "num_samples", "audio", "raw_transcription", "transcription", "gender"]
    for rg in range(pf.metadata.num_row_groups):
        if len(pw.records) >= n_cap:
            break
        t = pf.read_row_group(rg, columns=cols)
        for row in t.to_pylist():
            if len(pw.records) >= n_cap:
                break
            stem = row["audio"]["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            uid = f"fleurs-{lang}-{split}-{stem}"
            if uid in pw.ids:
                continue
            if row["num_samples"] / 16000 > MAX_DUR:
                continue
            x, sr = decode_audio(row["audio"]["bytes"])
            if len(x) / sr > MAX_DUR:
                continue
            out = root / "audio" / lang / split / f"{stem}.flac"
            dur = write_flac(out, x, sr)
            labels = {"fleurs_id": row["id"], "transcription": row["transcription"],
                      "gender": GENDER.get(row["gender"], str(row["gender"])), "config": config}
            pw.add(make_record(uid, out, lang, row["raw_transcription"], split, None, dur, labels))
        print(f"{lang}/{split}: row group {rg + 1}/{pf.metadata.num_row_groups} -> {len(pw.records)}",
              flush=True)
        del t
    pw.finish()
    return len(pw.records)


def _job(args):
    lang, split = args
    return lang, split, fetch(lang, split)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    jobs = [(lg, s) for lg in a.langs for s in a.splits]
    with ProcessPoolExecutor(a.workers) as ex:
        for lang, split, n in ex.map(_job, jobs):
            print(f"done {lang}/{split}: {n}", flush=True)
    order = [f"{lg}.{s}" for lg in CONFIGS for s in SPLITS]
    merge_parts(data_dir(NAME), order)
    from . import load

    print(summarize(load(NAME)))


if __name__ == "__main__":
    main()
