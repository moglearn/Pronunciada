"""Rebuild the round-19 practice ("dev") sets of planted changed words, unchanged rules, new folder.

    python -m bench.round20_native_fa.rebuild --lang de     # bench/round19/devset.py
    python -m bench.round20_native_fa.rebuild --lang en     # bench/round19_rollout/sources.py + devset.py (also es, fr)
    python -m bench.round20_native_fa.rebuild --lang ko     # bench/round19ko/cv_pool.py + plant.py (asr_check separately)

The round-19 builders are run as they are; only their work folders are pointed at
$PRONUNCIADA_CACHE/round20/nativefa/<set>/ (the round-19 caches were deleted). Same seeds, same
exclusions (no Lingolix clip, speaker or prompt; no cv-real test-half speaker; FLEURS test never read).
The Lingolix manifests are read only by the builders' exclusion code; nothing from them is printed.
Output per language: <work>/dev_manifest.jsonl (de, en, es, fr) or dev_items.jsonl (ko).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
ROOT = CACHE / "round20" / "nativefa"


def german_work() -> Path:
    return ROOT / "german"


def rollout_root() -> Path:
    return ROOT / "rollout"


def korean_work() -> Path:
    return ROOT / "korean"


def patch_german():
    import bench.round19.devset as D

    D.WORK = german_work()
    D.CV_AUDIO = D.WORK / "cv_audio"
    D.FLEURS_AUDIO = D.WORK / "fleurs_val_audio"
    for p in (D.WORK, D.CV_AUDIO, D.FLEURS_AUDIO, D.WORK / "cvmeta"):
        p.mkdir(parents=True, exist_ok=True)
    return D


def patch_rollout():
    import bench.round19_rollout.sources as S

    S.ROOT = rollout_root()
    return S


def patch_korean():
    import bench.round19ko.asr_check as A
    import bench.round19ko.cv_pool as C
    import bench.round19ko.plant as P

    def work() -> Path:
        d = korean_work()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def audio_dir() -> Path:
        d = work() / "cv_audio"
        d.mkdir(parents=True, exist_ok=True)
        return d

    C.work, C.audio_dir = work, audio_dir
    P.work, P.audio_dir = work, audio_dir
    A.work = work
    P.vocabulary = _ko_vocabulary
    return C, P, A


def _ko_vocabulary():
    """plant.vocabulary with the Common Voice TSV folder taken from the cache root (the round-19 code
    derived it from its own work folder, which now sits one level deeper)."""
    import csv
    from collections import Counter

    from bench.pipeline.common import records
    from bench.round19ko.plant import _HANGUL, _STRIP
    from bench.third_party import lingolix as LX

    bad = set()
    for m in LX.tests()["tier1_ko"]:
        for t in (m["prompt"], m["target"]):
            bad |= {_STRIP.sub("", x) for x in t.split()}
    voc = Counter()
    texts = []
    for split in ("train", "test"):
        texts += [r["text"] for r in records("fleurs", split=split, lang="ko")]
    texts += [r["text"] for r in records("fleurs-validation", lang="ko")]
    meta = CACHE / "data" / "cv-real" / "meta" / "ko"
    for f in ("dev", "test", "other", "invalidated"):
        with open(meta / f"{f}.tsv", encoding="utf-8") as fh:
            texts += [r["sentence"] for r in csv.DictReader(fh, delimiter="\t")]
    for t in texts:
        for tok in t.split():
            w = _STRIP.sub("", tok)
            if _HANGUL.match(w) and w not in bad:
                voc[w] += 1
    return voc


def _german_validated() -> None:
    """cvmeta/validated.tsv of Common Voice 22.0 de (the builder's word counts and speaker lookup)."""
    import requests

    from bench.cv_build import HF

    dst = german_work() / "cvmeta" / "validated.tsv"
    if dst.exists():
        return
    tmp = dst.with_suffix(".part")
    with requests.get(f"{HF}/transcript/de/validated.tsv", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dst)


def fetch_dev(lang: str) -> None:
    """sources.cmd_fetch restricted to the "dev" speakers (the training / pool clips are not needed here)."""
    import io
    import json
    import tarfile

    import requests
    import soundfile as sf

    from bench.cv_build import n_shards, shard_url
    from bench.data.common import resample, write_flac

    S = patch_rollout()
    sel = json.loads((S.work(lang) / "cv_select.json").read_text())
    todo = {c["stem"] for c in sel["dev"] if not (S.cv_audio_dir(lang) / f"{c['stem']}.flac").exists()}
    print(lang, "to fetch", len(todo), flush=True)
    try:
        ns = int(n_shards(S.locale(lang)).get("dev", 1))
    except Exception:  # noqa: BLE001
        ns = 1
    for k in range(ns):
        if not todo:
            break
        with requests.get(shard_url(S.locale(lang), "dev", k), stream=True, timeout=120) as r:
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
                    write_flac(S.cv_audio_dir(lang) / f"{stem}.flac", resample(x.mean(axis=1), sr), 16000)
                    todo.discard(stem)
                    if not todo:
                        break
        print(lang, "shard", k, "left", len(todo), flush=True)
    print(lang, "missing after fetch", len(todo), flush=True)


def run(lang: str, step: str) -> None:
    if lang == "de":
        D = patch_german()
        steps = ["meta", "speakers", "select", "fetch", "vocab", "build"] if step == "all" else [step]
        for s in steps:
            print("de", s, flush=True)
            if s == "meta":
                _german_validated()
            else:
                {"speakers": D.cmd_speakers, "select": D.cmd_select, "fetch": D.cmd_fetch, "vocab": D.cmd_vocab,
                 "build": D.cmd_build}[s]()
    elif lang == "ko":
        C, P, A = patch_korean()
        steps = ["select", "fetch", "plant"] if step == "all" else [step]
        for s in steps:
            print("ko", s, flush=True)
            {"select": C.select, "fetch": C.fetch, "plant": P.main, "asr": A.main}[s]()
    else:
        S = patch_rollout()
        import bench.round19_rollout.devset as V

        steps = ["select", "fetch", "vocab", "build"] if step == "all" else [step]
        for s in steps:
            print(lang, s, flush=True)
            {"select": S.cmd_select, "fetch": fetch_dev, "vocab": V.cmd_vocab, "build": V.cmd_build}[s](lang)


def manifest_path(lang: str) -> Path:
    if lang.endswith("@fresh"):
        return ROOT / "fresh" / lang.split("@")[0] / "dev_manifest.jsonl"
    if lang == "de":
        return german_work() / "dev_manifest.jsonl"
    if lang == "ko":
        return korean_work() / "dev_items.jsonl"
    return rollout_root() / lang / "dev_manifest.jsonl"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--step", default="all")
    a = ap.parse_args(argv)
    for l in a.lang.split(","):
        run(l, a.step)


if __name__ == "__main__":
    main()
