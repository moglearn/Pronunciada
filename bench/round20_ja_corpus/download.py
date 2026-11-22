"""Downloads for the Japanese length corpus (all resumable, all streamed).

    python -m bench.round20_ja_corpus.download cv [--shards train,dev,test,other_0,...]
        Common Voice 22.0 ja tar shards from the CC0 mirror (fsicoli/common_voice_22_0, pinned
        revision); only clips listed in <work>/cv_select.jsonl are written, as the original mp3
        bytes, to <work>/audio/cv/<cv_split>/<clip>.mp3 (no second copy).
    python -m bench.round20_ja_corpus.download fleurs
        FLEURS ja_jp train (CC BY 4.0, google/fleurs, pinned revision), 16 kHz FLAC, minus any
        sentence of FLEURS validation / test (sentence-disjoint by design; checked).
    python -m bench.round20_ja_corpus.download kokoro-list    # mp3 list for fetch_kokoro.sh
    python -m bench.round20_ja_corpus.download kokoro         # cut clips, 16 kHz FLAC
        Kokoro Speech Dataset v1.3 metadata (public domain) over the LibriVox mp3 chapters fetched
        by fetch_kokoro.sh; sample offsets are at 22050 Hz as in the dataset's extract.py.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

from bench.cv_build import norm_sentence
from bench.round20_ja_corpus.common import AUDIO, CV_URL, RAW, WORK

FLEURS_REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_URL = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REV}/data/ja_jp"
CV_SHARDS = ["train/ja_train_0", "dev/ja_dev_0", "test/ja_test_0"] + [f"other/ja_other_{k}" for k in range(7)]
# older releases on the same mirror family (CC0 per their cards), pinned; only validated-in-22.0 clips
OLDER_REV = {"15_0": "09f19a8a98229c26c2c6eaf5cb6419636cc1497d", "17_0": "8262c16bf297c87a9cd88c51997c4758ed7a8ba2",
             "19_0": "590c8abec6cf7c8d06e650f1438e60332a796e11", "21_0": "6516e5e8eef5f4dee99f431f502959ac78821909"}
OLDER_SHARDS = [f"v{v}_{s}" for v in OLDER_REV for s in ("train", "dev", "test")]


def _stream_tar(url: str, mode: str = "r|"):
    import requests

    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    r.raw.decode_content = True
    return r, tarfile.open(fileobj=r.raw, mode=mode)


def cv(shards: list[str]) -> None:
    rows = [json.loads(x) for x in open(WORK / "cv_select.jsonl", encoding="utf-8")]
    want: dict[str, dict[str, str]] = {}
    for r in rows:
        want.setdefault(r["cv_split"], {})[r["path"]] = r["cv_split"]
    for sh in shards:
        if sh.startswith("v"):            # older release: v15_0_train -> its own mirror, pinned revision
            ver, s = sh[1:].rsplit("_", 1)
            split = sh
            url = (f"https://huggingface.co/datasets/fsicoli/common_voice_{ver}/resolve/{OLDER_REV[ver]}"
                   f"/audio/ja/{s}/ja_{s}_0.tar")
        else:
            split = sh.split("/")[0]
            url = f"{CV_URL}/audio/ja/{sh}.tar"
        done = WORK / "logs" / f"cv_{sh.replace('/', '_')}.done"
        if done.exists():
            continue
        out = AUDIO / "cv" / split
        out.mkdir(parents=True, exist_ok=True)
        todo = {p for p in want.get(split, {}) if not (out / p).exists()}
        if not todo:
            continue
        n = 0
        r, t = _stream_tar(url)
        with r, t:
            for m in t:
                if not m.isfile():
                    continue
                name = m.name.rsplit("/", 1)[-1]
                if name not in todo:
                    continue
                data = t.extractfile(m).read()
                tmp = out / (name + ".part")
                tmp.write_bytes(data)
                tmp.rename(out / name)
                n += 1
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(str(n))
        print(sh, "wrote", n, file=sys.stderr, flush=True)


def fleurs() -> None:
    import soundfile as sf

    out = AUDIO / "fleurs"
    out.mkdir(parents=True, exist_ok=True)
    bad = set()
    for s in ("dev", "test"):
        for line in open(RAW / "fleurs" / f"{s}.tsv", encoding="utf-8"):
            bad.add(norm_sentence(line.split("\t")[2]))
    rows = {}
    dropped = 0
    for line in open(RAW / "fleurs" / "train.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if norm_sentence(p[2]) in bad:
            dropped += 1
            continue
        rows[p[1]] = {"fleurs_id": p[0], "file": p[1], "text": p[2], "gender": p[-1], "n": int(p[-2])}
    r, t = _stream_tar(f"{FLEURS_URL}/audio/train.tar.gz", mode="r|gz")
    n = 0
    with r, t:
        for m in t:
            if not m.isfile():
                continue
            name = m.name.rsplit("/", 1)[-1]
            if name not in rows:
                continue
            x, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32", always_2d=True)
            assert sr == 16000, sr
            dst = out / (name.rsplit(".", 1)[0] + ".flac")
            sf.write(dst, x.mean(axis=1), 16000, format="FLAC", subtype="PCM_16")
            rows[name]["audio"] = str(dst.relative_to(WORK))
            rows[name]["dur"] = len(x) / 16000
            n += 1
    with open(WORK / "fleurs_select.jsonl", "w", encoding="utf-8") as f:
        for v in rows.values():
            if "audio" in v:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print("fleurs train wrote", n, "dropped by sentence", dropped, file=sys.stderr)


def kokoro_list() -> None:
    meta = RAW / "kokoro_meta"
    out = []
    for b in json.load(open(meta / "index.json")):
        ident = b["archive_url"].rstrip("/").split("/")[-2]
        files = sorted({ln.split("|")[1] for ln in open(meta / f"{b['id']}.metadata.txt", encoding="utf-8")})
        out += [f"{b['id']}\t{ident}\t{f}" for f in files]
    (meta / "mp3_list.tsv").write_text("\n".join(out) + "\n")
    print(len(out), "files")


def _decode(path: Path, sr: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
                          "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def kokoro() -> None:
    import soundfile as sf

    from bench.data.common import resample

    meta = RAW / "kokoro_meta"
    out = AUDIO / "kokoro"
    out.mkdir(parents=True, exist_ok=True)
    bad = set()
    for s in ("dev", "test"):
        for line in open(RAW / "fleurs" / f"{s}.tsv", encoding="utf-8"):
            bad.add(norm_sentence(line.split("\t")[2]))
    rows, missing = [], set()
    for b in json.load(open(meta / "index.json")):
        cur, y = None, None
        for line in open(meta / f"{b['id']}.metadata.txt", encoding="utf-8"):
            cid, f, a, e, text, voca = line.rstrip("\r\n").split("|")
            src = RAW / "kokoro_mp3" / b["id"] / f
            if not src.exists():
                missing.add(str(f))
                continue
            dst = out / b["id"] / f"{cid}.flac"
            if not dst.exists():
                if cur != f:
                    y, cur = _decode(src, 22050), f
                seg = y[int(a):int(e)]
                if len(seg) < 22050:
                    continue
                dst.parent.mkdir(exist_ok=True)
                sf.write(dst, resample(seg, 22050), 16000, format="FLAC", subtype="PCM_16")
            rows.append({"id": f"kokoro-{cid}", "book": b["id"], "text": text.replace(" ", ""),
                         "reading": voca, "audio": str(dst.relative_to(WORK)), "dur": (int(e) - int(a)) / 22050,
                         "sentence_excluded": norm_sentence(text) in bad})
    with open(WORK / "kokoro_select.jsonl", "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("kokoro clips", len(rows), "missing mp3", len(missing), file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("cv", "fleurs", "kokoro-list", "kokoro"))
    ap.add_argument("--shards", default=",".join(CV_SHARDS + OLDER_SHARDS))
    a = ap.parse_args(argv)
    if a.cmd == "cv":
        cv(a.shards.split(","))
    elif a.cmd == "fleurs":
        fleurs()
    elif a.cmd == "kokoro-list":
        kokoro_list()
    else:
        kokoro()


if __name__ == "__main__":
    main()
