"""EpaDB (Open Data Commons Attribution 1.0): English read by 50 Argentinian Spanish L1 speakers,
phone-level expert annotations. Source: CONICET institutional repository, dataset handle
11336/150678 (open access, no registration). The 574 MB waveform zip is fetched in
one sequential (resumable) download, extracted to FLAC and then deleted; the small annotation zips
are kept in raw/.

    python -m bench.data.epadb

split: the corpus' own train (30 speakers) / test (20 speakers) lists from its readme.
labels:
    score: annotator 1's overall 1-5 score for the phrase (int or null)
    words: [{text, start, end}]                      (annotator-corrected alignment)
    annotation: [{phone, start, end}]                annotator 1's phone tier ("0" = deleted)
    phones: [{ref, annotated, correct}]              from labels/*.txt: each phone of the best-matching
            reference transcription vs annotator 1 (ARPAbet + extensions such as Th, Kh, DX, BH, RR;
            "0" = deletion); correct = the corpus' own "+" / "-" label
    annotation_2: second annotator's phone tier when it exists (subset of speakers), else absent
"""
from __future__ import annotations

import io
import time
import re
import zipfile

from .common import (PartWriter, check_disk, data_dir, decode_audio, make_record, merge_parts,
                     summarize, write_flac)
from .l2arctic import parse_textgrid

NAME = "epadb"
API = "https://ri.conicet.gov.ar/server/api/core/bitstreams/{}/content"
FILES = {"annotations.zip": "2363bfae-3b17-4f5a-9d8a-00b7a33501d4",
         "labels.zip": "be80f6d9-15c6-4cf9-a55e-ec66eeed0584",
         "transcriptions.zip": "521b6fd0-0146-4424-9cbc-2a23400af077",
         "readme.md": "c4fa776e-6a29-4e78-b47f-4c358db73da4",
         "reference_transcriptions.txt": "1e00101b-453c-4c29-9479-28f9c2853397"}
WAVEFORMS = "81df6663-bcbf-4620-869f-6790247e0a10"
WAVEFORMS_BYTES = 573667269
TRAIN = {14, 15, 17, 18, 22, 25, 27, 29, 30, 31, 32, 33, 34, 35, 41, 42, 45, 46, 50, 52, 55, 56,
         57, 64, 65, 89, 109, 110, 112, 114}
TEST = {28, 43, 51, 60, 61, 62, 66, 67, 68, 69, 70, 71, 82, 83, 90, 95, 103, 105, 111, 124}
UA = {"User-Agent": "Mozilla/5.0"}


def _download_small(raw):
    import requests

    raw.mkdir(parents=True, exist_ok=True)
    for name, uid in FILES.items():
        p = raw / name
        if not p.exists():
            r = requests.get(API.format(uid), headers=UA, timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)


def _download_waveforms(raw):
    """Resumable download of waveforms.zip (the server is slow for many small range reads, so
    one sequential download is ~30x faster). Deleted after extraction."""
    import requests

    dst = raw / "waveforms.zip"
    part = raw / "waveforms.zip.part"
    if dst.exists():
        return dst
    total = WAVEFORMS_BYTES
    for attempt in range(30):
        have = part.stat().st_size if part.exists() else 0
        if have >= total:
            break
        hdr = dict(UA, Range=f"bytes={have}-") if have else UA
        try:
            with requests.get(API.format(WAVEFORMS), headers=hdr, stream=True, timeout=120) as r:
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except (requests.RequestException, OSError) as e:
            print(f"waveforms.zip: {type(e).__name__} at {have / 1e6:.0f} MB, resuming", flush=True)
            time.sleep(5)
    if part.stat().st_size != total:
        raise RuntimeError("waveforms.zip incomplete")
    part.rename(dst)
    return dst


def _tiers(z: zipfile.ZipFile, name: str):
    return parse_textgrid(z.read(name).decode("utf-8", "replace"))


def main() -> None:
    check_disk()
    root = data_dir(NAME)
    raw = root / "raw"
    _download_small(raw)
    ann = zipfile.ZipFile(raw / "annotations.zip")
    lab = zipfile.ZipFile(raw / "labels.zip")
    trn = zipfile.ZipFile(raw / "transcriptions.zip")
    a1 = {n.rsplit("/", 1)[1][:-9]: n for n in ann.namelist() if "/annotations_1/" in n and n.endswith(".TextGrid")}
    a2 = {n.rsplit("/", 1)[1][:-9]: n for n in ann.namelist() if "/annotations_2/" in n and n.endswith(".TextGrid")}
    labs = {n.rsplit("/", 1)[1][:-4]: n for n in lab.namelist() if n.endswith(".txt")}
    labs_txt = {n.rsplit("/", 1)[1][:-4]: n for n in trn.namelist() if n.endswith(".lab")}

    wzip = _download_waveforms(raw)
    wz = zipfile.ZipFile(wzip)
    wavs = {n.rsplit("/", 1)[1].rsplit(".", 1)[0]: n for n in wz.namelist() if n.lower().endswith(".wav")}
    parts = {s: PartWriter(root, s) for s in ("train", "test")}
    for utt in sorted(a1, key=lambda u: (int(re.search(r"spkr(\d+)", u).group(1)), u)):
        spk_n = int(re.search(r"spkr(\d+)", utt).group(1))
        split = "test" if spk_n in TEST else "train" if spk_n in TRAIN else None
        if split is None or utt not in wavs or utt not in labs:
            continue
        uid = f"epadb-{utt}"
        if uid in parts[split].ids:
            continue
        t1 = _tiers(ann, a1[utt])
        words = [{"text": t, "start": round(a, 4), "end": round(b, 4)}
                 for a, b, t in t1.get("words", []) if t.strip() and t not in ("sil", "sp")]
        annotation = [{"phone": t, "start": round(a, 4), "end": round(b, 4)}
                      for a, b, t in t1.get("annotation", []) if t.strip() and t not in ("sil", "sp")]
        score = None
        for _, _, t in t1.get("score", []):
            m = re.match(r"\s*(\d)", t)
            if m:
                score = int(m.group(1))
        phones = []
        for line in lab.read(labs[utt]).decode("utf-8", "replace").splitlines():
            p = line.split()
            if len(p) >= 4:
                phones.append({"ref": p[1], "annotated": p[2], "correct": p[3] == "+"})
        labels = {"score": score, "words": words, "annotation": annotation, "phones": phones}
        if utt in a2:
            t2 = _tiers(ann, a2[utt])
            labels["annotation_2"] = [{"phone": t, "start": round(a, 4), "end": round(b, 4)}
                                      for a, b, t in t2.get("annotation", []) if t.strip() and t not in ("sil", "sp")]
        text = (trn.read(labs_txt[utt]).decode("utf-8", "replace").strip() if utt in labs_txt
                else " ".join(w["text"] for w in words))
        x, sr = decode_audio(wz.read(wavs[utt]))
        out = root / "audio" / f"spkr{spk_n}" / f"{utt}.flac"
        dur = write_flac(out, x, sr)
        parts[split].add(make_record(uid, out, "en", text, split, f"spkr{spk_n}", dur, labels))
        n = sum(len(p.records) for p in parts.values())
        if n % 200 == 0:
            print(n, flush=True)
    for p in parts.values():
        p.finish()
    wz.close()
    wzip.unlink()   # intermediate archive: delete after extraction
    merge_parts(root, ("train", "test"))
    from . import load

    print(summarize(load(NAME)))


if __name__ == "__main__":
    main()
