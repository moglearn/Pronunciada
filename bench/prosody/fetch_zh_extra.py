"""Extra Mandarin TRAINING data for the tone model (labels come from our front-end).

    .venv/bin/python -m bench.prosody.fetch_zh_extra aishell1 [n_speakers=300] [per_speaker=100]
    .venv/bin/python -m bench.prosody.fetch_zh_extra cvtw [per_speaker=50]

aishell1: HF AISHELL/AISHELL-1 (Apache-2.0 per the HF card), per-speaker archives
  data_aishell/wav/SXXXX.tar.gz of the corpus' TRAIN speakers (S0002-S0723), downloaded one by
  one, the first `per_speaker` utterances kept, archive deleted. AISHELL-1 and AISHELL-3 are
  separate recordings with their own speaker ids (BAC009S.. vs SSB..).
cvtw: Common Voice 25.0 zh-TW official TRAIN split (CC0-1.0, HF OpenFormosa/common_voice_25_zh-TW),
  contributors present in our zhtw test set excluded.
Output: ~/.cache/pronunciada/data/<name>/manifest.jsonl (+ audio/*.flac), records as in bench.data.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from pronunciada.acoustic import cache_root

DATA = Path(cache_root()) / "data"
HF = "https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/main/data_aishell"


def _write(recs, root):
    with open(root / "manifest.jsonl.tmp", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(root / "manifest.jsonl.tmp", root / "manifest.jsonl")


def aishell1(n_spk=300, per=100):
    root = DATA / "aishell1"
    (root / "audio").mkdir(parents=True, exist_ok=True)
    tp = root / "transcript.txt"
    if not tp.exists():
        urllib.request.urlretrieve(f"{HF}/transcript/aishell_transcript_v0.8.txt", tp)
    text = {}
    for line in open(tp, encoding="utf-8"):
        p = line.split()
        if p:
            text[p[0]] = "".join(p[1:])
    recs = [json.loads(l) for l in open(root / "manifest.jsonl", encoding="utf-8")] if (root / "manifest.jsonl").exists() else []
    done = {r["speaker"] for r in recs}
    tmp = root / "tmp"
    tmp.mkdir(exist_ok=True)
    for k in range(2, 724):
        if len(done) >= n_spk:
            break
        spk = f"S{k:04d}"
        if spk in done:
            continue
        arc = tmp / f"{spk}.tar.gz"
        r = subprocess.run(["curl", "-sfL", "-o", str(arc), f"{HF}/wav/{spk}.tar.gz"])
        if r.returncode != 0:
            arc.unlink(missing_ok=True)
            continue
        n = 0
        with tarfile.open(arc) as t:
            for m in sorted((m for m in t.getmembers() if m.name.endswith(".wav")), key=lambda m: m.name):
                uid = Path(m.name).stem
                if uid not in text:
                    continue
                x, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32")
                assert sr == 16000
                out = root / "audio" / spk / f"{uid}.flac"
                out.parent.mkdir(exist_ok=True)
                sf.write(out, x, 16000, subtype="PCM_16")
                recs.append({"id": f"aishell1-{uid}", "audio": str(out), "lang": "zh", "text": text[uid],
                             "split": "train", "speaker": f"aishell1-{spk}", "dur": round(len(x) / 16000, 3),
                             "labels": {}})
                n += 1
                if n >= per:
                    break
        arc.unlink()
        done.add(f"aishell1-{spk}")
        done.add(spk)
        if len(done) % 40 == 0:
            _write(recs, root)
            print(f"aishell1: {len(done) // 2} speakers, {len(recs)} utts", flush=True)
    _write(recs, root)
    print("aishell1 done", len({r['speaker'] for r in recs}), "speakers", len(recs), "utts",
          round(sum(r["dur"] for r in recs) / 3600, 2), "h")


def aishell1_parquet(n_spk=300, per=100, step=3):
    """More AISHELL-1 train speakers from the HF parquet mirror carlot/AIShell (same corpus, rows
    ordered by speaker): every `step`-th row group is range-read, at most `per` utterances per
    speaker, until `n_spk` speakers are in the manifest. Nothing but the kept clips is stored."""
    from bench.data.common import open_remote_parquet

    root = DATA / "aishell1"
    recs = [json.loads(l) for l in open(root / "manifest.jsonl", encoding="utf-8")]
    cnt = Counter(r["speaker"] for r in recs)
    d = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/carlot/AIShell"))
    names = sorted(x["rfilename"] for x in d["siblings"] if "/train-" in x["rfilename"])
    for name in names:
        if len(cnt) >= n_spk:
            break
        pf = open_remote_parquet("https://huggingface.co/datasets/carlot/AIShell/resolve/main/" + name)
        for rg in range(0, pf.metadata.num_row_groups, step):
            if len(cnt) >= n_spk:
                break
            rows = pf.read_row_group(rg).to_pylist()
            for row in rows:
                uid = row["audio"]["path"].rsplit(".", 1)[0]
                spk = "aishell1-" + uid[6:11]
                if cnt[spk] >= per or (spk not in cnt and len(cnt) >= n_spk):
                    continue
                x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                assert sr == 16000
                out = root / "audio" / uid[6:11] / f"{uid}.flac"
                out.parent.mkdir(exist_ok=True)
                sf.write(out, x, 16000, subtype="PCM_16")
                recs.append({"id": f"aishell1-{uid}", "audio": str(out), "lang": "zh",
                             "text": row["transcription"].replace(" ", ""), "split": "train", "speaker": spk,
                             "dur": round(len(x) / 16000, 3), "labels": {"source": "carlot/AIShell"}})
                cnt[spk] += 1
        _write(recs, root)
        print(name, len(cnt), "speakers", len(recs), "utts", flush=True)
    _write(recs, root)
    print("aishell1 done", len(cnt), "speakers", len(recs), "utts",
          round(sum(r["dur"] for r in recs) / 3600, 2), "h")


def cvtw(per=50):
    import pyarrow.parquet as pq

    from bench.data import load

    root = DATA / "cvtw"
    (root / "audio").mkdir(parents=True, exist_ok=True)
    pqf = root / "train-0.parquet"
    if not pqf.exists():
        urllib.request.urlretrieve("https://huggingface.co/api/datasets/OpenFormosa/common_voice_25_zh-TW/"
                                   "parquet/default/train/0.parquet", pqf)
    test_spk = {r["speaker"] for r in load("zhtw")}
    test_sent = {r["labels"]["sentence_id"] for r in load("zhtw")}
    per_spk = Counter()
    recs = []
    pf = pq.ParquetFile(pqf)
    cols = ["audio", "client_id", "path", "sentence_id", "sentence", "duration_ms"]
    skipped = Counter()
    for rg in range(pf.metadata.num_row_groups):
        for row in pf.read_row_group(rg, columns=cols).to_pylist():
            if row["client_id"] in test_spk:
                skipped["test_speaker"] += 1
                continue
            if row["sentence_id"] in test_sent:
                skipped["test_sentence"] += 1
                continue
            if per_spk[row["client_id"]] >= per or (row["duration_ms"] or 0) > 30000:
                skipped["cap"] += 1
                continue
            import subprocess as sp

            stem = row["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            raw = row["audio"]["bytes"]
            p = sp.run(["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                       input=raw, capture_output=True)
            x = np.frombuffer(p.stdout, dtype=np.float32)
            if len(x) < 1600:
                skipped["decode"] += 1
                continue
            out = root / "audio" / f"{stem}.flac"
            sf.write(out, x, 16000, subtype="PCM_16")
            recs.append({"id": f"cvtw-{stem}", "audio": str(out), "lang": "zh", "text": row["sentence"],
                         "split": "train", "speaker": row["client_id"], "dur": round(len(x) / 16000, 3),
                         "labels": {"sentence_id": row["sentence_id"]}})
            per_spk[row["client_id"]] += 1
    _write(recs, root)
    pqf.unlink()
    print("cvtw done", len(per_spk), "speakers", len(recs), "utts", round(sum(r["dur"] for r in recs) / 3600, 2),
          "h; skipped", dict(skipped))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "aishell1":
        aishell1(*(int(x) for x in a[1:]))
    elif a[0] == "aishell1_parquet":
        aishell1_parquet(*(int(x) for x in a[1:]))
    else:
        cvtw(*(int(x) for x in a[1:]))
