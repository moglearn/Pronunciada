"""Heroico / West Point Spanish speech (OpenSLR 39 = LDC2006S37): Spanish read by US Military
Academy learners and by native speakers, the same short prompts for both.

    python -m bench.learners.heroico            # fetch (resumable, sha256 recorded) + build + select

Source: https://www.openslr.org/39/ (file LDC2006S37.tar.gz, 2,141,920,338 bytes), which states
"License: apache 2.0". The LDC catalogue entry (https://catalog.ldc.upenn.edu/LDC2006S37) still
lists the LDC User Agreement for Non-Members and the archive's index.html says "All Rights
Reserved". Because the two statements disagree, this suite treats the corpus as held-out
evaluation data only (nothing is fitted on it).

What is used (read speech only, the text is known exactly):
  * USMA subcorpus, head-mounted microphone files ``s<N>.wav`` (throat-microphone ``t<N>.wav``
    excluded): 18 speakers whose directory names mark them ``native`` (8) or ``nonnative`` (10);
    every speaker read the same 206-sentence list (transcripts/usma-prompts.txt).
  * HEROICO subcorpus (Mexican Military Academy cadets, all native): only the recordings of the
    206 USMA sentences (prompt ids 355-560 of transcripts/heroico-recordings.txt) - they are the
    native counterparts of the learner prompts. Its other read sentences (no learner reads them)
    and the free-answer recordings (Answers_Spanish, not read speech) are excluded.
The directory names carry further fields (e.g. ``nonnative-m-mic2-152-34-70-3.3-tdn9604``) that
the documentation does not define; they are kept verbatim in ``labels.dir_fields`` and not used.
There is no proficiency level and no expert rating.

HEADER_RATE_NOTE: 14 of the 18 USMA speaker folders have WAV headers saying 20050 Hz (the others,
and all HEROICO files, say 22050 Hz). 20050 is almost certainly a typo for 22050: the Kaldi recipe
for this corpus (egs/heroico/s5/local/usma_*_make_lists.pl) forces ``sox -r 22050`` on every input
file. We do the same (the samples are read as 22050 Hz), so these recordings are not slowed down
by 9 %. Recordings of the 20050-header speakers also have a fixed, prompt-dependent length (the
recording window), so they contain more leading / trailing silence than the others.

Test selection (seed 19, fixed before anything was scored):
  * 60 of the 206 prompts drawn at random among those that at least one HEROICO native also read;
  * every USMA speaker contributes 25 items from these 60 prompts, assigned by rotating through the
    prompt list from a random offset separately for natives and for learners, so each prompt is
    read by several learners and several natives (a speaker missing a prompt takes the next one);
  * 30 of the HEROICO speakers who read at least 5 of the 60 prompts are drawn; each contributes 5
    items (secondary native group, different recording site and equipment).
  Total 18 x 25 + 30 x 5 = 600 items. ``dev.jsonl`` = every item of the HEROICO speakers not drawn
  (speaker-disjoint from the test); the unused items of test speakers are in neither file.
"""
from __future__ import annotations

import json
import random
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench.data.common import write_flac
from bench.learners import downloads, fetch, hours, make_record, root, sha256, write_jsonl

URL = "https://openslr.elda.org/resources/39/LDC2006S37.tar.gz"
SIZE = 2141920338
SHA256 = "e24646cff85e4a3c3be6491affd64a4cc50218d7b84a4f5294a2f0e490f8ed7a"  # measured 2026-11-22
LICENCE = ("OpenSLR 39 states Apache-2.0; LDC catalogue lists the LDC User Agreement for Non-Members. "
           "Used here for held-out evaluation only.")
SEED = 19
N_PROMPTS, PER_USMA, N_HEROICO_SPK, PER_HEROICO = 60, 25, 30, 5
CORPUS = "heroico"


def read_prompts(path: Path) -> dict[str, str]:
    txt = path.read_bytes().decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
    out = {}
    for line in txt.split("\n"):
        if "\t" in line:
            k, v = line.split("\t", 1)
            out[k.strip()] = " ".join(v.split())
    return out


def extract(tar_path: Path, dest: Path) -> Path:
    base = dest / "LDC2006S37"
    if (dest / ".extracted").exists():
        return base
    with tarfile.open(tar_path, "r:gz") as tf:
        def members():
            for m in tf:
                n = m.name
                if not (m.isfile() or m.isdir()) or n.startswith("/") or ".." in Path(n).parts:
                    continue
                if "/Answers_Spanish/" in n:
                    continue
                yield m
        tf.extractall(dest, members=members())
    (dest / ".extracted").touch()
    return base


def parse_dir(name: str) -> dict:
    parts = name.split("-")
    return {"native": parts[0] == "native", "gender": parts[1], "place": parts[2],
            "dir_fields": parts[3:-1], "code": parts[-1]}


def build() -> None:
    dl = downloads(CORPUS)
    tar = fetch(URL, dl / "LDC2006S37.tar.gz", size=SIZE, sha=SHA256)
    base = extract(tar, dl / "raw")
    usma_prompts = read_prompts(base / "data/transcripts/usma-prompts.txt")
    her_prompts = read_prompts(base / "data/transcripts/heroico-recordings.txt")
    by_text = {v: k for k, v in usma_prompts.items()}
    her_to_usma = {k: by_text[v] for k, v in her_prompts.items() if v in by_text}
    out = root(CORPUS)
    excl = Counter()
    recs = []
    import soundfile as sf

    def add(src: Path, rid: str, spk: str, prompt_id: str, labels: dict) -> None:
        dst = out / "audio" / spk / f"{rid}.flac"
        if dst.exists():
            dur = sf.info(str(dst)).duration
        else:
            try:
                x, sr = sf.read(str(src), dtype="float32", always_2d=True)
            except Exception:
                excl["unreadable wav"] += 1
                return
            x = x.mean(axis=1)
            if sr == 20050:  # see HEADER_RATE_NOTE
                sr = 22050
            if len(x) < 0.3 * sr or float(np.abs(x).max()) < 1e-4:
                excl["empty or < 0.3 s"] += 1
                return
            dur = write_flac(dst, x, sr)
        text = usma_prompts[prompt_id]
        recs.append(make_record(rid, dst, "es", text, "pool", spk, dur,
                                {"prompt_id": prompt_id, "source_file": str(src.relative_to(base)), **labels}))

    usma = sorted(p for p in (base / "data/speech/usma").iterdir() if p.is_dir())
    for d in usma:
        info = parse_dir(d.name)
        spk = f"usma-{info['code']}"
        lab = {"native": info["native"], "l1": "spanish" if info["native"] else "english (USMA learner)",
               "subcorpus": "usma", "gender": info["gender"], "place": info["place"],
               "dir_fields": info["dir_fields"], "dir": d.name}
        for w in sorted(d.glob("*.wav")):
            m = re.fullmatch(r"s(\d+)\.wav", w.name)
            if not m:
                excl["usma throat-microphone (t*.wav)" if w.name.startswith("t") else "usma other file"] += 1
                continue
            pid = f"s{m.group(1)}"
            if pid not in usma_prompts:
                excl["usma file without prompt"] += 1
                continue
            add(w, f"heroico-{spk}-{pid}", spk, pid, lab)
    her = base / "data/speech/heroico/Recordings_Spanish"
    for d in sorted((p for p in her.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        spk = f"heroico-{d.name}"
        lab = {"native": True, "l1": "spanish", "subcorpus": "heroico", "place": "Mexico City"}
        for w in sorted(d.rglob("*.wav")):
            if w.parent != d:
                excl["heroico file in a nested folder"] += 1
                continue
            hid = w.stem
            if hid not in her_to_usma:
                excl["heroico read sentence not in the USMA prompt list"] += 1
                continue
            add(w, f"heroico-{spk}-{hid}", spk, her_to_usma[hid], lab)
    tbl = (base / "docs/file.tbl").read_text(errors="replace").split()
    excl["heroico free answers (Answers_Spanish, not read speech; not extracted)"] = sum(
        1 for t in tbl if "/Answers_Spanish/" in t and t.endswith(".wav"))
    write_jsonl(out / "manifest.jsonl", recs)
    test, dev = select(recs)
    for r in test:
        r["split"] = "test"
    for r in dev:
        r["split"] = "dev"
    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "dev.jsonl", dev)
    summary = {
        "corpus": CORPUS, "source": URL, "archive_sha256": SHA256, "archive_bytes": SIZE, "licence": LICENCE,
        "seed": SEED, "pool": {"items": len(recs), "speakers": len({r["speaker"] for r in recs}), "hours": hours(recs)},
        "excluded": dict(excl),
        "test": describe(test), "dev": describe(dev),
    }
    (out / "build.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def describe(recs: list[dict]) -> dict:
    g = defaultdict(list)
    for r in recs:
        g[(r["labels"]["subcorpus"], "native" if r["labels"]["native"] else "learner")].append(r)
    return {f"{a}/{b}": {"items": len(v), "speakers": len({r['speaker'] for r in v}), "hours": hours(v)}
            for (a, b), v in sorted(g.items())} | {"total_items": len(recs), "total_hours": hours(recs),
                                                    "prompts": len({r['labels']['prompt_id'] for r in recs})}


def select(recs: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    have = defaultdict(dict)  # speaker -> prompt -> record
    for r in recs:
        have[r["speaker"]][r["labels"]["prompt_id"]] = r
    her_spk = sorted(s for s in have if s.startswith("heroico-"))
    usma_spk = sorted(s for s in have if s.startswith("usma-"))
    her_prompts = sorted({p for s in her_spk for p in have[s]}, key=lambda p: int(p[1:]))
    prompts = rng.sample(her_prompts, N_PROMPTS)
    test = []
    for native in (True, False):
        group = [s for s in usma_spk if next(iter(have[s].values()))["labels"]["native"] == native]
        off = rng.randrange(N_PROMPTS)
        for s in group:
            got, k = [], off
            while len(got) < PER_USMA and k < off + 4 * N_PROMPTS:
                p = prompts[k % N_PROMPTS]
                if p in have[s] and have[s][p] not in got:
                    got.append(have[s][p])
                k += 1
            off = (off + PER_USMA) % N_PROMPTS
            test.extend(got)
    pset = set(prompts)
    eligible = [s for s in her_spk if len(pset & set(have[s])) >= PER_HEROICO]
    chosen = sorted(rng.sample(eligible, min(N_HEROICO_SPK, len(eligible))))
    for s in chosen:
        ps = sorted(pset & set(have[s]), key=lambda p: int(p[1:]))
        for p in rng.sample(ps, PER_HEROICO):
            test.append(have[s][p])
    dev = [r for s in her_spk if s not in chosen for r in have[s].values()]
    return test, dev


if __name__ == "__main__":
    build()
