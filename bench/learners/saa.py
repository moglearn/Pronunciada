"""Speech Accent Archive (George Mason University): English, one fixed paragraph read by native
speakers and by speakers of about 200 other first languages.

    python -m bench.learners.saa            # fetch the selected speakers (resumable, sha256-checked) + build

Source: https://accent.gmu.edu/download/ -> OSF projects https://osf.io/yh23d (speaker_information.xlsx,
MP3) and https://osf.io/e2j5y (lossless FLAC, used here). Public, no account. Licence stated on the
download and about pages: CC BY-NC-SA 4.0 (older mirrors say 2.0) -> held-out evaluation only.
Labels: first language, age, onset age of English, country of English residence, length of
residence, learning style (self-reported). No pronunciation rating (the narrow phonetic
transcriptions exist for some speakers only as images and are not used).

Groups (fixed before anything was scored):
  native  = native_language "english" and no alternative native language;
  learner = native_language not "english" and English onset age >= 5 (early bilinguals with onset
            0-4 are excluded as ambiguous);
  everything else (unknown onset age, english plus another native language) is excluded.
Test selection (seed 19): 100 natives drawn at random, and 100 learners drawn at random with at
most 4 per first language (for diversity of L1). All speakers read the same paragraph, so the
comparison is prompt-matched by construction. ``dev.jsonl`` lists the remaining eligible speakers
(metadata only, audio not downloaded); nothing is fitted on this corpus.
Proficiency proxies for learners (secondary, weak): English onset age, log(1 + length of
residence in an English-speaking country).
"""
from __future__ import annotations

import json
import math
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

from bench.data.common import write_flac
from bench.learners import downloads, fetch, hours, make_record, root, sha256, write_jsonl

CORPUS = "saa"
LICENCE = "CC BY-NC-SA 4.0 (accent.gmu.edu/download, osf.io/yh23d, osf.io/e2j5y); evaluation only"
INFO_URL = "https://osf.io/download/9tupa/"
FLAC_FOLDER = "https://api.osf.io/v2/nodes/e2j5y/files/osfstorage/6a0f9d2e3c8c30eec4229a58/"
PARAGRAPH = ("Please call Stella. Ask her to bring these things with her from the store: Six spoons of "
             "fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. "
             "We also need a small plastic snake and a big toy frog for the kids. She can scoop these "
             "things into three red bags, and we will go meet her Wednesday at the train station.")
SEED = 19
N_NATIVE, N_LEARNER, MAX_PER_L1, MIN_ONSET = 100, 100, 4, 5
M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx(path: Path) -> list[dict]:
    """First sheet of an .xlsx as a list of dicts (header row = keys); no third-party package."""
    z = zipfile.ZipFile(path)
    ss = ["".join(t.text or "" for t in si.iter(M + "t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(M + "si")]
    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(M + "row"):
        row = {}
        for c in r.findall(M + "c"):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v = c.find(M + "v")
            row[col] = None if v is None else (ss[int(v.text)] if c.get("t") == "s" else v.text)
        rows.append(row)
    hdr = rows[0]
    return [{hdr[k]: v for k, v in r.items() if k in hdr} for r in rows[1:]]


def clean(v):
    return None if v in (None, "", "NULL") else v.strip()


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def flac_listing(dl: Path) -> dict:
    p = dl / "osf_flac_listing.json"
    if p.exists():
        return json.loads(p.read_text())
    import requests

    out, url = {}, FLAC_FOLDER + "?page%5Bsize%5D=100"
    while url:
        d = requests.get(url, timeout=60).json()
        for f in d["data"]:
            a = f["attributes"]
            out[a["name"]] = {"size": a["size"], "url": f["links"]["download"],
                              "sha256": (a.get("extra") or {}).get("hashes", {}).get("sha256")}
        url = d["links"].get("next")
    p.write_text(json.dumps(out, indent=0))
    return out


def eligible(info: list[dict], listing: dict) -> tuple[list[dict], Counter]:
    excl, out = Counter(), []
    for r in info:
        sample = clean(r.get("speech_sample"))
        if not sample:
            excl["no speech sample"] += 1
            continue
        stem = sample.rsplit(".", 1)[0]
        if f"{stem}.flac" not in listing:
            excl["no FLAC on OSF"] += 1
            continue
        l1 = (clean(r.get("native_language")) or "").lower()
        alt = (clean(r.get("alternative_native_language")) or "").lower()
        onset = num(r.get("onset_age"))
        if l1 == "english":
            if alt and alt != "english":
                excl["english plus another native language"] += 1
                continue
            native = True
        else:
            if onset is None:
                excl["learner without onset age"] += 1
                continue
            if onset < MIN_ONSET:
                excl[f"early bilingual (English onset < {MIN_ONSET})"] += 1
                continue
            native = False
        los = num(r.get("length_of_residence"))
        out.append({"stem": stem, "native": native, "l1": l1, "alt_l1": alt or None, "onset_age": onset,
                    "age": num(r.get("age")), "gender": clean(r.get("gender")), "country": clean(r.get("country")),
                    "english_residence": clean(r.get("english_residence")), "length_of_residence": los,
                    "learning_style": clean(r.get("learning_style")), "speakerid": clean(r.get("speakerid"))})
    return out, excl


def select(pool: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    nat = [p for p in pool if p["native"]]
    lea = [p for p in pool if not p["native"]]
    rng.shuffle(nat)
    rng.shuffle(lea)
    chosen = nat[:N_NATIVE]
    per = Counter()
    for p in lea:
        if sum(1 for c in chosen if not c["native"]) >= N_LEARNER:
            break
        if per[p["l1"]] < MAX_PER_L1:
            per[p["l1"]] += 1
            chosen.append(p)
    return chosen


def build() -> None:
    dl = downloads(CORPUS)
    info_path = dl / "speaker_information.xlsx"
    if not info_path.exists():
        fetch(INFO_URL, info_path)
    info = read_xlsx(info_path)
    listing = flac_listing(dl)
    pool, excl = eligible(info, listing)
    chosen = select(pool)
    out = root(CORPUS)
    import soundfile as sf

    recs = []
    for p in chosen:
        meta = listing[f"{p['stem']}.flac"]
        src = fetch(meta["url"], dl / "flac" / f"{p['stem']}.flac", size=meta["size"], sha=meta["sha256"])
        dst = out / "audio" / f"{p['stem']}.flac"
        if dst.exists():
            dur = sf.info(str(dst)).duration
        else:
            x, sr = sf.read(str(src), dtype="float32", always_2d=True)
            dur = write_flac(dst, x.mean(axis=1), sr)
        lab = {"native": p["native"], "l1": p["l1"], "alt_l1": p["alt_l1"], "onset_age": p["onset_age"],
               "length_of_residence": p["length_of_residence"],
               "log_residence": None if p["length_of_residence"] is None else math.log1p(max(0.0, p["length_of_residence"])),
               "english_residence": p["english_residence"], "learning_style": p["learning_style"],
               "age": p["age"], "gender": p["gender"], "country": p["country"], "prompt_id": "paragraph",
               "source_file": f"osf.io/e2j5y flac_files/{p['stem']}.flac", "source_sha256": meta["sha256"]}
        recs.append(make_record(f"saa-{p['stem']}", dst, "en", PARAGRAPH, "test", f"saa-{p['stem']}", dur, lab))
    write_jsonl(out / "manifest.jsonl", recs)
    write_jsonl(out / "test.jsonl", recs)
    chosen_stems = {p["stem"] for p in chosen}
    write_jsonl(out / "dev.jsonl", [{**p, "split": "dev"} for p in pool if p["stem"] not in chosen_stems])
    nat = [r for r in recs if r["labels"]["native"]]
    lea = [r for r in recs if not r["labels"]["native"]]
    summary = {"corpus": CORPUS, "licence": LICENCE, "seed": SEED, "speakers_in_metadata": len(info),
               "eligible": {"native": sum(p["native"] for p in pool), "learner": sum(not p["native"] for p in pool)},
               "excluded": dict(excl),
               "test": {"native": {"items": len(nat), "hours": hours(nat)},
                        "learner": {"items": len(lea), "hours": hours(lea), "l1s": len({r['labels']['l1'] for r in lea})},
                        "total_hours": hours(recs), "over_30s": {"native": sum(r["dur"] > 30 for r in nat),
                                                                  "learner": sum(r["dur"] > 30 for r in lea)}}}
    (out / "build.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    build()
