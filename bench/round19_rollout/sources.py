"""Round 19 rollout: native clips for the sound-identity head and its dev set, any language.

    python -m bench.round19_rollout.sources select --lang fr   # metadata only
    python -m bench.round19_rollout.sources fetch  --lang fr   # stream CV's dev tar shard, keep chosen clips

Everything comes from Common Voice 22.0's own DEV split (CC0; bench.cv_build mirror) and FLEURS
(CC BY 4.0). Rules for a Common Voice clip:
  * the speaker is in the cv-real DEV half (bench.cv_build.speaker_half), so no cv-real test-half
    speaker can occur; CV splits are speaker-disjoint, so no train / test speaker can either;
  * the speaker did not record any Lingolix clip (clips looked up in dev.tsv / test.tsv; Lingolix clips
    from CV's train split have train speakers, which are not in the dev split) and the sentence is no
    Lingolix prompt or target;
  * no cv-real clip (either half), accepted (up >= 2, down == 0), accent not marked non-native,
    1-12 s, at least 3 words.
Speakers are split by a hash: 35 % "dev" (planted-change dev set, at most 2 clips each), 65 % "train"
(at most 4 clips each; one fifth of those speakers is the held-out native pool used only to place
operating points). FLEURS: TRAIN for the head; VALIDATION for the dev set and the validation guards.
FLEURS TEST is never read here.
Work folder: $PRONUNCIADA_CACHE/round19/rollout/<lang>/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
ROOT = CACHE / "round19" / "rollout"
LINGOLIX = CACHE / "third_party" / "lingolix" / "repo_v2026.10" / "src" / "data" / "tier1"
CV_LOCALE = {"zh": "zh-CN"}
SEED = 1920
N_DEV_CV = 460
N_DEV_FLEURS = 260
N_TRAIN_CV = 3200
csv.field_size_limit(1 << 30)


def work(lang: str) -> Path:
    p = ROOT / lang
    p.mkdir(parents=True, exist_ok=True)
    return p


def locale(lang: str) -> str:
    return CV_LOCALE.get(lang, lang)


def cv_meta(lang: str) -> Path:
    return CACHE / "data" / "cv-real" / "meta" / locale(lang)


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def read_tsv(p: Path):
    with open(p, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def lingolix_items(lang: str) -> list[dict]:
    p = LINGOLIX / f"manifest_{lang}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def lingolix_exclusions(lang: str) -> tuple[set, set, set]:
    """(clips, speakers, normalised sentences) of the Lingolix items of this language."""
    items = lingolix_items(lang)
    clips = {x["cv_clip"] for x in items}
    spk = set()
    for name in ("dev.tsv", "test.tsv", "invalidated.tsv"):
        p = cv_meta(lang) / name
        if p.exists():
            spk |= {r["client_id"] for r in read_tsv(p) if r["path"] in clips}
    sents = {norm(x["prompt"]) for x in items} | {norm(x["target"]) for x in items}
    return clips, spk, sents


def cvreal_clips(lang: str) -> set:
    from bench.cv_build import OUT_DIR

    p = OUT_DIR / f"{locale(lang)}.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["path"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def speaker_role(cid: str) -> str:
    h = _h("role", cid) % 100
    if h < 35:
        return "dev"
    return "pool" if h < 48 else "train"


def cmd_select(lang: str) -> None:
    from bench.cv_build import nativeness, speaker_half

    bad_clip, bad_spk, bad_sent = lingolix_exclusions(lang)
    bad_clip |= cvreal_clips(lang)
    durs = {}
    with open(cv_meta(lang) / "clip_durations.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1].strip().isdigit():
                durs[p[0]] = int(p[1]) / 1000
    cands = []
    for r in read_tsv(cv_meta(lang) / "dev.tsv"):
        cid = r["client_id"]
        if speaker_half(cid) != "dev" or cid in bad_spk or r["path"] in bad_clip or norm(r["sentence"]) in bad_sent:
            continue
        if int(r["up_votes"] or 0) < 2 or int(r["down_votes"] or 0) != 0:
            continue
        if nativeness(locale(lang), r.get("accents") or "") == "nonnative":
            continue
        d = durs.get(r["path"])
        if d is None or not 1.0 <= d <= 12.0:
            continue
        if lang != "zh" and len(r["sentence"].split()) < 3:
            continue
        cands.append({"stem": r["path"].rsplit(".", 1)[0], "speaker": cid, "sentence": r["sentence"], "dur": d,
                      "role": speaker_role(cid)})
    rng = random.Random(SEED)
    rng.shuffle(cands)
    per, seen = Counter(), set()
    out = {"dev": [], "train": [], "pool": []}
    cap = {"dev": 2, "train": 4, "pool": 4}
    if len({c["speaker"] for c in cands}) < 1000:          # few speakers (it, pt, ru, zh): more clips each
        cap = {"dev": 4, "train": 16, "pool": 16}
    lim = {"dev": int(N_DEV_CV * 1.15), "train": N_TRAIN_CV, "pool": N_TRAIN_CV // 4}
    for c in cands:
        role = c["role"]
        if per[c["speaker"]] >= cap[role] or len(out[role]) >= lim[role]:
            continue
        if role == "dev" and norm(c["sentence"]) in seen:
            continue
        per[c["speaker"]] += 1
        seen.add(norm(c["sentence"]))
        out[role].append(c)
    # a sentence used in the dev set never appears in training / pool
    devs = {norm(c["sentence"]) for c in out["dev"]}
    for k in ("train", "pool"):
        out[k] = [c for c in out[k] if norm(c["sentence"]) not in devs]
    (work(lang) / "cv_select.json").write_text(json.dumps(out, ensure_ascii=False))
    print(lang, "candidates", len(cands), {k: (len(v), len({c['speaker'] for c in v})) for k, v in out.items()},
          "lingolix speakers excluded", len(bad_spk))


def cv_audio_dir(lang: str) -> Path:
    p = work(lang) / "cv_audio"
    p.mkdir(exist_ok=True)
    return p


def cmd_fetch(lang: str) -> None:
    import tarfile

    import requests
    import soundfile as sf

    from bench.cv_build import n_shards, shard_url
    from bench.data.common import resample, write_flac

    sel = json.loads((work(lang) / "cv_select.json").read_text())
    want = {c["stem"] for v in sel.values() for c in v}
    todo = {s for s in want if not (cv_audio_dir(lang) / f"{s}.flac").exists()}
    print(lang, "to fetch", len(todo), flush=True)
    try:
        ns = int(n_shards(locale(lang)).get("dev", 1))
    except Exception:  # noqa: BLE001
        ns = 1
    for k in range(ns):
        if not todo:
            break
        with requests.get(shard_url(locale(lang), "dev", k), stream=True, timeout=120) as r:
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
                    write_flac(cv_audio_dir(lang) / f"{stem}.flac", resample(x.mean(axis=1), sr), 16000)
                    todo.discard(stem)
                    if not todo:
                        break
    print(lang, "missing after fetch", len(todo), flush=True)


def cv_rows(lang: str, role: str) -> list[dict]:
    sel = json.loads((work(lang) / "cv_select.json").read_text())
    out = []
    for c in sel[role]:
        p = cv_audio_dir(lang) / f"{c['stem']}.flac"
        if p.exists():
            out.append({**c, "audio": str(p), "spk": "cv:" + hashlib.sha1(c["speaker"].encode()).hexdigest()[:12]})
    return out


def fleurs_val(lang: str) -> list[dict]:
    """FLEURS VALIDATION records (bench.pipeline.common), audio read from the parquet by row."""
    from bench.pipeline.common import records

    return records("fleurs-validation", lang=lang)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "fetch"])
    ap.add_argument("--lang", required=True, help="one language or several, comma-separated (fetched concurrently)")
    a = ap.parse_args(argv)
    langs = a.lang.split(",")
    if a.cmd == "fetch" and len(langs) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(len(langs)) as ex:
            list(ex.map(cmd_fetch, langs))
        return
    for l in langs:
        {"select": cmd_select, "fetch": cmd_fetch}[a.cmd](l)


if __name__ == "__main__":
    main()
