"""AISHELL-3 (Apache-2.0) capped subset: native Mandarin read speech with per-character pinyin+tone
labels (from the corpus' content.txt). Source: HF ``AISHELL/AISHELL-3`` (individual wav files are
fetched; the 19 GB archive is never downloaded).

    python -m bench.data.aishell3

The HF mirror holds only 11,750 of the 88,035 wavs (100 speakers under train/, 100 under test/ of
which only 7 are test-only speakers), so the requested ">= 100 train + >= 20 different test
speakers" cannot both be met. Split used here (no speaker overlap):
train: ~6000 utterances from 87 speakers of train/;
test:  800 utterances (40 each) from 20 speakers = the 7 test-only speakers + 13 speakers held out
       from train/ (seeded).
Selection is a seeded random sample per speaker (seed 0), so reruns pick the same files.

labels: {hanzi: [chars], pinyin: ["guang3", ...] (tone digit 1-5, 5 = neutral; "v" = ü),
         tones: [int], age_group: A <14 / B 14-25 / C 26-40 / D >41, gender, accent}
Note: the pinyin is the corpus annotation (lexical tones as transcribed; tone sandhi is NOT applied,
e.g. 3-3 sequences keep both 3s) and contains occasional annotation errors.
"""
from __future__ import annotations

import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .common import (PartWriter, check_disk, data_dir, decode_audio, make_record, merge_parts,
                     summarize, write_flac)

NAME = "aishell3"
BASE = "https://huggingface.co/datasets/AISHELL/AISHELL-3/resolve/main/"
N_TRAIN = 6000
N_TEST = 800
N_TEST_SPK = 20


# HF allows 3000 "resolve" requests per 5 min per IP, shared with everything else on this machine:
# stay at <= 4 requests/s (~40 % of the quota).
MAX_RPS = 4.0
_lock = threading.Lock()
_next = [0.0]


def _throttle() -> None:
    with _lock:
        now = time.monotonic()
        wait = _next[0] - now
        _next[0] = max(now, _next[0]) + 1.0 / MAX_RPS
    if wait > 0:
        time.sleep(wait)


def _get(session, url: str, tries: int = 10) -> bytes:
    delay = 5.0
    for _ in range(tries):
        _throttle()
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                raise FileNotFoundError(url)
            if r.status_code == 429:
                m = re.search(r"t=(\d+)", r.headers.get("ratelimit", ""))
                delay = max(delay, float(m.group(1)) + 1 if m else 30.0)
        except FileNotFoundError:
            raise
        except Exception:
            pass
        time.sleep(delay)
        delay = min(delay * 2, 300)
    raise RuntimeError(f"failed: {url}")


def _meta(session):
    root = data_dir(NAME)
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ("train/content.txt", "test/content.txt", "spk-info.txt"):
        p = root / name.replace("/", "_")
        if not p.exists():
            p.write_bytes(_get(session, BASE + name))
        out[name] = p.read_text(encoding="utf-8")
    return out


def _parse_content(txt: str) -> dict[str, list[tuple[str, str]]]:
    by_spk = defaultdict(list)
    for line in txt.splitlines():
        if "\t" not in line:
            continue
        fn, rest = line.split("\t", 1)
        toks = rest.split()
        pairs = list(zip(toks[0::2], toks[1::2]))
        by_spk[fn[:7]].append((fn[:-4], pairs))
    return by_spk


def available_files() -> set[str]:
    """Wav files actually present on the HF mirror (it holds only 11,750 of the 88,035)."""
    import requests

    r = requests.get("https://huggingface.co/api/datasets/AISHELL/AISHELL-3", timeout=60)
    r.raise_for_status()
    return {x["rfilename"] for x in r.json()["siblings"] if x["rfilename"].endswith(".wav")}


def select(meta, avail: set[str]) -> list[tuple[str, str, str, int, list]]:
    """-> [(our split, source dir, speaker, quota, shuffled [(utt, pairs)])].

    The mirror has 100 speakers in train/ and 7 test-only speakers (never in the corpus' train
    list) in test/. Test = the 7 test-only speakers + N_TEST_SPK-7 speakers held out from train/
    (seeded); train = the remaining train/ speakers. No speaker is in both of our splits."""
    tr = _parse_content(meta["train/content.txt"])
    te = _parse_content(meta["test/content.txt"])

    def present(src, by_spk):
        out = {}
        for spk, utts in by_spk.items():
            u = sorted(x for x in utts if f"{src}/wav/{spk}/{x[0]}.wav" in avail)
            if u:
                out[spk] = u
        return out

    tr, te = present("train", tr), present("test", te)
    rng = random.Random(0)
    test_only = sorted(set(te) - set(tr) - set(_parse_content(meta["train/content.txt"])))
    per_test = -(-N_TEST // N_TEST_SPK)
    cand = sorted(s for s in tr if len(tr[s]) >= per_test)
    rng.shuffle(cand)
    held = sorted(cand[:max(0, N_TEST_SPK - len(test_only))])
    train_spk = sorted(set(tr) - set(held))
    q = 1
    while sum(min(q, len(tr[s])) for s in train_spk) < N_TRAIN and q < 1000:
        q += 1
    jobs = []
    for spk in train_spk:
        u = list(tr[spk]); rng.shuffle(u)
        jobs.append(("train", "train", spk, q, u))
    tests = [("test", spk, te[spk]) for spk in test_only] + [("train", spk, tr[spk]) for spk in held]
    left = N_TEST
    for k, (src, spk, utts) in enumerate(tests):
        quota = min(per_test, left)
        left -= quota
        u = list(utts); rng.shuffle(u)
        jobs.append(("test", src, spk, quota, u))
    return jobs


def main() -> None:
    import requests

    check_disk()
    root = data_dir(NAME)
    s = requests.Session()
    meta = _meta(s)
    info = {}
    for line in meta["spk-info.txt"].splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        info[f[0]] = {"age_group": f[1], "gender": f[2], "accent": f[3] if len(f) > 3 else None}
    picks = select(meta, available_files())
    parts = {sp: PartWriter(root, sp) for sp in ("train", "test")}

    def one(split, src, utt, pairs):
        spk = utt[:7]
        try:
            data = _get(s, f"{BASE}{src}/wav/{spk}/{utt}.wav")
        except FileNotFoundError:
            return None
        x, sr = decode_audio(data)
        out = root / "audio" / split / spk / f"{utt}.flac"
        dur = write_flac(out, x, sr)
        hanzi = [h for h, _ in pairs]
        pinyin = [p for _, p in pairs]
        tones = [int(p[-1]) if p[-1:].isdigit() else None for p in pinyin]
        labels = {"hanzi": hanzi, "pinyin": pinyin, "tones": tones, **info.get(spk, {})}
        return make_record(f"aishell3-{utt}", out, "zh", "".join(hanzi), split, spk, dur, labels)

    def job(item):
        split, src, spk, quota, utts = item
        have = [r for r in parts[split].records if r["speaker"] == spk]
        new = []
        for utt, pairs in utts:
            if len(have) + len(new) >= quota:
                break
            if f"aishell3-{utt}" in parts[split].ids:
                continue
            rec = one(split, src, utt, pairs)
            if rec:
                new.append(rec)
        return split, new

    done = 0
    with ThreadPoolExecutor(12) as ex:
        for split, new in ex.map(job, picks):
            for rec in new:
                parts[split].add(rec)
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(picks)} speakers", flush=True)
                check_disk()
    for p in parts.values():
        p.finish()
    merge_parts(root, ("train", "test"))
    from . import load

    recs = load(NAME)
    print(summarize(recs))
    for sp in ("train", "test"):
        print(sp, "speakers:", len({r["speaker"] for r in recs if r["split"] == sp}))


if __name__ == "__main__":
    main()
