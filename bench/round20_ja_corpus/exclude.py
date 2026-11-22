"""Exclusion lists and the speaker split for the Japanese length corpus.

    python -m bench.round20_ja_corpus.exclude     # -> <work>/exclusions.json, <work>/cv_select.jsonl

Held-out and practice material is removed by clip, by speaker (Common Voice client id) and by
prompt sentence (NFKC, letters and digits only):

* Lingolix Japanese test (tier-1 manifests of v2026.10 and of main): only the manifest fields
  cv_clip / prompt / target are read; the clip ids are mapped to client ids through the CV TSVs.
  No audio, label or score of the benchmark is opened.
* cv-real ja (bench/data/cv_real/ja.jsonl): test half (held-out) and dev half (round-19 dev guard).
* round-19 practice pools: the Common Voice dev pool of bench/round19_ja/cv_pool.py, recomputed with
  its own code on the same metadata (fit pools are training data and stay in).
* FLEURS ja validation (round-19 dev) and test (held-out): their sentences are excluded from every
  source; their audio is never used.

Common Voice audio that exists on the mirror: the train / dev / test splits (validated) and `other`
(not yet validated, kept as a separate tier with down_votes == 0). Speakers are split by a salted
hash of the client id: train 85 %, validation 7.5 %, held-back 7.5 %.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bench.cv_build import norm_sentence, speaker_half
from bench.round20_ja_corpus.common import CVMETA, RAW, WORK, cv_durations, read_tsv

REPO = Path(__file__).resolve().parents[2]
LX = Path.home() / ".cache" / "pronunciada" / "third_party" / "lingolix"
LX_MANIFESTS = [LX / "repo_v2026.10" / "src/data/tier1/manifest_ja.jsonl",
                LX / "repo_main_f8750aa" / "src/data/tier1/manifest_ja.jsonl"]
SALT = "jacorpus-r20"
_BAD_TEXT = re.compile(r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ]")
MIN_DUR, MAX_DUR = 1.0, 15.0
OLDER = ("15_0", "17_0", "19_0", "21_0")     # fsicoli/common_voice_<v> mirrors (CC0), metadata in raw/cv_older


def spk_split(spk: str) -> str:
    h = int.from_bytes(hashlib.sha256(f"{SALT}|{spk}".encode()).digest()[:4], "big") / 2**32
    return "train" if h < 0.85 else ("validation" if h < 0.925 else "heldback")


def lingolix() -> tuple[set[str], set[str]]:
    """(clip paths, normalised sentences) from the manifests' id fields only."""
    clips, sents = set(), set()
    for m in LX_MANIFESTS:
        for line in open(m, encoding="utf-8"):
            it = json.loads(line)
            clips.add(it["cv_clip"])
            sents.add(norm_sentence(it["prompt"]))
            sents.add(norm_sentence(it["target"]))
    return clips, sents


def r19_dev_pool() -> set[str]:
    """Clip paths of the round-19 Common Voice dev pool, recomputed with cv_pool.select()."""
    import bench.round19_ja.cv_pool as cp

    tmp = WORK / "r19pool"
    (tmp / "cvmeta").mkdir(parents=True, exist_ok=True)
    for n in ("validated", "train"):
        link = tmp / "cvmeta" / f"{n}.tsv"
        if not link.exists():
            link.symlink_to(CVMETA / f"{n}.tsv")
    cp.WORK = tmp
    if not (tmp / "cv_dev.jsonl").exists():
        cp.select()
    return {json.loads(x)["stem"] + ".mp3" for x in open(tmp / "cv_dev.jsonl", encoding="utf-8")}


def fleurs_sentences() -> dict[str, set[str]]:
    out = {}
    for s in ("dev", "test"):
        out[s] = set()
        for line in open(RAW / "fleurs" / f"{s}.tsv", encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            out[s].add(norm_sentence(p[2]))
    return out


def main() -> None:
    dur = cv_durations()
    tsv = {n: list(read_tsv(CVMETA / f"{n}.tsv")) for n in ("validated", "train", "dev", "test", "other", "invalidated")}
    clip2spk = {}
    for rows in tsv.values():
        for r in rows:
            clip2spk[r["path"]] = r["client_id"]

    ex_clip: dict[str, set[str]] = {}
    ex_spk: dict[str, set[str]] = {}
    ex_sent: dict[str, set[str]] = {}
    c, s = lingolix()
    ex_clip["lingolix_ja"], ex_sent["lingolix_ja"] = c, s
    missing = [x for x in c if x not in clip2spk]
    assert not missing, f"{len(missing)} Lingolix clips not found in the CV TSVs"
    ex_spk["lingolix_ja"] = {clip2spk[x] for x in c}
    cvr = [json.loads(x) for x in open(REPO / "bench" / "data" / "cv_real" / "ja.jsonl", encoding="utf-8")]
    for half in ("test", "dev"):
        rr = [r for r in cvr if r["half"] == half]
        ex_clip[f"cv_real_{half}"] = {r["path"] for r in rr}
        ex_sent[f"cv_real_{half}"] = {norm_sentence(r["sentence"]) for r in rr}
        ex_spk[f"cv_real_{half}"] = {clip2spk[r["path"]] for r in rr if r["path"] in clip2spk}
    pool = r19_dev_pool()
    ex_clip["r19_cv_dev_pool"] = pool
    ex_spk["r19_cv_dev_pool"] = {clip2spk[p] for p in pool}
    ex_sent["r19_cv_dev_pool"] = {norm_sentence(r["sentence"]) for r in tsv["dev"] if r["path"] in pool}
    fs = fleurs_sentences()
    ex_sent["fleurs_validation"], ex_sent["fleurs_test"] = fs["dev"], fs["test"]

    all_clip = set().union(*ex_clip.values())
    all_spk = set().union(*ex_spk.values())
    all_sent = set().union(*ex_sent.values())
    # every source (FLEURS train, Kokoro) is checked against the same sentence list later
    (WORK / "excluded_sentences.txt").write_text("\n".join(sorted(all_sent)) + "\n", encoding="utf-8")

    # candidates: clips whose audio is on the mirror
    cand = []
    for split in ("train", "dev", "test", "other"):
        for r in tsv[split]:
            cand.append((split, r))
    # validated 22.0 clips whose audio is only in the split shards of older releases of the same
    # mirror family (raw/cv_older/<version>/<split>.tsv); the 22.0 validated.tsv row is used (votes, text)
    seen = {r["path"] for _, r in cand}
    vmap = {r["path"]: r for r in tsv["validated"]}
    for v in OLDER:
        for s in ("train", "dev", "test"):
            p = RAW / "cv_older" / v / f"{s}.tsv"
            if not p.exists():
                continue
            for r in read_tsv(p):
                if r["path"] in vmap and r["path"] not in seen:
                    seen.add(r["path"])
                    cand.append((f"v{v}_{s}", vmap[r["path"]]))
    reasons: Counter = Counter()
    by_set: dict[str, Counter] = defaultdict(Counter)
    keep = []
    for split, r in cand:
        p, spk, sent = r["path"], r["client_id"], norm_sentence(r["sentence"])
        d = dur.get(p)
        hit = False
        for k in ex_spk:
            if spk in ex_spk[k] or p in ex_clip.get(k, ()):
                by_set[k]["clips_by_speaker_or_clip"] += 1
                hit = True
        for k in ex_sent:
            if sent in ex_sent[k]:
                by_set[k]["clips_by_sentence"] += 1
                hit = True
        if hit:
            reasons["excluded_heldout_or_practice"] += 1
            continue
        up, down = int(r["up_votes"] or 0), int(r["down_votes"] or 0)
        if down > 0:
            reasons["down_votes"] += 1
            continue
        if not r["sentence"].strip() or _BAD_TEXT.search(r["sentence"]):
            reasons["digits_or_latin"] += 1
            continue
        if d is None or not MIN_DUR <= d <= MAX_DUR:
            reasons["duration"] += 1
            continue
        tier = "validated" if split != "other" else "other"
        keep.append({"id": f"cv22-ja-{p.rsplit('.', 1)[0].rsplit('_', 1)[-1]}", "path": p, "cv_split": split,
                     "tier": tier, "client_id": spk, "split": spk_split(spk),
                     "cv_real_hash_half": speaker_half(spk), "text": r["sentence"], "dur": d,
                     "up_votes": up, "down_votes": down})
        reasons[f"kept_{tier}"] += 1

    # sanity: no excluded speaker anywhere in the kept set
    assert not ({k["client_id"] for k in keep} & all_spk)
    spks = sorted({k["client_id"] for k in keep})
    pseud = {s: f"cvja_{i:05d}" for i, s in enumerate(spks)}
    with open(WORK / "cv_select.jsonl", "w", encoding="utf-8") as f:
        for k in keep:
            k["speaker"] = pseud[k.pop("client_id")]
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    hours = defaultdict(float)
    nspk = defaultdict(set)
    for k in keep:
        hours[(k["tier"], k["split"])] += k["dur"] / 3600
        nspk[(k["tier"], k["split"])].add(k["speaker"])
    rep = {
        "excluded_sets": {k: {"speakers": len(ex_spk.get(k, ())), "clips_listed": len(ex_clip.get(k, ())),
                              "sentences": len(ex_sent.get(k, ())), **by_set[k]} for k in ex_sent},
        "excluded_union": {"speakers": len(all_spk), "clips": len(all_clip), "sentences": len(all_sent)},
        "candidate_clips": len(cand),
        "filter_counts": dict(reasons),
        "kept": {f"{t}/{s}": {"hours": round(h, 2), "speakers": len(nspk[(t, s)])} for (t, s), h in sorted(hours.items())},
        "hash_test_half_speakers_kept": len({k["speaker"] for k in keep if k["cv_real_hash_half"] == "test"}),
    }
    json.dump(rep, open(WORK / "exclusions.json", "w"), indent=1)
    json.dump(rep, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
