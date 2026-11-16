"""Build the "ompal" benchmark: expert-marked French-L1 learners of (Taiwan) Mandarin.

    python -m bench.ompal_build [--clone]

Source: OMPAL corpus, github.com/phantomhsieh/OMPAL-corpus (Hsieh et al., Interspeech 2025,
doi 10.21437/Interspeech.2025-983), CC BY 4.0. Audio (16 kHz mono PCM WAV) and JSON scores are in
the repository itself; `--clone` makes a shallow clone into
``$PRONUNCIADA_CACHE/data/ompal/repo`` (about 625 MB on disk). Nothing is downloaded otherwise.

Outputs
  $PRONUNCIADA_CACHE/data/ompal/manifest.jsonl    one record per utterance (bench.data.load("ompal"))
  results/tables/ompal_manifest.json               ids / speakers / split / label summary, no audio

Annotation facts (from the data; the README says "four experts", but every learner item in
non-native_scores-detail.json carries exactly three ratings, one item has two):
  * word unit = one Hanzi character of the Traditional-script prompt (9 / 19,775 learner units
    have two characters, see below); each unit is marked 1 = correct / 0 = incorrect separately
    for tone, consonant (phoneme_consonant) and vowel (phoneme_vowel). No pinyin or tone targets
    are shipped; the target is the character as read in Taiwan Mandarin.
  * the shipped per-word label = majority of the three raters (checked: 0 iff <= 1 of 3 said
    correct, on all 59,325 learner judgements);
  * sentence accuracy / fluency / prosody = mean of the three 1-5 ratings (rounded to 2 dp);
  * native speakers (SPEAKER01*) have only aggregated labels (all sentences 5/5/5).

Split, fixed by speaker (nothing is tuned on either half in the benchmark task):
  TEST = every speaker appearing in the five shipped test/test_k_scores.json files, plus all
         three native speakers (SPEAKER01003 appears in no shipped split file).
  DEV  = the remaining learners.
NOTE: the shipped test files do not match the speaker combinations listed in the corpus README;
the files are used (they are what the repository's own experiments load).

Unit -> prompt-character spans (`unit_spans`): the unit characters are aligned to the prompt by
difflib; a unit's span covers the prompt characters matched to it. A unit with no prompt
character (e.g. units for "正月初一" on a native recording whose prompt omits it) is dropped; a
two-character unit with one character absent from the prompt (e.g. "還也" against prompt "還",
an inserted word) keeps the matched character. An utterance whose unit list has more than one
multi-character unit is treated as malformed (one native recording interleaves two word lists)
and contributes no word units; its sentence scores are kept.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "ompal"
REPO_URL = "https://github.com/phantomhsieh/OMPAL-corpus.git"
DIMS = ("tone", "phoneme_consonant", "phoneme_vowel")
MANIFEST_OUT = ROOT / "results" / "tables" / "ompal_manifest.json"
LICENCE = "CC BY 4.0"
CITATION = ("W.-W. Hsieh, H.-W. Chi, K.-C. Wang, P.-C. Yeh, T.-h. Liu, C.-Y. Chiang, \"OMPAL: Bridging "
            "Speech and Learning with an Open-Source Mandarin Pronunciation Assessment Corpus for Global "
            "Learners\", Interspeech 2025, pp. 2415-2419, doi:10.21437/Interspeech.2025-983")


def data_root() -> Path:
    from bench.data.common import data_dir
    return data_dir(NAME)


def repo_dir() -> Path:
    return data_root() / "repo"


# ----------------------------------------------------------------------------- pure logic

def speaker_of(uid: str) -> str:
    """'00200101' -> 'SPEAKER02001' (file name = '0' + 5-digit speaker + 2-digit utterance)."""
    if len(uid) != 8 or not uid.isdigit():
        raise ValueError(f"unexpected OMPAL id {uid!r}")
    return "SPEAKER" + uid[1:6]


def is_native(speaker: str) -> bool:
    return speaker.startswith("SPEAKER01")


def make_split(all_speakers, test_file_speakers) -> dict[str, str]:
    """speaker -> 'test' | 'dev'. TEST = speakers of the shipped test files + every native."""
    tset = set(test_file_speakers)
    return {s: ("test" if (s in tset or is_native(s)) else "dev") for s in sorted(set(all_speakers))}


def unit_spans(text: str, unit_texts: list[str]) -> tuple[list[tuple[int, int] | None], str | None]:
    """Character span in `text` of each unit (None = unit not in the prompt -> dropped), and a
    reason string when the whole utterance's unit list is malformed (then all spans are None)."""
    n_multi = sum(len(u) > 1 for u in unit_texts)
    if n_multi > 1:
        return [None] * len(unit_texts), f"malformed unit list ({n_multi} multi-character units)"
    uchars, owner = [], []
    for k, u in enumerate(unit_texts):
        for ch in u:
            uchars.append(ch)
            owner.append(k)
    pos: list[list[int]] = [[] for _ in unit_texts]
    sm = SequenceMatcher(None, uchars, list(text), autojunk=False)
    for b in sm.get_matching_blocks():
        for t in range(b.size):
            pos[owner[b.a + t]].append(b.b + t)
    spans = [(min(p), max(p) + 1) if p else None for p in pos]
    return spans, None


def unit_label(w: dict) -> dict:
    t, c, v = (int(w[d]) for d in DIMS)
    return {"tone": t, "consonant": c, "vowel": v, "any_wrong": int(min(t, c, v) == 0),
            "tone_only": int(t == 0 and c == 1 and v == 1), "seg_wrong": int(c == 0 or v == 0),
            "correct": int(t == c == v == 1)}


# ----------------------------------------------------------------------------- agreement

def fleiss_kappa(rows: list[list[int]]) -> float:
    """Fleiss' kappa for binary ratings, rows = list of per-item rating lists (>= 2 each)."""
    import numpy as np

    P, p1, n_tot = [], 0.0, 0
    for r in rows:
        n = len(r)
        if n < 2:
            continue
        k1 = sum(r)
        k0 = n - k1
        P.append((k1 * (k1 - 1) + k0 * (k0 - 1)) / (n * (n - 1)))
        p1 += k1
        n_tot += n
    pbar = float(np.mean(P))
    q = p1 / n_tot
    pe = q * q + (1 - q) * (1 - q)
    return float((pbar - pe) / (1 - pe))


def agreement(detail: dict) -> dict:
    """Inter-rater agreement among the three learner ratings. Pairwise numbers assume list
    position k is the same rater across items (not documented by the corpus)."""
    import numpy as np

    from bench.metrics import pearson

    out = {"n_items": len(detail), "ratings_per_item": dict(Counter(len(v["accuracy"]) for v in detail.values()))}
    for d in DIMS:
        rows = [[int(x) for x in w[d]] for v in detail.values() for w in v["words"]]
        out[f"word_{d}_fleiss_kappa"] = fleiss_kappa(rows)
        r3 = [r for r in rows if len(r) == 3]
        out[f"word_{d}_unanimous_share"] = float(np.mean([len(set(r)) == 1 for r in r3]))
    for s in ("accuracy", "fluency", "prosody"):
        m = np.array([v[s] for v in detail.values() if len(v[s]) == 3], float)
        pair = [pearson(m[:, i], m[:, j]) for i in range(3) for j in range(i + 1, 3)]
        loo = [pearson(m[:, i], np.delete(m, i, axis=1).mean(1)) for i in range(3)]
        out[f"sentence_{s}_pairwise_pearson_mean"] = float(np.mean(pair))
        out[f"sentence_{s}_rater_vs_mean_of_others_pearson_mean"] = float(np.mean(loo))
    return out


# ----------------------------------------------------------------------------- build

def clone() -> None:
    from bench.data.common import check_disk

    d = repo_dir()
    if (d / ".git").exists():
        print(f"{d} exists; not cloning again")
        return
    check_disk(3.0)
    d.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(d)], check=True)


def build() -> list[dict]:
    from bench.data.common import flac_duration

    rd = repo_dir()
    if not (rd / "README.md").exists():
        sys.exit(f"{rd} missing; run `python -m bench.ompal_build --clone`")
    nn = json.loads((rd / "non-native_scores.json").read_text())
    nat = json.loads((rd / "native_scores.json").read_text())
    detail = json.loads((rd / "non-native_scores-detail.json").read_text())
    test_sets: dict[str, list[int]] = {}
    for k in range(1, 6):
        for uid in json.loads((rd / "test" / f"test_{k}_scores.json").read_text()):
            test_sets.setdefault(uid, []).append(k)
    master = {**nn, **nat}
    wavs = {p.stem: p for p in (rd / "wav").glob("*/*.wav")}
    if set(wavs) != set(master):
        raise RuntimeError("audio files and score files do not match")
    for uid, p in wavs.items():
        if p.parent.name != speaker_of(uid):
            raise RuntimeError(f"{uid} in {p.parent.name}")
    split = make_split([speaker_of(u) for u in master], [speaker_of(u) for u in test_sets])
    commit = subprocess.run(["git", "-C", str(rd), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    recs = []
    for uid in sorted(master):
        v = master[uid]
        spk = speaker_of(uid)
        texts = ["".join(w["text"]) for w in v["words"]]
        spans, bad = unit_spans(v["text"], texts)
        units = []
        for i, (w, t, sp) in enumerate(zip(v["words"], texts, spans)):
            u = {"i": i, "text": t, "span": list(sp) if sp else None, **unit_label(w)}
            if uid in detail and i < len(detail[uid]["words"]):
                u["raters"] = {d: [int(x) for x in detail[uid]["words"][i][d]] for d in DIMS}
            units.append(u)
        labels = {"native": is_native(spk), "test_sets": test_sets.get(uid, []),
                  "sentence": {k: float(v[k]) for k in ("accuracy", "fluency", "prosody")},
                  "units": units, "word_level_dropped": bad,
                  "n_units_not_in_prompt": 0 if bad else sum(sp is None for sp in spans)}
        if uid in detail:
            labels["sentence_raters"] = {k: detail[uid][k] for k in ("accuracy", "fluency", "prosody")}
        recs.append({"id": f"ompal-{uid}", "audio": str(wavs[uid].resolve()), "lang": "zh",
                     "text": v["text"], "split": split[spk], "speaker": spk,
                     "dur": round(flac_duration(wavs[uid]), 3), "labels": labels})
    out = data_root() / "manifest.jsonl"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    # repo manifest (no audio, no paths)
    summ = {"dataset": NAME, "source": REPO_URL, "source_commit": commit, "licence": LICENCE,
            "citation": CITATION, "split_rule": "TEST = speakers of the shipped test/test_[1-5]_scores.json "
            "+ all native speakers; DEV = remaining learners",
            "speakers": {s: {"split": sp, "native": is_native(s),
                             "n_utts": sum(r["speaker"] == s for r in recs)} for s, sp in split.items()},
            "utterances": [{"id": r["id"], "speaker": r["speaker"], "split": r["split"],
                            "test_sets": r["labels"]["test_sets"], "dur": r["dur"],
                            "n_units": sum(u["span"] is not None for u in r["labels"]["units"]),
                            "word_level_dropped": r["labels"]["word_level_dropped"]} for r in recs],
            "agreement": agreement(detail)}
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    c = Counter((r["split"], r["labels"]["native"]) for r in recs)
    print(f"wrote {len(recs)} records -> {out}")
    print("utterances (split, native):", dict(c))
    print("speakers:", dict(Counter((sp, is_native(s)) for s, sp in split.items())))
    print("dropped word level:", [r["id"] for r in recs if r["labels"]["word_level_dropped"]])
    return recs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clone", action="store_true", help="shallow-clone the corpus repository first")
    a = ap.parse_args(argv)
    if a.clone:
        clone()
    build()


if __name__ == "__main__":
    main()
