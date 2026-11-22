"""Records and aligner inputs for the Japanese length corpus.

    python -m bench.round20_ja_corpus.prep records
        Merge the per-source selections into <work>/records.jsonl (id, source, licence, speaker, split,
        audio, text, dur, words/units from Pronunciada's front-end). Utterances whose text the front-end
        cannot turn into mora units, or that hit an excluded sentence, are counted and dropped.
    python -m bench.round20_ja_corpus.prep batch NAME --where EXPR [--dict canon|variants] [--limit N]
        Write an aligner corpus <work>/align/NAME/{corpus/<speaker>/<id>.wav,.lab, dict.txt, ids.txt}.
        Audio is decoded with libsndfile (soundfile), mixed to mono, resampled to 16 kHz; these WAVs
        are temporary (deleted after the batch is aligned).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from bench.cv_build import norm_sentence
from bench.round20_ja_corpus.common import ALIGN, AUDIO, WORK
from bench.round20_ja_corpus.units import UnitError, ref_units, token, variants

LIC = {"cv": "CC0-1.0", "fleurs": "CC-BY-4.0", "kokoro": "public-domain"}


def _sources():
    for r in map(json.loads, open(WORK / "cv_select.jsonl", encoding="utf-8")):
        yield {"id": r["id"], "source": "cv22", "licence": LIC["cv"], "speaker": r["speaker"],
               "split": r["split"], "tier": r["tier"], "audio": str((AUDIO / "cv" / r["cv_split"] / r["path"]).relative_to(WORK)),
               "text": r["text"], "dur": r["dur"], "cv_split": r["cv_split"], "cv_real_hash_half": r["cv_real_hash_half"]}
    p = WORK / "fleurs_select.jsonl"
    if p.exists():
        for r in map(json.loads, open(p, encoding="utf-8")):
            # FLEURS has no speaker ids: one pseudo-speaker per clip; the whole source goes to train
            yield {"id": f"fleurs-ja-{Path(r['file']).stem}", "source": "fleurs", "licence": LIC["fleurs"],
                   "speaker": f"fleurs_{Path(r['file']).stem}", "split": "train", "tier": "validated",
                   "audio": r["audio"], "text": r["text"], "dur": r["dur"], "fleurs_id": r["fleurs_id"],
                   "gender": r["gender"]}
    p = WORK / "kokoro_select.jsonl"
    if p.exists():
        for r in map(json.loads, open(p, encoding="utf-8")):
            yield {"id": r["id"], "source": "kokoro", "licence": LIC["kokoro"], "speaker": "kokoro_ekzemplaro",
                   "split": "train", "tier": "validated", "audio": r["audio"], "text": r["text"], "dur": r["dur"],
                   "book": r["book"], "kokoro_reading": r["reading"]}


def kokoro_counts(reading: str) -> dict:
    """Long vowels / geminates / moraic nasals in the dataset's own romanised reading (MeCab +
    UniDic Lite, independent of Open JTalk). 'q' = っ, 'N' = ん, 'V:' or a repeated vowel inside a
    word = long vowel; 'e i' / 'o u' inside a word count as long (the front-end's convention)."""
    n = {"long": 0, "gem": 0, "N": 0}
    for wd in reading.split("_"):
        toks = [t for t in wd.split() if t not in (".", ",", "、", "。", "?", "!")]
        prev_v = None
        for t in toks:
            if t == "q":
                n["gem"] += 1
                prev_v = None
            elif t == "N":
                n["N"] += 1
                prev_v = None
            elif t.endswith(":"):
                n["long"] += 1
                prev_v = None
            elif t in ("a", "i", "u", "e", "o"):
                if prev_v is not None and (t == prev_v or (prev_v, t) in (("e", "i"), ("o", "u"))):
                    n["long"] += 1
                    prev_v = None
                else:
                    prev_v = t
            else:
                prev_v = None if t not in ("a", "i", "u", "e", "o") else t
    return n


def kokoro_agree(reading: str, words: list[dict]) -> dict:
    k = kokoro_counts(reading)
    units = [u for wd in words for u in wd["units"]]
    o = {"long": sum(1 for u in units if u.get("long")), "gem": sum(1 for u in units if u["gem"] or u["final_q"]),
         "N": sum(1 for u in units if u["kind"] == "N")}
    return {"agree": k == o, "kokoro": k, "frontend": o}


def records() -> None:
    from pronunciada.text.ja import build

    g2p = build()
    bad = set((WORK / "excluded_sentences.txt").read_text(encoding="utf-8").split())
    cache: dict[str, object] = {}
    cnt: Counter = Counter()
    with open(WORK / "records.jsonl", "w", encoding="utf-8") as f:
        for r in _sources():
            if norm_sentence(r["text"]) in bad:
                cnt[f"{r['source']}:excluded_sentence"] += 1
                continue
            t = r["text"]
            if t not in cache:
                try:
                    ref = g2p(t)
                    cache[t] = "warn:" + ref.warnings[0] if ref.warnings else ref_units(ref)
                except (UnitError, Exception) as e:  # noqa: BLE001
                    cache[t] = f"error:{type(e).__name__}:{e}"
            w = cache[t]
            if isinstance(w, str):
                cnt[f"{r['source']}:frontend_{w.split(':')[0]}"] += 1
                continue
            if not any(u for wd in w for u in wd["units"]):
                cnt[f"{r['source']}:frontend_empty"] += 1
                continue
            r["words"] = w
            if r["source"] == "kokoro":
                r["reading_check"] = kokoro_agree(r["kokoro_reading"], w)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            cnt[f"{r['source']}:kept"] += 1
    json.dump(dict(cnt), open(WORK / "records_counts.json", "w"), indent=1)
    print(json.dumps(dict(cnt), indent=1))


def load_audio16(path: Path) -> np.ndarray:
    import soundfile as sf

    from bench.data.common import resample

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), sr, 16000)


def batch(name: str, where: str, mode: str, limit: int | None, seed: int = 20) -> None:
    import random

    import soundfile as sf

    out = ALIGN / name
    corp = out / "corpus"
    corp.mkdir(parents=True, exist_ok=True)
    rows = [r for r in map(json.loads, open(WORK / "records.jsonl", encoding="utf-8")) if eval(where, {}, {"r": r})]  # noqa: S307
    if limit and len(rows) > limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    prons: dict[str, dict[tuple, float]] = {}
    ids = []
    for r in rows:
        toks = []
        for wd in r["words"]:
            if not wd["units"]:
                continue
            canon = [p for u in wd["units"] for p in u["ph"]]
            tk = token(canon)
            toks.append(tk)
            vs = variants(wd["units"]) if mode == "variants" else [("canon", canon, 1.0)]
            d = prons.setdefault(tk, {})
            for _, ph, pr in vs:
                d[tuple(ph)] = max(pr, d.get(tuple(ph), 0.0))
        # the aligner parallelises by speaker: the single Kokoro reader is split by book
        spk = corp / (r["speaker"] if r["source"] != "kokoro" else f"kokoro_{r['book']}")
        spk.mkdir(exist_ok=True)
        wav = spk / f"{r['id']}.wav"
        if not wav.exists():
            try:
                x = load_audio16(WORK / r["audio"])
            except Exception as e:  # noqa: BLE001
                print("audio fail", r["id"], type(e).__name__, file=sys.stderr)
                continue
            sf.write(wav, x, 16000, subtype="PCM_16")
        (spk / f"{r['id']}.lab").write_text(" ".join(toks) + "\n")
        ids.append(r["id"])
    with open(out / "dict.txt", "w") as f:
        for tk in sorted(prons):
            for ph, pr in sorted(prons[tk].items(), key=lambda kv: -kv[1]):
                f.write(f"{tk}\t{pr:.2f}\t{' '.join(ph)}\n" if mode == "variants" else f"{tk}\t{' '.join(ph)}\n")
    (out / "ids.txt").write_text("\n".join(ids) + "\n")
    print(name, len(ids), "utterances", len(prons), "words", file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("records", "batch"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("--where", default="True")
    ap.add_argument("--dict", default="canon", choices=("canon", "variants"))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "records":
        records()
    else:
        batch(a.name, a.where, a.dict, a.limit)


if __name__ == "__main__":
    main()
