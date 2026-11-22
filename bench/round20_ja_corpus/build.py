"""Aligner output -> mora-level segments, alignment QC, and the per-split manifests.

    python -m bench.round20_ja_corpus.build parse NAME     # align/NAME/out/*.json -> aligned/NAME.jsonl
    python -m bench.round20_ja_corpus.build manifest [--sample N] [--out DIR] [--batches a,b,...]
        -> corpus/{train,validation,heldback}.jsonl (+ counts.json)

Segment fields (one per sound, in time order):
  s       sound in Pronunciada's ZIPA convention ("k", "kj", "oː", "a", "N", "Q")
  k       kind: C consonant, V vowel, N moraic nasal, Q closure of a geminate (or of a final っ)
  t0, t1  start / end in seconds (10 ms aligner frames)
  long    V only: long vowel (one unit, two morae)
  gem     Q: closure of a geminate; C: release of a geminate
  fq      Q only: word-final っ (glottal closure, no following consonant)
  mora    index of the first mora the segment belongs to (long vowels take two)
  unit    index of the front-end mora unit (Reference syllable); w word index; ph phrase index
  pf      phrase-final: in the last mora unit before a pause >= 150 ms or the utterance end
  pp      pre-pausal: the segment directly before a pause >= 150 ms (utterance end not counted)
  uf      utterance-final mora unit
  dv      vowel devoiced by Open JTalk's labels; de: devoicing environment (C_V_C voiceless)
  prev, next   neighbouring sounds ("#" = pause >= 150 ms or utterance edge, "_" short pause)
Utterance fields: id, source, licence, speaker, split, tier, audio (relative to the work folder), text,
duration, words [{text, t0, t1}], segments, n_morae, speech_s, rate (morae per second of speech),
n_phrases, plus source-specific fields (cv_split, cv_real_hash_half, fleurs_id, book, reading_check).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from bench.round20_ja_corpus.common import ALIGN, CORPUS, WORK
from bench.round20_ja_corpus.units import variants

PAUSE = 0.150
SIL = {"", "sil", "sp", "spn", "<eps>"}


def _entries(tier) -> list[tuple[float, float, str]]:
    return [(float(a), float(b), str(c)) for a, b, c in tier["entries"]]


def segments_for(rec: dict, ali: dict) -> tuple[list[dict] | None, str]:
    """(segments, status). status: ok | variant:<name> | mismatch."""
    words = [w for w in rec["words"] if w["units"]]
    tiers = ali["tiers"]
    wtier = [e for e in _entries(tiers["words"]) if e[2] not in SIL]
    ptier = [e for e in _entries(tiers["phones"]) if e[2] not in SIL]
    if len(wtier) != len(words):
        return None, "mismatch"
    segs: list[dict] = []
    pi = 0
    unit_base = 0
    mora = 0
    for wi, (wd, (ws, we, _)) in enumerate(zip(words, wtier)):
        phs = []
        while pi < len(ptier) and ptier[pi][0] < we - 1e-4:
            if ptier[pi][0] >= ws - 1e-4:
                phs.append(ptier[pi])
            pi += 1
        canon = [p for u in wd["units"] for p in u["ph"]]
        got = [p[2] for p in phs]
        if got != canon:
            for name, ph, _ in variants(wd["units"]):
                if got == ph:
                    return None, "variant:" + name.split("@")[0]
            return None, "mismatch"
        k = 0
        for ui, u in enumerate(wd["units"]):
            uidx = unit_base + ui
            for j, p in enumerate(u["ph"]):
                t0, t1, _ = phs[k]
                k += 1
                seg = {"t0": round(t0, 3), "t1": round(t1, 3), "unit": uidx, "w": wi, "mora": mora}
                if p == "cl" and j == 0 and u["gem"]:
                    seg.update(s="Q", k="Q", gem=True, fq=False)
                elif p == "cl":
                    seg.update(s="Q", k="Q", gem=False, fq=True,
                               mora=mora + (1 if u["gem"] else 0) + (2 if u.get("long") else 1))
                elif p == "N":
                    seg.update(s="N", k="N", nas=u["nuc"])
                elif p in ("a", "i", "u", "e", "o", "aa", "ii", "uu", "ee", "oo"):
                    seg.update(s=u["nuc"], k="V", long=bool(u["long"]), dv=bool(u["devoiced"]), de=bool(u["devoice_env"]))
                    if u["gem"]:
                        seg["mora"] = mora + 1
                else:
                    cons = list(u["cons"])
                    if u["gem"]:
                        c0 = cons[0]
                        cons[0] = {"ttʃ": "tʃ", "tts": "ts", "ddʒ": "dʒ", "ddz": "dz"}.get(c0, c0[1:])
                    seg.update(s="".join(cons), k="C", gem=bool(u["gem"]))
                    if u["gem"]:
                        seg["mora"] = mora + 1
                segs.append(seg)
            mora += (1 if u["gem"] else 0) + (2 if u.get("long") else 1) + (1 if u["final_q"] else 0)
        unit_base += len(wd["units"])
    # pauses, phrases, neighbours
    end = float(ali.get("end", segs[-1]["t1"]))
    ph = 0
    for i, s in enumerate(segs):
        gap_after = (segs[i + 1]["t0"] - s["t1"]) if i + 1 < len(segs) else None
        s["ph"] = ph
        s["pp"] = bool(gap_after is not None and gap_after >= PAUSE - 1e-6)
        if s["pp"]:
            ph += 1
    last_unit_of_phrase: dict[int, int] = {}
    for s in segs:
        last_unit_of_phrase[s["ph"]] = max(last_unit_of_phrase.get(s["ph"], -1), s["unit"])
    last_unit = segs[-1]["unit"]
    for i, s in enumerate(segs):
        s["pf"] = s["unit"] == last_unit_of_phrase[s["ph"]]
        s["uf"] = s["unit"] == last_unit
        if i == 0 or segs[i - 1]["pp"] or s["t0"] - segs[i - 1]["t1"] >= PAUSE - 1e-6:
            s["prev"] = "#"
        else:
            s["prev"] = segs[i - 1]["s"] if s["t0"] - segs[i - 1]["t1"] < 0.005 else "_"
        if i + 1 == len(segs) or s["pp"]:
            s["next"] = "#"
        else:
            s["next"] = segs[i + 1]["s"] if segs[i + 1]["t0"] - s["t1"] < 0.005 else "_"
    _ = end
    return segs, "ok"


def parse(name: str) -> None:
    B = ALIGN / name
    ids = set((B / "ids.txt").read_text().split())
    out = WORK / "aligned"
    out.mkdir(exist_ok=True)
    cnt: Counter = Counter()
    found = {p.stem: p for p in (B / "out").rglob("*.json")}
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for rec in map(json.loads, open(WORK / "records.jsonl", encoding="utf-8")):
            if rec["id"] not in ids:
                continue
            p = found.get(rec["id"])
            if p is None:
                cnt["alignment_failed"] += 1
                continue
            ali = json.load(open(p, encoding="utf-8"))
            try:
                segs, st = segments_for(rec, ali)
            except (KeyError, IndexError):
                segs, st = None, "mismatch"
            cnt[st] += 1
            if segs is None:
                continue
            speech = sum(s["t1"] - s["t0"] for s in segs)
            n_mora = len({(s["mora"]) for s in segs}) + sum(1 for s in segs if s.get("long"))
            wds = [w for w in rec.pop("words") if w["units"]]
            rec["words"] = []
            for wi, w in enumerate(wds):
                ws = [s for s in segs if s["w"] == wi]
                rec["words"].append({"text": w["text"], "t0": ws[0]["t0"], "t1": ws[-1]["t1"]})
            rec.update(segments=segs, n_morae=n_mora, speech_s=round(speech, 3),
                       rate=round(n_mora / max(speech, 1e-3), 2), n_phrases=segs[-1]["ph"] + 1)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    json.dump(dict(cnt), open(out / f"{name}.qc.json", "w"), indent=1)
    print(name, dict(cnt), file=sys.stderr)


def manifest(batches: list[str], out_dir: Path, sample: int | None) -> None:
    import random

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {s: open(out_dir / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "validation", "heldback")}
    seen = set()
    rows = []
    for b in batches:
        for rec in map(json.loads, open(WORK / "aligned" / f"{b}.jsonl", encoding="utf-8")):
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            rows.append(rec)
    if sample:
        random.Random(7).shuffle(rows)
        per = {"train": int(sample * 0.8), "validation": sample // 10, "heldback": sample // 10}
        keep, n = [], Counter()
        for r in rows:
            if n[r["split"]] < per[r["split"]]:
                keep.append(r)
                n[r["split"]] += 1
        rows = keep
    for r in rows:
        r["duration"] = round(r.pop("dur"), 3)
        files[r["split"]].write(json.dumps(r, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    print("wrote", len(rows), "utterances to", out_dir, file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("parse", "manifest"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("--batches", default="")
    ap.add_argument("--out", default=str(CORPUS))
    ap.add_argument("--sample", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "parse":
        parse(a.name)
    else:
        manifest([b for b in a.batches.split(",") if b], Path(a.out), a.sample)


if __name__ == "__main__":
    main()
