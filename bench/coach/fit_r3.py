"""Round 3 thresholds: extra speech and delivery flags from FLEURS VALIDATION natives
(bench/coach/dump_r3.py), sanity check on speechocean762 TRAIN expert fluency / prosody.

    python -m bench.coach.fit_r3

Rules (fixed in advance):
  extra_speech   per language, the smallest "decoded sounds in the segment" bar n with <= 2 %
                 of native clips (correct prompt) having any segment of >= n sounds. Hit rate =
                 share of clips scored against the prompt with one inner word removed where a
                 segment of >= n sounds overlaps that word (+- 0.1 s).
  long_pause     96th percentile of the longest pause per native clip
  pause_mid_phrase  96th percentile of the longest pause not at punctuation
  rushed_punctuation  per mark kind (stop / comma), enabled only where natives leave >= 0.1 s
                 at >= 60 % of such marks; bar = the largest gap g in {0.02 .. 0.30} with
                 <= 3 % (stop) and then <= 5 % in total (stop + comma) of native clips flagged
  flat_pitch     4th percentile of the pitch range (semitones), clips of >= 3 words
  slow / fast    4th / 96th percentile of the articulation rate (sounds / s of speech)
Writes the "extra_speech" and "delivery" entries of pronunciada/data/coach/model.json and
results/tables/coaching_r3.json.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.coach.gather import out_dir
from pronunciada.delivery_coach import delivery_flags

LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
GRID = np.round(np.arange(0.02, 0.32, 0.02), 2)


def extra_rate(rows, n):
    return float(np.mean([any(g["n_sounds"] >= n for g in r["extra"]) for r in rows])) if rows else None


def extra_hit(rows, n):
    hits = []
    for r in rows:
        R = r.get("removed")
        if not R:
            continue
        a, b = R["start"] - 0.1, R["end"] + 0.1
        hits.append(any(g["n_sounds"] >= n and g["start"] < b and g["end"] > a for g in R["extra"]))
    return (float(np.mean(hits)) if hits else None), len(hits)


def fit_lang(rows):
    T = {}
    mx = [max([p["duration"] for p in r["delivery"]["pauses"]] or [0.0]) for r in rows]
    T["long_pause_s"] = round(float(np.quantile(mx, 0.96)), 3)
    mm = [max([p["duration"] for p in r["delivery"]["pauses"] if p["punctuation"] is None] or [0.0]) for r in rows]
    T["mid_pause_s"] = round(max(float(np.quantile(mm, 0.96)), 0.25), 3)
    rushed, stats = {}, {}
    for kind, budget in (("stop", 0.03), ("comma", 0.05)):
        gaps = [m["gap"] for r in rows for m in r["delivery"]["marks"] if m["kind"] == kind]
        share = float(np.mean(np.asarray(gaps) >= 0.1)) if gaps else 0.0
        stats[kind] = {"n_marks": len(gaps), "native_pause_share": share}
        if len(gaps) < 30 or share < 0.6:
            rushed[kind] = None
            continue
        best = None
        for g in GRID:
            trial = dict(rushed, **{kind: float(g)})
            rate = np.mean([any(m["gap"] < trial.get(m["kind"]) for m in r["delivery"]["marks"]
                                if trial.get(m["kind"]) is not None) for r in rows])
            if rate <= budget:
                best = float(g)
        rushed[kind] = best
    T["rushed"] = rushed
    fr = [r["delivery"]["f0_range_st"] for r in rows if r["delivery"]["f0_range_st"] is not None and r["n_words"] >= 3]
    T["flat_range_st"] = round(float(np.quantile(fr, 0.04)), 2) if len(fr) >= 50 else None
    rt = [r["delivery"]["articulation_rate"] for r in rows if r["delivery"]["n_sounds"] >= 8]
    T["slow_rate"] = round(float(np.quantile(rt, 0.04)), 3)
    T["fast_rate"] = round(float(np.quantile(rt, 0.96)), 3)
    return T, stats


def flag_rates(rows, T):
    kinds = ("long_pause", "pause_mid_phrase", "rushed_punctuation", "flat_pitch", "slow", "fast")
    c = defaultdict(int)
    for r in rows:
        ks = {k for k, _ in delivery_flags(r["delivery"], T, r["n_words"])}
        for k in kinds:
            c[k] += k in ks
        c["any"] += bool(ks)
    return {k: c[k] / max(len(rows), 1) for k in kinds + ("any",)}


def main():
    D = pickle.loads((out_dir() / "r3.pkl").read_bytes())
    fl = defaultdict(list)
    for rid, r in D["fleurs"].items():
        if "error" not in r:
            fl[r["lang"]].append(r)
    rep = {"extra_speech": {}, "delivery": {}, "delivery_rates": {}, "punct_stats": {}}
    mdl_path = Path("pronunciada/data/coach/model.json")
    mdl = json.loads(mdl_path.read_text())
    ms = {}
    for l in LANGS:
        rows = fl[l]
        n = next((n for n in range(1, 13) if extra_rate(rows, n) <= 0.02), 13)
        ms[l] = n
        hit, nh = extra_hit(rows, n)
        rep["extra_speech"][l] = {"min_sounds": n, "native_rate": extra_rate(rows, n), "hit_rate": hit, "n_removed": nh,
                                  "curve": {k: {"native": extra_rate(rows, k), "hit": extra_hit(rows, k)[0]} for k in range(1, 9)}}
        T, st = fit_lang(rows)
        rep["delivery"][l] = T
        rep["punct_stats"][l] = st
        rep["delivery_rates"][l] = flag_rates(rows, T)
    mdl["extra_speech"] = {"min_sounds": {**ms, "default": max(ms.values())}}
    mdl["delivery"] = rep["delivery"]
    mdl_path.write_text(json.dumps(mdl, indent=1))
    # sanity: speechocean762 train, English thresholds
    T = rep["delivery"]["en"]
    S = [r for r in D["so762"].values() if "error" not in r]
    san = {}
    for kind in ("long_pause", "pause_mid_phrase", "rushed_punctuation", "flat_pitch", "slow", "fast", "any"):
        flu, pro, f = [], [], []
        for r in S:
            ks = {k for k, _ in delivery_flags(r["delivery"], T, r["n_words"])}
            f.append(bool(ks) if kind == "any" else kind in ks)
            flu.append(r["labels"]["fluency"])
            pro.append(r["labels"]["prosodic"])
        f, flu, pro = np.array(f), np.array(flu, float), np.array(pro, float)
        san[kind] = {"n_flagged": int(f.sum()), "share": float(f.mean()),
                     "fluency_flagged": float(flu[f].mean()) if f.any() else None,
                     "fluency_rest": float(flu[~f].mean()),
                     "prosodic_flagged": float(pro[f].mean()) if f.any() else None,
                     "prosodic_rest": float(pro[~f].mean())}
    rep["so762_train_sanity"] = san
    # extra speech on speechocean762 train (no labels for insertions; rate only)
    rep["so762_train_extra_rate"] = float(np.mean([any(g["n_sounds"] >= ms["en"] for g in r["extra"]) for r in S]))
    Path("results/tables/coaching_r3.json").write_text(json.dumps(rep, indent=1))
    for l in LANGS:
        e = rep["extra_speech"][l]
        print(l, "extra n", e["min_sounds"], "native", round(e["native_rate"], 3), "hit", e["hit_rate"] and round(e["hit_rate"], 3),
              "| delivery", rep["delivery"][l], {k: round(v, 3) for k, v in rep["delivery_rates"][l].items()})
    print(json.dumps(san, indent=0))


if __name__ == "__main__":
    main()
