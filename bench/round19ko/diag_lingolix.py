"""Round 19 (Korean): aggregate diagnosis of the Lingolix test 1 Korean results, by bucket.

    python -m bench.round19ko.diag_lingolix [--responses PATH] [--out results/tables/round19_ko_diag.json]

Diagnosis only (allowed by the held-out rule): AUC / FA@80 per edit bucket, where the false alarms
at the 80 % threshold come from, and how the lowest-scoring wrong picks arise. Nothing here is fitted.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from bench.third_party import lingolix as L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(L.responses_path("tier1_ko")))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    man = L.tests()["tier1_ko"]
    R = {}
    for line in Path(a.responses).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            R[r["id"]] = r
    c = L.collect(man, R)
    units = c["units"]
    labels = [l for u in units for l in u[0]]
    scores = [s for u in units for s in u[1]]
    pos = sorted((s for s, l in zip(scores, labels) if l), reverse=True)
    thr = pos[int(0.8 * len(pos) + 0.999) - 1]
    out = {"auc": L.roc_auc(labels, scores), "fa80": L.fpr_at_recall(labels, scores, 0.8), "thr80": thr,
           "n_pos": len(pos), "n_neg": len(scores) - len(pos)}
    # per bucket
    by = defaultdict(list)
    neg = [s for s, l in zip(scores, labels) if not l]
    for ls, ss, item, _ in units:
        lab = item.get("label") or {}
        b = lab.get("bucket") or lab.get("category") or "control"
        for l, s in zip(ls, ss):
            if l:
                by[b].append(s)
    out["bucket"] = {}
    for b, ps in sorted(by.items()):
        out["bucket"][b] = {"n": len(ps), "auc": L.roc_auc([1] * len(ps) + [0] * len(neg), ps + neg),
                            "caught_at_thr80": sum(p >= thr for p in ps), "median_signal": statistics.median(ps)}
    # sources of false alarms
    fa = []
    for ls, ss, item, _ in units:
        lab = item.get("label") or {}
        words = item["target"].split()
        wi = lab.get("word_index")
        for k, (l, s) in enumerate(zip(ls, ss)):
            if not l and s >= thr:
                fa.append({"id": item["id"], "word": words[k], "k": k, "n": len(words), "signal": round(s, 3),
                           "clip": "control" if not lab else lab.get("bucket") or lab.get("category"),
                           "dist_to_edit": None if wi is None else abs(k - wi),
                           "whisper_cer": (item.get("whisper") or {}).get("cer"),
                           "pos": "first" if k == 0 else "last" if k == len(words) - 1 else "mid"})
    out["false_alarms"] = fa
    out["fa_by_clip"] = Counter(f["clip"] for f in fa)
    out["fa_by_pos"] = Counter(f["pos"] for f in fa)
    out["fa_adjacent_to_edit"] = sum(1 for f in fa if f["dist_to_edit"] == 1)
    out["fa_whisper_mismatch"] = sum(1 for f in fa if (f["whisper_cer"] or 0) > 0)
    allpos = Counter("first" if k == 0 else "last" if k == len(u[0]) - 1 else "mid"
                     for u in units for k, l in enumerate(u[0]) if not l)
    out["neg_by_pos"] = allpos
    # localisation misses
    miss = []
    for ls, ss, item, _ in units:
        if 1 not in ls:
            continue
        p = ls.index(1)
        top = max(range(len(ss)), key=lambda i: ss[i])
        if top != p:
            words = item["target"].split()
            miss.append({"id": item["id"], "bucket": item["label"].get("bucket") or item["label"].get("category"),
                         "changed": words[p], "changed_signal": round(ss[p], 3), "picked": words[top],
                         "picked_signal": round(ss[top], 3), "adjacent": abs(top - p) == 1})
    out["loc_misses"] = miss
    out["loc_miss_adjacent"] = sum(m["adjacent"] for m in miss)
    for k in ("auc", "fa80", "thr80", "n_pos", "n_neg"):
        print(k, out[k])
    for b, v in out["bucket"].items():
        print(b, v)
    print("FA by clip", out["fa_by_clip"], "by pos", out["fa_by_pos"], "neg by pos", allpos)
    print("FA adjacent to edit", out["fa_adjacent_to_edit"], "whisper mismatch", out["fa_whisper_mismatch"])
    for f in sorted(fa, key=lambda f: -f["signal"]):
        print("  FA", f)
    print("loc misses", len(miss), "adjacent", out["loc_miss_adjacent"])
    for m in miss:
        print("  MISS", m)
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=dict) + "\n")


if __name__ == "__main__":
    main()
