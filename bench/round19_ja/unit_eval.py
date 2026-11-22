"""Benchmark-style word metrics on a dump (bench/round19_ja/dump.py) of clips that carry `units`
and a `label` like the Lingolix test-1 items (and the round-19 dev set, built the same way).

Each system word is placed in the text by character search (as the benchmark's align_spans does
with the vendor's word text) and a unit gets the lowest word score overlapping it. Error signal =
1 - score; AUC over all units (planted unit = positive). FA@80 = share of correct units at or above
the threshold that catches 80 % of planted units.

    python -m bench.round19_ja.unit_eval DUMP.jsonl [--by bucket] [--diag]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def roc_auc(labels, scores) -> float:
    from scipy.stats import rankdata

    y = np.asarray(labels, bool)
    s = np.asarray(scores, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return math.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fa_at(labels, scores, target=0.8) -> float:
    pos = sorted((s for s, l in zip(scores, labels) if l), reverse=True)
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return math.nan
    thr = pos[min(int(math.ceil(target * len(pos))) - 1, len(pos) - 1)]
    return sum(1 for s in neg if s >= thr) / len(neg)


def unit_words(rec: dict, prob_key: str = "prob"):
    """[(unit, [words overlapping])] with words placed by text search like the benchmark."""
    text = rec["text"]
    spans, cur = [], 0
    for w in rec["words"]:
        t = _PUNCT.sub("", w["text"])
        j = text.find(t, cur) if t else -1
        if j >= 0:
            spans.append((j, j + len(t), w))
            cur = j + len(t)
    out = []
    for u in rec["units"]:
        out.append((u, [w for s, e, w in spans if s < u["end"] and e > u["start"]]))
    return out


def units_of(recs, prob_fn=None):
    """[(clip_id, group, label(0/1), signal, bucket, unit, words)] for every unit."""
    prob_fn = prob_fn or (lambda w: w["prob"])
    rows = []
    for r in recs:
        if r.get("error"):
            continue
        lab = r.get("label") or {}
        if lab and not lab.get("word_level_scorable", True):
            continue
        pos = lab.get("word_index") if lab else None
        b = lab.get("bucket") or ("control" if not lab else lab.get("category"))
        for k, (u, ws) in enumerate(unit_words(r)):
            sig = 1.0 if not ws else 1.0 - min(prob_fn(w) for w in ws)
            rows.append((r["id"], r.get("group", r["id"]), int(k == pos), sig, b if k == pos else "control", u, ws))
    return rows


def summary(rows, boot: int = 1000, seed: int = 0) -> dict:
    y = [x[2] for x in rows]
    s = [x[3] for x in rows]
    out = {"n_units": len(y), "n_pos": int(sum(y)), "auc": roc_auc(y, s), "fa80": fa_at(y, s, 0.8)}
    by = defaultdict(list)
    for x in rows:
        by[x[4]].append(x)
    neg = [x for x in rows if not x[2]]
    out["by_bucket"] = {b: {"n": len(v), "auc": roc_auc([1] * len(v) + [0] * len(neg),
                                                       [x[3] for x in v] + [x[3] for x in neg])}
                        for b, v in sorted(by.items()) if b != "control"}
    if rows and all(w.get("flag") is not None for x in rows for w in x[6]):
        fl = [(not x[6]) or any(w["flag"] for w in x[6]) for x in rows]
        out["flag"] = {"recall": float(np.mean([f for f, x in zip(fl, rows) if x[2]])),
                       "fa": float(np.mean([f for f, x in zip(fl, rows) if not x[2]])),
                       "length_caught": int(sum(f for f, x in zip(fl, rows) if x[2] and x[4] == "length")),
                       "length_n": int(sum(1 for x in rows if x[2] and x[4] == "length"))}
    if boot:
        groups = defaultdict(list)
        for x in rows:
            groups[x[1]].append(x)
        keys = sorted(groups)
        rng = random.Random(seed)
        A, F = [], []
        for _ in range(boot):
            pick = [x for _ in keys for x in groups[keys[rng.randrange(len(keys))]]]
            yy, ss = [x[2] for x in pick], [x[3] for x in pick]
            A.append(roc_auc(yy, ss))
            F.append(fa_at(yy, ss))
        out["auc_ci"] = [float(np.nanpercentile(A, 2.5)), float(np.nanpercentile(A, 97.5))]
        out["fa80_ci"] = [float(np.nanpercentile(F, 2.5)), float(np.nanpercentile(F, 97.5))]
    return out


def paired(rows_a, rows_b, boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap (resampling groups) of AUC and FA@80 differences b - a on the same units."""
    assert [(x[0], x[5]["start"]) for x in rows_a] == [(x[0], x[5]["start"]) for x in rows_b]
    groups = defaultdict(list)
    for i, x in enumerate(rows_a):
        groups[x[1]].append(i)
    keys = sorted(groups)
    y = np.array([x[2] for x in rows_a])
    sa = np.array([x[3] for x in rows_a])
    sb = np.array([x[3] for x in rows_b])
    d0 = (roc_auc(y, sb) - roc_auc(y, sa), fa_at(y, sb) - fa_at(y, sa))
    rng = random.Random(seed)
    D = []
    for _ in range(boot):
        idx = [i for _ in keys for i in groups[keys[rng.randrange(len(keys))]]]
        D.append((roc_auc(y[idx], sb[idx]) - roc_auc(y[idx], sa[idx]),
                  fa_at(y[idx], sb[idx]) - fa_at(y[idx], sa[idx])))
    D = np.array(D)
    res = {}
    for j, k in enumerate(("auc", "fa80")):
        v = D[:, j]
        v = v[np.isfinite(v)]
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        res[k] = {"delta": float(d0[j]), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                  "p": float(min(1.0, p))}
    return res


def load(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args(argv)
    compare(a.dumps, a.boot)



def compare(paths: list[str], boot: int = 500, sources=("all", "cv", "fleurs")) -> None:
    """One line per dump and source: AUC, FA@80, AUC by bucket; paired deltas against the first."""
    base = None
    recs0 = load(paths[0])
    for p in paths:
        recs = load(p)
        for src in sources:
            rs = [r for r in recs if src == "all" or r.get("source") == src]
            rows = units_of(rs)
            s = summary(rows, boot)
            bb = " ".join(f"{b[:6]}={v['auc']:.3f}" for b, v in s["by_bucket"].items())
            line = (f"{Path(p).stem:28s} {src:6s} n+={s['n_pos']:3d} AUC {s['auc']:.4f} "
                    f"[{s['auc_ci'][0]:.3f},{s['auc_ci'][1]:.3f}] FA80 {s['fa80']:.3f} | {bb}")
            if "flag" in s:
                f = s["flag"]
                line += f" | flag R {f['recall']:.2f} FA {f['fa']:.3f} length {f['length_caught']}/{f['length_n']}"
            line += f" | FA80 CI [{s['fa80_ci'][0]:.3f},{s['fa80_ci'][1]:.3f}]"
            if p != paths[0]:
                r0 =units_of([r for r in recs0 if src == "all" or r.get("source") == src])
                d = paired(r0, rows, boot)
                line += f" | dAUC {d['auc']['delta']:+.4f} p={d['auc']['p']:.3f} dFA {d['fa80']['delta']:+.3f} p={d['fa80']['p']:.3f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
