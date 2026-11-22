"""Word rows of a scored practice set and the benchmark-style metric (bench.third_party.lingolix rules).

A word row is one prompt token of one item: label 1 for the changed word of a changed item, else 0
(control items: every word; changed items: every other word). Its error signal is
1 - benchmark word score, exactly as lingolix.collect computes it (align on keys, missing word = 1).
FA@80: the error-signal threshold that catches 80 % of the changed words; a correct word at or
above it is a false alarm. Intervals: 1,000 bootstrap draws over speakers (FLEURS clips count as
their own speaker; FLEURS ko clips are grouped by sentence id, as in round 19).
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict

from bench.round20_native_fa.collect import items as load_items
from bench.round20_native_fa.collect import load as load_dump
from bench.round20_native_fa.rebuild import ROOT
from bench.third_party import lingolix as L

LANGS = ("en", "de", "ko", "es", "fr")
APP_THRESHOLD = 0.6
MAX_CER = 0.1            # Korean CV read-the-prompt rule of round 19 (Whisper large-v3)


def practice_items(lang: str) -> list[dict]:
    its = load_items(lang)
    if lang == "ko":                     # round-19 rule: CV clips kept only if Whisper CER <= 0.1
        its = [it for it in its if it["source"] != "cv" or (it["asr_cer"] is not None and it["asr_cer"] <= MAX_CER)]
    return its


def word_rows(lang: str) -> list[dict]:
    its = practice_items(lang)
    dump = load_dump(lang)
    rows = []
    for it in its:
        rec = dump.get(it["id"])
        if rec is None or rec.get("error"):
            continue
        lab = it.get("label") or {}
        toks = it["target"].split()
        al = L.align(toks, rec["words"])
        pos = lab.get("word_index")
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", it["target"])]
        for k, tok in enumerate(toks):
            a, b = spans[k]
            iw = [j for j, w in enumerate(rec["rwords"]) if w["span"][0] < b and w["span"][1] > a]
            rows.append({"lang": lang, "item": it["id"], "clip": it["clip"], "speaker": it["speaker"],
                         "source": it["source"], "k": k, "tok": tok, "label": int(k == pos), "pos": pos,
                         "err": L.error_signal(al[k]), "score": 1.0 - L.error_signal(al[k]),
                         "kind": lab.get("bucket") or "control", "n_tok": len(toks), "iw": iw,
                         "changed_item": bool(lab), "unaligned": al[k] is None})
    return rows


def threshold(rows, target=0.8) -> float:
    pos = sorted((r["err"] for r in rows if r["label"]), reverse=True)
    return pos[min(int(math.ceil(target * len(pos))) - 1, len(pos) - 1)]


def metrics(rows) -> tuple[float, float]:
    labels = [r["label"] for r in rows]
    scores = [r["err"] for r in rows]
    return L.roc_auc(labels, scores), L.fpr_at_recall(labels, scores, 0.8)


def boot(rows, fn, n=1000, seed=0):
    """Percentile interval of fn(rows) over speaker-cluster bootstrap draws."""
    g = defaultdict(list)
    for r in rows:
        g[r["speaker"]].append(r)
    groups = list(g.values())
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        v = fn([r for gg in pick for r in gg])
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return [math.nan, math.nan]
    if isinstance(vals[0], (tuple, list)):
        out = []
        for j in range(len(vals[0])):
            s = sorted(v[j] for v in vals)
            out.append([s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))]])
        return out
    s = sorted(vals)
    return [s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))]]


def dump_rec(lang: str) -> dict:
    return load_dump(lang)


def save_json(name: str, obj) -> None:
    p = ROOT / "tables" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
