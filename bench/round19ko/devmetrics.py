"""Round 19 (Korean): Lingolix-style word metrics on the planted dev set.

Units = whitespace tokens of the reference text (bench.textswap.units); a unit's score is the lowest
probability of our words that overlap it (the benchmark's align_spans rule); error signal = 1 - score.
Positives = the changed unit of each changed item; negatives = every other unit of every item
(controls and changed items). Metrics: pooled ROC AUC, FA@80 (share of negatives at or above the
signal that catches 80 % of the positives), lowest-in-sentence rate, AUC per kind, the same per
source (cv / fleurs-val), and 95 % intervals by resampling speakers (CV speaker / FLEURS sentence id).
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np

from bench.third_party.lingolix import fpr_at_recall, roc_auc


@lru_cache(maxsize=None)
def _unit_spans(text: str) -> tuple:
    from bench import textswap as TS
    from pronunciada.text import get_g2p

    ref = get_g2p("ko")(text)
    return tuple(tuple(u.span) for u in TS.units(ref, "ko"))


def unit_rows(rows: dict, probs: dict | None = None) -> list[dict]:
    """rows: collect.load(...) output; probs: optional {item: per-word probabilities} replacing the
    stored ones. Returns per item {labels, signals, bucket, src, speaker}."""
    out = []
    for item, r in rows.items():
        if "error" in r:
            continue
        m = r["meta"]
        wp = np.asarray(probs[item] if probs is not None else r["words"]["prob"], np.float64)
        spans = r["words"]["span"]
        us = _unit_spans(m["target"])
        sig = []
        for a, b in us:
            hit = [wp[i] for i, (c, d) in enumerate(spans) if c < b and d > a]
            sig.append(1.0 - min(hit) if hit else 1.0)
        lab = m.get("label") or {}
        k = lab.get("word_index")
        labels = [int(i == k) for i in range(len(us))]
        out.append({"item": item, "labels": labels, "signals": sig, "bucket": lab.get("bucket", "control"),
                    "src": m["src"], "speaker": m["speaker"]})
    return out


def _core(units: list[dict]) -> dict:
    L = [l for u in units for l in u["labels"]]
    S = [s for u in units for s in u["signals"]]
    neg = [s for l, s in zip(L, S) if not l]
    res = {"auc": roc_auc(L, S), "fa80": fpr_at_recall(L, S, 0.8), "n_pos": int(sum(L)), "n_neg": len(neg)}
    hit = n = 0
    for u in units:
        if 1 in u["labels"]:
            n += 1
            hit += int(int(np.argmax(u["signals"])) == u["labels"].index(1))
    res["loc"] = hit / n if n else float("nan")
    by = defaultdict(list)
    for u in units:
        for l, s in zip(u["labels"], u["signals"]):
            if l:
                by[u["bucket"]].append(s)
    res["bucket"] = {b: roc_auc([1] * len(p) + [0] * len(neg), p + neg) for b, p in sorted(by.items())}
    return res


def metrics(rows: dict, probs: dict | None = None, boot: int = 0, seed: int = 0) -> dict:
    units = unit_rows(rows, probs)
    out = {"all": _core(units)}
    for src in sorted({u["src"] for u in units}):
        out[src] = _core([u for u in units if u["src"] == src])
    if boot:
        spk = defaultdict(list)
        for u in units:
            spk[u["speaker"]].append(u)
        keys = sorted(spk)
        rng = np.random.default_rng(seed)
        vals = defaultdict(list)
        for _ in range(boot):
            pick = [u for k in rng.choice(len(keys), len(keys)) for u in spk[keys[k]]]
            c = _core(pick)
            vals["auc"].append(c["auc"])
            vals["fa80"].append(c["fa80"])
        out["all"]["ci"] = {k: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))] for k, v in vals.items()}
    return out


def paired(rows: dict, probs_a: dict | None, probs_b: dict | None, boot: int = 1000, seed: int = 0) -> dict:
    """Speaker-bootstrap of the difference b - a in AUC and FA@80 (same items)."""
    ua, ub = unit_rows(rows, probs_a), unit_rows(rows, probs_b)
    spk = defaultdict(list)
    for i, u in enumerate(ua):
        spk[u["speaker"]].append(i)
    keys = sorted(spk)
    rng = np.random.default_rng(seed)
    d = defaultdict(list)
    for _ in range(boot):
        idx = [i for k in rng.choice(len(keys), len(keys)) for i in spk[keys[k]]]
        a, b = _core([ua[i] for i in idx]), _core([ub[i] for i in idx])
        d["auc"].append(b["auc"] - a["auc"])
        d["fa80"].append(b["fa80"] - a["fa80"])
    out = {}
    for k, v in d.items():
        v = np.asarray(v)
        out[k] = {"ci": [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))],
                  "p_not_better": float(np.mean(v <= 0) if k == "auc" else np.mean(v >= 0))}
    return out


def line(m: dict) -> str:
    a = m["all"]
    s = f"AUC {a['auc']:.4f} FA@80 {a['fa80']:.4f} loc {a['loc']:.3f}"
    if "ci" in a:
        s += f" [AUC {a['ci']['auc'][0]:.3f},{a['ci']['auc'][1]:.3f}; FA {a['ci']['fa80'][0]:.3f},{a['ci']['fa80'][1]:.3f}]"
    s += " | " + " ".join(f"{b} {v:.3f}" for b, v in a["bucket"].items())
    for src in ("cv", "fleurs-val"):
        if src in m:
            s += f" | {src} AUC {m[src]['auc']:.4f} FA {m[src]['fa80']:.4f}"
    return s
