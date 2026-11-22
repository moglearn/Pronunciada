"""Round 19 (Korean): our own Korean rows before / after the identity factor, exactly paired.

    python -m bench.round19ko.guards --split val     # FLEURS validation (dev): fleurs-val-cor + dev controls
    python -m bench.round19ko.guards --split test    # HELD-OUT, once, after freezing: swap-test + fleurs-test-cor

Rows are collected with the factor off (bench.round19ko.collect); "after" applies the shipped
parameters (pronunciada/data/calib/ko_identity_word.json + data/coach/ko_identity.npz) with the
runtime's own functions, at strictness 0.5, so before / after differ only by the factor.
Rows reported (the ten-language and swap benchmarks' definitions):
  swap: near / far word AUC (changed word vs the words of the original text), hit@5 % (changed words
        below the 5 % quantile of native words), native words < 0.6;
  planted: native word acceptance (P >= 0.5), word AUC of the eval_languages planted errors vs clean words.
95 % intervals and paired differences: bootstrap over source clips (1000).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round19ko import collect as CO
from bench.round19ko.select import _ref_from_row
from bench.third_party.lingolix import roc_auc
from pronunciada import ko_identity as KI


def after_probs(rows):
    from pronunciada.text import get_g2p

    g = get_g2p("ko")
    cal = KI._word_params()
    out = {}
    for k, r in rows.items():
        if "error" in r:
            continue
        m = r["meta"]
        ref = g(m["target"]) if m.get("cor_kind") in (None, "clean") else _ref_from_row(r, g)
        p = np.asarray(r["words"]["prob"], np.float64)
        if ref is None or [q.ipa for q in ref.phones] != list(r["phones"]["ipa"]):
            out[k] = p
            continue
        s = KI.evidence(ref, r["phones"]["V"].astype(np.float32))
        w = r["phones"]["word"]
        new = p.copy()
        for j in range(len(p)):
            f = KI.word_factor(cal, s[w == j], 0.5, sounds_prob=float(r["words"]["sounds"][j]))
            if f is not None:
                new[j] = p[j] * f
        out[k] = new
    return out


def swap_rows(rows, P):
    by = defaultdict(dict)
    for k, r in rows.items():
        if "error" in r:
            continue
        by[r["meta"]["id"]][k] = r
    units = []
    for sid, d in by.items():
        orig = [k for k, r in d.items() if not r["meta"]["label"]]
        if not orig:
            continue
        nat = list(P[orig[0]])
        ch = {}
        for k, r in d.items():
            lab = r["meta"]["label"]
            if lab:
                ch[lab["bucket"]] = _unit_prob(r, P[k], lab["word_index"])
        units.append({"nat": nat, "ch": ch})
    return units


def _unit_prob(r, p, k):
    import re

    text = r["meta"]["target"]
    toks = [m for m in re.finditer(r"\S+", text)]
    spans = r["words"]["span"]
    a, b = toks[k].span() if k < len(toks) else (0, 0)
    hit = [p[i] for i, (c, d) in enumerate(spans) if c < b and d > a]
    return min(hit) if hit else 0.0


def swap_metrics(units):
    nat = np.array([x for u in units for x in u["nat"]])
    out = {"native_below_0.6": float(np.mean(nat < 0.6))}
    t5 = np.quantile(nat, 0.05)
    for v in ("swap_near", "swap_far"):
        pos = np.array([u["ch"][v] for u in units if v in u["ch"]])
        out[v] = {"auc": roc_auc([1] * len(pos) + [0] * len(nat), list(1 - pos) + list(1 - nat)),
                  "hit5": float(np.mean(pos < t5)), "n": int(len(pos))}
    return out


def planted_rows(rows, P):
    by = defaultdict(lambda: {"neg": [], "pos": []})
    for k, r in rows.items():
        if "error" in r:
            continue
        sid = r["meta"]["id"]
        if r["meta"]["cor_kind"] == "clean":
            by[sid]["neg"] += list(P[k])
        else:
            by[sid]["pos"] += [p for p, w in zip(P[k], r.get("word_wrong") or []) if w]
    return list(by.values())


def planted_metrics(units):
    neg = [x for u in units for x in u["neg"]]
    pos = [x for u in units for x in u["pos"]]
    return {"native_accept": float(np.mean(np.array(neg) >= 0.5)),
            "planted_auc": roc_auc([1] * len(pos) + [0] * len(neg), [1 - x for x in pos + neg])}


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[pre + k] = v
    return out


def boot(units_a, units_b, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a0, b0 = flat(fn(units_a)), flat(fn(units_b))
    vals = defaultdict(list)
    for _ in range(n):
        idx = rng.integers(0, len(units_a), len(units_a))
        fa, fb = flat(fn([units_a[i] for i in idx])), flat(fn([units_b[i] for i in idx]))
        for k in fa:
            if isinstance(fa[k], float):
                vals[k].append(fb[k] - fa[k])
    out = {}
    for k in a0:
        if k in vals:
            v = np.array(vals[k])
            out[k] = {"before": a0[k], "after": b0[k], "delta_ci": [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), required=True)
    a = ap.parse_args()
    rep = {}
    if a.split == "test":
        S = CO.load("swap-test", "base")
        PS0 = {k: np.asarray(r["words"]["prob"]) for k, r in S.items() if "error" not in r}
        PS1 = after_probs(S)
        rep["swap"] = boot(swap_rows(S, PS0), swap_rows(S, PS1), swap_metrics)
        C = CO.load("fleurs-test-cor", "base")
    else:
        C = CO.load("fleurs-val-cor", "base")
    PC0 = {k: np.asarray(r["words"]["prob"]) for k, r in C.items() if "error" not in r}
    PC1 = after_probs(C)
    rep["planted"] = boot(planted_rows(C, PC0), planted_rows(C, PC1), planted_metrics)
    Path(f"results/tables/round19_ko_guards_{a.split}.json").write_text(json.dumps(rep, indent=1, default=float))
    for grp, d in rep.items():
        for k, v in d.items():
            print(f"{grp:8s} {k:28s} {v['before']:.4f} -> {v['after']:.4f}  delta CI [{v['delta_ci'][0]:+.4f}, {v['delta_ci'][1]:+.4f}]")


if __name__ == "__main__":
    main()
