"""Round 19 (Korean, part 2): HELD-OUT guards on FLEURS ko TEST, before / after, exactly paired.

    python -m bench.round19ko.guards2        # after: bench.round19ko.build2 test

Rows were collected with the switch off; "after" applies the shipped part-2 parameters with
bench.round19ko.fit2's formula, which equals the runtime (max |dP| 2.5e-4 on the dev items).
  swap      bench.eval_swap fleurs-swap test items (ko): near / far word AUC, hit@5 %, native words < 0.6
  planted   bench.eval_languages controlled errors on FLEURS test: native acceptance, words < 0.6,
            planted word-error AUC
  audit     word-level contrast audit (round-17 test plan): per contrast and direction, share of
            edited words below the 5 % quantile of native words (the round-18 "word hit@5 %")
95 % intervals of the differences: bootstrap over source clips (1000).
"""
from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round19ko import fit2 as F2
from bench.round19ko.cv_pool import work
from bench.third_party.lingolix import roc_auc
from pronunciada import ko_identity as KI


def after(rows):
    cal = KI._word_params()
    b = np.asarray(cal["coef"])
    return {k: F2.apply(r, F2.feats(r, cal["thr"], cal.get("ins_c", 2.0)), b, cal["cap_up"], F2.gate_of(cal)) for k, r in rows.items()}


def _unit(r, p, k):
    toks = list(re.finditer(r"\S+", r["meta"]["target"]))
    a, bb = toks[k].span()
    hit = [p[i] for i, (c, d) in enumerate(r["span"]) if c < bb and d > a]
    return min(hit) if hit else 0.0


def swap_units(rows, P):
    by = defaultdict(dict)
    for k, r in rows.items():
        by[r["meta"]["id"]][k] = r
    out = []
    for d in by.values():
        orig = [k for k, r in d.items() if not r["meta"]["label"]]
        if not orig:
            continue
        ch = {r["meta"]["label"]["bucket"]: _unit(r, P[k], r["meta"]["label"]["word_index"])
              for k, r in d.items() if r["meta"]["label"]}
        out.append({"nat": list(P[orig[0]]), "ch": ch})
    return out


def swap_m(units):
    nat = np.array([x for u in units for x in u["nat"]])
    t5 = np.quantile(nat, 0.05)
    out = {"native_below_0.6": float(np.mean(nat < 0.6))}
    for v in ("swap_near", "swap_far"):
        pos = np.array([u["ch"][v] for u in units if v in u["ch"]])
        out[f"{v}.auc"] = roc_auc([1] * len(pos) + [0] * len(nat), list(1 - pos) + list(1 - nat))
        out[f"{v}.hit5"] = float(np.mean(pos < t5))
    return out


def cor_units(rows, P):
    by = defaultdict(lambda: {"neg": [], "pos": []})
    for k, r in rows.items():
        if r["meta"]["cor_kind"] == "clean":
            by[r["meta"]["id"]]["neg"] += list(P[k])
        else:
            by[r["meta"]["id"]]["pos"] += [p for p, w in zip(P[k], r["word_wrong"] or []) if w]
    return list(by.values())


def cor_m(units):
    neg = np.array([x for u in units for x in u["neg"]])
    pos = [x for u in units for x in u["pos"]]
    return {"native_accept": float(np.mean(neg >= 0.5)), "native_below_0.6": float(np.mean(neg < 0.6)),
            "planted_auc": roc_auc([1] * len(pos) + [0] * len(neg), [1 - x for x in pos] + list(1 - neg))}


def audit_units(rows, P):
    by = defaultdict(lambda: {"nat": [], "ed": []})
    for k, r in rows.items():
        if r.get("site") is None:
            by[r["meta"]["id"]]["nat"] += list(P[k])
        else:       # every edit, as the audit counts them (a "*" marks edits whose new sound was already an alternative)
            s = r["site"]
            tag = f"{s['contrast']}/{s['direction']}" + ("*" if r.get("partner_in_alts") else "")
            by[r["meta"]["id"]]["ed"].append((tag, float(P[k][r["edit_word"]])))
    return list(by.values())


def audit_m(units):
    nat = np.array([x for u in units for x in u["nat"]])
    t5 = np.quantile(nat, 0.05)
    by = defaultdict(list)
    for u in units:
        for c, p in u["ed"]:
            by[c].append(p < t5)
            by["pooled"].append(p < t5)
    return {c: float(np.mean(v)) for c, v in sorted(by.items())}


def boot(ua, ub, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a0, b0 = fn(ua), fn(ub)
    d = defaultdict(list)
    for _ in range(n):
        idx = rng.integers(0, len(ua), len(ua))
        fa, fb = fn([ua[i] for i in idx]), fn([ub[i] for i in idx])
        for k in a0:
            if k in fa and k in fb:
                d[k].append(fb[k] - fa[k])
    return {k: {"before": a0[k], "after": b0[k], "delta_ci": [float(np.quantile(d[k], 0.025)), float(np.quantile(d[k], 0.975))],
                "n_boot": len(d[k])} for k in a0}


def main():
    D = pickle.loads((work() / "words_r2_test.pkl").read_bytes())
    rep = {}
    for name, units, fn in (("swap", swap_units, swap_m), ("planted", cor_units, cor_m), ("audit", audit_units, audit_m)):
        rows = D[{"swap": "swap", "planted": "test_cor", "audit": "audit"}[name]]
        P0 = {k: r["prob"] for k, r in rows.items()}
        P1 = after(rows)
        rep[name] = boot(units(rows, P0), units(rows, P1), fn)
        if name == "audit":
            rep[name]["n_edits"] = {c: sum(1 for r in rows.values() if r.get("site")
                                           and f"{r["site"]["contrast"]}/{r["site"]["direction"]}" + ("*" if r.get("partner_in_alts") else "") == c)
                                    for c in rep[name] if c != "pooled"}
    Path("results/tables/round19_ko2_guards_test.json").write_text(json.dumps(rep, indent=1, default=float))
    for g, d in rep.items():
        for k, v in d.items():
            if isinstance(v, dict) and "before" in v:
                print(f"{g:8s} {k:28s} {v['before']:.4f} -> {v['after']:.4f}  [{v['delta_ci'][0]:+.4f}, {v['delta_ci'][1]:+.4f}]")
    print("audit n", rep["audit"].get("n_edits"))


if __name__ == "__main__":
    main()
