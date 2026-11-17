"""Round 17, Mandarin: does the aspiration evidence help on real learners (OMPAL)?

    python -m bench.round17.ompal [--split dev] [--model path/to/contrast_heads.json]

Per character (learners only): the round-12b coaching score of the syllable's initial (max
over its initial sounds) vs the same maxed with 1 - sigmoid(z) of the head-based aspiration
evidence; AUC for "consonant marked wrong", and the precision / recall of characters with an
aspiration item at each bar. Inputs: bench/coach/dump_r4.py (ompal-<split>).
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from bench.coach.gather import out_dir


def main():
    from bench.coach.fit_v2 import add_initial
    from bench.data import load
    from bench.eval_ompal import overlap, syllable_char_spans, word_char_spans
    from pronunciada import attr_heads as AH
    from pronunciada import coach as Co
    from scipy.stats import rankdata

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="pronunciada/data/coach/contrast_heads.json")
    a = ap.parse_args()
    cfg = [e for e in json.load(open(a.model))["langs"]["zh"] if e["contrast"] == "aspiration"][0]
    D = pickle.loads((out_dir() / f"dump-r4-ompal-{a.split}.pkl").read_bytes())
    M = Co.model()
    heads = AH.shipped()
    rows = []
    for rec in load("ompal", split=a.split):
        lab = rec["labels"]
        if lab["native"] or lab["word_level_dropped"]:
            continue
        o = D.get(rec["id"])
        if o is None or "error" in o:
            continue
        X = add_initial(dict(o["sounds"]))
        ep = Co.error_prob(X, M, "zh")
        V = o["vec"].astype(np.float32)
        mem = set(cfg["members"])
        idx = [i for i, e in enumerate(X["expected"]) if e in mem]
        pc = np.full(len(ep), np.nan)
        if idx:
            P = heads.predict(V, idx, AH.context_vectors(V))
            for i, p in zip(idx, P):
                cls = AH.labels(X["expected"][i])["aspirated"]
                q = min(max(p["aspirated"][cls], 1e-6), 1 - 1e-6)
                pc[i] = 1 / (1 + np.exp(-cfg["slope"] * (np.log(q / (1 - q)) - cfg["s0"])))
        spans = word_char_spans(rec["text"], [w["text"] for w in o["words"]])
        syl = {}
        for wi, w in enumerate(o["words"]):
            if spans[wi] is not None:
                for k, sp in enumerate(syllable_char_spans(spans[wi], w["n_syl"])):
                    syl[(wi, k)] = sp
        for u in lab["units"]:
            if u["span"] is None:
                continue
            ini = [i for i in range(len(ep)) if X["initial"][i] and (int(X["word"][i]), int(X["syl"][i])) in syl
                   and overlap(*u["span"], *syl[(int(X["word"][i]), int(X["syl"][i]))])]
            if not ini:
                continue
            e0 = max(ep[i] for i in ini)
            pmin = min([pc[i] for i in ini if np.isfinite(pc[i])] or [1.0])
            asp = [X["expected"][i] for i in ini if np.isfinite(pc[i])]
            rows.append({"wrong": u["consonant"] == 0, "ep": e0, "pc": pmin, "member": bool(asp),
                         "aspirated_expected": any(AH.labels(x)["aspirated"] == "yes" for x in asp)})
    y = np.array([r["wrong"] for r in rows])
    e0 = np.array([r["ep"] for r in rows])
    pc = np.array([r["pc"] for r in rows])

    def auc(s):
        r = rankdata(s)
        n1 = y.sum()
        return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))

    out = {"units": len(rows), "consonant_wrong": int(y.sum()),
           "auc_coaching_score": auc(e0), "auc_with_aspiration": auc(np.maximum(e0, 1 - pc)),
           "auc_aspiration_alone": auc(1 - pc)}
    mem = np.array([r["member"] for r in rows])
    out["auc_aspiration_alone_members"] = float(
        (lambda yy, s: (rankdata(s)[yy].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (len(yy) - yy.sum())))(y[mem], 1 - pc[mem]))
    out["bars"] = []
    for bar in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        f = pc < bar
        out["bars"].append({"bar": bar, "flagged": int(f.sum()), "precision": float(y[f].mean()) if f.any() else None,
                            "recall": float(y[f].sum() / y.sum())})
    for s in ("careful", "balanced", "eager"):
        t = Co.sound_threshold(M, "zh", s, True)
        f0 = e0 >= t
        f1 = f0 | (pc < cfg["coach"][s])
        out[s] = {"coach_only": {"flagged": int(f0.sum()), "precision": float(y[f0].mean()) if f0.any() else None,
                                 "recall": float(y[f0].sum() / y.sum())},
                  "with_aspiration": {"flagged": int(f1.sum()), "precision": float(y[f1].mean()) if f1.any() else None,
                                      "recall": float(y[f1].sum() / y.sum())}}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"results/tables/round17_ompal_{a.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
