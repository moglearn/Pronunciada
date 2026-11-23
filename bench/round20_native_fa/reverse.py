"""Reverse safety check of each fix on the CHANGED words (all practice items): how many changed words a fix raises,
how many it lifts out of the caught region at main's FA@80 threshold, and the largest per-kind AUC drop (check
half and fresh sets, from tables/gate_*.json).

    python -m bench.round20_native_fa.reverse
"""
from __future__ import annotations

import json

from bench.round20_native_fa import core as K
from bench.round20_native_fa.gate import KINDS
from bench.round20_native_fa.rebuild import ROOT

TAGS = {"B": "B0.02", "C": "C", "D1": "D1", "D2": "D2", "E": None, "B+C+E": "ALLnoD", "ALL": "ALL"}


def main():
    out = {}
    for lang in K.LANGS:
        base = K.word_rows(lang, None, fix_labels=True, measure=True)
        thr = K.threshold(base)
        b = {(r["item"], r["k"]): r for r in base if r["label"]}
        for name, tag in TAGS.items():
            if name == "E":
                tag = {"es": "Ees", "de": "Ede", "ko": "Eko"}.get(lang)
                if tag is None:
                    continue
            try:
                new = {(r["item"], r["k"]): r for r in K.word_rows(lang, tag, fix_labels=True, measure=True) if r["label"]}
            except FileNotFoundError:
                continue
            raised = sum(1 for k, r in new.items() if k in b and r["score"] > b[k]["score"] + 1e-9)
            lost = sum(1 for k, r in new.items() if k in b and b[k]["err"] >= thr > r["err"])
            gained = sum(1 for k, r in new.items() if k in b and r["err"] >= thr > b[k]["err"])
            out[f"{lang}|{name}"] = {"changed": len(b), "raised": raised, "caught_lost": lost, "caught_gained": gained}
    worst = {}
    for gname in ("gate_check.json", "gate_fresh.json"):
        G = json.loads((ROOT / "tables" / gname).read_text())
        for lang, d in G.items():
            for tag, v in d.items():
                if tag == "main":
                    continue
                drops = {k: v[f"auc_{k}"] - d["main"][f"auc_{k}"] for k in KINDS}
                k = min(drops, key=drops.get)
                worst[f"{gname[5:-5]}|{lang}|{tag}"] = {"kind": k, "d_auc": drops[k]}
    K.save_json("r20_reverse.json", {"changed_words": out, "worst_kind_auc_drop": worst})
    for k, v in out.items():
        print(k, v)
    for k, v in worst.items():
        if v["d_auc"] < -0.002:
            print("KIND DROP > 0.002:", k, v)


if __name__ == "__main__":
    main()
