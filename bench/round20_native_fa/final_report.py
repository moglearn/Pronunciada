"""Markdown tables of the round-20 fix gate (reads tables/gate_check.json, gate_fresh.json and the dumps).

    python -m bench.round20_native_fa.final_report > fixes_tables.md
"""
from __future__ import annotations

import json

from bench.round20_native_fa import core as K
from bench.round20_native_fa.rebuild import ROOT

NAMES = {"B0.02": "B", "B": "B", "C": "C", "D1": "D1", "D2": "D2", "Ees": "E", "Ede": "E", "Eko": "E", "E": "E",
         "ALL": "ALL (B+C+D1+E)", "ALLnoD": "ALL without D (B+C+E)"}
KINDS = ("sub_subtle", "sub_moderate", "sub_gross", "indel", "word_omission")


def j(n):
    return json.loads((ROOT / "tables" / n).read_text())


def sig(p):
    return "" if p is None else (" *" if p < 0.05 else "")


def measuring_stick():
    print("### 0a / A: the measuring stick (all practice items)\n")
    print("| lang | as diagnosed: AUC / FA@80 | 0a label index fixed | 0a + A (punctuation tokens out, no-break-space numbers one token) |")
    print("|---|---|---|---|")
    out = {}
    for l in K.LANGS:
        vals = []
        for fx, ms in ((False, False), (True, False), (True, True)):
            rr = K.word_rows(l, None, fix_labels=fx, measure=ms)
            a, f = K.metrics(rr)
            neg = [r for r in rr if not r["label"]]
            vals.append((a, f, sum(r["score"] < 0.6 for r in neg) / len(neg)))
        out[l] = vals
        print(f"| {l} | " + " | ".join(f"{a:.4f} / {100 * f:.2f} % (< 0.6: {100 * b:.2f} %)" for a, f, b in vals) + " |")
    K.save_json("r20_measuring_stick.json", out)


def fix_table(name, title, langs):
    G = j(name)
    print(f"\n### {title}\n")
    print("FA@80 % and correct words < 0.6 % (relative change vs main), AUC; * = paired speaker-bootstrap p < 0.05.\n")
    print("| lang | config | FA@80 % | < 0.6 % | AUC | ΔFA@80 [95 % CI] p | ΔAUC [95 % CI] p |")
    print("|---|---|---|---|---|---|---|")
    for l in langs:
        if l not in G:
            continue
        m = G[l]["main"]
        print(f"| {l} | main | {100 * m['fa80']:.2f} | {100 * m['low06']:.2f} | {m['auc']:.4f} | | |")
        for t, v in G[l].items():
            if t == "main":
                continue
            if t.startswith("E") and len(t) == 3 and not t.endswith(l.split("@")[0]):
                continue
            pf, pa = v["paired"]["fa80"], v["paired"]["auc"]
            rf = (v["fa80"] - m["fa80"]) / m["fa80"]
            rl = (v["low06"] - m["low06"]) / m["low06"]
            print(f"| {l} | {NAMES.get(t, t)} | {100 * v['fa80']:.2f} ({100 * rf:+.0f} %) | {100 * v['low06']:.2f} "
                  f"({100 * rl:+.0f} %) | {v['auc']:.4f} | {100 * pf['d']:+.2f} [{100 * pf['ci'][0]:+.2f}, {100 * pf['ci'][1]:+.2f}]"
                  f" p={pf['p']:.3f}{sig(pf['p'])} | {pa['d']:+.4f} [{pa['ci'][0]:+.4f}, {pa['ci'][1]:+.4f}] p={pa['p']:.3f}{sig(pa['p'])} |")


def kind_table(name, title, langs, tags):
    G = j(name)
    print(f"\n### {title}: changed-word AUC by kind (each kind's changed words vs all correct words)\n")
    print("| lang | config | " + " | ".join(KINDS) + " |")
    print("|---|---|" + "---|" * len(KINDS))
    for l in langs:
        if l not in G:
            continue
        for t in ["main"] + [x for x in tags if x in G[l]]:
            if t.startswith("E") and len(t) == 3 and not t.endswith(l.split("@")[0]):
                continue
            v = G[l][t]
            print(f"| {l} | {NAMES.get(t, t)} | " + " | ".join(f"{v[f'auc_{k}']:.4f}" for k in KINDS) + " |")


def remaining(tag="ALLnoD", fold="check"):
    """Composition of the English false alarms left under `tag` (check half), with the diagnosis tests."""
    rows = [r for r in K.word_rows("en", tag, fix_labels=True, measure=True) if r["fold"] == fold]
    thr = K.threshold(rows)
    fa = [r for r in rows if not r["label"] and r["err"] >= thr]
    diag = {}
    for line in (ROOT / "tables" / "rows_en.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        diag.setdefault((d["item"], d["tok"]), d)
    from collections import Counter

    c = Counter()
    for r in fa:
        d = diag.get((r["item"], r["tok"]))
        if d is None:
            c["unmatched"] += 1
        elif d.get("asr_mismatch") and not d.get("text_risk") or d.get("truncated") or d.get("low_snr"):
            c["corpus noise (Whisper disagrees, truncated, low SNR)"] += 1
        elif d.get("text_risk"):
            c["text front-end (name, unknown word, number)"] += 1
        elif d.get("driver") == "stress":
            c["stress factor"] += 1
        elif d.get("neighbour"):
            c["near the change (not resolved)"] += 1
        elif 0 <= d.get("tail_margin", 9) <= 1:
            c["tail (within 1 logit)"] += 1
        else:
            c["other"] += 1
    print(f"\n### English false alarms left under {NAMES.get(tag, tag)} (check half, n = {len(fa)})\n")
    for k, v in c.most_common():
        print(f"- {k}: {v} ({100 * v / len(fa):.0f} %)")
    K.save_json("r20_remaining_en.json", {"n": len(fa), "composition": dict(c)})


def main():
    measuring_stick()
    fix_table("gate_check.json", "Practice sets, check half (speaker-disjoint from the fit half)", K.LANGS)
    fix_table("gate_fresh.json", "Fresh confirmation sets (scored once, frozen fixes)",
              ["en@fresh", "de@fresh", "es@fresh", "fr@fresh"])
    kind_table("gate_check.json", "Practice check half", K.LANGS, ["B0.02", "C", "D1", "D2", "Ees", "Ede", "Eko", "ALL", "ALLnoD"])
    kind_table("gate_fresh.json", "Fresh sets", ["en@fresh", "de@fresh", "es@fresh", "fr@fresh"],
               ["B", "C", "D1", "D2", "E", "ALL", "ALLnoD"])
    remaining("ALLnoD")
    remaining("ALL")


if __name__ == "__main__":
    main()
