"""Round 19 merge: trim the *_round19 rival tables to the re-run mode and print the summary table
(test, Pronunciada 1.0.2, Pronunciada round 19, OpenPronounce 1.1.0, round 19 - OpenPronounce).

    python bench/round19_merge/rivals_summary.py [--trim]
"""
from __future__ import annotations

import json
import sys

P = "results/tables/"
NOTE = ("round 19 merged code (frozen run 2, 2026-11-22); OpenPronounce 1.1.0 {mode} mode from its cached raw "
        "outputs (no OpenPronounce model was run again); the 1.0.2 table is {old}")


def trim():
    for f, keep, mode, old in (("rivals_so762_round19.json", ("protocol", "default"), "default", "rivals_so762.json"),
                               ("rivals_lingolix_round19.json", ("protocol", "fast"), "fast", "rivals_lingolix.json")):
        a = json.load(open(P + f))
        a = {k: v for k, v in a.items() if k in keep}
        a["note"] = NOTE.format(mode=mode, old=old)
        json.dump(a, open(P + f, "w"), indent=1, ensure_ascii=False)
    s = json.load(open(P + "rivals_shared_fast_round19.json"))
    s["note"] = NOTE.format(mode="fast", old="rivals_shared_fast.json") + \
        " (L2-ARCTIC and EpaDB were only run in fast and student modes)"
    json.dump(s, open(P + "rivals_shared_fast_round19.json", "w"), indent=1, ensure_ascii=False)


def table():
    L = ["| test | Pronunciada 1.0.2 | Pronunciada round 19 | OpenPronounce 1.1.0 | round 19 − OpenPronounce [95 % CI] | p |",
         "|---|---|---|---|---|---|"]
    o_so = json.load(open(P + "rivals_so762.json"))["default"]["groups"]
    n_so = json.load(open(P + "rivals_so762_round19.json"))["default"]["groups"]
    for g in ("everyone", "adults (18+)", "children (12 and under)", "teenagers (13-17)"):
        for t in ("sentence total", "sentence accuracy", "sentence fluency", "sentence prosodic", "word Pearson",
                  "sound Pearson"):
            o, n = o_so[g][t], n_so[g][t]
            d = n["delta"]
            L.append(f"| speechocean762, {g}: {t} | {o['pronunciada']['value']:.3f} | {n['pronunciada']['value']:.3f} | "
                     f"{n['openpronounce']['value']:.3f} | {d['value']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {d['p']:.3f} |")
    o_sh = {r["test"]: r for r in json.load(open(P + "rivals_shared_fast.json"))["rows"]}
    for n in json.load(open(P + "rivals_shared_fast_round19.json"))["rows"]:
        if n["group"] != "learner errors":
            continue
        o = o_sh[n["test"]]
        L.append(f"| {n['test']}, error-detection AUC (OpenPronounce fast) | {o['pronunciada']['value']:.3f} | "
                 f"{n['pronunciada']['value']:.3f} | {n['openpronounce']['value']:.3f} | {n['delta']:+.3f} "
                 f"[{n['delta_ci'][0]:+.3f}, {n['delta_ci'][1]:+.3f}] | {n['p']:.3f} |")
    o_l = json.load(open(P + "rivals_lingolix.json"))["fast"]
    n_l = json.load(open(P + "rivals_lingolix_round19.json"))["fast"]
    for lang in ("en", "fr", "es", "de"):
        for m, name in (("roc_auc", "word AUC"), ("fpr_at_recall_80", "FA@80")):
            o, n = o_l["tier1"][lang][m], n_l["tier1"][lang][m]
            L.append(f"| Lingolix test 1 {lang}, {name} (OpenPronounce fast) | {o['pronunciada']['value']:.3f} | "
                     f"{n['pronunciada']['value']:.3f} | {n['openpronounce']['value']:.3f} | {n['delta']:+.3f} "
                     f"[{n['delta_ci'][0]:+.3f}, {n['delta_ci'][1]:+.3f}] | {n['p']:.3f} |")
    for m in sorted(n_l.get("tier2", {})):
        o, n = o_l["tier2"].get(m), n_l["tier2"][m]
        if isinstance(n, dict) and "pronunciada" in n and "delta" in n:
            L.append(f"| Lingolix test 2, {m} (OpenPronounce fast) | {o['pronunciada']['value']:.3f} | "
                     f"{n['pronunciada']['value']:.3f} | {n['openpronounce']['value']:.3f} | {n['delta']:+.3f} "
                     f"[{n['delta_ci'][0]:+.3f}, {n['delta_ci'][1]:+.3f}] | {n['p']:.3f} |")
    print("\n".join(L))


if __name__ == "__main__":
    if "--trim" in sys.argv:
        trim()
    table()
