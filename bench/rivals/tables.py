"""Markdown tables for results/round19_rivals.md from the rivals_* JSON tables.

    python -m bench.rivals.tables > /tmp/tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

T = Path(__file__).resolve().parents[2] / "results" / "tables"


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def ci(v: dict) -> str:
    c = v.get("ci") or [None, None]
    return f"{f3(v['value'])} [{f3(c[0])}, {f3(c[1])}]"


def pv(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def mark(d: float, p: float) -> str:
    if p >= 0.05:
        return "level"
    return "Pronunciada ahead" if d > 0 else "OpenPronounce ahead"


def so762() -> None:
    p = T / "rivals_so762.json"
    if not p.exists():
        return
    J = json.loads(p.read_text())
    for mode in ("default", "fast", "student"):
        if mode not in J:
            continue
        R = J[mode]
        print(f"\n### speechocean762 test, OpenPronounce {mode} ({R['system']})\n")
        print(f"counts: {R['counts']}\n")
        print("| group | test | n | Pronunciada 1.0.2 | OpenPronounce 1.1.0 | difference (P - O) [95 % CI] | p | verdict |")
        print("|---|---|--:|---|---|---|--:|---|")
        for g, rows in R["groups"].items():
            for t, r in rows.items():
                d = r["delta"]
                print(f"| {g} | {t} | {r['n']} | {ci(r['pronunciada'])} | {ci(r['openpronounce'])} | "
                      f"{d['value']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {pv(d['p'])} | {mark(d['value'], d['p'])} |")


def shared() -> None:
    for mode in ("default", "fast", "student"):
        p = T / f"rivals_shared_{mode}.json"
        if not p.exists():
            continue
        J = json.loads(p.read_text())
        print(f"\n### shared tests, OpenPronounce {mode}\n")
        print("| group | test | n | Pronunciada 1.0.2 | OpenPronounce 1.1.0 | difference (P - O) [95 % CI] | p | verdict |")
        print("|---|---|--:|---|---|---|--:|---|")
        for r in J["rows"]:
            dc = r.get("delta_ci") or [float("nan")] * 2
            print(f"| {r['group']} | {r['test']} | {r['n']} | {ci(r['pronunciada'])} | {ci(r['openpronounce'])} | "
                  f"{r['delta']:+.3f} [{dc[0]:+.3f}, {dc[1]:+.3f}] | {pv(r['p'])} | {mark(r['delta'], r['p'])} |")


def lingolix() -> None:
    p = T / "rivals_lingolix.json"
    if not p.exists():
        return
    J = json.loads(p.read_text())
    for mode in ("default", "fast", "student"):
        if mode not in J:
            continue
        R = J[mode]
        print(f"\n### Lingolix benchmark, OpenPronounce {mode}\n")
        print("| test | metric | Pronunciada 1.0.2 | OpenPronounce 1.1.0 | difference (P - O) [95 % CI] | p |")
        print("|---|---|---|---|---|--:|")
        for lang, row in R["tier1"].items():
            for k, name in (("roc_auc", "word AUC (higher better)"), ("fpr_at_recall_80", "FA@80 (lower better)")):
                r = row[k]
                print(f"| test 1 {lang} | {name} | {ci(r['pronunciada'])} | {ci(r['openpronounce'])} | "
                      f"{r['delta']:+.3f} [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] | {pv(r['p'])} |")
        for k in ("pcc_word", "pcc_utterance"):
            r = R["tier2"][k]
            print(f"| test 2 | {k} | {ci(r['pronunciada'])} | {ci(r['openpronounce'])} | "
                  f"{r['delta']:+.3f} [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] | {pv(r['p'])} |")


def ondevice() -> None:
    d = T / "ondevice"
    if not d.exists():
        return
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    order = ("pronunciada-fp32", "pronunciada-int8", "openpronounce-default", "openpronounce-fast",
             "openpronounce-student")
    print("\n### on-device (first 200 speechocean762 test clips, CPU)\n")
    print("| system | threads | load s | cold start s | p50 s/clip | p95 s/clip | RTF | peak RSS MB | weights MB |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in sorted(rows, key=lambda r: (order.index(r["system"]), r["threads"])):
        print(f"| {r['system']} | {r['threads']} | {r['load_s']:.1f} | {r['cold_start_s']:.1f} | {r['p50_s']:.3f} | "
              f"{r['p95_s']:.3f} | {r['rtf']:.3f} | {r['peak_rss_mb']:.0f} | {r['weights_mb']:.0f} |")


if __name__ == "__main__":
    so762()
    shared()
    lingolix()
    ondevice()
