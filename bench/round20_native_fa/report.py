"""Markdown tables for results/round20_native_fa.md from the JSON tables of this folder's scripts, and a copy of
those tables to results/tables/round20_native_fa_*.json.

    python -m bench.round20_native_fa.report --lang en,de,ko,es,fr > tables.md
"""
from __future__ import annotations

import argparse
import json
import shutil

from bench.round20_native_fa.rebuild import ROOT

B_NAMES = {"noise": "corpus noise", "neighbour": "neighbour of the change", "variant": "variant rescues",
           "text": "text front-end", "boundary": "boundary", "function": "function / short", "tail": "tail",
           "residual": "residual"}


def j(name):
    p = ROOT / "tables" / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(v, ci=None, nd=1):
    s = f"{v:.{nd}f}"
    if ci:
        s += f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--copy", action="store_true")
    a = ap.parse_args(argv)
    langs = a.lang.split(",")
    T = {l: j(f"taxonomy_{l}.json") for l in langs}
    P = {l: j(f"patterns_{l}.json") for l in langs}
    print("### Baseline (practice sets, current main)\n")
    print("| lang | items | speakers | correct words | changed | AUC [95 % CI] | FA@80 % [95 % CI] | threshold score | n FA | n < 0.6 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for l in langs:
        t = T[l]
        print(f"| {l} | {t['n_items']} | {t['n_speakers']} | {t['n_correct']} | {t['n_changed']} | "
              f"{t['auc']:.4f} [{t['auc_ci'][0]:.4f}, {t['auc_ci'][1]:.4f}] | {100 * t['fa80']:.2f} "
              f"[{100 * t['fa80_ci'][0]:.2f}, {100 * t['fa80_ci'][1]:.2f}] | {t['thr_score']:.3f} | {t['FA']['n']} | "
              f"{t['low06']['n']} |")
    for pop, title in (("FA", "false alarms at FA@80"), ("low06", "correct words below 0.6")):
        print(f"\n### Exclusive buckets, {title} (% of the population [95 % CI])\n")
        order = list(T[langs[0]][pop]["buckets"])
        print("| bucket | " + " | ".join(f"{l} (n {T[l][pop]['n']})" for l in langs) + " |")
        print("|---|" + "---|" * len(langs))
        for b in order:
            print(f"| {B_NAMES.get(b, b)} | " + " | ".join(pct(T[l][pop]["buckets"][b]["pct"], T[l][pop]["buckets"][b]["ci"])
                                                     for l in langs) + " |")
    print("\n### Overlapping hit rate of every test (% of FA; base rate on 3,000 random correct words in brackets)\n")
    tests = list(T[langs[0]]["FA"]["overlap"])
    print("| test | " + " | ".join(langs) + " |")
    print("|---|" + "---|" * len(langs))
    for t in tests:
        cells = []
        for l in langs:
            o = T[l]["FA"]["overlap"].get(t)
            br = T[l]["FA"]["base_rate"].get(t)
            cells.append("-" if o is None else (f"{o:.1f}" + (f" ({br:.1f})" if br is not None else "")))
        print(f"| {t} | " + " | ".join(cells) + " |")
    print("\n### Other priority orders (% of FA per bucket)\n")
    for l in langs:
        ao = T[l]["FA"]["alt_orders"]
        print(f"- {l}: " + "; ".join(f"{nm}: " + ", ".join(f"{b} {v:.0f}" for b, v in d.items() if v >= 0.5)
                                      for nm, d in ao.items()))
    print("\n### Candidate shared causes (overlapping families, % of FA [95 % CI])\n")
    fams = list(P[langs[0]]["families"])
    print("| family | " + " | ".join(langs) + " |")
    print("|---|" + "---|" * len(langs))
    for f in fams:
        print(f"| {f} | " + " | ".join(pct(P[l]["families"][f]["pct"], P[l]["families"][f]["ci"]) for l in langs) + " |")
    print("\n### Which evidence drives the low score (FA; all correct words in brackets)\n")
    for l in langs:
        t = T[l]
        n = t["FA"]["n"]
        dall = t["driver_all"]
        nall = sum(dall.values())
        print(f"- {l}: " + ", ".join(f"{k} {100 * v / n:.0f} % ({100 * dall.get(k, 0) / nall:.1f} %)"
                                      for k, v in t["FA"]["driver"].items())
              + f"; lowest sound of sound-driven FA: " + ", ".join(f"{k} {v}" for k, v in t["FA"]["low_kind"].items())
              + f"; caught changed words: " + ", ".join(f"{k} {v}" for k, v in t["changed_low_kind"].items()))
    print("\n### Neighbour effect, paired with the same word of the control item (same audio)\n")
    print("| lang | change kind | pairs | FA % in changed item | FA % in control item |")
    print("|---|---|---|---|---|")
    for l in langs:
        for k, v in T[l]["neighbour_by_kind"].items():
            print(f"| {l} | {k} | {v['n']} | {v['fa_changed']:.1f} | {v['fa_control']:.1f} |")
    ea = j("ea_check.json")
    if ea:
        print("\n### Word-omission items with the explain-away pass unbounded (existing option; diagnosis only)\n")
        print("| lang | items | neighbour FA % default -> explain-away | inserted word caught % | other words FA % |")
        print("|---|---|---|---|---|")
        for l, v in ea.items():
            print(f"| {l} | {v['items']} | {v['neighbour_fa_default']:.1f} -> {v['neighbour_fa_explain_away']:.1f} | "
                  f"{v['inserted_caught_default']:.1f} -> {v['inserted_caught_explain_away']:.1f} | "
                  f"{v['other_fa_default']:.1f} -> {v['other_fa_explain_away']:.1f} |")
    pr = j("presence.json")
    if pr:
        print("\n### Presence factor below 0.5 (share of words)\n")
        groups = list(next(iter(pr.values())))
        print("| group | " + " | ".join(pr) + " |")
        print("|---|" + "---|" * len(pr))
        for g in groups:
            print(f"| {g} | " + " | ".join(f"{100 * pr[l][g]['share_below_0.5']:.1f} (n {pr[l][g]['n']})"
                                          if g in pr[l] else "-" for l in pr) + " |")
    print("\n### Best of N alternative pronunciations (counterfactual re-scoring; every word re-scored takes its best)\n")
    print("| lang | variants | AUC | FA@80 % | new threshold | FA rescued % | caught changed words lifted out % | changed raised |")
    print("|---|---|---|---|---|---|---|---|")
    for l in langs:
        t = T[l]
        print(f"| {l} | none | {t['auc']:.4f} | {100 * t['fa80']:.2f} | {t['thr_score']:.3f} | - | - | - |")
        for k, v in t["best_of_n"].items():
            if k == "brit" and l != "en":
                continue
            if k == "lex" and l in ("es", "ko"):
                continue
            print(f"| {l} | {k} | {v['auc']:.4f} | {100 * v['fa80']:.2f} | {v['thr_score']:.3f} | "
                  f"{100 * v['rescued_share_FA']:.1f} | {100 * v['rescued_share_caught_changed']:.1f} | {v['changed_raised']} |")
    print("\n### Rescues by variant kind (FA rescued by the best alternative)\n")
    for l in langs:
        print(f"- {l}: " + ", ".join(f"{k} {v}" for k, v in T[l]["FA"]["rescue_by_rule"].items()))
    print("\n### FA rate by number of sounds (correct words) and catch rate (changed words) at the one threshold\n")
    keys = sorted({k for l in langs for k in T[l]["by_n_sounds"]}, key=int)
    print("| sounds | " + " | ".join(f"{l} FA % / catch %" for l in langs) + " |")
    print("|---|" + "---|" * len(langs))
    for k in keys:
        if k == "0":
            continue
        cells = []
        for l in langs:
            v = T[l]["by_n_sounds"].get(k)
            cells.append("-" if not v else f"{v['fa_pct']:.1f} / {v['catch_pct']:.0f} (n {v['n_correct']})")
        print(f"| {k}{'+' if k == '12' else ''} | " + " | ".join(cells) + " |")
    for curve, title in (("dur_50ms", "aligned duration (ms)"), ("sounds_per_s", "sounds per second")):
        print(f"\n### FA rate by {title}: function words / content words (% FA, n in brackets)\n")
        keys = []
        for l in langs:
            for nm in ("function", "content"):
                for k in T[l]["curves"][nm].get(curve, {}):
                    if k not in keys:
                        keys.append(k)
        keys.sort(key=lambda s: float(s.split("-")[0]))
        print("| bin | " + " | ".join(langs) + " |")
        print("|---|" + "---|" * len(langs))
        for k in keys:
            cells = []
            for l in langs:
                f = T[l]["curves"]["function"].get(curve, {}).get(k)
                c = T[l]["curves"]["content"].get(curve, {}).get(k)
                fs = "-" if not f else f"{f[1]:.1f} ({f[0]})"
                cs = "-" if not c else f"{c[1]:.1f} ({c[0]})"
                cells.append(f"{fs} / {cs}")
            print(f"| {k} | " + " | ".join(cells) + " |")
    nbd = {l: T[l].get("neighbour_by_distance") for l in langs}
    if all(nbd.values()):
        print("\n### Effect of the change by distance (FA % in the changed item vs the same word in the control item)\n")
        keys = sorted({k for v in nbd.values() for k in v})
        print("| change / distance | " + " | ".join(langs) + " |")
        print("|---|" + "---|" * len(langs))
        for k in keys:
            print(f"| {k} | " + " | ".join(
                (f"{nbd[l][k]['fa_changed']:.1f} vs {nbd[l][k]['fa_control']:.1f} (n {nbd[l][k]['n']})" if k in nbd[l] else "-")
                for l in langs) + " |")
    print("\n### Function vs content words\n")
    for l in langs:
        t = T[l]
        print(f"- {l}: function words {t['share_function_all']:.0f} % of correct words, FA rate {t['fa_rate_function']:.1f} %;"
              f" content words FA rate {t['fa_rate_content']:.1f} %")
    print("\n### Tail: correct words by distance to the threshold (logit units of the score; % of correct words)\n")
    bins = list(T[langs[0]]["tail_mass"])
    print("| distance below threshold | " + " | ".join(langs) + " |")
    print("|---|" + "---|" * len(langs))
    for b in bins:
        print(f"| {b} | " + " | ".join(f"{T[l]['tail_mass'][b]:.2f}" for l in langs) + " |")
    print("| FA@80 % if the threshold moved by -0.5 / -0.25 / 0 / +0.25 / +0.5 logit | " + " | ".join(
        " / ".join(f"{v:.2f}" for v in T[l]["fa_if_thr_shift"].values()) for l in langs) + " |")
    for l in langs:
        h = j(f"handcheck_{l}.json")
        if h:
            print(f"\nHand check {l}: n {h['n']}, agreement {100 * h['agreement']:.0f} %; hand counts {h['hand_counts']}; "
                  "precision of the automatic bucket: " + ", ".join(
                      f"{b} {v['precision']:.2f} ({v['auto_n']})" for b, v in h["per_bucket"].items() if v["precision"] is not None))
    if a.copy:
        from bench.pipeline.common import repo_root

        dst = repo_root() / "results" / "tables"
        for p in (ROOT / "tables").glob("*.json"):
            shutil.copy(p, dst / f"round20_native_fa_{p.stem}.json")


if __name__ == "__main__":
    main()
