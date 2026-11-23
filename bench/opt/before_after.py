"""The before / after section of docs/optimization_and_mobile_report.md.

    python -m bench.opt.before_after <after dir> <before profile dir> [--fill]

<after dir> holds run_profile_after.sh's outputs (paired latency before vs after, RSS, footprint,
first-ever load); <before profile dir> the profile of record of the "before" code (RSS, footprint).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGES = ("load", "g2p", "fbank", "model", "adapter", "align", "features", "delivery", "calibrate", "prosody",
          "assemble", "wall")


def _j(p):
    return json.loads(Path(p).read_text())


def latency_md(P: dict) -> list[str]:
    a, b = P["configs"][0]["name"], P["configs"][1]["name"]
    L = [f"**{a} -> {b}** ({P['clip_set']} clips, {P['repeats']} rounds, load average "
         f"{P['machine']['load_avg_1min']:.1f}; {P['date']}). Per thread count: median over the ten languages of "
         "each language's value.", "",
         "| threads | load s, before -> after | cold start s | wall p50 ms | paired mean diff ms (languages faster / slower, "
         "95 % CI excludes 0) | relative |", "|--:|---|---|---|---|--:|"]
    R = {(r["config"], r["lang"], r["threads"]): r for r in P["results"]}
    for th in sorted({r["threads"] for r in P["results"]}):
        langs = sorted({r["lang"] for r in P["results"] if r["threads"] == th})
        med = lambda c, f: float(np.median([f(R[(c, l, th)]) for l in langs]))  # noqa: E731
        pp = [p for p in P["paired"] if p["threads"] == th]
        fast = sum(p["mean_diff_ci"][1] < 0 for p in pp)
        slow = sum(p["mean_diff_ci"][0] > 0 for p in pp)
        L.append(f"| {th} | {med(a, lambda r: r['load_s']):.2f} -> {med(b, lambda r: r['load_s']):.2f} | "
                 f"{med(a, lambda r: r['cold_start_s']):.2f} -> {med(b, lambda r: r['cold_start_s']):.2f} | "
                 f"{med(a, lambda r: r['stages']['wall']['p50']):.1f} -> {med(b, lambda r: r['stages']['wall']['p50']):.1f} | "
                 f"{np.median([p['mean_diff_ms'] for p in pp]):+.2f} ({fast} / {slow} of {len(pp)}) | "
                 f"{100 * np.median([p['rel_mean'] for p in pp]):+.1f} % |")
    L += ["", "Per stage, p50 ms, median over languages (before -> after):", "",
          "| threads | " + " | ".join(STAGES) + " |", "|--:|" + "---|" * len(STAGES)]
    for th in sorted({r["threads"] for r in P["results"]}):
        langs = sorted({r["lang"] for r in P["results"] if r["threads"] == th})
        cells = []
        for s in STAGES:
            v = [[R[(c, l, th)]["stages"].get(s, {}).get("p50") for l in langs] for c in (a, b)]
            if any(x is None for x in v[0] + v[1]):
                cells.append("–")
            else:
                cells.append(f"{np.median(v[0]):.2f} -> {np.median(v[1]):.2f}")
        L.append(f"| {th} | " + " | ".join(cells) + " |")
    return L


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("after")
    ap.add_argument("before")
    ap.add_argument("--fill", action="store_true")
    a = ap.parse_args(argv)
    A, B = Path(a.after), Path(a.before)
    L = ["Code before = cbc6d6a, after = 7834007 (opt-phase1-code; passes the full gate by identity). Timings paired "
         "and interleaved on the 20 practice clips per language (about 10 s each).", ""]
    for prec in ("fp32", "int8"):
        L += latency_md(_j(A / f"opt_profile_after_{prec}.json")) + [""]
    rb = {(r["config"], r["lang"], r["threads"]): r for r in _j(B / "opt_rss_baseline.json")["results"]}
    ra = {(r["config"].replace("after-", ""), r["lang"], r["threads"]): r for r in _j(A / "opt_rss_after.json")["results"]}
    L += ["Peak RSS, fresh process, 20 clips (before = profile of record, after = this run):", "",
          "| precision | lang | threads | load only MB, before -> after | after scoring MB, before -> after |",
          "|---|---|--:|---|---|"]
    for k in sorted(ra):
        if k in rb:
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {rb[k]['load_only_mb']:.0f} -> {ra[k]['load_only_mb']:.0f} | "
                     f"{rb[k]['after_n_mb']:.0f} -> {ra[k]['after_n_mb']:.0f} |")
    L += ["", "Disk: smallest install per language, MB (code + backbone file actually loaded + data opened; "
          "before -> after):", "", "| lang | fp32 | int8 |", "|---|---|---|"]
    fb = {p: _j(B / f"opt_footprint_{p}.json")["smallest_install"] for p in ("fp32", "int8")}
    fa = {p: _j(A / f"opt_footprint_after_{p}.json")["smallest_install"] for p in ("fp32", "int8")}
    for lang in fb["fp32"]:
        L.append(f"| {lang} | " + " | ".join(f"{fb[p][lang]['total_bytes'] / 1e6:.1f} -> {fa[p][lang]['total_bytes'] / 1e6:.1f}"
                                             for p in ("fp32", "int8")) + " |")
    pk = {p: (_j(B / f"opt_footprint_{p}.json")["package_bytes"], _j(A / f"opt_footprint_after_{p}.json")["package_bytes"])
          for p in ("fp32",)}
    L += ["", f"Package (pronunciada/, the wheel's contents): {pk['fp32'][0] / 1e6:.2f} -> {pk['fp32'][1] / 1e6:.2f} MB.", ""]
    F = _j(A / "opt_first_load.json")
    L += ["First-ever load (a model cache holding only the downloaded files; en, 4 threads) against later loads "
          "(median of 3 later fresh processes on the same cache), and the files the first load writes:", "",
          "| config | first load s | first cold start s | later load s | later cold start s | files written by the first load (MB) |",
          "|---|--:|--:|--:|--:|---|"]
    for r in F["results"]:
        made = ", ".join(f"`{n}` {s / 1e6:.1f}" for n, s in r["first"]["files_created"].items()) or "none"
        L.append(f"| {r['config']} | {r['first']['load_s']:.2f} | {r['first']['cold_start_s']:.2f} | "
                 f"{r['later_load_s_median']:.2f} | {r['later_cold_start_s_median']:.2f} | {made} |")
    md = "\n".join(L)
    if a.fill:
        from .report import DOC, fill

        fill(DOC, "before-after", md)
    print(md)


if __name__ == "__main__":
    main()
