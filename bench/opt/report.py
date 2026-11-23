"""Markdown tables for docs/optimization_and_mobile_report.md from the bench/opt JSON outputs.

    python -m bench.opt.report fill --section baseline-latency --from results/tables/opt_profile_baseline.json
    python -m bench.opt.report show results/tables/opt_footprint.json        # print the table(s)

A section of the report is the text between `<!-- opt:NAME -->` and `<!-- /opt:NAME -->`; `fill`
replaces it with the tables made from the given file(s) (several files are concatenated).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import REPO

DOC = REPO / "docs" / "optimization_and_mobile_report.md"
MAIN_STAGES = ("load", "fbank", "model", "adapter", "align", "features", "delivery", "calibrate", "sentence_head",
               "prosody", "length", "word_step", "assemble")


def _f(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def md_latency(P: dict) -> str:
    L = [f"Clips: {P['clip_set']}, {P['repeats']} rounds; machine: {P['machine'].get('cpu')}, "
         f"{P['machine'].get('ncpu')} cores, load average {P['machine'].get('load_avg_1min', 0):.1f} at the start, "
         f"{P.get('machine_after', {}).get('load_avg_1min', 0):.1f} at the end; onnxruntime "
         f"{P['machine'].get('onnxruntime')}; {P['date']}.", "",
         "| config | lang | threads | load s | cold start s | wall p50 ms | mean | p90 | peak RSS MB |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for r in P["results"]:
        w = r["stages"]["wall"]
        L.append(f"| {r['config']} | {r['lang']} | {r['threads']} | {_f(r['load_s'], 2)} | {_f(r['cold_start_s'], 2)} | "
                 f"{_f(w.get('p50'))} | {_f(w.get('mean'))} | {_f(w.get('p90'))} | {_f(r.get('maxrss_after_mb'), 0)} |")
    L += ["", "Per stage, p50 ms (SentenceResult.timing_ms; load = audio decode + resample, model = ZIPA backbone):", "",
          "| config | lang | threads | " + " | ".join(MAIN_STAGES) + " |", "|---|---|--:|" + "--:|" * len(MAIN_STAGES)]
    for r in P["results"]:
        L.append(f"| {r['config']} | {r['lang']} | {r['threads']} | " +
                 " | ".join(_f(r["stages"].get(s, {}).get("p50"), 2) for s in MAIN_STAGES) + " |")
    if P.get("paired"):
        L += ["", "Paired comparison (per clip median over rounds, B - A; 95 % clip-bootstrap interval of the mean):", "",
              "| lang | threads | A | B | mean B - A ms [95 % CI] | median ms | relative |", "|---|--:|---|---|--:|--:|--:|"]
        for p in P["paired"]:
            L.append(f"| {p['lang']} | {p['threads']} | {p['a']} | {p['b']} | {p['mean_diff_ms']:+.2f} "
                     f"[{p['mean_diff_ci'][0]:+.2f}, {p['mean_diff_ci'][1]:+.2f}] | {p['median_diff_ms']:+.2f} | "
                     f"{100 * p['rel_mean']:+.1f} % |")
    return "\n".join(L)


def md_rss(R: dict) -> str:
    L = [f"Fresh process per row; peak resident set size from getrusage; {R['n_clips']} clips ({R['clip_set']}); "
         f"{R['date']}.", "", "| config | lang | threads | load only MB | after scoring MB | load s |", "|---|---|--:|--:|--:|--:|"]
    for r in R["results"]:
        L.append(f"| {r['config']} | {r['lang']} | {r['threads']} | {r['load_only_mb']:.0f} | {r['after_n_mb']:.0f} | "
                 f"{r['load_s']:.2f} |")
    return "\n".join(L)


def md_footprint(F: dict) -> str:
    mb = lambda b: f"{b / 1e6:.2f}"  # noqa: E731
    L = [f"Source tree {F['tree']}; {F['date']}. {F['note']}", "", "| category | files | MB |", "|---|--:|--:|"]
    for c, v in sorted(F["by_category"].items(), key=lambda kv: -kv[1]["bytes"]):
        L.append(f"| {c} | {v['files']} | {mb(v['bytes'])} |")
    L += ["", f"Package (pronunciada/, as in the wheel): {mb(F['package_bytes'])} MB.", "",
          "Smallest install that can score each language (code + backbone + what it opens):", "",
          "| lang | code MB | backbone MB | adapter | lexicon | calibration | coaching | other data | total MB | "
          "third-party data MB |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for lang, s in F["smallest_install"].items():
        c = s["by_category"]
        bb = sum(v for k, v in c.items() if k.startswith("backbone"))
        other = s["data_bytes"] - bb - sum(c.get(k, 0) for k in ("adapter", "lexicon", "calibration", "coaching"))
        L.append(f"| {lang} | {mb(s['code_bytes'])} | {mb(bb)} | {mb(c.get('adapter', 0))} | {mb(c.get('lexicon', 0))} | "
                 f"{mb(c.get('calibration', 0))} | {mb(c.get('coaching', 0))} | {mb(other)} | {mb(s['total_bytes'])} | "
                 f"{mb(s['third_party_bytes'])} |")
    unused = F.get("data_files_opened_by_no_language", [])
    if unused:
        L += ["", f"Data files no language opened while scoring ({len(unused)}): " + ", ".join(f"`{u}`" for u in unused[:40])]
    return "\n".join(L)


def md_parity(P: dict) -> str:
    L = [f"{P['ref']['name']} vs {P['cand']['name']}, {P['threads']} threads, {P['clip_set']}: **{P['verdict']}**.", "",
         "| lang | clips | clips with a differing score | max abs diff, scores | raw posteriors | adapted posteriors |",
         "|---|--:|--:|--:|--:|--:|"]
    for lang, r in P["by_lang"].items():
        L.append(f"| {lang} | {r['n_clips']} | {r['n_clips_any_score_differs']} | {r['max_abs_diff']:.3g} | "
                 f"{r['posteriors']['raw']['max_abs_diff']:.3g} | {r['posteriors']['adapted']['max_abs_diff']:.3g} |")
    return "\n".join(L)


def md_gate(G: dict) -> str:
    L = [f"**{G['summary']}**", ""]
    if G.get("fails"):
        L += ["Failing rows:", ""] + [f"- {f['text']}" for f in G["fails"]] + [""]
    if G.get("moved"):
        L += ["| row | baseline | candidate | delta | paired p | class |", "|---|--:|--:|--:|--:|---|"]
        for m in G["moved"]:
            cls = "" if m.get("cat_base") == m.get("cat_cand") else f"{m['cat_base']} -> {m['cat_cand']}"
            p = "–" if m.get("p") is None else f"{m['p']:.3g}"
            L.append(f"| {m['key']} | {m['base']:.4f} | {m['cand']:.4f} | {m['delta']:+.4f} | {p} | {cls} |")
    return "\n".join(L)


def render(obj: dict) -> str:
    k = obj.get("kind")
    return {"opt_profile": md_latency, "opt_rss": md_rss, "opt_footprint": md_footprint, "opt_parity": md_parity,
            "opt_gate": md_gate}[k](obj)


def fill(doc: Path, section: str, text: str) -> None:
    s = doc.read_text(encoding="utf-8")
    pat = re.compile(rf"(<!-- opt:{re.escape(section)} -->\n)(.*?)(<!-- /opt:{re.escape(section)} -->)", re.S)
    if not pat.search(s):
        raise SystemExit(f"no section {section} in {doc}")
    doc.write_text(pat.sub(lambda m: m.group(1) + text.rstrip() + "\n" + m.group(3), s), encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fill")
    f.add_argument("--section", required=True)
    f.add_argument("--from", dest="src", nargs="+", required=True)
    f.add_argument("--doc", default=str(DOC))
    s = sp.add_parser("show")
    s.add_argument("src", nargs="+")
    a = ap.parse_args(argv)
    text = "\n\n".join(render(json.loads(Path(p).read_text())) for p in (a.src if a.cmd == "show" else a.src))
    if a.cmd == "show":
        print(text)
    else:
        fill(Path(a.doc), a.section, text)
        print(f"filled {a.section} in {a.doc}")


if __name__ == "__main__":
    main()
