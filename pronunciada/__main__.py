"""Command line: python -m pronunciada score AUDIO "TEXT" --lang en [--strictness 0.5] [--json] [--coach]"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys


def _fmt_p(p) -> str:
    return "  –  " if p is None else f"{p:5.2f}"


def table(res) -> str:
    L = [f"{res.lang}  strictness {res.strictness:g}   sentence {res.prob:.2f}   "
         f"fluency {_fmt_p(res.fluency).strip()}   completeness {_fmt_p(res.completeness).strip()}", ""]
    L.append(f"{'word':<18} {'prob':>5} {'sounds':>6} {'stress':>6} {'tone':>5}  {'time':>11}  sounds (prob, heard)")
    for w in res.words:
        snd = "  ".join(f"{p.ipa}:{p.prob:.2f}" + (f"→{p.heard or '∅'}" if p.heard is not None else "")
                        for p in w.phones)
        flag = " (omitted)" if w.omitted else ""
        L.append(f"{(w.text + flag)[:18]:<18} {w.prob:5.2f} {_fmt_p(w.sounds_prob):>6} "
                 f"{_fmt_p(w.stress_prob):>6} {_fmt_p(w.tone_prob):>5}  {w.start:5.2f}-{w.end:5.2f}  {snd}")
    if res.warnings:
        L += ["", "warnings: " + "; ".join(res.warnings)]
    L += ["", "timing ms: " + ", ".join(f"{k} {v:.1f}" for k, v in res.timing_ms.items())]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pronunciada", description="pronunciation scoring")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score one recording against its prompt text")
    s.add_argument("audio")
    s.add_argument("text")
    s.add_argument("--lang", required=True)
    s.add_argument("--strictness", type=float, default=0.5)
    s.add_argument("--precision", default="fp32", choices=("fp32", "fp16", "int8"))
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--json", action="store_true", help="print the SentenceResult as JSON")
    s.add_argument("--coach", action="store_true",
                   help="add coaching items and default tips (pronunciada.coach, pronunciada.tips)")
    s.add_argument("--max-tips", type=int, default=3)
    a = ap.parse_args(argv)
    from .scorer import Scorer

    try:
        sc = Scorer(a.lang, precision=a.precision, threads=a.threads)
        res = sc.score(a.audio, a.text, strictness=a.strictness)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    items = tp = None
    if a.coach:
        from .tips import tips

        items = res.coaching()
        tp = [{k: v for k, v in t.items() if k != "item"} for t in tips(items, a.max_tips, res)]
    if a.json:
        d = dataclasses.asdict(res)
        if a.coach:
            from .coach import SCHEMA_VERSION

            d["coaching"] = {"schema_version": SCHEMA_VERSION, "items": items, "tips": tp}
        print(json.dumps(d, ensure_ascii=False, indent=1))
    else:
        print(table(res))
        if res.audio is not None:
            iss = ", ".join(res.audio["issues"]) or "none"
            print(f"audio: issues {iss}" + ("  (retry recommended)" if res.audio["retry_recommended"] else ""))
        if a.coach:
            print("\ncoaching:")
            for it in items:
                h = "?" if it["heard"] is None else (it["heard"] or "∅")
                print(f"  {it['kind']:<12} {it['word'][:16]:<16} {it['expected']!s:>4} -> {h!s:<4} "
                      f"p {it['prob']:.2f}  conf {it['confidence'] if it['confidence'] is not None else '-'}  "
                      f"{','.join(it['diff_keys'])}")
            print("tips:")
            for t in tp:
                print(f"  [{t['tip_key']}] {t['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
