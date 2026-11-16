"""Command line: python -m pronunciada score AUDIO "TEXT" --lang en [--strictness 0.5] [--json]"""
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
    a = ap.parse_args(argv)
    from .scorer import Scorer

    try:
        sc = Scorer(a.lang, precision=a.precision, threads=a.threads)
        res = sc.score(a.audio, a.text, strictness=a.strictness)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(dataclasses.asdict(res), ensure_ascii=False, indent=1))
    else:
        print(table(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
