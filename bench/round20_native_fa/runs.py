"""Scoring runs of the round-20 fixes (each a tag = one switch configuration), resumable.

    python -m bench.round20_native_fa.runs fit        # practice sets: B threshold grid, C, D1, E (all languages)
    python -m bench.round20_native_fa.runs final      # frozen configurations on practice and on the fresh sets

Every run is bench.round20_native_fa.collect with the switches in its environment; untouched items of a subset
run are taken from the untagged (main) dump. B runs only re-score items in which some word has a presence
probability below 0.5 under main (no other item can change).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from bench.round20_native_fa.rebuild import ROOT

LANGS = ("en", "de", "ko", "es", "fr")
TAUS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
OFF = {"PRONUNCIADA_R20_UNSAID": "0", "PRONUNCIADA_R20_NUMALT": "0", "PRONUNCIADA_R20_GUESS": "0",
       "PRONUNCIADA_R20_DE_LEXALT": "0", "PRONUNCIADA_R20_ES_GLIDE": "0", "PRONUNCIADA_R20_KO_NINS": "0"}


def b_subset(lang: str) -> list[str]:
    from bench.round20_native_fa.collect import load

    out = []
    for iid, r in load(lang).items():
        if not r.get("error") and any((w.get("present") is not None and w["present"] < 0.5) for w in r["rwords"]):
            out.append(iid)
    return out


def collect(lang: str, tag: str | None, env: dict, only: list | None = None, workers: int = 2) -> None:
    e = {**os.environ, **OFF, **env}
    for k in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "LINGOLIX_API_KEY"):
        e.pop(k, None)
    cmd = [sys.executable, "-m", "bench.round20_native_fa.collect", "--lang", lang, "--workers", str(workers)]
    if tag:
        cmd += ["--tag", tag]
    if only is not None:
        p = ROOT / "subsets" / f"{lang}_{tag}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(only))
        cmd += ["--only", str(p)]
    print("run", lang, tag, {k: v for k, v in env.items()}, "items", "all" if only is None else len(only), flush=True)
    subprocess.run(cmd, env=e, check=True)


def fit_runs() -> None:
    for lang in LANGS:
        sub = b_subset(lang)
        for tau in TAUS:
            collect(lang, f"B{tau}", {"PRONUNCIADA_R20_UNSAID": "1", "PRONUNCIADA_R20_UNSAID_TAU": str(tau)}, sub)
    for lang in LANGS:
        collect(lang, "C", {"PRONUNCIADA_R20_NUMALT": "1"})
    for lang in ("en", "de", "es", "fr"):
        collect(lang, "D1", {"PRONUNCIADA_R20_GUESS": "d1"})
    collect("es", "Ees", {"PRONUNCIADA_R20_ES_GLIDE": "1"})
    collect("de", "Ede", {"PRONUNCIADA_R20_DE_LEXALT": "1"})
    collect("ko", "Eko", {"PRONUNCIADA_R20_KO_NINS": "1"})


FIXES = {"B": {"PRONUNCIADA_R20_UNSAID": "1"}, "C": {"PRONUNCIADA_R20_NUMALT": "1"},
         "D1": {"PRONUNCIADA_R20_GUESS": "d1"}, "D2": {"PRONUNCIADA_R20_GUESS": "d2"},
         "E": {"PRONUNCIADA_R20_ES_GLIDE": "1", "PRONUNCIADA_R20_DE_LEXALT": "1", "PRONUNCIADA_R20_KO_NINS": "1"}}


def final_runs(all_fixes: str) -> None:
    """Practice: D2 at the fitted factor and the all-together configuration. Fresh sets (en, de, es, fr; no
    Korean pool left): main, each fix alone, all together. Thresholds come from data/calib/r20.json."""
    for lang in ("en", "de", "es", "fr"):
        collect(lang, "D2", FIXES["D2"])
    allenv = {k: v for f in all_fixes.split("+") for k, v in FIXES[f].items()}
    for lang in LANGS:
        collect(lang, "ALL", allenv)
    for lang in ("en", "de", "es", "fr"):
        fl = f"{lang}@fresh"
        collect(fl, None, {})
        for name in ("B", "C", "D1", "D2", "E"):
            if name == "E" and lang == "en":
                continue
            collect(fl, name, FIXES[name])
        collect(fl, "ALL", allenv)


def combo_runs(name: str, fixes: str) -> None:
    """One more combined configuration (tag `name`) on the practice sets and the fresh sets."""
    env = {k: v for f in fixes.split("+") for k, v in FIXES[f].items()}
    for lang in LANGS:
        collect(lang, name, env)
    for lang in ("en", "de", "es", "fr"):
        collect(f"{lang}@fresh", name, env)


def repair_runs() -> None:
    """Resume the runs whose items failed before the word-value copy fix of pronunciada.r20 (error lines removed)."""
    for lang in ("en", "de", "fr", "es", "ko"):
        collect(lang, "C", FIXES["C"])
        collect(lang, "D1", FIXES["D1"])
        collect(lang, "ALL", {**FIXES["B"], **FIXES["C"], **FIXES["D1"], **FIXES["E"]})
        collect(lang, "ALLnoD", {**FIXES["B"], **FIXES["C"], **FIXES["E"]})
    collect("de", "Ede", {"PRONUNCIADA_R20_DE_LEXALT": "1"})
    for lang in ("en", "de", "es", "fr"):
        fl = f"{lang}@fresh"
        collect(fl, "C", FIXES["C"])
        collect(fl, "D1", FIXES["D1"])
        if lang != "en":
            collect(fl, "E", FIXES["E"])
        collect(fl, "ALL", {**FIXES["B"], **FIXES["C"], **FIXES["D1"], **FIXES["E"]})
        collect(fl, "ALLnoD", {**FIXES["B"], **FIXES["C"], **FIXES["E"]})


def main():
    if sys.argv[1] == "repair":
        repair_runs()
        return
    if sys.argv[1] == "fit":
        fit_runs()
    elif sys.argv[1] == "combo":
        combo_runs(sys.argv[2], sys.argv[3])
    else:
        final_runs(sys.argv[2])


if __name__ == "__main__":
    main()
