"""OpenPronounce 1.1.0 against Pronunciada on the shared held-out tests other than speechocean762.

    python -m bench.rivals.eval_shared --mode default [--tests errdet,swap,cv]

Runs the test code of bench/eval_zipa_baseline.py unchanged (as bench/eval_external.py did for the
2026-11-18 snapshot), with OpenPronounce's per-record files (bench.rivals.convert) in place of the
ZIPA baseline's: the same items, labels, fallback rules, resampling units, bootstrap sizes and seeds as
results/SCOREBOARD.md. A record OpenPronounce could not score gets its floor (0) and is counted.
Pronunciada's column is recomputed from its cached outputs and checked against results/tables/.

  errdet  L2-ARCTIC and EpaDB test: ROC AUC of each system's per-sound error signal for the sounds the
          annotators marked (OpenPronounce: 1 - its phone model's probability, see bench.rivals.convert).
  swap    fleurs-swap (FLEURS test, 150 recordings per language, a word of the prompt changed for a
          distant or a near word): far AUC, near AUC, near hit at 5 % false alarms, from word scores.
          Only the languages OpenPronounce covers (en fr es de it pt); every record of those is required.
  cv      cv-real test (Common Voice clips the community accepted vs rejected), ROC AUC of the sentence
          score, for the locales OpenPronounce covers.
Writes results/tables/rivals_shared_<mode>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COVERED = ("en", "fr", "es", "de", "it", "pt")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--tests", default="errdet,swap,cv")
    a = ap.parse_args(argv)

    import bench.data as BD
    import bench.eval_zipa_baseline as Z
    from bench.rivals.common import coverage, ext_dir
    from bench.rivals.convert import SYSTEM

    system = SYSTEM[a.mode]
    Z.cache_dir = lambda dataset: ext_dir(system, dataset)
    Z.RANK.clear()
    Z.GOP_FLOOR = 0.0
    cov = coverage(system)
    print(json.dumps(cov, indent=1), flush=True)
    t0 = time.time()
    tests = a.tests.split(",")
    R = {"rows": [], "counts": {}}
    if "errdet" in tests:
        r, c = Z.eval_errdet({ds: Z.baseline(ds) for ds in ("l2arctic", "epadb")})
        R["rows"] += r
        R["counts"]["learner errors"] = c
    swap_langs = []
    if "swap" in tests:
        B = Z.baseline("fleurs-swap")
        n_lang, got = defaultdict(int), defaultdict(int)
        for t in Z.tasks()["fleurs-swap"]:
            n_lang[t["lang"]] += 1
        for o in B.values():
            got[o["lang"]] += 1
        swap_langs = [x for x in COVERED if got.get(x) == n_lang[x]]
        print("swap languages with every record:", swap_langs, flush=True)
        _load = BD.load
        BD.load = lambda name, **kw: ([r for r in _load(name, **kw) if r["lang"] in swap_langs]
                                      if name == "fleurs-swap" else _load(name, **kw))
        r, c = Z.eval_swap(B)
        BD.load = _load
        R["rows"] += r
        R["counts"]["swap benchmark"] = c
    if "cv" in tests:
        Bc = Z.baseline("cv-real")
        r, c = Z.eval_cv(Bc)
        cv_langs = {t["lang"] for t in Z.tasks()["cv-real"]}
        n_cv, got_cv = defaultdict(int), defaultdict(int)
        for t in Z.tasks()["cv-real"]:
            n_cv[t["lang"]] += 1
        for o in Bc.values():
            got_cv[o["lang"]] += 1
        full = {x for x in COVERED if got_cv.get(x) == n_cv[x]}
        loc_lang = {}
        from bench import eval_cv as EC

        for rec in EC.records():
            loc_lang[rec["locale"]] = rec["lang"] if "lang" in rec else rec["locale"].split("-")[0]
        keep = [x for x in r if loc_lang.get(x["test"].split("/")[0], "") in full]
        R["rows"] += keep
        R["counts"]["cv-real"] = c
        R["counts"]["cv-real_languages_in_corpus"] = sorted(cv_langs)
        R["counts"]["cv-real_languages_scored"] = sorted(full)
    for r in R["rows"]:
        r["verdict"] = Z.verdict(r["delta"], r["p"])

    def rename(r: dict) -> dict:
        r = dict(r)
        r["openpronounce"] = r.pop("gop")
        r["openpronounce_flags_or_secondary"] = r.pop("match")
        r.pop("delta_vs_match", None)
        return r

    out = {"system": system, "mode": a.mode, "floor_for_unscored": 0.0, "coverage": cov,
           "rows": [rename(r) for r in R["rows"]], "counts": R["counts"], "swap_languages": swap_langs,
           "pronunciada_matches_scoreboard": all(r.get("pronunciada_matches_table", True) for r in R["rows"]),
           "seconds": round(time.time() - t0), "protocol": __doc__.split("\n\n", 1)[1]}
    p = ROOT / "results" / "tables" / f"rivals_shared_{a.mode}.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    if old:                                               # keep tests run earlier with another --tests
        groups = {r["group"] for r in out["rows"]}
        out["rows"] = [r for r in old.get("rows", []) if r["group"] not in groups] + out["rows"]
        out["counts"] = {**old.get("counts", {}), **out["counts"]}
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float) + "\n")
    print(f"wrote {p} ({len(out['rows'])} rows)")
    for r in out["rows"]:
        print(f"  {r['group']:<18} {r['test']:<22} Pronunciada {r['pronunciada']['value']:.3f}  "
              f"OpenPronounce {r['openpronounce']['value']:.3f}  d {r['delta']:+.3f} p {r['p']:.3f}  {r['verdict']}")


if __name__ == "__main__":
    main()
