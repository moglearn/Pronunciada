"""Round 19 (Korean): HELD-OUT Lingolix test 1 Korean, before / after, with the benchmark runner's code.

    python -m bench.round19ko.lingolix_run score --tag before    # PRONUNCIADA_KO_IDENTITY=0
    python -m bench.round19ko.lingolix_run score --tag after     # shipped default
    python -m bench.round19ko.lingolix_run report

Scoring and reduction are bench.third_party.lingolix._score_chunk (Scorer defaults -> to_azure ->
the benchmark's Azure adapter); metrics are that module's re-implementation of report.py (v2026.10
rules, and the main-branch `unverified` rule as a variant). Responses go to
$CACHE/round19/korean/lingolix/<tag>.jsonl; the shared responses folder is not touched.
Paired differences: the same clip bootstrap as report.py (random.Random(0), 1000 draws) applied to
both systems at once.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

from bench.round19ko.cv_pool import work
from bench.third_party import lingolix as L


def out_path(tag: str) -> Path:
    d = work() / "lingolix"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}.jsonl"


def score(tag: str):
    if tag == "before":
        os.environ["PRONUNCIADA_KO_IDENTITY"] = "0"
    man = L.tests()["tier1_ko"]
    jobs = [(it["id"], "ko", str(L.tier1_wav(it)), it["target"]) for it in man]
    recs = L._score_chunk(jobs, threads=2)
    with open(out_path(tag), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(tag, len(recs), "failed", sum(1 for r in recs if r.get("error")), file=sys.stderr)


def read(tag):
    return {json.loads(x)["id"]: json.loads(x) for x in out_path(tag).read_text(encoding="utf-8").splitlines() if x.strip()}


def paired(man, ra, rb, unv=None, n=1000):
    ca = L.collect(man, ra, exclude_unverified=unv)["units"]
    cb = L.collect(man, rb, exclude_unverified=unv)["units"]
    ida = [u[2]["id"] for u in ca]
    assert ida == [u[2]["id"] for u in cb]
    rng = random.Random(0)
    d = {"auc": [], "fa80": [], "loc": []}

    def loc(units):
        h = t = 0
        for ls, ss, _, _ in units:
            if 1 in ls:
                t += 1
                h += int(max(range(len(ss)), key=lambda i: ss[i]) == ls.index(1))
        return h / t

    for _ in range(n):
        idx = [rng.randrange(len(ca)) for _ in range(len(ca))]
        A = [ca[i] for i in idx]
        B = [cb[i] for i in idx]
        la, sa = [l for u in A for l in u[0]], [s for u in A for s in u[1]]
        lb, sb = [l for u in B for l in u[0]], [s for u in B for s in u[1]]
        d["auc"].append(L.roc_auc(lb, sb) - L.roc_auc(la, sa))
        d["fa80"].append(L.fpr_at_recall(lb, sb, 0.8) - L.fpr_at_recall(la, sa, 0.8))
        d["loc"].append(loc(B) - loc(A))
    out = {}
    for k, v in d.items():
        v = sorted(x for x in v if not math.isnan(x))
        better = (lambda x: x > 0) if k in ("auc", "loc") else (lambda x: x < 0)
        out[k] = {"ci": [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]],
                  "p_not_better": sum(1 for x in v if not better(x)) / len(v)}
    return out


def report(after: str = "after"):
    man = L.tests()["tier1_ko"]
    unv = {it["id"]: it.get("unverified", []) for it in L.load_manifest("repo_main_f8750aa", "tier1/manifest_ko.jsonl")}
    ra, rb = read("before"), read(after)
    out = {}
    for name, kw in (("v2026.10", {}), ("main_rules", {"exclude_unverified": unv})):
        ma, mb = L.metrics(man, ra, **kw), L.metrics(man, rb, **kw)
        out[name] = {"before": ma, "after": mb, "paired": paired(man, ra, rb, kw.get("exclude_unverified"))}
    out["published_not_rerun_here"] = {k: L.PUBLISHED[k] for k in ("tier1_auc", "tier1_fa_at_80")}
    p = Path("results/tables/round19_ko_lingolix.json" if after == "after" else f"results/tables/round19_ko_lingolix_{after}.json")
    p.write_text(json.dumps(L._clean(out), indent=1, ensure_ascii=False))
    for name in ("v2026.10", "main_rules"):
        b, a, pd = out[name]["before"], out[name]["after"], out[name]["paired"]
        print(f"{name}: AUC {b['roc_auc']:.3f} {b['roc_auc_ci']} -> {a['roc_auc']:.3f} {a['roc_auc_ci']}  "
              f"FA@80 {b['fpr_at_recall_80']:.3f} {b['fpr_at_recall_80_ci']} -> {a['fpr_at_recall_80']:.3f} {a['fpr_at_recall_80_ci']}  "
              f"loc {b['localisation_top1']:.3f} -> {a['localisation_top1']:.3f}")
        print("   by bucket", {k: (round(b['by_bucket'][k], 3), round(a['by_bucket'][k], 3)) for k in b["by_bucket"]})
        print("   verdicts R/FA", round(b["verdicts"]["recall"], 3), round(b["verdicts"]["fa"], 3), "->",
              round(a["verdicts"]["recall"], 3), round(a["verdicts"]["fa"], 3))
        print("   paired", pd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("--tag")
    a = ap.parse_args()
    score(a.tag) if a.cmd == "score" else report(a.tag or "after")
