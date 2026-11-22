"""Lingolix test 1 Japanese through the benchmark runner's own code path (bench/third_party/lingolix.py:
Scorer -> to_azure -> the benchmark's Azure reduction -> its metrics), into this round's own folder
(the published responses are not touched), under both rule sets (v2026.10 and main f8750aa).

    python -m bench.round19_ja.lx_run score TAG      # TAG = before (run with PRONUNCIADA_JA_ROUND19=0) | after
    python -m bench.round19_ja.lx_run report         # -> results/tables/round19_ja_lingolix_runner.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bench.round19_ja.cv_pool import WORK

OUT = WORK / "lx_runner"


def score(tag: str) -> None:
    from bench.third_party import lingolix as LX

    items = LX.tests()["tier1_ja"]
    jobs = [(it["id"], "ja", str(LX.tier1_wav(it)), it["target"]) for it in items]
    OUT.mkdir(parents=True, exist_ok=True)
    recs = LX._score_chunk(jobs, 2)  # noqa: SLF001
    with open(OUT / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(tag, len(recs), sum(1 for r in recs if r.get("error")), "errors", file=sys.stderr)


def _paired(manifest, A, B, n=2000, seed=0, **kw):
    import random

    import numpy as np

    from bench.third_party import lingolix as LX

    ua = LX.collect(manifest, A, **kw)["units"]
    ub = LX.collect(manifest, B, **kw)["units"]
    assert [u[2]["id"] for u in ua] == [u[2]["id"] for u in ub]
    rng = random.Random(seed)
    D = {"auc": [], "fa80": []}
    for _ in range(n):
        idx = [rng.randrange(len(ua)) for _ in range(len(ua))]
        for k, fn in (("auc", LX.roc_auc), ("fa80", lambda l, s: LX.fpr_at_recall(l, s, 0.8))):
            fa = fn([l for i in idx for l in ua[i][0]], [s for i in idx for s in ua[i][1]])
            fb = fn([l for i in idx for l in ub[i][0]], [s for i in idx for s in ub[i][1]])
            D[k].append(fb - fa)
    out = {}
    for k, v in D.items():
        v = np.array([x for x in v if x == x])
        out[k] = {"ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                  "p": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))}
    return out


def report() -> None:
    from bench.third_party import lingolix as LX

    T = LX.tests()["tier1_ja"]
    main_unv = {it["id"]: it.get("unverified", []) for it in LX.load_manifest("repo_main_f8750aa", "tier1/manifest_ja.jsonl")}
    R = {t: {json.loads(x)["id"]: json.loads(x) for x in open(OUT / f"{t}.jsonl", encoding="utf-8")}
         for t in ("before", "after", "final") if (OUT / f"{t}.jsonl").exists()}
    res = {"note": "after = candidate 2 (third look at the benchmark in round 19); final = the round-19 final "
                   "candidate (fourth and last look)"}
    for rules, kw in (("v2026.10", {}), ("main_f8750aa", {"exclude_unverified": main_unv})):
        res[rules] = {t: LX._clean(LX.metrics(T, R[t], **kw)) for t in R}  # noqa: SLF001
        res[rules]["paired_after_minus_before"] = _paired(T, R["before"], R["after"], **kw)
        if "final" in R:
            res[rules]["paired_final_minus_before"] = _paired(T, R["before"], R["final"], **kw)
            res[rules]["paired_final_minus_after"] = _paired(T, R["after"], R["final"], **kw)
        for t in R:
            m = res[rules][t]
            v = m["verdicts"]
            print(f"{rules:12s} {t:6s} AUC {m['roc_auc']:.3f} {m['roc_auc_ci']} FA80 {m['fpr_at_recall_80']:.3f} "
                  f"{m['fpr_at_recall_80_ci']} buckets {m['by_bucket']} flag R {v['recall']:.2f} P {v['precision']:.2f} "
                  f"FA {v['fa']:.3f} length {v['recall_by_bucket'].get('length')}")
        for k in [x for x in res[rules] if x.startswith("paired")]:
            print(rules, k, res[rules][k])
    p = Path(__file__).resolve().parents[2] / "results" / "tables" / "round19_ja_lingolix_runner.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"score": lambda: score(sys.argv[2]), "report": report}[sys.argv[1]]()
