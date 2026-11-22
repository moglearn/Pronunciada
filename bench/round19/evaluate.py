"""Round 19 (German): the Lingolix test-1 metrics on a dump (bench.round19.collect).

    python -m bench.round19.evaluate dev-base [dev-cand ...] [--fold fit|check|all] [--source cv|fleurs]
    python -m bench.round19.evaluate lingolix-base lingolix-cand --paired

Metrics are those of bench.third_party.lingolix (re-implemented report.py rules of v2026.10):
word ROC AUC of 1 - word score, FA@80 (share of correct words at or above the error-signal
threshold that catches 80 % of the changed words), per-bucket AUC, and the system's own flag
(recall / false alarms). Intervals: 1000 bootstrap draws over speakers (dev; FLEURS clips count
as their own speaker) or over clips (Lingolix, as report.py). --paired adds the paired difference
of two dumps with a one-sided bootstrap p (share of draws in which the second is not better).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict

from bench.round19.devset import WORK
from bench.third_party import lingolix as L


def load_dump(name: str) -> dict[str, dict]:
    out = {}
    for l in (WORK / "dumps" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = r
    return out


def manifest_for(name: str) -> list[dict]:
    from bench.round19.collect import manifest

    return manifest("lingolix" if name.startswith("lingolix") else "dev")


MAIN_RULES = False      # Lingolix: the later "main" rule (unverified words dropped as negatives)


def _unverified():
    return {it["id"]: it.get("unverified", []) for it in L.load_manifest("repo_main_f8750aa", "tier1/manifest_de.jsonl")}


def units(items, dump):
    c = L.collect(items, dump, exclude_unverified=_unverified() if MAIN_RULES else None)
    return c["units"]


def _metrics(us) -> dict:
    labels = [l for u in us for l in u[0]]
    scores = [s for u in us for s in u[1]]
    flags = [f for u in us for f in u[3]]
    neg = [s for s, l in zip(scores, labels) if not l]
    out = {"auc": L.roc_auc(labels, scores), "fa80": L.fpr_at_recall(labels, scores, 0.8),
           "n_pos": sum(labels), "n_neg": len(neg)}
    tp = sum(1 for l, f in zip(labels, flags) if l and f)
    fp = sum(1 for l, f in zip(labels, flags) if not l and f)
    out["flag_recall"] = tp / max(1, sum(labels))
    out["flag_fa"] = fp / max(1, len(neg))
    by = defaultdict(list)
    for ls, ss, it, _ in us:
        b = (it.get("label") or {}).get("bucket") or (it.get("label") or {}).get("category") or "control"
        for l, s in zip(ls, ss):
            if l:
                by[b].append(s)
    out["bucket_auc"] = {b: L.roc_auc([1] * len(v) + [0] * len(neg), v + neg) for b, v in sorted(by.items())}
    return out


def _groups(us, dev: bool):
    g = defaultdict(list)
    for u in us:
        key = u[2].get("speaker") if dev else u[2]["id"]
        g[key].append(u)
    return list(g.values())


def boot(us_list, dev: bool, fn, n=1000, seed=0):
    """Paired bootstrap over speakers / clips of fn(units) for each dump in us_list (same items)."""
    keys = [u[2]["id"] for u in us_list[0]]
    idx = {k: i for i, k in enumerate(keys)}
    grp = defaultdict(list)
    for u in us_list[0]:
        grp[u[2].get("speaker") if dev else u[2]["id"]].append(idx[u[2]["id"]])
    gl = list(grp.values())
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [gl[rng.randrange(len(gl))] for _ in range(len(gl))]
        ii = [i for g in pick for i in g]
        vals.append([fn([us[i] for i in ii]) for us in us_list])
    return vals


def summary(names, fold="all", source=None, paired=False, n_boot=1000):
    dev = not names[0].startswith("lingolix")
    items = manifest_for(names[0])
    if dev:
        items = [it for it in items if (fold == "all" or it["fold"] == fold) and (source is None or it["source"] == source)]
    dumps = [load_dump(n) for n in names]
    ids = set.intersection(*[set(d) for d in dumps])
    items = [it for it in items if it["id"] in ids]
    us_list = [units(items, d) for d in dumps]
    # same item order in every unit list
    for us in us_list:
        us.sort(key=lambda u: u[2]["id"])
    res = {}
    for n, us in zip(names, us_list):
        res[n] = _metrics(us)
    fns = {"auc": lambda us: _metrics_light(us)[0], "fa80": lambda us: _metrics_light(us)[1]}
    if n_boot:
        draws = boot(us_list, dev, lambda us: _metrics_light(us), n=n_boot)
        for j, n in enumerate(names):
            for k, m in (("auc", 0), ("fa80", 1)):
                v = sorted(d[j][m] for d in draws if not math.isnan(d[j][m]))
                res[n][f"{k}_ci"] = [v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]]
        if paired and len(names) >= 2:
            for j in range(1, len(names)):
                for k, m, better in (("auc", 0, 1), ("fa80", 1, -1)):
                    dd = sorted(d[j][m] - d[0][m] for d in draws)
                    p = sum(1 for x in dd if x * better <= 0) / len(dd)
                    res[names[j]][f"d_{k}"] = res[names[j]][k] - res[names[0]][k]
                    res[names[j]][f"d_{k}_ci"] = [dd[int(0.025 * len(dd))], dd[int(0.975 * len(dd))]]
                    res[names[j]][f"d_{k}_p"] = p
    del fns
    return res


def _metrics_light(us):
    labels = [l for u in us for l in u[0]]
    scores = [s for u in us for s in u[1]]
    return (L.roc_auc(labels, scores), L.fpr_at_recall(labels, scores, 0.8))


def fmt(res) -> str:
    lines = []
    for n, m in res.items():
        s = (f"{n}: AUC {m['auc']:.4f} {_ci(m.get('auc_ci'))}  FA@80 {100 * m['fa80']:.2f}% "
             f"{_ci(m.get('fa80_ci'), 100)}  pos {m['n_pos']} neg {m['n_neg']}  flag recall "
             f"{m['flag_recall']:.2f} FA {100 * m['flag_fa']:.1f}%")
        if "d_auc" in m:
            s += (f"\n    dAUC {m['d_auc']:+.4f} {_ci(m['d_auc_ci'])} p={m['d_auc_p']:.3f}   dFA80 "
                  f"{100 * m['d_fa80']:+.2f} {_ci(m['d_fa80_ci'], 100)} p={m['d_fa80_p']:.3f}")
        s += "\n    " + "  ".join(f"{b} {v:.3f}" for b, v in m["bucket_auc"].items())
        lines.append(s)
    return "\n".join(lines)


def _ci(ci, f=1.0):
    return "" if not ci else f"[{f * ci[0]:.3f}, {f * ci[1]:.3f}]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--fold", default="all")
    ap.add_argument("--source")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--json")
    ap.add_argument("--main-rules", action="store_true")
    a = ap.parse_args(argv)
    global MAIN_RULES
    MAIN_RULES = a.main_rules
    res = summary(a.names, a.fold, a.source, a.paired, a.boot)
    print(fmt(res))
    if a.json:
        from pathlib import Path

        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
