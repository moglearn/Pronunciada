"""Pronunciada on the same manifests and with the same statistic as OpenPronounce's `openpronounce-evaluate`.

    python -m bench.rivals.evaluate_compare

For every manifest of bench.rivals.evaluate_manifests that has a results file: Pearson and Spearman of
each system's score with the manifest's `score`, with the 95 % interval of openpronounce-evaluate
(agreement(): 1,000 resamples of recordings, numpy default_rng(0)), for all rows and per value of the
grouping columns, plus the paired difference Pronunciada - OpenPronounce on the same resamples.
OpenPronounce = the scores openpronounce-evaluate wrote; Pronunciada = its cached outputs
(cv-real: SentenceResult.prob, the scoreboard's primary score; speechocean762: scores["total"]).
Writes results/tables/rivals_evaluate.json.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np

from bench.rivals.openpronounce import WORK

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tables" / "rivals_evaluate.json"


def _pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def _spearman(a, b):
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)  # noqa: E731
    return _pearson(rank(np.asarray(a)), rank(np.asarray(b)))


def agree(h, o, p) -> dict:
    h, o, p = (np.asarray(x, float) for x in (h, o, p))
    out = {"n": int(len(h)), "openpronounce": {"pearson": _pearson(h, o), "spearman": _spearman(h, o)},
           "pronunciada": {"pearson": _pearson(h, p), "spearman": _spearman(h, p)}}
    if len(h) >= 10:
        rng = np.random.default_rng(0)
        vo, vp = [], []
        for _ in range(1000):
            i = rng.integers(0, len(h), len(h))
            vo.append(_pearson(h[i], o[i]))
            vp.append(_pearson(h[i], p[i]))
        vo, vp = np.asarray(vo), np.asarray(vp)
        d = vp - vo
        d = d[np.isfinite(d)]
        out["openpronounce"]["ci"] = [float(x) for x in np.nanquantile(vo, [0.025, 0.975])]
        out["pronunciada"]["ci"] = [float(x) for x in np.nanquantile(vp, [0.025, 0.975])]
        out["delta"] = {"value": out["pronunciada"]["pearson"] - out["openpronounce"]["pearson"],
                        "ci": [float(x) for x in np.quantile(d, [0.025, 0.975])],
                        "p": float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))}
    return out


def main() -> None:
    from bench import eval_cv as EC
    from bench.calib.common import calib_cache

    d = WORK / "evaluate"
    recs = EC.records()
    ours, _ = EC.ours_scores(recs)
    by_audio = {r["audio"]: r["id"] for r in recs}
    so = pickle.loads((calib_cache() / "api" / "speechocean762-test-fp32.pkl").read_bytes())
    res = {}
    for man in sorted(d.glob("*.csv")):
        if man.name.endswith(".results.csv"):
            continue
        rp = man.with_suffix(".results.csv")
        if not rp.exists():
            continue
        with open(rp, newline="", encoding="utf-8") as f:
            R = {os.path.normpath(x["audio"]): x for x in csv.DictReader(f)}
        rows = []
        with open(man, newline="", encoding="utf-8") as f:
            for x in csv.DictReader(f):
                r = R.get(os.path.normpath(x["audio"]))
                if r is None or r.get("error") or not r.get("score"):
                    continue
                if man.name.startswith("speechocean762"):
                    sid = "so762-" + Path(x["audio"]).stem
                    p = so[sid]["sentence"]["total"]
                else:
                    p = ours[by_audio[x["audio"]]][EC.OURS_PRIMARY]
                rows.append((x, float(x["score"]), float(r["score"]), float(p)))
        groups = [c for c in rows[0][0] if c not in ("audio", "score", "text")]
        out = {"all": agree(*zip(*[(h, o, p) for _, h, o, p in rows]))}
        for c in groups:
            for v in sorted({x[c] for x, *_ in rows}):
                sel = [(h, o, p) for x, h, o, p in rows if x[c] == v]
                out[f"{c}={v}"] = agree(*zip(*sel))
        res[man.stem] = out
        print(man.stem, {k: (round(v["pronunciada"]["pearson"], 3), round(v["openpronounce"]["pearson"], 3),
                             round(v.get("delta", {}).get("p", float("nan")), 3)) for k, v in out.items()}, flush=True)
    OUT.write_text(json.dumps({"protocol": __doc__.split("\n\n", 1)[1], "results": res}, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
