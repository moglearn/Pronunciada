"""Round-19 Japanese guards, before / after on the same clips (own caches; shared caches untouched).

    python -m bench.round19_ja.guards score --set swap|cvreal --tag before|after
    python -m bench.round19_ja.guards report             # -> results/tables/round19_ja_guards.json

* swap: fleurs-swap TEST, Japanese (bench/textswap.py; metrics of bench.eval_swap: near / far AUC,
  hit@5 %FA, natives below 0.6), paired bootstrap over source utterances.
* cvreal: cv-real Test A, Japanese TEST half (rejected vs accepted clips): AUC of the sentence
  probability and of the mean / lowest word probability, paired speaker bootstrap.
Run "before" with PRONUNCIADA_JA_ROUND19=0 in the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from bench.round19_ja.cv_pool import WORK

OUT = WORK / "guards"
ROOT = Path(__file__).resolve().parents[2]


def _records(which):
    if which == "swap":
        from bench.data import load

        return [r for r in load("fleurs-swap", split="test") if r["lang"] == "ja"]
    from bench.cv_build import load_manifest

    return load_manifest(["ja"], half="test", test="A")


def score(which: str, tag: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import warnings

    warnings.simplefilter("ignore")
    from bench.eval_swap import sentence_result_output
    from pronunciada.scorer import Scorer

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{which}_{tag}.jsonl"
    done = set()
    if p.exists():
        done = {json.loads(x)["id"] for x in open(p, encoding="utf-8")}
    sc = Scorer("ja", threads=2)
    with open(p, "a", encoding="utf-8") as f:
        for k, r in enumerate(_records(which)):
            if r["id"] in done:
                continue
            try:
                res = sc.score(r["audio"], r.get("text") or r.get("sentence"))
                spans = [tuple(w.char_span) for w in res._pronunciada["ref"].words]  # noqa: SLF001
                o = sentence_result_output(res, spans)
                o.update(prob=res.prob, word_probs=[w.prob for w in res.words])
            except ValueError as e:
                o = {"status": f"error: {e}", "words": []}
            f.write(json.dumps({"id": r["id"], **o}, ensure_ascii=False) + "\n")
            if k % 100 == 99:
                f.flush()
                print(which, tag, k + 1, file=sys.stderr, flush=True)


def _load(which, tag):
    return {json.loads(x)["id"]: json.loads(x) for x in open(OUT / f"{which}_{tag}.jsonl", encoding="utf-8")}


def cvreal_table(recs, A, B, n_boot=2000, seed=0):
    from bench.metrics import roc_auc

    rows = [r for r in recs if r["id"] in A and r["id"] in B and A[r["id"]].get("word_probs")
            and B[r["id"]].get("word_probs")]
    y = np.array([1 if r["side"] == "rejected" else 0 for r in rows]) if "side" in rows[0] else \
        np.array([1 if r.get("label") in ("rejected", 1, True) else 0 for r in rows])
    spk = np.array([r.get("client_id") or r.get("speaker") for r in rows])
    out = {"n": len(rows), "n_rejected": int(y.sum())}
    feats = {"sentence_prob": lambda o: o["prob"], "word_mean": lambda o: float(np.mean(o["word_probs"])),
             "word_min": lambda o: float(np.min(o["word_probs"]))}
    rng = np.random.default_rng(seed)
    us = np.unique(spk)
    idx_by = {s: np.flatnonzero(spk == s) for s in us}
    for name, fn in feats.items():
        a = np.array([fn(A[r["id"]]) for r in rows])
        b = np.array([fn(B[r["id"]]) for r in rows])
        aa, bb = roc_auc(y, -a), roc_auc(y, -b)
        D = []
        for _ in range(n_boot):
            ii = np.concatenate([idx_by[s] for s in rng.choice(us, len(us))])
            if 0 < y[ii].sum() < len(ii):
                D.append(roc_auc(y[ii], -b[ii]) - roc_auc(y[ii], -a[ii]))
        D = np.array(D)
        out[name] = {"before": aa, "after": bb, "delta": bb - aa, "ci": [float(np.percentile(D, 2.5)),
                     float(np.percentile(D, 97.5))], "p": float(min(1, 2 * min((D <= 0).mean(), (D >= 0).mean())))}
    return out


def report() -> None:
    from bench.eval_swap import evaluate, paired

    res = {}
    recs = _records("swap")
    A, B = _load("swap", "before"), _load("swap", "after")
    ea, eb = evaluate(recs, A), evaluate(recs, B)
    res["fleurs_swap_ja"] = {"before": ea, "after": eb, "paired": paired(recs, A, B)}
    crecs = _records("cvreal")
    res["cv_real_ja_testA"] = cvreal_table(crecs, _load("cvreal", "before"), _load("cvreal", "after"))
    p = ROOT / "results" / "tables" / "round19_ja_guards.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["cv_real_ja_testA"], indent=1, default=float))
    for k in ("before", "after"):
        print(k, json.dumps(res["fleurs_swap_ja"][k], default=float)[:1500])
    print("paired", json.dumps(res["fleurs_swap_ja"]["paired"], default=float)[:1500])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("--set", default="swap")
    ap.add_argument("--tag", default="after")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        score(a.set, a.tag)
    else:
        report()


if __name__ == "__main__":
    main()
