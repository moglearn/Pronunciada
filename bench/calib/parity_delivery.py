"""Round 5 parity: delivery inputs of the SHIPPED public scorer vs the training cache.

    python -m bench.calib.parity_delivery [--n 200] [--threads 1]

The training inputs ($PRONUNCIADA_CACHE/calib/delivery_speechocean762_train.npz) were produced
by bench/calib/delivery_extract.py through Scorer.score. Here a sample of the same train clips is
scored again with the default public scorer (shipped sentence model and projection) and
compared: every delivery column, the projection outputs (runtime vs the shipped projection
applied to the cached pooled activations) and the sentence-model input vector; round 6: the
word-pooled projection input of the reported word score (runtime vs the training cache
delivery_speechocean762_train_align.pkl).
Writes results/tables/delivery_parity.json.
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    from bench.calib.common import calib_cache, save_json
    from bench.pipeline.common import records
    from pronunciada import delivery as D
    from pronunciada.scorer import Scorer

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    z = np.load(calib_cache() / "delivery_speechocean762_train.npz")
    names = [str(x) for x in z["names"]]
    pos = {u: i for i, u in enumerate(z["utt"])}
    recs = {r["id"]: r for r in records("speechocean762") if r["split"] == "train"}
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(sorted(pos), a.n, replace=False))
    sc = Scorer("en", threads=a.threads)
    proj = sc.sentence_proj
    cols = sc._dcols  # noqa: SLF001
    import pickle

    al = pickle.loads((calib_cache() / "delivery_speechocean762_train_align.pkl").read_bytes())
    wp = sc.word_score_proj
    wacc = []
    acc = {c: [] for c in cols}
    nan_mm = {c: 0 for c in cols}
    for u in pick:
        res = sc.score(recs[u]["audio"], recs[u]["text"])
        c = res._pronunciada  # noqa: SLF001
        i = pos[u]
        if wp is not None:
            P = al[u]["word_pooled"].astype(np.float64)
            zw = (((P - wp["mu"]) / wp["sd"]) @ wp["W"] + wp["b"])[:, 0]
            wacc.append(float(np.max(np.abs(zw - c["W"]["emb_word"]))))
        cached = {k: float(z["D"][i, names.index(k)]) for k in D.ALL}
        if proj is not None:
            pooled = {k: z[f"emb_{k}"][i] for k in proj["layers"]}
            cached.update(D.project(pooled, proj))
        for k in cols:
            x, y = float(c["S"][k][0]), cached[k]
            if np.isnan(x) or np.isnan(y):
                nan_mm[k] += int(np.isnan(x) != np.isnan(y))
                continue
            acc[k].append(abs(x - y))
    out = {"n_utts": len(pick), "threads": a.threads, "sentence_model": sc.sentence_name,
           "columns": {k: {"max_abs": float(max(v)) if v else 0.0,
                           "n_diff_gt_1e-6": int(sum(d > 1e-6 for d in v)), "nan_mismatch": nan_mm[k]}
                       for k, v in acc.items()}}
    out["word_emb_word_max_abs"] = float(max(wacc)) if wacc else None
    worst = sorted(((k, v["max_abs"]) for k, v in out["columns"].items()), key=lambda x: -x[1])[:5]
    out["worst"] = worst
    save_json("delivery_parity.json", out)
    print(len(pick), "utts; word emb max", out["word_emb_word_max_abs"], "; worst:", worst, "nan mismatches:", sum(nan_mm.values()))


if __name__ == "__main__":
    main()
