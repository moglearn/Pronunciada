"""Round 19 (encoder adaptation): paired comparison of out-of-fold sentence-total predictions.

    python -m bench.round19_encoder.report --pairs lora8_e6:frozen_e6,full_e6:frozen_e6 [--dir DIR]

For each pair (variant:control) and each seed present for both: Pearson r of the sentence total, the
difference, and a 95% speaker-bootstrap interval of the difference (2000 resamples of speakers, the
same resample for both) with a two-sided bootstrap p-value; also for children <= 12 and the others.
Reads oof_<name>_seed<k>.npz written by train_cv. Writes results/tables/round19_encoder_paired.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bench.round19_encoder import en_data as RD


def main():
    from bench.calib.common import load_json, save_json
    from bench.round19_encoder import train_cv as TC

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--dir", default="")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else TC.work_dir()
    U = RD.load("train", keep=["speaker", "labels"])
    lab = {u["id"]: (u["speaker"], u["labels"]["age"], u["labels"]["sentence"]["total"]) for u in U}
    out = load_json("round19_encoder_paired.json") or {}
    rng = np.random.default_rng(0)
    for pair in a.pairs.split(","):
        v, c = pair.split(":")
        res = {}
        for seed in range(3):
            pv, pc = d / f"oof_{v}_seed{seed}.npz", d / f"oof_{c}_seed{seed}.npz"
            if not (pv.exists() and pc.exists()):
                continue
            zv, zc = np.load(pv), np.load(pc)
            assert (zv["ids"] == zc["ids"]).all()
            ids = zv["ids"]
            ok = np.array([i in lab for i in ids])
            spk = np.array([lab[i][0] for i in ids[ok]])
            age = np.array([lab[i][1] for i in ids[ok]])
            y = np.array([lab[i][2] for i in ids[ok]])
            sv = np.clip(zv["S"][ok, 3], 0, 10)
            sc = np.clip(zc["S"][ok, 3], 0, 10)
            r = {}
            for grp, m in (("all", np.ones(len(y), bool)), ("child", age <= 12), ("adult", age > 12)):
                rv, rc = RD.pearson(sv[m], y[m]), RD.pearson(sc[m], y[m])
                us = np.unique(spk[m])
                idx = {s: np.flatnonzero((spk == s) & m) for s in us}
                diffs = []
                for _ in range(a.n):
                    j = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
                    diffs.append(RD.pearson(sv[j], y[j]) - RD.pearson(sc[j], y[j]))
                diffs = np.array(diffs)
                p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
                r[grp] = {"r_variant": rv, "r_control": rc, "diff": rv - rc,
                          "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "p": float(p)}
            res[seed] = r
            print(v, "vs", c, "seed", seed, {g: (round(x["diff"], 4), [round(q, 4) for q in x["ci95"]], round(x["p"], 3))
                                              for g, x in r.items()}, flush=True)
        out[pair] = res
    save_json("round19_encoder_paired.json", out)


if __name__ == "__main__":
    main()
